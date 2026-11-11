"""Hybrid MD + Model D: střídá krátkou MD s ML skokem v torzích a měří zkreslení i skutečný čas.

Cyklus jedné trajektorie:
  1. MD --md-ps (torze peptidu se ukládají po 1 ps),
  2. model z historie torzí (t, t−τ, t−2τ) navzorkuje nové φ/ψ/χ1 v čase t+τ (--jump-ps = τ),
  3. „zpětné mapování“: rotace kolem vazeb nastaví nové torze (rotuje se menší část peptidu),
  4. minimalizace s torzními restrainty (voda se přizpůsobí), nové rychlosti, --relax-ps restrained MD,
  5. restrainty vypnout, čas += τ. Pokud energie po skoku není konečná, skok se zamítne (pokračuje MD).
     Když MD po přijatém skoku vybuchne (NaN), návrat na konec předchozího MD úseku a skok se počítá jako zamítnutý.
  Alternativa --backmap steer: místo 3–4 řízená MD (restrainty se plynule posouvají k cíli během --steer-ps).
--jump-ps 0 = čistá MD stejným kódem (srovnání času a kontrola metriky).

Srovnání s referenčními replikami systému (100 replik ze stejného startu): JS Ramachandranova rozdělení
a χ1 po reziduích, populace hrubých stavů v čase; šum = JS mezi náhodnými K replikami a zbytkem.

    python hybrid.py --sys ala5 --model model_d_full --n-traj 10 --total-ps 1000 --md-ps 10 --jump-ps 10
"""
import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as u
import torch

from model_c import js, rama_hist
from model_d import chi_hist, rollout
from model_e import backbone_index, load_model_e, step_e
from outcond import WINDOWS, dihedral, load_system, pop_curve, torsions
from run_replicas import load, make_sim


def ring_size(adj, b, c):
    """Velikost nejmenšího kruhu přes vazbu b–c (0 = vazba není v kruhu)."""
    dist, front = {c: 0}, [c]
    while front:
        nxt = []
        for x in front:
            for y in adj[x]:
                if (x == c and y == b) or y in dist:
                    continue
                dist[y] = dist[x] + 1
                if y == b:
                    return dist[y] + 1
                nxt.append(y)
        front = nxt
    return 0


def rotation_sets(omm_top, n_sol, Q):
    """Pro každou torzi (n, k): (atomy k rotaci, znaménko) – menší strana vazby b–c; None pro malý kruh
    (Pro φ); "ring" pro vazbu ve velkém kruhu přes disulfidický můstek: rotací ji nastavit nelze,
    řízená MD (--backmap steer) ji ale táhne restraintem stejně jako ostatní torze."""
    adj = [[] for _ in range(n_sol)]
    for b in omm_top.bonds():
        i, j = b[0].index, b[1].index
        if i < n_sol and j < n_sol:
            adj[i].append(j)
            adj[j].append(i)
    sets = {}
    for n in range(Q.shape[0]):
        for k in range(3):
            a, b, c, d = Q[n, k]
            if a < 0:
                continue
            seen, stack = {c}, [c]
            while stack:
                x = stack.pop()
                for y in adj[x]:
                    if y not in seen and not (x == c and y == b):
                        seen.add(y)
                        stack.append(y)
            if b in seen:
                sets[(n, k)] = None if ring_size(adj, b, c) <= 8 else "ring"
                continue
            side_c = np.array(sorted(seen))
            side_b = np.array(sorted(set(range(n_sol)) - seen))
            sets[(n, k)] = (side_c, 1.0) if len(side_c) <= len(side_b) else (side_b, -1.0)
    return sets


def set_torsions(X, Q, sets, target):
    """Nastaví torze peptidu rotacemi kolem vazeb; X (N, 3) nm, target (L, 3). Vrací nové X a nastavené torze."""
    X = X.copy()
    done = []
    for (n, k), s in sets.items():
        if s is None or isinstance(s, str) or np.isnan(target[n, k]):
            continue
        atoms, sign = s
        a, b, c, d = Q[n, k]
        delta = target[n, k] - dihedral(X[[a, b, c, d]])
        th = sign * np.arctan2(np.sin(delta), np.cos(delta))
        u_ = X[c] - X[b]
        u_ = u_ / np.linalg.norm(u_)
        v = X[atoms] - X[b]
        X[atoms] = X[b] + v * np.cos(th) + np.cross(u_, v) * np.sin(th) + np.outer(v @ u_, u_) * (1 - np.cos(th))
        done.append((n, k))
    return X, done


def add_restraints(system, Q, sets):
    f = mm.CustomTorsionForce("0.5*k_rest*dth^2; dth = min(dt, 2*pi_c-dt); dt = abs(theta-theta0); "
                              "pi_c = 3.141592653589793")
    f.addGlobalParameter("k_rest", 0.0)
    f.addPerTorsionParameter("theta0")
    index = {}
    for (n, k), s in sets.items():
        if s is not None:
            index[(n, k)] = f.addTorsion(*[int(x) for x in Q[n, k]], [0.0])
    f.setForceGroup(31)
    system.addForce(f)
    return f, index


def add_ca_restraints(system, ca):
    """Posuvné polohové restrainty Cα pro skok v polohách (--pos-jump); k_pos = 0 mimo řízení."""
    f = mm.CustomExternalForce("0.5*k_pos*periodicdistance(x, y, z, x0, y0, z0)^2")
    f.addGlobalParameter("k_pos", 0.0)
    for p in ("x0", "y0", "z0"):
        f.addPerParticleParameter(p)
    idx = [f.addParticle(int(a), [0.0, 0.0, 0.0]) for a in ca]
    f.setForceGroup(30)
    system.addForce(f)
    return f, idx


def anm_modes(Xca, cutoff, n_modes, k_seq=1.0):
    """Nejměkčí módy elastické sítě (ANM, pružiny mezi Cα do cutoff nm, sousedé v řetězci k_seq× tužší,
    aby se nenatahovaly virtuální vazby Cα–Cα): vlastní čísla (n,) a ortonormální módy (3N, n), bez 6 tuhých pohybů."""
    N = len(Xca)
    Hm = np.zeros((3 * N, 3 * N))
    for i in range(N):
        d = Xca[i + 1:] - Xca[i]
        r2 = (d ** 2).sum(1)
        for jj in np.where(r2 < cutoff ** 2)[0]:
            j = i + 1 + jj
            b = -np.outer(d[jj], d[jj]) / r2[jj] * (k_seq if j == i + 1 else 1.0)
            Hm[3 * i:3 * i + 3, 3 * j:3 * j + 3] = b
            Hm[3 * j:3 * j + 3, 3 * i:3 * i + 3] = b
            Hm[3 * i:3 * i + 3, 3 * i:3 * i + 3] -= b
            Hm[3 * j:3 * j + 3, 3 * j:3 * j + 3] -= b
    lam, V = np.linalg.eigh(Hm)
    keep = np.where(lam > 1e-6 * lam.max())[0][:n_modes]  # bez 6 tuhých pohybů
    return lam[keep], V[:, keep]


def anm_displacement(Xca, rng, cutoff, n_modes, rmsd, k_seq=1.0):
    """Náhodný posun Cα podél nejměkčích módů ANM. Koeficienty ~ N(0, 1/λ), celé škálované tak,
    aby střední kvadratické RMSD posunu bylo rmsd (nm)."""
    lam, V = anm_modes(Xca, cutoff, n_modes, k_seq)
    s = rmsd / np.sqrt((1.0 / lam).sum() / len(Xca))
    return (V @ (rng.standard_normal(len(lam)) * s / np.sqrt(lam))).reshape(len(Xca), 3)


def is_ar(ang):
    from analyze import assign
    phi, psi = ang[:, 0], ang[:, 1]
    return np.isfinite(phi) & np.isfinite(psi) & (assign(np.nan_to_num(phi), np.nan_to_num(psi)) == 0)


def ar_runs(h, min_len):
    """Úseky (i, j) po sobě jdoucích True v h dlouhé aspoň min_len (j bez)."""
    L, runs, i = len(h), [], 0
    while i < L:
        if not h[i]:
            i += 1
            continue
        j = i
        while j < L and h[j]:
            j += 1
        if j - i >= min_len:
            runs.append((i, j))
        i = j
    return runs


def helix_segments(ang, min_len):
    """Úseky (i, j) Cα, které tvoří šroubovici: ≥ min_len reziduí po sobě v aR podle aktuálních φ, ψ,
    rozšířené o krajní Cα (vzdálenosti Cα i, i+3 a i, i+4 určují φ, ψ reziduí mezi nimi)."""
    L = len(ang)
    return [(max(0, i - 1), min(L, j + 1)) for i, j in ar_runs(is_ar(ang), min_len)]


def rigid_helices(Xc, tgt, ang, min_len, ref=None):
    """Cíl Cα s tuhými šroubovicemi: každý úsek se v cíli jen posune a natočí (Kabsch tvaru úseku na jeho cíl),
    takže si šroubovice drží tvar; mimo úseky zůstane cíl beze změny. Překrývající se úseky se zpracují postupně.
    ref = None: úseky a tvar podle aktuálních úhlů a poloh.
    ref = (Cα, úhly) startovní struktury: šroubovice ze startu, z konců zkrácené o rezidua, která teď nejsou v aR
    (konec se smí rozplést), dostanou tvar ze startu i s vadami uvnitř (vada se nezakonzervuje, ale opraví);
    aktuální šroubovice mimo ně drží aktuální tvar. Vrací (cíl, počet držených Cα, počet Cα s tvarem ze startu)."""
    L = len(Xc)
    out = np.array(tgt, dtype=np.float64, copy=True)
    held, from_ref = np.zeros(L, bool), np.zeros(L, bool)
    hc = is_ar(ang)
    jobs = []
    if ref is not None:
        core = np.zeros(L, bool)
        for i, j in ar_runs(is_ar(ref[1]), min_len):
            while i < j and not hc[i]:
                i += 1
            while j > i and not hc[j - 1]:
                j -= 1
            if j - i >= min_len:
                jobs.append((max(0, i - 1), min(L, j + 1), ref[0]))
                core[i:j] = True
        for i, j in ar_runs(hc & ~core, min_len):  # nové šroubovice mimo startovní
            jobs.append((max(0, i - 1), min(L, j + 1), Xc))
    else:
        jobs = [(i, j, Xc) for i, j in helix_segments(ang, min_len)]
    for i, j, src in jobs:
        A, B = np.asarray(src[i:j], np.float64), out[i:j]
        ca_, cb = A.mean(0), B.mean(0)
        U, _, Vt = np.linalg.svd((A - ca_).T @ (B - cb))
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
        out[i:j] = (A - ca_) @ R.T + cb
        held[i:j] = True
        if src is not Xc:
            from_ref[i:j] = True
    return out, int(held.sum()), int(from_ref.sum())


def steer(sim, saved, Q, rest, rest_idx, target, args, spp, n_sol, ca_target=None, posres=None):
    """Řízená MD: theta0 restraintů se v --steer-stages krocích posouvá z aktuálních torzí k cíli
    (po kratším oblouku) během --steer-ps; voda se přizpůsobuje průběžně, bez minimalizace a nových rychlostí.
    target None = torze se neřídí; ca_target (n_Cα, 3) = navíc posuvné polohové restrainty Cα k cíli (--k-pos)."""
    X = saved.getPositions(asNumpy=True).value_in_unit(u.nanometer)[:n_sol]
    if target is not None:
        cur = {key: float(dihedral(X[Q[key[0], key[1]]])) for key in rest_idx}
        dlt = {key: (float(np.arctan2(np.sin(target[key] - cur[key]), np.cos(target[key] - cur[key])))
                     if not np.isnan(target[key]) else 0.0) for key in rest_idx}
    if ca_target is not None:
        pres, pidx, ca = posres
        c0 = X[ca]
    n_st = max(1, args.steer_stages)
    steps = max(1, int(round(args.steer_ps * spp / n_st)))
    try:
        if target is not None:
            sim.context.setParameter("k_rest", args.k_rest)
        if ca_target is not None:
            sim.context.setParameter("k_pos", args.k_pos)
        for st in range(1, n_st + 1):
            if target is not None:
                for key, ri in rest_idx.items():
                    rest.setTorsionParameters(ri, *[int(x) for x in Q[key[0], key[1]]], [cur[key] + dlt[key] * st / n_st])
                rest.updateParametersInContext(sim.context)
            if ca_target is not None:
                tgt = c0 + (ca_target - c0) * st / n_st
                for j, pi in enumerate(pidx):
                    pres.setParticleParameters(pi, int(ca[j]), [float(v) for v in tgt[j]])
                pres.updateParametersInContext(sim.context)
            sim.step(steps)
        sim.context.setParameter("k_rest", 0.0)
        if ca_target is not None:
            sim.context.setParameter("k_pos", 0.0)
        e = sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
        return bool(np.isfinite(e)), e
    except Exception:
        sim.context.setParameter("k_rest", 0.0)
        if ca_target is not None:
            sim.context.setParameter("k_pos", 0.0)
        return False, float("nan")


def run_traj(sim, state0, meta, S, sets, rest, rest_idx, net, cfg, dev, args, seed, rng):
    n_sol = meta["n_solute"]
    Q = S["Q"]
    kT = meta["temperature_K"] * u.kelvin
    spp = int(round(1000 / meta["dt_fs"]))  # kroků na ps
    sim.context.setState(state0)
    sim.context.setVelocitiesToTemperature(kT, seed)
    sim.context.setParameter("k_rest", 0.0)
    pos = lambda: sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(u.nanometer)
    X0 = pos()[:n_sol]
    times, angs, poss = [0.0], [torsions(X0, Q)], [X0.astype(np.float32)]
    hist = deque(maxlen=100000)
    hist.append((0.0, angs[0]))
    wall = dict(md=0.0, model=0.0, backmap=0.0, relax=0.0)
    stats = dict(jumps=0, rejected=0, md_nan=0, rest_E_before_min=[], E_after=[], E_md=[], steer_dev_deg=[],
                 ca_dev_nm=[], helix_held=[], helix_ref=[])
    ref0 = (X0[S["posres"][2]], angs[0]) if args.pos_helix_template == "start" and "posres" in S else None
    H, tau = cfg["hist"], args.jump_ps
    t = 0.0
    # stav po posledním úspěšném MD úseku: když MD po přijatém skoku vybuchne (NaN), skok se zamítne
    good, jumped, fails = (sim.context.getState(getPositions=True, getVelocities=True), 1, 0.0), False, 0
    while t < args.total_ps - 1e-9:
        t0 = time.time()
        try:
            for _ in range(int(min(args.md_ps, args.total_ps - t))):
                sim.step(spp)
                t += 1.0
                X = pos()[:n_sol]
                if not np.isfinite(X).all():
                    raise ValueError("NaN v polohách")
                a = torsions(X, Q)
                times.append(t)
                angs.append(a)
                poss.append(X.astype(np.float32))
                hist.append((t, a))
            st_e = sim.context.getState(getEnergy=True)
            e_md = st_e.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
            if not np.isfinite(e_md):
                raise ValueError("energie není konečná")
        except Exception as ex:
            wall["md"] += time.time() - t0
            fails += 1
            if fails > 10:
                raise
            print(f"  MD po skoku selhala v {t:g} ps ({str(ex).splitlines()[0][:80]}), návrat na {good[2]:g} ps",
                  flush=True)
            sim.context.setState(good[0])
            del times[good[1]:], angs[good[1]:], poss[good[1]:]
            t = good[2]
            while hist[-1][0] > t + 1e-9:
                hist.pop()
            stats["md_nan"] += 1
            if jumped:  # vybuchl přijatý skok: počítat jako zamítnutý
                stats["jumps"] -= 1
                stats["rejected"] += 1
                stats["E_after"].pop()
                if args.backmap == "steer":
                    stats["steer_dev_deg"].pop()
                if args.pos_jump != "none":
                    stats["ca_dev_nm"].pop()
            jumped = False
            continue
        stats["E_md"].append(e_md)
        wall["md"] += time.time() - t0
        good, jumped, fails = (sim.context.getState(getPositions=True, getVelocities=True), len(times), t), False, 0
        if tau <= 0 or t + tau > args.total_ps + 1e-9:
            continue
        # skok modelu
        t0 = time.time()
        frames = []
        for k in range(H):
            want = t - k * tau
            frames.append(next((a for (ti, a) in reversed(hist) if ti <= want + 1e-9), hist[0][1]))
        ang_full = np.full(((H - 1) * tau + 1,) + frames[0].shape, np.nan, np.float32)
        for k in range(H):
            ang_full[(H - 1) * tau - k * tau] = frames[k]
        if cfg.get("spatial"):  # Model E: navíc polohy páteře v čase t
            bb = pos()[:n_sol][S["bbi"]][None]
            target = step_e(net, dev, S["types"], [f[None] for f in frames], bb, tau, cfg["pred_chi"], rng)[0]
        else:
            target = rollout(net, dev, [dict(types=S["types"], ang=ang_full)], tau, 2, 1, cfg["pred_chi"],
                             rng)[0, 0, 1]
        wall["model"] += time.time() - t0
        t0 = time.time()
        saved = sim.context.getState(getPositions=True, getVelocities=True)
        if args.backmap == "steer":
            ca_target = None
            if args.pos_jump == "anm":
                Xc = saved.getPositions(asNumpy=True).value_in_unit(u.nanometer)[S["posres"][2]]
                ca_target = Xc + anm_displacement(Xc, rng, args.anm_cutoff, args.anm_modes, args.anm_rmsd,
                                                  args.anm_kseq)
            elif args.pos_jump == "model":  # Model Q: kolektivní posun Cα za τ naučený z ATLAS
                from model_q import kabsch_onto, sample_q
                Xs = saved.getPositions(asNumpy=True).value_in_unit(u.nanometer)[:n_sol]
                Xc = Xs[S["posres"][2]]
                prev = None
                ip = next((i for i in range(len(times) - 1, -1, -1) if times[i] <= t - tau + 1e-9), None)
                if ip is not None:  # posun Cα(t) − Cα(t − τ) po přeložení (směr dosavadního driftu)
                    prev = Xc - kabsch_onto(poss[ip][S["posres"][2]][None].astype(np.float64), Xc[None])[0]
                dq = sample_q(S["qnet"], dev, S["types"], Xs[S["bbi"]], frames[0], tau, prev, rng, args.pos_diag)
                if args.pos_proj > 0:  # jen část vzorku v měkkých módech ANM (fyzikálně dosažitelná, páteř se netrhá)
                    V = anm_modes(Xc, args.anm_cutoff, args.pos_proj, args.anm_kseq)[1]
                    dq = (V @ (V.T @ dq.reshape(-1))).reshape(-1, 3)
                ca_target = Xc + args.pos_scale * dq
            if ca_target is not None and args.pos_helix_rigid > 0:  # šroubovice v cíli Cα jako tuhá tělesa
                ca_target, n_held, n_ref = rigid_helices(Xc, ca_target, frames[0], args.pos_helix_rigid, ref0)
                stats["helix_held"].append(n_held)
                stats["helix_ref"].append(n_ref)
            ok, e = steer(sim, saved, Q, rest, rest_idx, None if args.pos_jump_hold_tors else target, args, spp,
                          n_sol, ca_target, S.get("posres"))
            wall["backmap"] += time.time() - t0
            if not ok:
                sim.context.setState(saved)
                stats["rejected"] += 1
                continue
            stats["jumps"] += 1
            stats["E_after"].append(e)
            jumped = True
            t += tau
            X = pos()[:n_sol]
            a = torsions(X, Q)
            dev_ = np.abs(np.arctan2(np.sin(a - target), np.cos(a - target)))
            stats["steer_dev_deg"].append(float(np.degrees(np.nanmean(dev_))))
            if ca_target is not None:  # odchylka Cα od cíle po řízení (bez přeložení)
                stats["ca_dev_nm"].append(float(np.sqrt(((X[S["posres"][2]] - ca_target) ** 2).sum(1).mean())))
            times.append(t)
            angs.append(a)
            poss.append(X.astype(np.float32))
            hist.append((t, a))
            continue
        X = saved.getPositions(asNumpy=True).value_in_unit(u.nanometer)
        Xp, done = set_torsions(X[:n_sol], Q, sets, target)
        X = X.copy()
        X[:n_sol] = Xp
        for key, ri in rest_idx.items():  # nenastavené torze drží svou současnou hodnotu
            n, k = key
            th0 = target[n, k] if key in done else dihedral(Xp[Q[n, k]])
            rest.setTorsionParameters(ri, *[int(x) for x in Q[n, k]], [float(th0)])
        rest.updateParametersInContext(sim.context)
        sim.context.setPositions(X)
        sim.context.setParameter("k_rest", args.k_rest)
        e_rest = sim.context.getState(getEnergy=True, groups={31}).getPotentialEnergy() \
            .value_in_unit(u.kilojoule_per_mole)
        stats["rest_E_before_min"].append(e_rest)
        if e_rest > 1.0 * len(done):  # torze z rotace musí sedět s OpenMM konvencí
            raise SystemExit(f"Restrainty nesedí s nastavenými torzemi (E = {e_rest:.1f} kJ/mol), končím")
        ok = True
        try:
            sim.minimizeEnergy(maxIterations=args.min_iter)
            wall["backmap"] += time.time() - t0
            t0 = time.time()
            sim.context.setVelocitiesToTemperature(kT, int(rng.integers(1 << 30)))
            sim.step(int(round(args.relax_ps * spp)))
            sim.context.setParameter("k_rest", 0.0)
            e = sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
            ok = np.isfinite(e)
            wall["relax"] += time.time() - t0
        except Exception:
            ok = False
        if not ok:
            sim.context.setParameter("k_rest", 0.0)
            sim.context.setState(saved)
            stats["rejected"] += 1
            continue
        stats["jumps"] += 1
        stats["E_after"].append(e)
        jumped = True
        t += tau
        X = pos()[:n_sol]
        a = torsions(X, Q)
        times.append(t)
        angs.append(a)
        poss.append(X.astype(np.float32))
        hist.append((t, a))
    return np.array(times), np.stack(angs), np.stack(poss), wall, stats


def compare(S, times, angs, n_traj, rng, skip_ps=20):
    """angs (K, T, L, 3) hybrid na společné časové ose times vs. referenční repliky."""
    T = min(r.shape[0] for r in S["reps"])
    reps = np.stack([r[:T] for r in S["reps"]])
    md_t = np.arange(1, T + 1, dtype=float)
    t_max = min(times.max(), md_t.max())
    hsel = (times >= skip_ps) & (times <= t_max)
    dsel = (md_t >= skip_ps) & (md_t <= t_max)
    R, L = reps.shape[0], reps.shape[2]
    draws = [rng.permutation(R) for _ in range(20)]
    names = [str(r) for r in S["top"].residues]
    per = {}
    for n in range(L):
        d = {}
        if not np.isnan(reps[0, 0, n, :2]).any():
            h_md = rama_hist(reps[:, dsel, n, 0].ravel(), reps[:, dsel, n, 1].ravel())
            d["JS_rama"] = js(rama_hist(angs[:, hsel, n, 0].ravel(), angs[:, hsel, n, 1].ravel()), h_md)
            d["JS_rama_noise"] = float(np.median([js(
                rama_hist(reps[p[:n_traj]][:, dsel, n, 0].ravel(), reps[p[:n_traj]][:, dsel, n, 1].ravel()),
                rama_hist(reps[p[n_traj:]][:, dsel, n, 0].ravel(), reps[p[n_traj:]][:, dsel, n, 1].ravel()))
                for p in draws]))
            pm, pd = pop_curve(angs, times, n, WINDOWS), pop_curve(reps, md_t, n, WINDOWS)
            k = min(len(pm), len(pd))
            d["pop_err"] = float(np.abs(pm[:k] - pd[:k]).mean())
            d["pop_err_noise"] = float(np.median([np.abs(pop_curve(reps[p[:n_traj]], md_t, n, WINDOWS)[:k]
                                                         - pd[:k]).mean() for p in draws]))
        if not np.isnan(reps[0, 0, n, 2]):
            h_md = chi_hist(reps[:, dsel, n, 2])
            d["JS_chi"] = js(chi_hist(angs[:, hsel, n, 2]), h_md)
            d["JS_chi_noise"] = float(np.median([js(chi_hist(reps[p[:n_traj]][:, dsel, n, 2]),
                                                    chi_hist(reps[p[n_traj:]][:, dsel, n, 2])) for p in draws]))
        if d:
            per[names[n]] = d
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sys", default="ala5")
    ap.add_argument("--model", default="model_d_full")
    ap.add_argument("--n-traj", type=int, default=10)
    ap.add_argument("--total-ps", type=float, default=1000.0)
    ap.add_argument("--md-ps", type=float, default=10.0)
    ap.add_argument("--jump-ps", type=int, default=10, help="τ skoku modelu v ps (1, 10, 100); 0 = čistá MD")
    ap.add_argument("--relax-ps", type=float, default=1.0)
    ap.add_argument("--min-iter", type=int, default=200)
    ap.add_argument("--k-rest", type=float, default=1000.0, help="kJ/mol/rad²")
    ap.add_argument("--backmap", choices=["rotate", "steer"], default="rotate",
                    help="rotate = rotace + minimalizace + relaxace; steer = řízená MD k cílovým torzím")
    ap.add_argument("--steer-ps", type=float, default=2.0)
    ap.add_argument("--steer-stages", type=int, default=20)
    ap.add_argument("--pos-jump", choices=["none", "anm", "model"], default="none",
                    help="ke skoku torzí přidat řízení Cα k posunu: anm = náhodně podél měkkých módů ANM, "
                         "model = vzorek Modelu Q (--pos-model); jen --backmap steer")
    ap.add_argument("--pos-model", default=None, help="adresář Modelu Q (model_q.py) pro --pos-jump model")
    ap.add_argument("--pos-scale", type=float, default=1.0, help="násobek vzorku Modelu Q")
    ap.add_argument("--pos-diag", type=float, default=1.0, help="násobek nezávislého šumu σε ve vzorku Modelu Q")
    ap.add_argument("--pos-proj", type=int, default=0,
                    help="vzorek Modelu Q promítnout na tolik nejměkčích módů ANM (--anm-cutoff, --anm-kseq); 0 = ne")
    ap.add_argument("--pos-jump-hold-tors", action="store_true", help="při --pos-jump neřídit torze k cíli modelu")
    ap.add_argument("--anm-rmsd", type=float, default=0.08, help="střední RMSD posunu Cα za skok (nm)")
    ap.add_argument("--anm-cutoff", type=float, default=1.5, help="dosah pružin ANM mezi Cα (nm)")
    ap.add_argument("--anm-modes", type=int, default=20)
    ap.add_argument("--anm-kseq", type=float, default=10.0, help="tuhost pružin sousedů v řetězci (× ostatní)")
    ap.add_argument("--k-pos", type=float, default=5000.0, help="polohové restrainty Cα při řízení, kJ/mol/nm²")
    ap.add_argument("--pos-helix-rigid", type=int, default=0,
                    help="při --pos-jump drží šroubovice (≥ N reziduí po sobě v aR, i s krajními Cα) v cíli Cα "
                         "jako tuhá tělesa; 0 = vypnuto")
    ap.add_argument("--pos-helix-template", choices=["current", "start"], default="current",
                    help="tvar tuhé šroubovice: aktuální, nebo ze startovní struktury (jen úseky, které byly "
                         "šroubovicí i na startu)")
    ap.add_argument("--platform", default=None)
    ap.add_argument("--model-device", default="cpu")
    ap.add_argument("--seed", type=int, default=0, help="i počáteční rychlosti (víc běhů po 1 trajektorii)")
    ap.add_argument("--save-pos", action="store_true", help="uložit i polohy peptidu (pos.npz) pro RMSD/kontakty")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    dev = torch.device(args.model_device)
    out = Path(args.out or f"hybrid_{args.sys}_md{args.md_ps:g}_jump{args.jump_ps}")
    out.mkdir(parents=True, exist_ok=True)

    S = load_system(args.sys)
    meta, system, state0, omm_top = load(args.sys)
    sets = rotation_sets(omm_top, meta["n_solute"], S["Q"])
    rest, rest_idx = add_restraints(system, S["Q"], sets)
    if args.pos_jump != "none":
        if args.backmap != "steer":
            raise SystemExit("--pos-jump potřebuje --backmap steer")
        ca = np.array([a.index for a in S["top"].atoms if a.name == "CA"])
        S["posres"] = add_ca_restraints(system, ca) + (ca,)
    if args.pos_jump == "model":
        from model_q import load_model_q
        S["qnet"], qck = load_model_q(args.pos_model, torch.device(args.model_device))
        print(f"Model Q {args.pos_model}: lagy {qck['args']['lags']} ps, K {qck['cfg']['K']}", flush=True)
    sim = make_sim(meta, system, state0, omm_top, seed=1, platform_name=args.platform)
    net, cfg = load_model_e(args.model, dev)
    S["bbi"] = backbone_index(S["top"])
    print(f"{args.sys}: {meta['n_atoms']} atomů, platforma {sim.context.getPlatform().getName()}; "
          f"MD {args.md_ps:g} ps + skok {args.jump_ps} ps, {args.n_traj} × {args.total_ps:g} ps; "
          f"torze k nastavení {sum(s is not None for s in sets.values())}, z toho ve velkém kruhu přes S–S "
          f"(jen řízená MD) {sum(isinstance(s, str) for s in sets.values())}; malý kruh vynechán: "
          f"{[k for k, s in sets.items() if s is None]}", flush=True)
    if args.backmap == "rotate" and any(isinstance(s, str) for s in sets.values()):
        print("Pozor: --backmap rotate torze ve velkém kruhu nenastaví (drží je), použij --backmap steer", flush=True)

    all_t, all_a, all_x, walls, stats_all = None, [], [], [], []
    t_start = time.time()
    for i in range(args.n_traj):
        t0 = time.time()
        times, angs, poss, wall, stats = run_traj(sim, state0, meta, S, sets, rest, rest_idx, net, cfg, dev,
                                            args, 5000 + i + 1000 * args.seed, rng)
        w = time.time() - t0
        all_t = times if all_t is None else all_t
        if len(times) != len(all_t) or not np.allclose(times, all_t):  # zamítnuté skoky posunou osu
            grid = all_t
            idx = np.clip(np.searchsorted(times, grid, side="right") - 1, 0, len(times) - 1)
            angs, poss = angs[idx], poss[idx]
        all_a.append(angs)
        if args.save_pos:
            all_x.append(poss)
        walls.append(dict(wall, total=w))
        stats_all.append(dict(jumps=stats["jumps"], rejected=stats["rejected"], md_nan=stats["md_nan"],
                              rest_E_before_min_max=max(stats["rest_E_before_min"], default=0.0),
                              E_md_mean=float(np.mean(stats["E_md"])),
                              E_after_jump_mean=float(np.mean(stats["E_after"])) if stats["E_after"] else None,
                              steer_dev_deg=float(np.mean(stats["steer_dev_deg"])) if stats["steer_dev_deg"] else None,
                              ca_dev_nm=float(np.mean(stats["ca_dev_nm"])) if stats["ca_dev_nm"] else None,
                              helix_held_ca=float(np.mean(stats["helix_held"])) if stats["helix_held"] else None,
                              helix_ref_ca=float(np.mean(stats["helix_ref"])) if stats["helix_ref"] else None))
        print(f"traj {i}: {w:.1f} s (MD {wall['md']:.1f}, model {wall['model']:.2f}, zpětné mapování+min "
              f"{wall['backmap']:.2f}, relaxace {wall['relax']:.2f}); skoky {stats['jumps']}, zamítnuto "
              f"{stats['rejected']} (z toho NaN v MD {stats['md_nan']}); E MD {stats_all[-1]['E_md_mean']:.0f}, po skoku "
              f"{stats_all[-1]['E_after_jump_mean']} kJ/mol; odchylka od cíle po řízení "
              f"{stats_all[-1]['steer_dev_deg']}°" + (f", Cα od cíle {stats_all[-1]['ca_dev_nm']:.3f} nm"
                                                      if stats_all[-1]['ca_dev_nm'] is not None else "")
              + (f", tuhé šroubovice v průměru {stats_all[-1]['helix_held_ca']:.1f} Cα"
                 f" (tvar ze startu {stats_all[-1]['helix_ref_ca']:.1f})"
                 if stats_all[-1]['helix_held_ca'] is not None else ""), flush=True)
    angs = np.stack(all_a)
    tot = time.time() - t_start
    sim_ns = args.n_traj * args.total_ps / 1000
    per = compare(S, all_t, angs, args.n_traj, rng) if S["reps"] else {}  # bez vlastní MD srovnání jinde
    res = dict(args=vars(args), wall_s=round(tot, 1), simulated_ns=sim_ns,
               effective_ns_per_day=sim_ns / tot * 86400,
               wall_breakdown_s={k: round(sum(w[k] for w in walls), 2) for k in walls[0]},
               jumps=sum(s["jumps"] for s in stats_all), rejected=sum(s["rejected"] for s in stats_all),
               traj_stats=stats_all, per_residue=per)
    np.savez_compressed(out / "torsions.npz", times=all_t, angs=angs)
    if args.save_pos:
        np.savez_compressed(out / "pos.npz", times=all_t, pos=np.stack(all_x))
    (out / "hybrid_summary.json").write_text(json.dumps(res, indent=2))
    print(f"Celkem {tot:.0f} s za {sim_ns:g} ns simulovaného času = {res['effective_ns_per_day']:.0f} ns/den "
          f"(rozpad {res['wall_breakdown_s']}); skoky {res['jumps']}, zamítnuto {res['rejected']}")
    for nm, d in per.items():
        parts = []
        if "JS_rama" in d:
            parts.append(f"JS Rama {d['JS_rama']:.4f} (šum {d['JS_rama_noise']:.4f}), populace v čase "
                         f"{d['pop_err']:.3f} (šum {d['pop_err_noise']:.3f})")
        if "JS_chi" in d:
            parts.append(f"JS χ1 {d['JS_chi']:.4f} (šum {d['JS_chi_noise']:.4f})")
        print(f"   {nm}: " + "; ".join(parts))
    print("Uloženo", out / "hybrid_summary.json")


if __name__ == "__main__":
    main()
