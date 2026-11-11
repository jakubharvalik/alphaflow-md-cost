"""Samostudium 38 / kontrola S18 (Gemini 38m): stejně drahé nezávislé krátké MD ze stejných vzorků AF jako C18E.
Pro každý z K cílů ED (oracle38 v13b ens_setup, seed 0, ens_m 0, tedy stejné indexy jako běh C18E) se model ze souboru
AF postaví jako p2_<N> (build_system: CHARMM36m + CHARMM TIP3P, dodekaedr, NaCl, HMR, PME 0,9 nm; parametry z
p2_<N>/meta.json), protonace HIS se převezme z p2_<N> (stejná chemie jako hybrid), minimalizace, --equil-ps ekvilibrace
s polohovými restrainty těžkých atomů proteinu (k = --restrain-k), pak --md-ps volné MD (restrainty vypnuté), polohy
solutu po 1 ps. Každý box má vlastní kontext a seed. Wall-clock včetně stavby a minimalizace se loguje po boxech i celkem.
Výstup: <work>/bKK.npz (hotové boxy se přeskočí; SIGTERM → rozpracovaný box se zahodí, stejný příkaz pokračuje) a po
doběhnutí všech pseudo-hybrid ve formátu oracle38 (torsions.npz, pos.npz, hybrid_summary.json): box k = úsek,
t_after_ps = (k + 1) · 1000 (konec ekvilibrace), snímky t_after_ps + 0 … + md_ps po 1 ps, ens_idx = cíl AF. Polohy jsou
v pořadí atomů p2_<N>/solute.pdb (mapování podle rezidua a jména atomu), takže core36, af38, rel38 a win38 jdou beze změny.
    python s18.py --sys p2_7jfl_C --ens-pdb public/af38/alphaflow_md_templates_base_202402/7jfl_C.pdb \
        --work s18_7jfl_C --out hyb38_7jfl_C_S18_s0 --platform OpenCL
"""
import argparse
import json
import signal
import time
from pathlib import Path

import mdtraj as md
import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as u

from build_system import FORCEFIELDS, peptide_from_pdb
from outcond import torsion_quads, torsions

STOP = [False]


class Stop(Exception):
    pass


def ens_models(path):
    """Řádky ATOM/HETATM jednotlivých modelů PDB s více modely (AlphaFlow píše „MODEL 0“)."""
    models, cur = [], []
    for ln in Path(path).read_text().splitlines():
        if ln.startswith("MODEL"):
            cur = []
        elif ln.startswith(("ATOM", "HETATM")):
            cur.append(ln)
        elif ln.startswith("ENDMDL"):
            models.append(cur)
            cur = []
    if cur:
        models.append(cur)
    return models


def ed_targets(ens_pdb, K, trim, pcs):
    """Cíle ED přesně jako oracle38 v13b v běhu C18E (--ens-select ed, seed 0 = start v medoidu, ens_m 0)."""
    import oracle38
    tr = md.load(str(ens_pdb))
    Y = tr.xyz[:, tr.topology.select("name CA")].astype(np.float64)
    ns = argparse.Namespace(ens_trim=trim, ens_m=0, seed=0, ens_lam=1.0, ens_cap=0.0, ens_noise="ou", ens_ends="drift",
                            ens_once=False, ens_once_target="medoid", ens_strat=K, ens_select="ed", ens_ed_pcs=pcs)
    return [int(i) for i in oracle38.ens_setup(Y, ns)["strat"]]


def his_variants(top):
    """Protonace HIS z topologie solutu (OpenMM): HID / HIE / HIP, jinak None (standardní pravidla)."""
    out = []
    for r in top.residues():
        names = {a.name for a in r.atoms()}
        if r.name in ("HIS", "HID", "HIE", "HIP"):
            out.append("HIP" if {"HD1", "HE2"} <= names else ("HID" if "HD1" in names else "HIE"))
        else:
            out.append(None)
    return out


def kabsch_rmsd(P, Q):
    P, Q = P - P.mean(0), Q - Q.mean(0)
    V, S, Wt = np.linalg.svd(P.T @ Q)
    S[-1] *= np.sign(np.linalg.det(V @ Wt))
    return float(np.sqrt(max(((P ** 2).sum() + (Q ** 2).sum() - 2 * S.sum()) / len(P), 0.0)))


def run_box(k, idx, lines, args, meta, ref_top, variants, work, seed):
    """Postaví box ze vzorku AF, zekvilibruje s restrainty a pustí volnou MD. Vrací (polohy v pořadí p2, info)."""
    w = {}
    t0 = time.time()
    src = work / f"b{k:02d}_model.pdb"
    src.write_text("\n".join(lines) + "\nEND\n")
    raw = work / f"b{k:02d}_raw.pdb"
    seq = peptide_from_pdb(str(src), None, raw)
    if seq != meta["seq"]:
        raise ValueError(f"sekvence vzorku {idx} ({seq}) ≠ p2 ({meta['seq']})")
    ff = app.ForceField(*FORCEFIELDS["charmm36m"][0])
    pdb = app.PDBFile(str(raw))
    ca_af = np.array([p.value_in_unit(u.nanometer) for a, p in zip(pdb.topology.atoms(), pdb.positions) if a.name == "CA"])
    mod = app.Modeller(pdb.topology, pdb.positions)
    mod.addHydrogens(ff, pH=7.0, variants=variants)
    n_sol = mod.topology.getNumAtoms()
    sol_pdb = work / f"b{k:02d}_solute.pdb"
    with open(sol_pdb, "w") as f:
        app.PDBFile.writeFile(mod.topology, mod.positions, f)
    # mapování na atomy p2 (rezidua a jména, stejná cesta zápis/čtení PDB jako p2)
    top_b = md.load(str(sol_pdb)).topology
    key = {(a.residue.index, a.name): a.index for a in top_b.atoms}
    ca_b = {a.residue.index: a.index for a in top_b.atoms if a.name == "CA"}
    imap, miss_h, miss_heavy = [], 0, []
    for a in ref_top.atoms:
        j = key.get((a.residue.index, a.name))
        if j is None:
            if a.element.symbol != "H":
                miss_heavy.append(f"{a.residue}{a.name}")
            miss_h += 1
            j = ca_b[a.residue.index]
        imap.append(j)
    if miss_heavy:
        raise ValueError(f"těžké atomy p2 chybí ve vzorku {idx}: {miss_heavy[:5]}")
    imap = np.array(imap)
    ca_sol = np.array([a.index for a in top_b.atoms if a.name == "CA"])
    mod.addSolvent(ff, padding=meta["padding_nm"] * u.nanometer, boxShape="dodecahedron", neutralize=True,
                   ionicStrength=meta["ionic_molar"] * u.molar)
    system = ff.createSystem(mod.topology, nonbondedMethod=app.PME, nonbondedCutoff=0.9 * u.nanometer,
                             constraints=app.HBonds, hydrogenMass=meta["hmr_amu"] * u.amu)
    temp = meta["temperature_K"] * u.kelvin
    system.addForce(mm.MonteCarloBarostat(1.0 * u.bar, temp))
    posres = mm.CustomExternalForce("0.5*k_pos*periodicdistance(x, y, z, x0, y0, z0)^2")
    posres.addGlobalParameter("k_pos", args.restrain_k)
    for p in ("x0", "y0", "z0"):
        posres.addPerParticleParameter(p)
    for atom in list(mod.topology.atoms())[:n_sol]:
        if atom.element is not None and atom.element.symbol != "H":
            posres.addParticle(atom.index, mod.positions[atom.index].value_in_unit(u.nanometer))
    system.addForce(posres)
    dt_fs = float(meta["dt_fs"])
    integ = mm.LangevinMiddleIntegrator(temp, meta.get("friction_per_ps", 1.0) / u.picosecond, dt_fs * u.femtoseconds)
    integ.setRandomNumberSeed(seed)
    plat = mm.Platform.getPlatformByName(args.platform) if args.platform else None
    ctx = mm.Context(system, integ, plat) if plat else mm.Context(system, integ)
    ctx.setPositions(mod.positions)
    w["build"] = time.time() - t0
    t1 = time.time()
    mm.LocalEnergyMinimizer.minimize(ctx, 10.0, 0)
    w["min"] = time.time() - t1
    ctx.setVelocitiesToTemperature(temp, seed)
    spp = int(round(1000 / dt_fs))
    t1 = time.time()
    for _ in range(int(round(args.equil_ps))):
        integ.step(spp)
        if STOP[0]:
            raise Stop()
    w["equil"] = time.time() - t1
    ctx.setParameter("k_pos", 0.0)

    def frame():
        X = ctx.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(u.nanometer)[:n_sol]
        if not np.isfinite(X).all():
            raise ValueError("NaN v MD")
        return np.asarray(X)

    X = frame()
    dev_eq = kabsch_rmsd(X[ca_sol], ca_af)
    pos = [X[imap].astype(np.float32)]
    t1 = time.time()
    for _ in range(int(round(args.md_ps))):
        integ.step(spp)
        pos.append(frame()[imap].astype(np.float32))
        if STOP[0]:
            raise Stop()
    w["md"] = time.time() - t1
    w["total"] = time.time() - t0
    X = pos[-1].astype(np.float64)
    info = dict(box=k, ens_idx=int(idx), seed=seed, n_atoms=system.getNumParticles(), n_solute=n_sol,
                n_water=sum(1 for r in mod.topology.residues() if r.name == "HOH"), missing_h_mapped_to_ca=miss_h,
                ca_dev_after_equil=round(dev_eq, 4), platform=ctx.getPlatform().getName(),
                wall_s={kk: round(v, 1) for kk, v in w.items()})
    return np.stack(pos), info


def assemble(args, targets, work, out, Q, failed):
    T, P, A, recs, wall, missing = [], [], [], [], {}, []
    for k, idx in enumerate(targets):
        f = work / f"b{k:02d}.npz"
        if not f.exists():
            missing.append(k)
            continue
        d = np.load(f, allow_pickle=False)
        info = json.loads(str(d["info"]))
        J = (k + 1) * 1000.0
        t = J + np.arange(len(d["pos"]), dtype=np.float64)
        T.append(t)
        P.append(d["pos"])
        A.append(torsions(d["pos"].astype(np.float64), Q))
        rec = dict(ok=True, t_after_ps=J, t_before_ps=J, ens_idx=int(idx), s18_box=k)
        rec.update({kk: v for kk, v in info.items() if kk not in ("box", "ens_idx")})
        recs.append(rec)
        for kk, v in info["wall_s"].items():
            wall[kk] = wall.get(kk, 0.0) + v
    if not recs:
        raise SystemExit("žádný hotový box")
    partial = len(missing) > args.allow_missing
    out.mkdir(parents=True, exist_ok=True)
    Tc = np.concatenate(T)
    np.savez_compressed(out / "torsions.npz", times=Tc, angs=np.concatenate(A)[None].astype(np.float32))
    np.savez_compressed(out / "pos.npz", times=Tc, pos=np.concatenate(P)[None])
    md_ns = len(recs) * (args.equil_ps + args.md_ps) / 1000.0
    tot = wall.get("total", 0.0)
    res = dict(args=vars(args), s18=True, partial=partial, targets=targets, missing_boxes=missing, failed=failed,
               wall_s=round(tot, 1), wall_breakdown_s={kk: round(v, 1) for kk, v in wall.items()},
               simulated_ns=md_ns, md_free_ns=len(recs) * args.md_ps / 1000.0,
               effective_ns_per_day=md_ns / tot * 86400 if tot > 0 else None,
               jumps=len(recs), rejected=0, jump_records=recs,
               mean=dict(ca_dev_after_equil=round(float(np.mean([r["ca_dev_after_equil"] for r in recs])), 4)))
    (out / ("hybrid_summary_partial.json" if partial else "hybrid_summary.json")).write_text(
        json.dumps(res, indent=2, ensure_ascii=False))
    print(f"Pseudo-hybrid {out}: {len(recs)}/{len(targets)} boxů, {len(Tc)} snímků, MD {md_ns:g} ns za {tot:.0f} s "
          f"(stavba {wall.get('build', 0):.0f}, minimalizace {wall.get('min', 0):.0f}, ekvilibrace {wall.get('equil', 0):.0f}, "
          f"MD {wall.get('md', 0):.0f} s) = {res['effective_ns_per_day'] or 0:.0f} ns/den; Cα po ekvilibraci od cíle "
          f"{res['mean']['ca_dev_after_equil']:.3f} nm" + (" — NEÚPLNÉ, uloženo jako partial" if partial else ""), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sys", required=True, help="p2_<N>: parametry stavby, protonace HIS a pořadí atomů pro výstup")
    ap.add_argument("--ens-pdb", required=True)
    ap.add_argument("--k", type=int, default=18)
    ap.add_argument("--targets", type=int, nargs="*", default=None, help="indexy vzorků místo výběru ED")
    ap.add_argument("--hyb-check", default=None, help="běh C18E: ověří, že cíle jsou stejné")
    ap.add_argument("--trim", type=int, default=4)
    ap.add_argument("--ed-pcs", type=int, default=10)
    ap.add_argument("--equil-ps", type=float, default=80.0)
    ap.add_argument("--md-ps", type=float, default=520.0)
    ap.add_argument("--restrain-k", type=float, default=1000.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-boxes", type=int, default=0, help="jen zkouška: nejvýš tolik nových boxů (0 = všechny)")
    ap.add_argument("--allow-missing", type=int, default=2, help="kolik selhaných boxů ještě dovolí plný souhrn")
    ap.add_argument("--platform", default=None)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: STOP.__setitem__(0, True))
    sysdir, work, out = Path(args.sys), Path(args.work), Path(args.out)
    work.mkdir(parents=True, exist_ok=True)
    meta = json.loads((sysdir / "meta.json").read_text())
    ref_top = md.load(str(sysdir / "solute.pdb")).topology
    variants = his_variants(app.PDBFile(str(sysdir / "solute.pdb")).topology)
    Q = torsion_quads(ref_top)
    models = ens_models(args.ens_pdb)
    targets = args.targets if args.targets else ed_targets(args.ens_pdb, args.k, args.trim, args.ed_pcs)
    print(f"{time.strftime('%T')} S18 {sysdir.name}: {len(targets)} boxů ze {len(models)} vzorků AF, cíle {targets}; "
          f"ekvilibrace {args.equil_ps:g} ps (k {args.restrain_k:g}) + MD {args.md_ps:g} ps; HIS "
          f"{[v for v in variants if v] or 'žádné'}; {meta['forcefield']}, padding {meta['padding_nm']} nm, "
          f"{meta['ionic_molar']} M, HMR {meta['hmr_amu']}, dt {meta['dt_fs']} fs", flush=True)
    if args.hyb_check and Path(args.hyb_check, "hybrid_summary.json").exists():
        used = {j["ens_idx"] for j in json.loads(Path(args.hyb_check, "hybrid_summary.json").read_text())["jump_records"]
                if j.get("ens_idx") is not None}
        print(f"  kontrola proti {args.hyb_check}: cíle běhu {sorted(used)}; shoda množin {used == set(targets)}, "
              f"společných {len(used & set(targets))}", flush=True)
    failed, ran, t_loop = [], 0, time.time()
    for k, idx in enumerate(targets):
        f = work / f"b{k:02d}.npz"
        if f.exists():
            continue
        if args.max_boxes and ran >= args.max_boxes:
            break
        seed = 38000 + 100 * args.seed + k
        try:
            pos, info = run_box(k, idx, models[idx], args, meta, ref_top, variants, work, seed)
        except Stop:
            print(f"{time.strftime('%T')} SIGTERM: box {k} zahozen, hotové boxy zůstávají; stejný příkaz pokračuje",
                  flush=True)
            return
        except (Exception, SystemExit) as e:  # noqa: BLE001 – selhání jednoho boxu nesmí shodit celý běh
            failed.append(dict(box=k, ens_idx=int(idx), error=f"{type(e).__name__}: {e}"))
            (work / f"b{k:02d}_fail.json").write_text(json.dumps(failed[-1], ensure_ascii=False))
            print(f"{time.strftime('%T')} box {k} (vzorek {idx}) SELHAL: {type(e).__name__}: {e}", flush=True)
            continue
        tmp = work / f"b{k:02d}.tmp.npz"
        np.savez(tmp, pos=pos, info=json.dumps(info, ensure_ascii=False))
        tmp.rename(f)
        ran += 1
        el = time.time() - t_loop
        done = sum((work / f"b{j:02d}.npz").exists() for j in range(len(targets)))
        ws = info["wall_s"]
        print(f"{time.strftime('%T')} box {k + 1}/{len(targets)} (vzorek {idx}): {info['n_atoms']} atomů, stavba "
              f"{ws['build']:.0f} s, min {ws['min']:.0f} s, ekv {ws['equil']:.0f} s, MD {ws['md']:.0f} s "
              f"({args.md_ps / 1000 / max(ws['md'], 1e-9) * 86400:.0f} ns/den), Cα po ekvilibraci od cíle "
              f"{info['ca_dev_after_equil']:.3f} nm; hotovo {done}, odhad konce za "
              f"{el / ran * (len(targets) - k - 1) / 60:.0f} min", flush=True)
    assemble(args, targets, work, out, Q, failed)


if __name__ == "__main__":
    main()
