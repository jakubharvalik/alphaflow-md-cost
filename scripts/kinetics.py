"""Ověření kinetiky a stability: hybrid (hybrid.py) vs. referenční repliky MD ze stejného startu.

Vše se počítá na časové mřížce hybridu (MD body po 1 ps + konce skoků); MD se na stejnou mřížku
podvzorkuje, takže mezery po skocích zkreslují obě strany stejně. Šum = rozdíl náhodných K replik
MD vs. zbytek (K = počet trajektorií hybridu), „× šum“ = |hybrid − MD| / medián tohoto rozdílu.

Torze (každé reziduum s φ i ψ):
  - autokorelace ψ (kruhová, e^{iψ}) na lazích uvnitř MD úseků a v násobcích cyklu,
  - rychlosti αR ↔ rozbalený stav (β+PPII) z jader stavů (přechod = vstup do druhého jádra),
  - implikované časy z přechodové matice 4 hrubých stavů na lazích = násobky cyklu.
Sbalení (--fold, potřebuje pos.npz z hybrid.py --save-pos a polohy v rep_*.npz):
  - RMSD Cα vůči referenční struktuře (peptide_raw.pdb = krystal při build_system --pdb), podíl
    nativních kontaktů Q, kontakty dvojic reziduí (--pairs 2-9), podíl sbalených v časových oknech,
    přežití sbaleného stavu (rozbalení = RMSD > --unfold-nm).

    python kinetics.py --sys ala5 --hyb hyb_a hyb_b
    python kinetics.py --sys cln025f --hyb hyb_c --fold --pairs 2-9
"""
import argparse
import json
import warnings
from pathlib import Path

import mdtraj as md
import numpy as np

from analyze import assign
from model_c import js
from outcond import torsion_quads, torsions

BACKBONE = {"N", "CA", "C", "O", "OXT"}


def load_md(sysdir, with_pos):
    """Repliky ze stejného startu → torze (R, T+1, L, 3), index = čas v ps (0 = start)."""
    sysdir = Path(sysdir)
    top = md.load(str(sysdir / "solute.pdb")).topology
    Q = torsion_quads(top)
    n_sol = top.n_atoms
    x0 = md.load(str(sysdir / "start.pdb")).xyz[0, :n_sol]
    files = [f for f in sorted((sysdir / "runs").glob("rep_*.npz")) if ".tmp" not in f.name]
    if not files:
        raise SystemExit(f"Žádné repliky v {sysdir / 'runs'}")
    angs, poss = [], []
    T = None
    for f in files:
        d = np.load(f)
        if abs(float(d["save_ps"]) - 1.0) > 1e-6:
            raise SystemExit(f"{f}: save_ps {float(d['save_ps'])} ≠ 1 ps, mřížka hybridu by neseděla")
        p = d["pos"]
        T = len(p) if T is None else min(T, len(p))
        angs.append(np.concatenate([torsions(x0, Q)[None], torsions(p, Q)]))
        if with_pos:
            poss.append(np.concatenate([x0[None], p]).astype(np.float32))
    angs = np.stack([a[:T + 1] for a in angs])
    pos = np.stack([p[:T + 1] for p in poss]) if with_pos else None
    return top, angs, pos


def load_hyb(d):
    d = Path(d)
    a = np.load(d / "torsions.npz")
    summ = json.loads((d / "hybrid_summary.json").read_text())
    pos = np.load(d / "pos.npz")["pos"] if (d / "pos.npz").exists() else None
    return dict(name=d.name, times=a["times"], angs=a["angs"], pos=pos, args=summ["args"],
                ns_per_day=summ.get("effective_ns_per_day"))


def grid_pairs(ti, lag, skip):
    where = {t: i for i, t in enumerate(ti)}
    i0 = np.array([i for i, t in enumerate(ti) if t >= skip and t + lag in where], int)
    return i0, np.array([where[ti[i] + lag] for i in i0], int)


def psi_acf(psi, pairs, sel):
    """psi (K, T); kruhová autokorelace normovaná na 1 v lagu 0 (střední hodnota z bodů sel)."""
    z = np.exp(1j * psi)
    m = np.abs(z[:, sel].mean()) ** 2
    out = []
    for i0, i1 in pairs:
        c = np.real(np.mean(z[:, i1] * np.conj(z[:, i0]))) if len(i0) else np.nan
        out.append((c - m) / (1 - m) if m < 1 else np.nan)
    return np.array(out)


def core_states(phi, psi):
    """0 = jádro αR, 1 = jádro rozbaleného stavu (β+PPII), −1 = mimo jádra (stupně)."""
    phi, psi = np.degrees(phi), np.degrees(psi)
    s = np.full(phi.shape, -1, np.int8)
    s[(phi < 0) & (psi > -90) & (psi < 20)] = 0
    s[(phi < 0) & ((psi > 100) | (psi < -160))] = 1
    return s


def core_rates(phi, psi, ti, skip):
    """Rychlosti αR→rozb. a rozb.→αR v 1/ns; čas mezi body mřížky patří poslednímu navštívenému jádru."""
    sel = ti >= skip
    s = core_states(phi[:, sel], psi[:, sel])
    dt = np.diff(ti[sel].astype(float))
    T = s.shape[1]
    last = np.maximum.accumulate(np.where(s >= 0, np.arange(T), -1), axis=1)
    cur = np.where(last >= 0, np.take_along_axis(s, np.maximum(last, 0), 1), -1)
    a, b = cur[:, :-1], cur[:, 1:]
    tr = (a >= 0) & (b >= 0) & (a != b)
    n = np.array([(tr & (a == 0)).sum(), (tr & (a == 1)).sum()], float)
    stay = np.array([((a == 0) * dt).sum(), ((a == 1) * dt).sum()])
    with np.errstate(divide="ignore", invalid="ignore"):
        return n / stay * 1000


def its(states, pairs, lags):
    """Implikované časy (ps) dvou nejpomalejších procesů ze symetrizovaných počtů 4 stavů."""
    out = []
    for (i0, i1), lag in zip(pairs, lags):
        C = np.zeros((4, 4))
        if len(i0):
            np.add.at(C, (states[:, i0].ravel(), states[:, i1].ravel()), 1)
        C = C + C.T
        keep = C.sum(1) > 0
        C = C[keep][:, keep]
        r = np.full(2, np.nan)
        if len(C) >= 2:
            ev = np.sort(np.real(np.linalg.eigvals(C / C.sum(1, keepdims=True))))[::-1][1:3]
            for k, e in enumerate(ev):
                if 0 < e < 1:
                    r[k] = -lag / np.log(e)
        out.append(r)
    return np.array(out)


def torsion_metrics(angs, ti, n, acf_pairs, its_pairs, its_lags, skip):
    phi, psi = angs[:, :, n, 0], angs[:, :, n, 1]
    return dict(acf=psi_acf(psi, acf_pairs, ti >= skip), rates=core_rates(phi, psi, ti, skip),
                its=its(assign(phi, psi), its_pairs, its_lags))


def kabsch_rmsd(X, ref):
    """X (..., N, 3), ref (N, 3) → RMSD po optimálním přeložení."""
    Xc = X - X.mean(-2, keepdims=True)
    Rc = ref - ref.mean(0)
    H = np.einsum("...ni,nj->...ij", Xc, Rc)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U @ Vt))
    S[..., -1] *= d
    e = (Xc ** 2).sum((-1, -2)) + (Rc ** 2).sum() - 2 * S.sum(-1)
    return np.sqrt(np.maximum(e, 0) / len(ref))


def fold_setup(sysdir, top, pairs_arg):
    """Reference z peptide_raw.pdb (těžké atomy, krystal) namapovaná na atomy solute.pdb."""
    raw = md.load(str(Path(sysdir) / "peptide_raw.pdb"))
    key = {(a.residue.index, a.name): a.index for a in top.atoms}
    ref_idx, sol_idx = [], []
    for a in raw.topology.atoms:
        k = (a.residue.index, a.name)
        if k in key and a.element.symbol != "H":
            ref_idx.append(a.index)
            sol_idx.append(key[k])
    ref = raw.xyz[0, ref_idx]
    sol_idx = np.array(sol_idx)
    res = np.array([top.atom(i).residue.index for i in sol_idx])
    names = [top.atom(i).name for i in sol_idx]
    ca = np.array([k for k, nm in enumerate(names) if nm == "CA"])
    D = np.linalg.norm(ref[:, None] - ref[None], axis=-1)
    iu = np.triu_indices(len(ref), 1)
    nat = (np.abs(res[iu[0]] - res[iu[1]]) > 3) & (D[iu] < 0.45)
    contacts = (iu[0][nat], iu[1][nat], D[iu][nat])
    pairs = []
    for p in pairs_arg:
        a, b = (int(v) - 1 for v in p.split("-"))
        sa = [k for k in range(len(sol_idx)) if res[k] == a and names[k] not in BACKBONE]
        sb = [k for k in range(len(sol_idx)) if res[k] == b and names[k] not in BACKBONE]
        pairs.append((p, np.array(sa), np.array(sb)))
    return dict(sol_idx=sol_idx, ref=ref, ca=ca, contacts=contacts, pairs=pairs)


def fold_series(pos, F):
    """pos (K, T, n_sol, 3) → RMSD Cα (nm), Q, min. vzdálenost postranních řetězců dvojic (nm)."""
    X = pos[:, :, F["sol_idx"]]
    rmsd = kabsch_rmsd(X[:, :, F["ca"]], F["ref"][F["ca"]])
    i, j, r0 = F["contacts"]
    r = np.linalg.norm(X[:, :, i] - X[:, :, j], axis=-1)
    q = (1 / (1 + np.exp(50 * (r - 1.8 * r0)))).mean(-1)
    pd = {}
    for name, a, b in F["pairs"]:
        pd[name] = np.linalg.norm(X[:, :, a][:, :, :, None] - X[:, :, b][:, :, None], axis=-1).min((-1, -2))
    return rmsd, q, pd


def fold_metrics(series, ti, args):
    rmsd, q, pd = series
    s = np.zeros(rmsd.shape, np.int8)  # 0 sbalený, 1 rozbalený; hystereze mezi prahy
    for k in range(rmsd.shape[1]):
        prev = s[:, k - 1] if k else np.zeros(len(rmsd), np.int8)
        s[:, k] = np.where(rmsd[:, k] < args.fold_nm, 0, np.where(rmsd[:, k] > args.unfold_nm, 1, prev))
    edges = [e for e in args.windows if e <= ti.max()] + [ti.max() + 1]
    folded = np.array([np.mean(s[:, (ti >= a) & (ti < b)] == 0) for a, b in zip(edges[:-1], edges[1:])])
    ever = np.maximum.accumulate(s, axis=1)
    surv = np.array([1 - ever[:, ti <= t].max(1).mean() if (ti <= t).any() else np.nan for t in args.surv_ps])
    sel = ti >= args.skip_ps
    h = lambda v, lo, hi: np.histogram(v[:, sel].ravel(), bins=40, range=(lo, hi))[0].astype(float) + 1e-9
    out = dict(folded_frac=folded, survival=surv, rmsd_mean=np.array([rmsd[:, sel].mean()]),
               q_mean=np.array([q[:, sel].mean()]), h_rmsd=h(rmsd, 0, 1.2), h_q=h(q, 0, 1))
    for name, dmin in pd.items():
        out[f"contact_{name}"] = np.array([np.mean(dmin[:, sel] < 0.45)])
    return out


def compare_metric(fn, hyb, md_all, K, rng, n_noise):
    """fn(pole (k, ...)) → dict polí; vrací hybrid, MD, šum (medián |K vs. zbytek|) a poměr."""
    R = len(md_all)
    h, m = fn(hyb), fn(md_all)
    draws = []
    for _ in range(n_noise):
        p = rng.permutation(R)
        a, b = fn(md_all[p[:K]]), fn(md_all[p[K:]])
        draws.append({k: np.abs(a[k] - b[k]) for k in a})
    out = {}
    for k in h:
        if k.startswith("h_"):
            d_h = js(h[k], m[k])
            noise = float(np.median([js(fn(md_all[p[:K]])[k], fn(md_all[p[K:]])[k])
                                     for p in [rng.permutation(R) for _ in range(min(n_noise, 20))]]))
            out[k.replace("h_", "JS_")] = dict(hyb=d_h, noise=noise, x_noise=d_h / noise if noise > 0 else None)
            continue
        noise = np.nanmedian(np.stack([d[k] for d in draws]), axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.abs(h[k] - m[k]) / noise
        out[k] = dict(hyb=h[k].tolist(), md=m[k].tolist(), noise=noise.tolist(), x_noise=ratio.tolist())
    # nejhorší poměr přes všechny veličiny: hybrid vs. totéž pro MD proti MD (kolik dává samotná náhoda)
    keys = [k for k in h if not k.startswith("h_")]
    nz = {k: np.asarray(out[k]["noise"], float) for k in keys}
    with np.errstate(divide="ignore", invalid="ignore"):
        null = [np.nanmax(np.concatenate([np.ravel(d[k] / nz[k]) for k in keys])) for d in draws]
    out["_worst"] = dict(hyb=worst({k: out[k] for k in keys}), null_median=float(np.nanmedian(null)),
                         null_p95=float(np.nanpercentile(null, 95)))
    return out


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else round(float(o), 5)
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    return o


def worst(d):
    v = np.concatenate([np.ravel(np.array(e["x_noise"], float)) for e in d.values()])
    v = v[np.isfinite(v)]
    return float(v.max()) if len(v) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sys", required=True)
    ap.add_argument("--hyb", nargs="+", required=True, help="adresáře z hybrid.py")
    ap.add_argument("--fold", action="store_true")
    ap.add_argument("--pairs", nargs="*", default=[], help="dvojice reziduí od 1, např. 2-9")
    ap.add_argument("--fold-nm", type=float, default=0.20)
    ap.add_argument("--unfold-nm", type=float, default=0.40)
    ap.add_argument("--windows", type=float, nargs="+", default=[0, 100, 250, 500, 1000, 2000, 3000])
    ap.add_argument("--surv-ps", type=float, nargs="+", default=[100, 250, 500, 1000, 2000])
    ap.add_argument("--skip-ps", type=float, default=100.0, help="vynechat relaxaci ze startu")
    ap.add_argument("--max-cycles", type=int, default=8)
    ap.add_argument("--n-noise", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    warnings.filterwarnings("ignore", category=RuntimeWarning)  # NaN tam, kde chybí přechody nebo stav

    top, md_ang, md_pos = load_md(args.sys, args.fold)
    names = [str(r) for r in top.residues]
    res_ok = [n for n in range(md_ang.shape[2]) if not np.isnan(md_ang[0, 0, n, :2]).any()]
    F = fold_setup(args.sys, top, args.pairs) if args.fold else None
    report = dict(sys=args.sys, md_reps=len(md_ang), md_ps=md_ang.shape[1] - 1, hybrids={})

    for hd in args.hyb:
        H = load_hyb(hd)
        ti = np.round(H["times"]).astype(int)
        keep = ti <= md_ang.shape[1] - 1
        ti, h_ang = ti[keep], H["angs"][:, keep]
        K = len(h_ang)
        cyc = int(round(H["args"]["md_ps"] + H["args"]["jump_ps"]))
        md_g = md_ang[:, ti]
        acf_lags = sorted({1, 2, 5, int(H["args"]["md_ps"]) - 1} - {0}) if H["args"]["jump_ps"] else [1, 2, 5]
        its_lags = [cyc * m for m in (1, 2, 4, 8) if m <= args.max_cycles and cyc * m * 3 <= ti.max()]
        acf_lags += [cyc * m for m in range(1, args.max_cycles + 1) if cyc * m * 3 <= ti.max()]
        acf_lags = sorted(set(acf_lags))
        acf_p = [grid_pairs(ti, lag, args.skip_ps) for lag in acf_lags]
        its_p = [grid_pairs(ti, lag, args.skip_ps) for lag in its_lags]
        rep = dict(K=K, cycle_ps=cyc, ns_per_day=H["ns_per_day"], acf_lags_ps=acf_lags, its_lags_ps=its_lags,
                   residues={})
        # plné rozlišení MD (1 ps) jen pro informaci, jak mřížka hybridu zkresluje rychlosti
        t_full = np.arange(md_ang.shape[1])
        for n in res_ok:
            fn = lambda arr, n=n: torsion_metrics(arr, ti, n, acf_p, its_p, its_lags, args.skip_ps)
            r = compare_metric(fn, h_ang, md_g, K, rng, args.n_noise)
            r["rates_md_full_res"] = core_rates(md_ang[:, :, n, 0], md_ang[:, :, n, 1], t_full, args.skip_ps).tolist()
            rep["residues"][names[n]] = r
        if args.fold:
            if H["pos"] is None:
                raise SystemExit(f"{hd}: chybí pos.npz (spusť hybrid.py se --save-pos)")
            h_ser = fold_series(H["pos"][:, keep], F)
            m_ser = fold_series(md_pos[:, ti], F)
            idx = lambda ser, sel: (ser[0][sel], ser[1][sel], {k: v[sel] for k, v in ser[2].items()})
            ser_all = dict(h=h_ser, m=m_ser)

            def ffn(arr):  # arr = indexy trajektorií se značkou zdroje (aby šel použít compare_metric)
                src = "h" if arr[0] < 0 else "m"
                sel = (-arr - 1) if src == "h" else arr
                return fold_metrics(idx(ser_all[src], sel), ti, args)

            rep["fold"] = compare_metric(ffn, -np.arange(K) - 1, np.arange(len(md_pos)), K, rng, args.n_noise)
            rep["fold"]["n_native_contacts"] = int(len(F["contacts"][0]))
        report["hybrids"][H["name"]] = rep

        print(f"\n== {H['name']}: {K} traj., cyklus {cyc} ps, {H['ns_per_day'] or 0:.0f} ns/den ==")
        print(f"{'reziduum':10s} {'α→rozb hyb/MD 1/ns':>20s} {'rozb→α hyb/MD':>16s} {'ITS1 hyb/MD ps (lag)':>24s} "
              f"{'nejhorší × šum (MD↔MD medián/95 %)':>36s}")
        for nm, r in rep["residues"].items():
            ra, ri = r["rates"], r["its"]
            j = 0
            print(f"{nm:10s} {ra['hyb'][0]:9.2f}/{ra['md'][0]:<9.2f} {ra['hyb'][1]:7.2f}/{ra['md'][1]:<7.2f} "
                  f"{(ri['hyb'][j][0] if ri['hyb'] else float('nan')):9.0f}/{(ri['md'][j][0] if ri['md'] else float('nan')):<6.0f}"
                  f"({its_lags[j] if its_lags else '-'}) {r['_worst']['hyb']:10.1f} ({r['_worst']['null_median']:.1f}/{r['_worst']['null_p95']:.1f})")
        if args.fold:
            f = rep["fold"]
            print(f"sbalený podíl po oknech hyb {np.round(f['folded_frac']['hyb'], 2).tolist()} "
                  f"MD {np.round(f['folded_frac']['md'], 2).tolist()}")
            print(f"přežití sbaleného {args.surv_ps} ps: hyb {np.round(f['survival']['hyb'], 2).tolist()} "
                  f"MD {np.round(f['survival']['md'], 2).tolist()}")
            w = f["_worst"]
            print(f"nejhorší × šum sbalení {w['hyb']:.1f} (MD↔MD {w['null_median']:.1f}/{w['null_p95']:.1f})")
            print(f"RMSD Cα hyb {f['rmsd_mean']['hyb'][0]:.3f} / MD {f['rmsd_mean']['md'][0]:.3f} nm, "
                  f"Q {f['q_mean']['hyb'][0]:.2f} / {f['q_mean']['md'][0]:.2f}, JS RMSD {f['JS_rmsd']['x_noise'] or 0:.1f}× šum"
                  + "".join(f", kontakt {k[8:]} {f[k]['hyb'][0]:.2f} / {f[k]['md'][0]:.2f}"
                            for k in f if k.startswith("contact_")))

    out = Path(args.out or f"kinetics_{Path(args.sys).name}.json")
    out.write_text(json.dumps(clean(report), indent=1))
    print(f"\nUloženo {out}")


if __name__ == "__main__":
    main()
