"""sub40 (prereg39 Z10): opravený přepočet t*, LOO a kontroly stejného rozpočtu MD po externím posudku. Jen čte.
Foldy k = 1..3: reference R_-k = dvě zbylé repliky ATLAS (snímky od --skip-ps). Zarovnání Cα jádra (trim) na krystal,
PCA a TICA (prvních --tica-dim PCA složek, lag --tica-lag-ps jen uvnitř každé souvislé trajektorie) a hrany 2D histogramu
(okraj 25 %, body mimo hrany do krajních binů) se fitují jen na R_-k. JS v bitech (log2), neodmocněná.
Větev MD: prefix vynechané repliky R_k od --skip-ps do t (mřížka sub38). Větev běhu: okno --win po začátku MD každého
platného boxu. Stejné n: n(t) = min(|R_k[skip, t]|, n_běh); replika rovnoměrným krokem v čase s náhodným posunem, běh
stejným počtem snímků z každého boxu (uvnitř boxu také rovnoměrný krok s náhodným posunem); --draws opakování, JS průměr.
t* = první t, kdy průměr JS_MD(t) přes foldy ≤ průměr JS_běh(t) (lineární interpolace rozdílu mezi body mřížky);
citlivost „last“ = t, od kterého MD zůstává ≤ běh; None = nedosaženo do 100 ns (cenzura).
Citlivost (nerozhoduje): hrany --bins2 (15 × 15); energy distance na týchž 2D projekcích bez binování (V-statistika,
reference R_-k; každá strana nejvýš --ed-cap bodů rovnoměrným krokem z téhož výběru, --ed-draws losů).
LOO (exploratorní): bod 100 ns téže křivky (běh < replika při shodném n). (iii): bod 10,9 ns (běh > replika).
Kontrola stejného rozpočtu MD: fold k, projekce a hrany z R_k samotné, reference R_k (od skip); kontrola = snímky
[skip, --ctrl-ps] obou zbylých replik spojené; běh podvzorkovaný na stejný počet (stejně z každého boxu), --draws losů.
    python sub40.py --sys p2_7jfl_C --atlas atlas/cache/7jfl_C --runs S18_s0=runs/hyb39_7jfl_C_S18_s0 --out sub40_7jfl_C.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

GRID_NS = [0.5, 1, 2, 3, 5, 7.5, 10.9, 15, 20, 30, 50, 75, 100]
SPACES = (("jádro", "pca"), ("TICA", "tica"))


def js(h1, h2):
    """Jensenova–Shannonova divergence v bitech (stejně jako core36.js)."""
    p, q = h1 / h1.sum(), h2 / h2.sum()
    m = 0.5 * (p + q)
    kl = lambda a, b: float(np.sum(a[a > 0] * np.log2(a[a > 0] / b[a > 0])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def align(X, ref):
    """Kabsch na pevnou referenci (krystal), stejně jako core36.align."""
    Xc = X - X.mean(1, keepdims=True)
    Rc = ref - ref.mean(0)
    H = np.einsum("kni,nj->kij", Xc, Rc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.ones((len(X), 3))
    D[:, 2] = d
    R = np.einsum("kij,kj,kjl->kil", U, D, Vt)
    return np.einsum("kni,kij->knj", Xc, R) + ref.mean(0)


def fit_proj(flat_list, kind, n_pca, lag):
    """PCA (2 složky) nebo TICA (n_pca PC → 2 TIC) fitované jen na flat_list (stejné výpočty jako core36.Space)."""
    allA = np.concatenate(flat_list)
    mu = allA.mean(0)
    _, S, Vt = np.linalg.svd(allA - mu, full_matrices=False)
    var = S ** 2 / (S ** 2).sum()
    if kind == "pca":
        return mu, Vt[:2].T, dict(var=var[:2].round(3).tolist())
    V = Vt[:n_pca].T
    C0 = np.zeros((n_pca, n_pca))
    Ct = np.zeros((n_pca, n_pca))
    n = 0
    for F in flat_list:
        y = (F - mu) @ V
        if len(y) <= lag:
            continue
        a, b = y[:-lag], y[lag:]
        C0 += a.T @ a + b.T @ b
        Ct += a.T @ b + b.T @ a
        n += 2 * len(a)
    if n == 0:
        raise ValueError("žádná trajektorie delší než lag TICA")
    C0 /= n
    Ct /= n
    ev, U = np.linalg.eigh(C0)
    keep = ev > 1e-10 * ev.max()
    Wh = U[:, keep] / np.sqrt(ev[keep])
    lam, Q = np.linalg.eigh(Wh.T @ Ct @ Wh)
    order = np.argsort(-lam)[:2]
    return mu, V @ (Wh @ Q[:, order]), dict(eig=lam[order].round(4).tolist(), pca_var_used=float(var[:n_pca].sum().round(3)))


def edges(P, bins):
    lo, hi = P.min(0), P.max(0)
    pad = 0.25 * (hi - lo)
    return [np.linspace(lo[i] - pad[i], hi[i] + pad[i], bins + 1) for i in (0, 1)]


def hist(P, ed):
    return np.histogram2d(np.clip(P[:, 0], ed[0][0], ed[0][-1]), np.clip(P[:, 1], ed[1][0], ed[1][-1]), bins=ed)[0].ravel()


def _md(X, Y):
    return float(np.sqrt(((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1)).mean())


def edist(X, Y):
    """Energy distance (V-statistika) dvou 2D množin bodů (jen numpy)."""
    return 2.0 * _md(X, Y) - _md(X, X) - _md(Y, Y)


def stride_pick(N, n, rng):
    """n z N indexů rovnoměrným krokem s náhodným posunem (vše, když n ≥ N)."""
    if n >= N:
        return np.arange(N)
    step = N / n
    return np.minimum(np.floor(rng.uniform(0.0, step) + step * np.arange(n)).astype(int), N - 1)


def box_pick(boxes, n, rng):
    """n indexů snímků běhu se stejným počtem z každého boxu (zbytek náhodně po jednom; plný box se přeskočí)."""
    sizes = np.array([len(b) for b in boxes])
    if n >= sizes.sum():
        return np.concatenate(boxes)
    q = np.zeros(len(boxes), int)
    rem = n
    while rem > 0:
        idx = np.nonzero(q < sizes)[0]
        add = np.full(len(idx), rem // len(idx))
        extra = rem % len(idx)
        if extra:
            add[rng.choice(len(idx), extra, replace=False)] += 1
        add = np.minimum(add, sizes[idx] - q[idx])
        q[idx] += add
        rem -= int(add.sum())
    return np.concatenate([b[stride_pick(len(b), k, rng)] for b, k in zip(boxes, q) if k > 0])


def t_first(md_c, run_c, ts):
    """První t, kde MD ≤ běh (interpolace rozdílu); None, když ani na konci mřížky."""
    d = [a - b for a, b in zip(md_c, run_c)]
    if d[0] <= 0:
        return float(ts[0])
    for i in range(1, len(d)):
        if d[i] <= 0:
            f = d[i - 1] / (d[i - 1] - d[i])
            return float(ts[i - 1] + f * (ts[i] - ts[i - 1]))
    return None


def t_last(md_c, run_c, ts):
    """t, od kterého MD zůstává ≤ běh (interpolace v posledním přechodu); None, když ani na konci ne."""
    d = [a - b for a, b in zip(md_c, run_c)]
    if d[-1] > 0:
        return None
    pos = [i for i, x in enumerate(d) if x > 0]
    if not pos:
        return float(ts[0])
    i = pos[-1] + 1
    f = d[i - 1] / (d[i - 1] - d[i])
    return float(ts[i - 1] + f * (ts[i] - ts[i - 1]))


def analyze(flatA, tA, runs, args, rng, log=print):
    """flatA: 3 pole (snímky od skip, zarovnané Cα jádra, zploštělé); tA: časy těchto snímků (ps);
    runs: {jméno: (zploštělé zarovnané snímky okna, [indexy snímků po boxech])}."""
    nR = len(flatA)
    lag = int(round(args.tica_lag_ps / args.dt_ps))
    grid = [t for t in GRID_NS if t * 1000.0 >= tA[0][0]]
    bins = {"js30": args.bins, "js15": args.bins2}
    nrun = {name: int(sum(len(b) for b in bx)) for name, (_, bx) in runs.items()}
    res = dict(grid_ns=grid, n_run=nrun, boxes={name: len(bx) for name, (_, bx) in runs.items()}, folds=[], runs={}, ctrl={})
    acc = {name: {lab: {m: dict(md=[], run=[]) for m in list(bins) + ["ed"]} for lab, _ in SPACES} for name in runs}
    nn = {name: [] for name in runs}
    ctrl = {name: {lab: dict(ctrl=[], run=[], n=[]) for lab, _ in SPACES} for name in runs}
    for k in range(nR):
        fit = [flatA[j] for j in range(nR) if j != k]
        finfo = dict(fold=k + 1, ref=[j + 1 for j in range(nR) if j != k])
        for lab, kind in SPACES:
            mu, W, info = fit_proj(fit, kind, args.tica_dim, lag)
            finfo[lab] = info
            Pfit = np.concatenate([(F - mu) @ W for F in fit])
            Pk = (flatA[k] - mu) @ W
            eds = {m: edges(Pfit, b) for m, b in bins.items()}
            href = {m: hist(Pfit, e) for m, e in eds.items()}
            for name, (Xr, bx) in runs.items():
                Pr = (Xr - mu) @ W
                cur = {m: dict(md=[], run=[]) for m in list(bins) + ["ed"]}
                ns = []
                for tns in grid:
                    irep = np.nonzero(tA[k] <= tns * 1000.0)[0]
                    n = min(len(irep), nrun[name])
                    ns.append(int(n))
                    v = {m: ([], []) for m in cur}
                    for dr in range(args.draws):
                        pm = Pk[irep[stride_pick(len(irep), n, rng)]]
                        pr = Pr[box_pick(bx, n, rng)]
                        for m, e in eds.items():
                            v[m][0].append(js(hist(pm, e), href[m]))
                            v[m][1].append(js(hist(pr, e), href[m]))
                        if dr < args.ed_draws:
                            c = min(n, args.ed_cap)
                            pref = Pfit[stride_pick(len(Pfit), args.ed_cap, rng)]
                            v["ed"][0].append(edist(pm[stride_pick(n, c, rng)], pref))
                            v["ed"][1].append(edist(pr[stride_pick(n, c, rng)], pref))
                    for m in cur:
                        cur[m]["md"].append(float(np.mean(v[m][0])))
                        cur[m]["run"].append(float(np.mean(v[m][1])))
                for m in cur:
                    acc[name][lab][m]["md"].append(cur[m]["md"])
                    acc[name][lab][m]["run"].append(cur[m]["run"])
                if lab == SPACES[0][0]:
                    nn[name].append(ns)
            # kontrola stejného rozpočtu MD: projekce, hrany a reference jen z R_k
            mu1, W1, _ = fit_proj([flatA[k]], kind, args.tica_dim, lag)
            P1 = (flatA[k] - mu1) @ W1
            e1 = edges(P1, args.bins)
            h1 = hist(P1, e1)
            pc = np.concatenate([(flatA[j][tA[j] <= args.ctrl_ps] - mu1) @ W1 for j in range(nR) if j != k])
            for name, (Xr, bx) in runs.items():
                Pr = (Xr - mu1) @ W1
                n = min(len(pc), nrun[name])
                vc, vr = [], []
                for _ in range(args.draws):
                    vc.append(js(hist(pc[stride_pick(len(pc), n, rng)], e1), h1))
                    vr.append(js(hist(Pr[box_pick(bx, n, rng)], e1), h1))
                ctrl[name][lab]["ctrl"].append(float(np.mean(vc)))
                ctrl[name][lab]["run"].append(float(np.mean(vr)))
                ctrl[name][lab]["n"].append(int(n))
        res["folds"].append(finfo)
        log(f"  fold {k + 1} hotov")
    i100 = grid.index(100) if 100 in grid else len(grid) - 1
    i109 = grid.index(10.9) if 10.9 in grid else None
    for name in runs:
        row = dict(boxes=res["boxes"][name], frames=nrun[name], n_grid=nn[name])
        for lab, _ in SPACES:
            d = {}
            for m, a in acc[name][lab].items():
                md_m = np.mean(a["md"], axis=0).tolist()
                run_m = np.mean(a["run"], axis=0).tolist()
                d[m] = dict(md_by_fold=[[round(x, 5) for x in c] for c in a["md"]],
                            run_by_fold=[[round(x, 5) for x in c] for c in a["run"]],
                            md=md_m, run=run_m, t_first=t_first(md_m, run_m, grid), t_last=t_last(md_m, run_m, grid))
            j30 = d["js30"]
            d["loo100"] = dict(md=j30["md"][i100], run=j30["run"][i100], run_beats=bool(j30["run"][i100] < j30["md"][i100]),
                               by_fold=[[round(a[i100], 4), round(b[i100], 4)] for a, b in zip(j30["md_by_fold"], j30["run_by_fold"])])
            if i109 is not None:
                d["at10_9"] = dict(md=j30["md"][i109], run=j30["run"][i109], run_worse=bool(j30["run"][i109] > j30["md"][i109]))
            c = ctrl[name][lab]
            d["ctrl_2x5_4"] = dict(ctrl_by_fold=c["ctrl"], run_by_fold=c["run"], n=c["n"], ctrl=float(np.mean(c["ctrl"])),
                                   run=float(np.mean(c["run"])), run_closer=bool(np.mean(c["run"]) < np.mean(c["ctrl"])))
            row[lab] = d
        res["runs"][name] = row
    return res


def load_protein(args):
    import mdtraj as md
    from atlas_compare import load_atlas
    from kinetics import fold_setup
    top = md.load(str(Path(args.sys) / "solute.pdb")).topology
    F = fold_setup(args.sys, top, [])
    ref_all, ca_sol = F["ref"][F["ca"]], F["sol_idx"][F["ca"]]
    L = len(ref_all)
    core = np.arange(args.trim, L - args.trim)
    ref = ref_all[core]
    meta, A = load_atlas(args.atlas)
    args.dt_ps = float(meta["dt_ps"])
    flatA, tA = [], []
    for R in A:
        m = R["t"] >= args.skip_ps
        X = np.asarray(R["ca"][m], np.float64)[:, core]
        flatA.append(align(X, ref).reshape(len(X), -1))
        tA.append(np.asarray(R["t"][m], float))
    wa, wb = (float(x) for x in args.win.split(":"))
    rms = lambda F, r: round(float(np.sqrt(((F.reshape(len(F), -1, 3) - r) ** 2).sum(-1).mean())), 4)
    rmsd = dict(ATLAS=[rms(F, ref) for F in flatA])
    runs = {}
    for spec in args.runs:
        name, d = spec.split("=", 1)
        d = Path(d)
        sf = d / "hybrid_summary.json"
        if not sf.exists():
            print(f"  {name}: chybí {sf}, vynechán")
            continue
        summ = json.loads(sf.read_text())
        T = np.load(d / "torsions.npz")["times"].astype(float)
        recs = sorted((j for j in summ["jump_records"] if j.get("ok") and "t_after_ps" in j), key=lambda j: j["t_after_ps"])
        J = np.array([j["t_after_ps"] for j in recs], float)
        jidx = np.searchsorted(J, T + 1e-6, side="right") - 1
        since = np.where(jidx >= 0, T - J[np.clip(jidx, 0, None)], -1.0)
        win = (jidx >= 0) & (since > wa) & (since <= wb)
        pos = np.load(d / "pos.npz")["pos"]
        X = np.asarray(pos[0][:, ca_sol][:, core][win], np.float64)
        del pos
        bw = jidx[win]
        runs[name] = (align(X, ref).reshape(len(X), -1), [np.nonzero(bw == b)[0] for b in np.unique(bw)])
        rmsd[name] = rms(runs[name][0], ref)
        print(f"  {name}: {len(runs[name][1])} boxů, {len(X)} snímků v okně {args.win} ps, Cα jádra od krystalu "
              f"{rmsd[name]:.3f} nm (ATLAS {', '.join(f'{v:.3f}' for v in rmsd['ATLAS'])})")
    return dict(L=int(L), core=[int(core[0]) + 1, int(core[-1]) + 1], n_atlas=[len(f) for f in flatA],
                rmsd_core_nm=rmsd), flatA, tA, runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sys", required=True)
    ap.add_argument("--atlas", required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="jméno=adresář běhu (hybrid_summary.json, torsions.npz, pos.npz)")
    ap.add_argument("--win", default="50:520")
    ap.add_argument("--trim", type=int, default=4)
    ap.add_argument("--tica-lag-ps", type=float, default=5000.0)
    ap.add_argument("--tica-dim", type=int, default=10)
    ap.add_argument("--skip-ps", type=float, default=100.0)
    ap.add_argument("--bins", type=int, default=30)
    ap.add_argument("--bins2", type=int, default=15)
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--ed-draws", type=int, default=5)
    ap.add_argument("--ed-cap", type=int, default=1500)
    ap.add_argument("--ctrl-ps", type=float, default=5400.0)
    ap.add_argument("--seed", type=int, default=40)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    print(f"=== sub40 {args.sys}")
    info, flatA, tA, runs = load_protein(args)
    if not runs:
        raise SystemExit("žádný běh")
    res = analyze(flatA, tA, runs, args, rng)
    res.update(sys=args.sys, atlas=args.atlas, args={k: v for k, v in vars(args).items()}, **info)
    for name, row in res["runs"].items():
        print(f"{name}: {row['boxes']} boxů, {row['frames']} snímků")
        for lab, _ in SPACES:
            d = row[lab]
            f = lambda t: "≥100" if t is None else f"{t:.1f}"
            print(f"  {lab:<5} t* JS30 {f(d['js30']['t_first'])} (last {f(d['js30']['t_last'])}) ns, JS15 "
                  f"{f(d['js15']['t_first'])}, ED {f(d['ed']['t_first'])}; LOO 100 ns běh {d['loo100']['run']:.3f} vs "
                  f"replika {d['loo100']['md']:.3f}; 2×5,4 ns proti R_k: kontrola {d['ctrl_2x5_4']['ctrl']:.3f} vs běh "
                  f"{d['ctrl_2x5_4']['run']:.3f}")
    Path(args.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print("Uloženo", args.out)


if __name__ == "__main__":
    main()
