"""Linie 39 (prereg39 Z4, Z7a, Z7c): doplňková hodnocení běhů S18/K18 na jednom proteinu. Jen čte.
1) Leave-one-out se shodným n: pro každou repliku r reference = sjednocení dvou ostatních (histogramy jako sub38);
   běh (okno --win po konci ekvilibrace každého boxu, všechny snímky) proti 100ns replice r podvzorkované rovnoměrně
   na stejný počet snímků (od --skip-ps). Průměr přes 3 repliky, jádro PCA a TICA (stejné prostory jako sub38).
   Navíc replika r do 10,9 ns (~1080 snímků) proti běhu podvzorkovanému na stejný počet (--draws losů).
2) Kontrola bez AI (Z7a): ATLAS 3 × 3,6 ns z krystalu (sloučené, od --skip-ps) proti sjednocení, a běh náhodně
   podvzorkovaný na stejný počet snímků (--draws losů) proti sjednocení.
    python eval39.py --sys p2_7jfl_C --atlas atlas/cache/7jfl_C --runs S18=hyb39_7jfl_C_S18_s0 --out eval39_7jfl_C.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

from atlas_compare import load_atlas
from core36 import Space, js
from kinetics import fold_setup, load_hyb
import mdtraj as md


def even(n_tot, n):
    return np.unique(np.linspace(0, n_tot - 1, n).round().astype(int)) if n_tot > n else np.arange(n_tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sys", required=True)
    ap.add_argument("--atlas", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--win", default="50:520")
    ap.add_argument("--trim", type=int, default=4)
    ap.add_argument("--tica-lag-ps", type=float, default=5000.0)
    ap.add_argument("--tica-dim", type=int, default=10)
    ap.add_argument("--skip-ps", type=float, default=100.0)
    ap.add_argument("--bins", type=int, default=30)
    ap.add_argument("--t-short-ps", type=float, default=10900.0)
    ap.add_argument("--t-split-ps", type=float, default=3600.0)
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--seed", type=int, default=39)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    wa, wb = (float(x) for x in args.win.split(":"))
    rng = np.random.default_rng(args.seed)
    top = md.load(str(Path(args.sys) / "solute.pdb")).topology
    F = fold_setup(args.sys, top, [])
    ref_all, ca_sol = F["ref"][F["ca"]], F["sol_idx"][F["ca"]]
    L = len(ref_all)
    core = np.arange(args.trim, L - args.trim)
    meta, A = load_atlas(args.atlas)
    dt = float(meta["dt_ps"])
    XA = [np.asarray(R["ca"], np.float64)[:, core] for R in A]
    sp = {"jádro": Space(XA, ref_all[core], "pca", 0, 0, args.bins),
          "TICA": Space(XA, ref_all[core], "tica", args.tica_dim, int(round(args.tica_lag_ps / dt)), args.bins)}
    nR = len(A)
    keep = [np.nonzero(R["t"] >= args.skip_ps)[0] for R in A]
    short = [np.nonzero((R["t"] >= args.skip_ps) & (R["t"] <= args.t_short_ps))[0] for R in A]
    split = [np.nonzero((R["t"] >= args.skip_ps) & (R["t"] <= args.t_split_ps))[0] for R in A]
    out = dict(sys=args.sys, atlas=args.atlas, win=args.win, n_split=int(sum(len(s) for s in split)),
               n_short=[int(len(s)) for s in short], runs={})
    # kontrola bez AI: 3 × 3,6 ns sloučené proti sjednocení
    ctrl = {}
    for k, s in sp.items():
        P3 = np.concatenate([s.PA[r][split[r]] for r in range(nR)])
        ctrl[k] = dict(js_union=float(js(s.hist(P3), s.hA)),
                       md_10_9ns_js_union=float(np.mean([js(s.hist(s.PA[r][short[r]]), s.hA) for r in range(nR)])))
    out["atlas_3x3_6ns"] = ctrl
    for spec in args.runs:
        name, d = spec.split("=", 1)
        H = load_hyb(d)
        T = H["times"].astype(float)
        summ = json.loads((Path(d) / "hybrid_summary.json").read_text())
        recs = sorted((j for j in summ["jump_records"] if j.get("ok") and "t_after_ps" in j), key=lambda j: j["t_after_ps"])
        J = np.array([j["t_after_ps"] for j in recs], float)
        jidx = np.searchsorted(J, T + 1e-6, side="right") - 1
        since = np.where(jidx >= 0, T - J[np.clip(jidx, 0, None)], -1.0)
        win = (jidx >= 0) & (since > wa) & (since <= wb)
        X = np.asarray(H["pos"][0][:, ca_sol], np.float64)[:, core][win]
        n = len(X)
        row = dict(boxes=int(len(np.unique(jidx[win]))), frames=int(n))
        for k, s in sp.items():
            P = s.proj(X)
            loo_run, loo_rep, loo_rep_n, sh_run, sh_rep = [], [], [], [], []
            for r in range(nR):
                href = sum(s.hR[j] for j in range(nR) if j != r)
                loo_run.append(js(s.hist(P), href))
                kr = keep[r][even(len(keep[r]), n)]
                loo_rep.append(js(s.hist(s.PA[r][kr]), href))
                loo_rep_n.append(int(len(kr)))
                m = len(short[r])
                sh_rep.append(js(s.hist(s.PA[r][short[r]]), href))
                sh_run.append(float(np.mean([js(s.hist(P[rng.choice(n, min(m, n), replace=False)]), href)
                                             for _ in range(args.draws)])))
            m3 = out["n_split"]
            row[k] = dict(loo_run=float(np.mean(loo_run)), loo_rep100_matched=float(np.mean(loo_rep)),
                          loo_rep100_n=loo_rep_n, loo_run_by_rep=[round(float(v), 4) for v in loo_run],
                          loo_rep100_by_rep=[round(float(v), 4) for v in loo_rep],
                          loo_rep10_9=float(np.mean(sh_rep)), loo_run_sub_to_10_9=float(np.mean(sh_run)),
                          union_run_sub_to_3x3_6=float(np.mean([js(s.hist(P[rng.choice(n, min(m3, n), replace=False)]), s.hA)
                                                                 for _ in range(args.draws)])),
                          union_atlas_3x3_6=ctrl[k]["js_union"])
            row[k]["run_beats_rep100_loo"] = bool(row[k]["loo_run"] < row[k]["loo_rep100_matched"])
        out["runs"][name] = row
        print(f"{name}: {row['boxes']} boxů, {n} snímků")
        for k in sp:
            r = row[k]
            print(f"  {k:<5} LOO běh {r['loo_run']:.3f} vs 100ns replika (n={n}) {r['loo_rep100_matched']:.3f} "
                  f"→ {'běh lepší' if r['run_beats_rep100_loo'] else 'replika lepší'}; 10,9 ns: replika {r['loo_rep10_9']:.3f} "
                  f"vs běh {r['loo_run_sub_to_10_9']:.3f}; sjednocení: ATLAS 3×3,6 ns {r['union_atlas_3x3_6']:.3f} vs běh "
                  f"{r['union_run_sub_to_3x3_6']:.3f}")
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("Uloženo", args.out)


if __name__ == "__main__":
    main()
