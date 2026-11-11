"""diag40 (post hoc, exploratorní, NEROZHODUJE): rozklad poklesu LOO mezi původním vyhodnocením (eval39) a Z10 (sub40).

LOO na 100 ns: běh (S18) i vynechaná replika R_k (prefix do 100 ns, stejné n) proti referenci R_-k (dvě zbylé repliky).
Liší se jen to, z čeho se fituje báze (PCA/TICA) a z čeho hrany 2D histogramu:
  A  báze všechny 3 repliky, hrany všechny 3   (= původní eval39)
  B  báze všechny 3,         hrany R_-k
  C  báze R_-k,              hrany všechny 3
  D  báze R_-k,              hrany R_-k        (= Z10, sub40)
Pro každou variantu a prostor: průměr přes foldy JS(běh) a JS(replika), výhra běhu, a podíl snímků běhu a repliky mimo
rozsah projekce reference (body, které se ořežou do krajních binů). Navíc „tic_sep“: pro TICA s bází ze všech 3 rozdíl
středu vynechané repliky a reference na 1. TIC v jednotkách směrodatné odchylky reference (velký = 1. TIC odděluje R_k).
Funkce bere ze sub40 (stejné zarovnání, fit_proj, edges, hist, js, stride_pick, box_pick, load_protein).
    python diag40.py --sys p2_6ro6_A --atlas atlas/cache/6ro6_A --runs S18d50_s0=runs/hyb39_6ro6_A_S18d50_s0 --out diag40_6ro6_A.json
"""
import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

import sub40 as S

VARIANTS = (("A", "all", "all"), ("B", "all", "ref"), ("C", "ref", "all"), ("D", "ref", "ref"))


def outside(P, ed):
    return float(np.mean((P[:, 0] < ed[0][0]) | (P[:, 0] > ed[0][-1]) | (P[:, 1] < ed[1][0]) | (P[:, 1] > ed[1][-1])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sys", required=True)
    ap.add_argument("--atlas", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--allow-partial", action="store_true", help="když chybí hybrid_summary.json, použij hybrid_summary_partial.json")
    ap.add_argument("--win", default="50:520")
    ap.add_argument("--trim", type=int, default=4)
    ap.add_argument("--tica-lag-ps", type=float, default=5000.0)
    ap.add_argument("--tica-dim", type=int, default=10)
    ap.add_argument("--skip-ps", type=float, default=100.0)
    ap.add_argument("--bins", type=int, default=30)
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--seed", type=int, default=40)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    tmp = None
    if args.allow_partial:
        specs = []
        for spec in args.runs:
            name, d = spec.split("=", 1)
            d = Path(d)
            if not (d / "hybrid_summary.json").exists() and (d / "hybrid_summary_partial.json").exists():
                tmp = tmp or Path(tempfile.mkdtemp(prefix="diag40_"))
                t = tmp / name
                t.mkdir()
                for f in ("torsions.npz", "pos.npz"):
                    (t / f).symlink_to((d / f).resolve())
                shutil.copy(d / "hybrid_summary_partial.json", t / "hybrid_summary.json")
                print(f"  {name}: použit hybrid_summary_partial.json (fyzikální selhání, jen diagnostika)")
                d = t
            specs.append(f"{name}={d}")
        args.runs = specs
    print(f"=== diag40 {args.sys}")
    info, flatA, tA, runs = S.load_protein(args)
    nR = len(flatA)
    lag = int(round(args.tica_lag_ps / args.dt_ps))
    res = dict(sys=args.sys, args=vars(args), info=info, runs={})
    for name, (Xr, bx) in runs.items():
        nrun = int(sum(len(b) for b in bx))
        out = {}
        for lab, kind in S.SPACES:
            agg = {v: dict(run=[], md=[], out_run=[], out_md=[]) for v, _, _ in VARIANTS}
            sep = []
            for k in range(nR):
                ref = [flatA[j] for j in range(nR) if j != k]
                fits = {"all": S.fit_proj(list(flatA), kind, args.tica_dim, lag), "ref": S.fit_proj(ref, kind, args.tica_dim, lag)}
                irep = np.nonzero(tA[k] <= 100000.0)[0]
                n = min(len(irep), nrun)
                for v, b, e in VARIANTS:
                    mu, W, _ = fits[b]
                    Pref = np.concatenate([(F - mu) @ W for F in ref])
                    Pall = np.concatenate([(F - mu) @ W for F in flatA])
                    ed = S.edges(Pall if e == "all" else Pref, args.bins)
                    href = S.hist(Pref, ed)
                    Pk = (flatA[k] - mu) @ W
                    Pr = (Xr - mu) @ W
                    jr, jm = [], []
                    for _ in range(args.draws):
                        jm.append(S.js(S.hist(Pk[irep[S.stride_pick(len(irep), n, rng)]], ed), href))
                        jr.append(S.js(S.hist(Pr[S.box_pick(bx, n, rng)], ed), href))
                    agg[v]["run"].append(float(np.mean(jr)))
                    agg[v]["md"].append(float(np.mean(jm)))
                    edr = S.edges(Pref, args.bins)
                    agg[v]["out_run"].append(outside(Pr, edr))
                    agg[v]["out_md"].append(outside(Pk[irep], edr))
                if kind == "tica":
                    mu, W, _ = fits["all"]
                    Pref = np.concatenate([(F - mu) @ W for F in ref])
                    Pk = (flatA[k] - mu) @ W
                    sep.append(round(float(abs(Pk[:, 0].mean() - Pref[:, 0].mean()) / Pref[:, 0].std()), 3))
            d = {}
            for v, b, e in VARIANTS:
                a = agg[v]
                r, m = float(np.mean(a["run"])), float(np.mean(a["md"]))
                d[v] = dict(base=b, edges=e, run=round(r, 4), md=round(m, 4), run_beats=bool(r < m),
                            run_by_fold=[round(x, 4) for x in a["run"]], md_by_fold=[round(x, 4) for x in a["md"]],
                            outside_ref_range_run=round(float(np.mean(a["out_run"])), 4),
                            outside_ref_range_md=round(float(np.mean(a["out_md"])), 4))
            if sep:
                d["tic1_sep_heldout_sd"] = sep
            out[lab] = d
            print(f"  {name} {lab:<5} " + "; ".join(f"{v} běh {d[v]['run']:.3f} vs rep {d[v]['md']:.3f}"
                                                   f"{' ✓' if d[v]['run_beats'] else ''}" for v, _, _ in VARIANTS)
                  + (f"; 1. TIC (báze 3) oddělení R_k {sep} sd" if sep else ""))
        res["runs"][name] = out
    Path(args.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print("Uloženo", args.out)
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
