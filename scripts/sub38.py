"""sub38: kolik MD stojí výsledek S18 a jak se zlepšuje s počtem boxů. Jen čte.
1) Křivka MD: JS jádra PCA a TICA prvních t ns každé repliky ATLAS (od --skip-ps), a) proti sjednocení všech replik
   (jako core36, replika je v referenci sama), b) leave-one-out proti sjednocení ostatních replik (férovější).
2) Běhy (S18 apod.): JS okna --win po začátku MD každého boxu proti stejným referencím, a křivka podle počtu boxů
   (náhodné podmnožiny n boxů, --subsets losů, průměr ± sd).
3) Ekvivalentní čas MD: nejkratší t, při kterém průměr replik dosáhne JS běhu (lineární interpolace mezi body
   mřížky); poměr t / cena běhu (--cost-ns, výchozí 18 × 0,6 ns = 10,8 ns) = zrychlení proti MD z krystalu.
   JS závisí na počtu snímků, proto i varianta se shodným n: v každém bodě t se z běhu náhodně vybere tolik snímků,
   kolik jich má okno repliky (--draws losů), a hledá se první t, kde replika ≤ běh.
    python sub38.py --sys p2_6ro6_A --atlas public/atlas/cache/6ro6_A --runs S18=hyb38_6ro6_A_S18_s0 --out sub38_6ro6_A.json
"""
import argparse
import json
from pathlib import Path

import mdtraj as md
import numpy as np

from atlas_compare import load_atlas
from core36 import Space, js
from kinetics import fold_setup, load_hyb

GRID_NS = [0.5, 1, 2, 3, 5, 7.5, 10.9, 15, 20, 30, 50, 75, 100]


def whist(sp, P):
    return np.histogram2d(np.clip(P[:, 0], sp.ed[0][0], sp.ed[0][-1]), np.clip(P[:, 1], sp.ed[1][0], sp.ed[1][-1]),
                          bins=sp.ed)[0].ravel()


def t_equiv(curve, val):
    """Nejkratší t (ns), kdy průměrná křivka replik klesne na val; None, pokud ji nedosáhne; 0, pokud je pod ní hned."""
    ts = [c[0] for c in curve]
    ys = [c[1] for c in curve]
    if ys[0] <= val:
        return float(ts[0])
    for i in range(1, len(ts)):
        if ys[i] <= val:
            f = (ys[i - 1] - val) / (ys[i - 1] - ys[i])
            return float(ts[i - 1] + f * (ts[i] - ts[i - 1]))
    return None


def t_equiv_matched(rep, run):
    """rep, run: seznamy [t, JS] na stejné mřížce; první t, kde rep ≤ run (interpolace rozdílu)."""
    dlt = [(a[0], a[1] - b[1]) for a, b in zip(rep, run)]
    if dlt[0][1] <= 0:
        return float(dlt[0][0])
    for i in range(1, len(dlt)):
        if dlt[i][1] <= 0:
            f = dlt[i - 1][1] / (dlt[i - 1][1] - dlt[i][1])
            return float(dlt[i - 1][0] + f * (dlt[i][0] - dlt[i - 1][0]))
    return None


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
    ap.add_argument("--subsets", type=int, default=50)
    ap.add_argument("--cost-ns", type=float, default=10.8)
    ap.add_argument("--draws", type=int, default=10)
    ap.add_argument("--nboxes", default="3,6,9,12,15,18,24,30", help="velikosti náhodných podmnožin boxů")
    ap.add_argument("--seed", type=int, default=38)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    wa, wb = (float(x) for x in args.win.split(":"))
    rng = np.random.default_rng(args.seed)
    top = md.load(str(Path(args.sys) / "solute.pdb")).topology
    F = fold_setup(args.sys, top, [])
    ref_all, ca_sol = F["ref"][F["ca"]], F["sol_idx"][F["ca"]]
    L = len(ref_all)
    core = np.arange(args.trim, L - args.trim)
    ref = ref_all[core]
    meta, A = load_atlas(args.atlas)
    dt = float(meta["dt_ps"])
    XA = [np.asarray(R["ca"], np.float64)[:, core] for R in A]
    sp = {"jádro": Space(XA, ref, "pca", 0, 0, args.bins),
          "TICA": Space(XA, ref, "tica", args.tica_dim, int(round(args.tica_lag_ps / dt)), args.bins)}
    nR = len(A)
    # 1) křivky MD
    curves = {k: {"union": [], "loo": []} for k in sp}
    nframes = []
    for k, s in sp.items():
        for tns in GRID_NS:
            u, lo = [], []
            for r, R in enumerate(A):
                m = (R["t"] >= args.skip_ps) & (R["t"] <= tns * 1000.0)
                if m.sum() < 2:
                    continue
                if k == "jádro" and r == 0:
                    nframes.append([tns, int(m.sum())])
                h = whist(s, s.PA[r][m])
                u.append(js(h, s.hA))
                lo.append(js(h, sum(s.hR[j] for j in range(nR) if j != r)))
            if u:
                curves[k]["union"].append([tns, float(np.mean(u)), float(np.min(u)), float(np.max(u))])
                curves[k]["loo"].append([tns, float(np.mean(lo)), float(np.min(lo)), float(np.max(lo))])
    out = dict(sys=args.sys, win=args.win, cost_ns=args.cost_ns, curves=curves, nframes=nframes, runs={})
    print(f"=== {args.sys}: MD z krystalu, průměr {nR} replik ATLAS (jádro / TICA), proti sjednocení | leave-one-out")
    for i, tns in enumerate(GRID_NS[:len(curves['jádro']['union'])]):
        print(f"  {tns:>5g} ns: {curves['jádro']['union'][i][1]:.3f} / {curves['TICA']['union'][i][1]:.3f}   |   "
              f"{curves['jádro']['loo'][i][1]:.3f} / {curves['TICA']['loo'][i][1]:.3f}")
    # 2) běhy
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
        bw = jidx[win]
        ub = np.unique(bw)
        idx = {b: np.nonzero(bw == b)[0] for b in ub}
        row = dict(boxes=int(len(ub)), frames=int(len(X)))
        for k, s in sp.items():
            P = s.proj(X)
            h = whist(s, P)
            ju = js(h, s.hA)
            jl = float(np.mean([js(h, sum(s.hR[j] for j in range(nR) if j != r)) for r in range(nR)]))
            scal = []
            for n in sorted({int(x) for x in args.nboxes.split(",")} | {len(ub)}):
                if n > len(ub):
                    continue
                v = []
                for _ in range(args.subsets if n < len(ub) else 1):
                    pick = rng.choice(ub, n, replace=False)
                    v.append(js(whist(s, P[np.concatenate([idx[b] for b in pick])]), s.hA))
                scal.append([int(n), float(np.mean(v)), float(np.std(v))])
            tu, tl = t_equiv(curves[k]["union"], ju), t_equiv(curves[k]["loo"], jl)
            runc = []
            for tns, nf in nframes:
                v = [js(whist(s, P[rng.choice(len(P), min(nf, len(P)), replace=False)]), s.hA) for _ in range(args.draws)]
                runc.append([tns, float(np.mean(v))])
            tm = t_equiv_matched(curves[k]["union"], runc)
            row[k + "_nmatched"] = dict(curve=runc, t_equiv_ns=tm, speedup=None if tm is None else tm / args.cost_ns)
            row[k] = dict(js_union=float(ju), js_loo=jl, t_equiv_union_ns=tu, t_equiv_loo_ns=tl,
                          speedup_union=None if tu is None else tu / args.cost_ns,
                          speedup_loo=None if tl is None else tl / args.cost_ns, by_boxes=scal)
        out["runs"][name] = row
        print(f"\n{name}: {row['boxes']} boxů, {row['frames']} snímků (okno {args.win} ps), cena {args.cost_ns:g} ns")
        for k in sp:
            r = row[k]
            fmt = lambda t: "nedosáhne do 100 ns" if t is None else f"{t:.1f} ns ({t / args.cost_ns:.1f}×)"
            print(f"  {k:<5} JS sjednocení {r['js_union']:.3f} → MD {fmt(r['t_equiv_union_ns'])};  leave-one-out "
                  f"{r['js_loo']:.3f} → MD {fmt(r['t_equiv_loo_ns'])}")
            rm = row[k + "_nmatched"]
            print(f"        se shodným počtem snímků: MD {fmt(rm['t_equiv_ns'])}")
            print("        podle počtu boxů: " + ", ".join(f"{n}: {m:.3f}±{sd:.3f}" for n, m, sd in r["by_boxes"]))
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("Uloženo", args.out)


if __name__ == "__main__":
    main()
