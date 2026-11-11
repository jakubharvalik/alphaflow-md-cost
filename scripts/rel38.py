"""Samostudium 38 (Gemini 38e): jak rychle MD po řízení opraví lokální strukturu (Rama, χ1) u hybridu se skoky.
Snímky po každém přijatém skoku se zařadí podle času od konce řízení (t - t_after_ps), podvzorkují na krok ATLAS
a rozdělí do oken (a, b] ps; okno „0“ = stav hned po řízení. Srovnání ve stejném návrhu vzorkování:
reference = K náhodných bloků ATLAS téže délky (K = počet skoků), --reps opakování → průměr, sd a podíl
referencí, které jsou horší než hybrid (≈ 0,5 = rovnováha, ≈ 0 = hybrid horší než všechny reference).
Jen čte (torsions.npz, hybrid_summary.json, cache ATLAS).
    python rel38.py --atlas public/atlas/cache/7jfl_C --hyb hyb38_7jfl_C_ensC_100ns_s0 --out rel38_7jfl_C_ensC.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

from atlas_compare import load_atlas
from model_c import js, rama_hist
from model_d import chi_hist


def parse_windows(s):
    out = []
    for w in s.split(","):
        if ":" in w:
            a, b = w.split(":")
            out.append((float(a), float(b)))
        else:
            out.append((float(w), float(w)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", required=True)
    ap.add_argument("--hyb", required=True)
    ap.add_argument("--windows", default="0,0:50,50:100,100:200,200:300,300:500,500:1000,0:300,0:500,0:1000")
    ap.add_argument("--skip-ps", type=float, default=100.0)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=38)
    ap.add_argument("--top", type=int, default=5, help="nejhorší rezidua (rozdíl JS hybrid - průměr reference)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    meta, A = load_atlas(args.atlas)
    dt = float(meta["dt_ps"])
    L = A[0]["ang"].shape[1]
    keepA = [R["t"] >= args.skip_ps for R in A]
    angU = np.concatenate([R["ang"][k] for R, k in zip(A, keepA)]).astype(np.float64)
    hR, hC = {}, {}
    for r in range(L):  # stejná pravidla jako af38
        if not (np.isnan(angU[:, r, :2]).all() or np.isnan(angU[0, r, :2]).any()):
            hR[r] = rama_hist(angU[:, r, 0], angU[:, r, 1])
        if not np.isnan(angU[:, r, 2]).all():
            hC[r] = chi_hist(angU[:, r, 2])
    seq = meta.get("seq")

    def per_res(ang):
        ang = ang.astype(np.float64)
        jr = {r: js(rama_hist(ang[:, r, 0], ang[:, r, 1]), h) for r, h in hR.items() if not np.isnan(ang[:, r, :2]).all()}
        jc = {r: js(chi_hist(ang[:, r, 2]), h) for r, h in hC.items() if not np.isnan(ang[:, r, 2]).all()}
        return jr, jc

    d = Path(args.hyb)
    tor = np.load(d / "torsions.npz")
    T = np.asarray(tor["times"], float)
    ang = tor["angs"]
    ang = ang[0] if ang.ndim == 4 else ang
    summ = json.loads((d / "hybrid_summary.json").read_text())
    J = np.array(sorted(j["t_after_ps"] for j in summ["jump_records"] if j.get("ok") and "t_after_ps" in j), float)
    K = len(J)
    if K == 0:
        raise SystemExit("žádný přijatý skok")
    jidx = np.searchsorted(J, T + 1e-6, side="right") - 1
    since = np.where(jidx >= 0, T - J[np.clip(jidx, 0, None)], -1.0)
    seg_len = np.array([since[jidx == k].max() for k in range(K)])
    starts = [np.nonzero(k)[0] for k in keepA]  # povolené indexy startu bloků v každé replice
    lens = np.array([len(s) for s in starts], float)
    res = dict(atlas=args.atlas, hyb=args.hyb, dt_ps=dt, jumps=K, seg_len_ps=seg_len.round(1).tolist(), reps=args.reps,
               n_rama_res=len(hR), n_chi_res=len(hC), windows={})
    print(f"{d.name}: {K} skoků, délky úseků po řízení {seg_len.min():g}–{seg_len.max():g} ps, ATLAS dt {dt:g} ps")
    print(f"{'okno (ps)':<12} {'n':>5} {'Rama':>6} {'ref':>13} {'p≥':>5} {'χ1':>6} {'ref':>13} {'p≥':>5}  nejhorší Rama")
    for a, b in parse_windows(args.windows):
        offs = np.array([0.0]) if a == b == 0 else np.arange(a + dt, b + dt / 2, dt)
        offs = offs[offs <= seg_len.min() + 1e-6]
        if len(offs) == 0:
            continue
        sel = np.nonzero((jidx >= 0) & np.isin(np.round(since, 3), np.round(offs, 3)))[0]
        if len(sel) == 0:
            continue
        jr, jc = per_res(ang[sel])
        steps = np.round(offs / dt).astype(int)
        ref_r, ref_c, ref_rr = [], [], {r: [] for r in hR}
        for _ in range(args.reps):
            blocks = []
            for _k in range(K):
                i = rng.choice(len(A), p=lens / lens.sum())
                s0 = rng.choice(starts[i][: max(1, len(starts[i]) - steps.max())])
                blocks.append(A[i]["ang"][s0 + steps - steps.min()])
            rr, rc = per_res(np.concatenate(blocks))
            ref_r.append(np.mean(list(rr.values())))
            ref_c.append(np.mean(list(rc.values())))
            for r, v in rr.items():
                ref_rr[r].append(v)
        mr, mc = float(np.mean(list(jr.values()))), float(np.mean(list(jc.values())))
        ref_r, ref_c = np.array(ref_r), np.array(ref_c)
        exc = sorted(((v - np.mean(ref_rr[r]), r) for r, v in jr.items()), reverse=True)[: args.top]
        worst = [dict(res=int(r) + 1, name=(seq[r] if seq else None), excess=round(float(e), 3)) for e, r in exc]
        lab = "po řízení" if a == b == 0 else f"{a:g}–{b:g}"
        res["windows"][lab] = dict(n=int(len(sel)), rama=round(mr, 3), rama_ref=round(float(ref_r.mean()), 3),
                                   rama_ref_sd=round(float(ref_r.std()), 3), rama_p=round(float((ref_r >= mr).mean()), 3),
                                   chi1=round(mc, 3), chi1_ref=round(float(ref_c.mean()), 3),
                                   chi1_ref_sd=round(float(ref_c.std()), 3), chi1_p=round(float((ref_c >= mc).mean()), 3),
                                   worst_rama=worst)
        w = res["windows"][lab]
        print(f"{lab:<12} {w['n']:>5} {mr:>6.3f} {w['rama_ref']:>6.3f}±{w['rama_ref_sd']:<6.3f} {w['rama_p']:>5.2f} "
              f"{mc:>6.3f} {w['chi1_ref']:>6.3f}±{w['chi1_ref_sd']:<6.3f} {w['chi1_p']:>5.2f}  "
              + ", ".join(f"{x['name'] or ''}{x['res']} {x['excess']:+.3f}" for x in worst[:3]))
    Path(args.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print("Uloženo", args.out)


if __name__ == "__main__":
    main()
