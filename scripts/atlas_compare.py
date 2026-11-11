"""Srovnání vlastní MD a hybridů s ATLAS (stejný protein i silové pole), i když mají různou délku.

Reference rovnováhy = ATLAS po konci okna (t > --t-max-ps, všechny repliky). Pro každý zdroj:
  - RMSD Cα vůči krystalu (peptide_raw.pdb) v časových oknech od startu a podíl snímků nad --far-nm,
  - JS Ramachandranova rozdělení a χ1 vůči referenci, průměr a nejhorší reziduum.
Měřítko „kolik dává krátká MD“: ATLAS repliky v okně stejné délky od startu (0 … délka zdroje)
vůči téže referenci. ATLAS má snímky po 10 ps, vlastní MD se podvzorkuje na stejný krok,
hybrid bere všechny své snímky (po 1 ps v MD úsecích a konce skoků).

    python atlas_compare.py --sys p_6okd_C --atlas public/atlas/cache/6okd_C --md \
        --hyb hyb14_6okd_C_j100_s2_md10_p1 hyb15_6okd_C_j100_s2_md10_p1_5ns --out cmp15_6okd_C.json
--pool JMÉNO sloučí všechny --hyb (např. samostatné běhy po 1 trajektorii s různým --seed) do jednoho zdroje.
"""
import argparse
import json
import warnings
from pathlib import Path

import mdtraj as md
import numpy as np

from kinetics import fold_setup, kabsch_rmsd, load_hyb, load_md
from model_c import js, rama_hist
from model_d import chi_hist


def load_atlas(cache):
    cache = Path(cache)
    meta = json.loads((cache / "meta.json").read_text())
    reps = []
    for i, _ in enumerate(meta["reps"], 1):
        ang = np.load(cache / f"R{i}_ang.npy").astype(np.float32)
        ca = np.load(cache / f"R{i}_bb.npy", mmap_mode="r")[:, :, 1].astype(np.float32)
        reps.append(dict(t=np.arange(len(ang)) * meta["dt_ps"], ang=ang, ca=ca))
    return meta, reps


def windows_for(t_end, edges):
    e = [x for x in edges if x < t_end] + [t_end]
    return list(zip(e[:-1], e[1:]))


def stats(t, ang, rmsd, ref_h, wins, args):
    """t (N,), ang (N, L, 3), rmsd (N,) jednoho zdroje (sloučené trajektorie) → metriky."""
    sel = t >= args.skip_ps
    rw = [float(rmsd[(t >= a) & (t < b) & sel].mean()) if ((t >= a) & (t < b) & sel).any() else None
          for a, b in wins]
    out = dict(frames=int(sel.sum()), t_end_ps=float(t.max()), rmsd_windows=rw,
               rmsd_mean=float(rmsd[sel].mean()), far_frac=float((rmsd[sel] > args.far_nm).mean()))
    jr, jc = {}, {}
    for n, (hr, hc) in ref_h.items():
        if hr is not None:
            jr[n] = js(rama_hist(ang[sel, n, 0], ang[sel, n, 1]), hr)
        if hc is not None:
            jc[n] = js(chi_hist(ang[sel, n, 2]), hc)
    out["JS_rama"], out["JS_chi"] = jr, jc
    out["JS_rama_mean"] = float(np.mean(list(jr.values()))) if jr else None
    out["JS_chi_mean"] = float(np.mean(list(jc.values()))) if jc else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sys", required=True, help="adresář systému z build_system.py (peptide_raw.pdb = krystal)")
    ap.add_argument("--atlas", required=True, help="cache/<jméno> z atlas_fetch.py")
    ap.add_argument("--md", action="store_true", help="přidat vlastní repliky MD ze --sys/runs")
    ap.add_argument("--hyb", nargs="*", default=[])
    ap.add_argument("--t-max-ps", type=float, default=10000.0, help="ATLAS do tohoto času = okna od startu")
    ap.add_argument("--skip-ps", type=float, default=100.0)
    ap.add_argument("--far-nm", type=float, default=0.3)
    ap.add_argument("--edges", type=float, nargs="+", default=[0, 500, 1000, 2000, 3000, 5000, 10000])
    ap.add_argument("--pool", default=None, help="sloučit všechny --hyb do jednoho zdroje s tímto jménem")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    top = md.load(str(Path(args.sys) / "solute.pdb")).topology
    F = fold_setup(args.sys, top, [])
    ref_ca = F["ref"][F["ca"]]
    ca_sol = F["sol_idx"][F["ca"]]
    meta, A = load_atlas(args.atlas)
    L = top.n_residues
    if meta["L"] != L or len(ref_ca) != L:
        raise SystemExit(f"Počet reziduí nesedí: systém {L}, Cα v krystalu {len(ref_ca)}, ATLAS {meta['L']}")
    names = [str(r) for r in top.residues]

    # reference rovnováhy: ATLAS po konci okna
    late = [(r["ang"][r["t"] > args.t_max_ps]) for r in A]
    late = np.concatenate(late)
    if not len(late):
        raise SystemExit("ATLAS nemá snímky po --t-max-ps")
    ref_h = {}
    for n in range(L):
        hr = None if np.isnan(late[:, n, :2]).all() or np.isnan(late[0, n, :2]).any() else \
            rama_hist(late[:, n, 0], late[:, n, 1])
        hc = None if np.isnan(late[:, n, 2]).all() else chi_hist(late[:, n, 2])
        ref_h[n] = (hr, hc)
    for r in A:
        r["rmsd"] = kabsch_rmsd(r["ca"][None], ref_ca)[0]
    late_rmsd = np.concatenate([r["rmsd"][r["t"] > args.t_max_ps] for r in A])

    def atlas_window(t_end):
        rows, wins = [], windows_for(t_end, args.edges)
        for r in A:
            m = r["t"] <= t_end
            rows.append(stats(r["t"][m], r["ang"][m], r["rmsd"][m], ref_h, wins, args))
        cat = lambda k: np.concatenate([r[k][r["t"] <= t_end] for r in A])
        return dict(reps=rows, pooled=stats(cat("t"), cat("ang"), cat("rmsd"), ref_h, wins, args))

    report = dict(sys=args.sys, atlas=args.atlas, t_max_ps=args.t_max_ps, skip_ps=args.skip_ps,
                  reference=dict(frames=int(len(late)), rmsd_mean=float(late_rmsd.mean()),
                                 far_frac=float((late_rmsd > args.far_nm).mean())), sources={}, atlas_windows={})
    sources = []
    if args.md:
        _, ang, pos = load_md(args.sys, True)
        step = int(round(meta["dt_ps"]))
        t = np.arange(ang.shape[1], dtype=float)[::step]
        ang, pos = ang[:, ::step], pos[:, ::step]
        rmsd = kabsch_rmsd(pos[:, :, ca_sol], ref_ca)
        K = len(ang)
        sources.append((f"MD {K} × {t.max() / 1000:g} ns", np.tile(t, K), ang.reshape(-1, L, 3), rmsd.ravel()))
    for hd in args.hyb:
        H = load_hyb(hd)
        if H["pos"] is None:
            raise SystemExit(f"{hd}: chybí pos.npz (hybrid.py --save-pos)")
        K = len(H["angs"])
        rmsd = kabsch_rmsd(H["pos"][:, :, ca_sol], ref_ca)
        sources.append((H["name"], np.tile(H["times"], K), H["angs"].reshape(-1, L, 3), rmsd.ravel()))
    if args.pool and args.hyb:
        hs = sources[-len(args.hyb):]
        sources = sources[:-len(args.hyb)] + [(args.pool,) + tuple(np.concatenate([h[i] for h in hs])
                                                                    for i in (1, 2, 3))]

    print(f"ATLAS {meta['name']}: {len(A)} × {A[0]['t'].max() / 1000:g} ns; reference rovnováhy t > "
          f"{args.t_max_ps / 1000:g} ns: {len(late)} snímků, RMSD Cα vůči krystalu {late_rmsd.mean():.3f} nm, "
          f"nad {args.far_nm} nm {100 * (late_rmsd > args.far_nm).mean():.1f} %")
    for name, t, ang, rmsd in sources:
        t_end = float(t.max())
        wins = windows_for(t_end, args.edges)
        key = f"{t_end:g}"
        if key not in report["atlas_windows"]:
            report["atlas_windows"][key] = atlas_window(t_end)
        aw, ap_ = report["atlas_windows"][key]["reps"], report["atlas_windows"][key]["pooled"]
        s = stats(t, ang, rmsd, ref_h, wins, args)
        jr_aw = [w["JS_rama_mean"] for w in aw]
        worst = sorted(s["JS_rama"].items(), key=lambda kv: -kv[1])[:3]
        s["worst_rama"] = [(names[n], v, float(np.max([w["JS_rama"][n] for w in aw]))) for n, v in worst]
        report["sources"][name] = s
        fmt = lambda v: " / ".join("—" if x is None else f"{x:.3f}" for x in v)
        print(f"\n== {name} ({s['frames']} snímků, do {t_end / 1000:g} ns) ==")
        print(f"  RMSD Cα v oknech {', '.join(f'{a / 1000:g}–{b / 1000:g}' for a, b in wins)} ns: {fmt(s['rmsd_windows'])}")
        for i, w in enumerate(aw, 1):
            print(f"     ATLAS R{i} ve stejných oknech:      {fmt(w['rmsd_windows'])}")
        far_aw = ", ".join("%.1f" % (100 * w["far_frac"]) for w in aw)
        f4 = lambda x: "—" if x is None else "%.4f" % x
        jr_txt = ", ".join(f4(x) for x in jr_aw)
        jc_txt = ", ".join(f4(w["JS_chi_mean"]) for w in aw)
        print(f"  nad {args.far_nm} nm: {100 * s['far_frac']:.1f} % (ATLAS R1–R{len(aw)}: {far_aw} %)")
        print(f"  JS Rama průměr přes rezidua {f4(s['JS_rama_mean'])} (ATLAS okno stejné délky, repliky "
              f"{jr_txt}, všechny 3 dohromady {f4(ap_['JS_rama_mean'])}); JS χ1 {f4(s['JS_chi_mean'])} "
              f"(ATLAS {jc_txt}, dohromady {f4(ap_['JS_chi_mean'])})")
        print("  nejhorší Rama: " + ", ".join(f"{n} {v:.3f} (ATLAS okno max {m:.3f})" for n, v, m in s["worst_rama"]))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))
        print("\nUloženo", args.out)


if __name__ == "__main__":
    main()
