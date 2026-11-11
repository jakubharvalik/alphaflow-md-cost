"""af38: vygenerované soubory (AlphaFlow/ESMFlow, Jing et al. 2024) proti ATLAS stejnými metrikami jako core36,
srovnané při stejném počtu snímků n (samostudium 7. 11. 2026). Jen čte, nic nesimuluje.

Naše testovací proteiny jsou v testovací části ATLAS rozdělení AlphaFlow (splits/atlas_test.csv), model je při
tréninku neviděl. Soubory: huggingface.co/bjing-mit/alphaflow/samples/*.zip (250 struktur na protein).

Metriky (jako core36 / atlas_compare): JS 2D histogramu PCA celé, PCA jádra (bez --trim reziduí na koncích), TICA jádra
(lag 5 ns) proti sjednocení ATLAS (bez 100 ps), podíl mimo ATLAS, JS proti jednotlivým replikám; průměrné JS Rama a χ1
po reziduích; korelace RMSF jádra. Malé n zvyšuje JS, proto i ATLAS a hybridy při stejném n.

    python af38.py --name 7jfl_C --sys p2_7jfl_C --atlas public/atlas/cache/7jfl_C \
        --ens AF_MD=public/af38/x/7jfl_C.pdb --runs orakulum_s0=hyb36_7jfl_C_local_100ns_s0 --out af38_7jfl_C.json
"""
import argparse
import json
from pathlib import Path

import mdtraj as md
import numpy as np

from atlas_compare import load_atlas
from core36 import Space, align
from model_c import js as js_t, rama_hist
from model_d import chi_hist

ALIAS = {"HSD": "HIS", "HSE": "HIS", "HSP": "HIS", "HID": "HIS", "HIE": "HIS", "HIP": "HIS", "CYX": "CYS"}


def ens_load(path):
    tr = md.load(str(path))
    top = tr.topology
    seq = [ALIAS.get(r.name, r.name) for r in top.residues]
    ca = top.select("name CA")
    L = top.n_residues
    ang = np.full((tr.n_frames, L, 3), np.nan, np.float32)
    for k, fn in enumerate((md.compute_phi, md.compute_psi, md.compute_chi1)):
        idx, val = fn(tr)
        for j, quad in enumerate(idx):
            ang[:, top.atom(int(quad[1])).residue.index, k] = val[:, j]
    return np.asarray(tr.xyz[:, ca], np.float64), ang, seq


def even(n_tot, n):
    return np.unique(np.linspace(0, n_tot - 1, n).round().astype(int)) if n_tot > n else np.arange(n_tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--sys", default="")
    ap.add_argument("--atlas", required=True)
    ap.add_argument("--ens", nargs="*", default=[], help="jméno=PDB s více modely")
    ap.add_argument("--runs", nargs="*", default=[], help="jméno=adresář hybridu")
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--reps", type=int, default=20, help="opakování náhodného výběru n snímků z ATLAS")
    ap.add_argument("--trim", type=int, default=4)
    ap.add_argument("--tica-lag-ps", type=float, default=5000.0)
    ap.add_argument("--tica-dim", type=int, default=10)
    ap.add_argument("--t-short-ps", type=float, default=10900.0)
    ap.add_argument("--skip-ps", type=float, default=100.0)
    ap.add_argument("--bins", type=int, default=30)
    ap.add_argument("--seed", type=int, default=38)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    meta, A = load_atlas(args.atlas)
    dt = float(meta["dt_ps"])
    L = A[0]["ca"].shape[1]
    ca_sol = None
    if args.sys and Path(args.sys, "solute.pdb").exists():
        from kinetics import fold_setup
        top = md.load(str(Path(args.sys) / "solute.pdb")).topology
        F = fold_setup(args.sys, top, [])
        ref, ca_sol = F["ref"][F["ca"]].astype(np.float64), F["sol_idx"][F["ca"]]
        ref_src = "peptide_raw.pdb (jako core36)"
    else:
        ref = np.asarray(A[0]["ca"][0], np.float64)
        ref_src = "ATLAS R1 snímek 0 (bez --sys)"
    assert len(ref) == L, (len(ref), L)
    core = np.arange(args.trim, L - args.trim)
    XA = [np.asarray(R["ca"], np.float64) for R in A]
    lag = int(round(args.tica_lag_ps / dt))
    spaces = {"full": Space(XA, ref, "pca", 0, 0, args.bins),
              "core": Space([X[:, core] for X in XA], ref[core], "pca", 0, 0, args.bins),
              "tica": Space([X[:, core] for X in XA], ref[core], "tica", args.tica_dim, lag, args.bins)}
    sel_atoms = {"full": slice(None), "core": core, "tica": core}
    # sjednocení ATLAS bez skip: torzní histogramy a RMSF jádra
    keepA = [R["t"] >= args.skip_ps for R in A]
    angU = np.concatenate([R["ang"][k] for R, k in zip(A, keepA)]).astype(np.float64)
    caU = np.concatenate([X[k] for X, k in zip(XA, keepA)])
    hR, hC = {}, {}
    for r in range(L):
        if not (np.isnan(angU[:, r, :2]).all() or np.isnan(angU[0, r, :2]).any()):
            hR[r] = rama_hist(angU[:, r, 0], angU[:, r, 1])
        if not np.isnan(angU[:, r, 2]).all():
            hC[r] = chi_hist(angU[:, r, 2])

    def rmsf(X):
        Y = align(X[:, core], ref[core])
        return np.sqrt(((Y - Y.mean(0)) ** 2).sum(-1).mean(0))

    rmsfA = rmsf(caU)

    def measure(X, ang):
        out = {n: s.describe(s.proj(X[:, sel_atoms[n]])) for n, s in spaces.items()}
        ang = ang.astype(np.float64)
        jr = [js_t(rama_hist(ang[:, r, 0], ang[:, r, 1]), h) for r, h in hR.items()
              if not np.isnan(ang[:, r, :2]).all()]
        jc = [js_t(chi_hist(ang[:, r, 2]), h) for r, h in hC.items() if not np.isnan(ang[:, r, 2]).all()]
        rf = rmsf(X)
        out["rama"] = round(float(np.mean(jr)), 3) if jr else None
        out["chi1"] = round(float(np.mean(jc)), 3) if jc else None
        out["rmsf_r"] = round(float(np.corrcoef(rf, rmsfA)[0, 1]), 3)
        out["rmsf_ratio"] = round(float(rf.mean() / rmsfA.mean()), 3)
        out["n"] = int(len(X))
        return out

    rows = {}
    # ATLAS: náhodných n ze sjednocení (opakovaně), n rovnoměrně z každé repliky, n z prvních 10,9 ns, všechny snímky
    idxU = np.arange(len(caU))
    rnd = [measure(caU[s], angU[s]) for s in (np.sort(rng.choice(idxU, args.n, replace=False)) for _ in range(args.reps))]
    agg = {}
    for key in ("full", "core", "tica"):
        v = np.array([r[key]["JS"] for r in rnd])
        o = np.array([r[key]["outside"] for r in rnd])
        agg[key] = dict(JS=round(float(v.mean()), 3), JS_sd=round(float(v.std()), 3), outside=round(float(o.mean()), 3),
                        JS_rep=np.mean([r[key]["JS_rep"] for r in rnd], 0).round(3).tolist())
    for key in ("rama", "chi1", "rmsf_r", "rmsf_ratio"):
        vals = [r[key] for r in rnd if r[key] is not None]
        agg[key] = round(float(np.mean(vals)), 3) if vals else None
    agg["n"] = args.n
    rows[f"ATLAS náhodných {args.n} ze sjednocení (průměr {args.reps}×)"] = agg
    for k, R in enumerate(A, 1):
        t = R["t"]
        m_all = np.nonzero(t >= args.skip_ps)[0]
        m_sh = np.nonzero((t >= args.skip_ps) & (t <= args.t_short_ps))[0]
        X, an = XA[k - 1], R["ang"]
        rows[f"ATLAS R{k} celá, {args.n} rovnoměrně"] = measure(X[m_all[even(len(m_all), args.n)]], an[m_all[even(len(m_all), args.n)]])
        rows[f"ATLAS R{k} do {args.t_short_ps / 1000:g} ns, {args.n}"] = measure(X[m_sh[even(len(m_sh), args.n)]], an[m_sh[even(len(m_sh), args.n)]])
        rows[f"ATLAS R{k} celá, všechny"] = measure(X[m_all], an[m_all])
    seq_note = {}
    for spec in args.ens:
        name, p = spec.split("=", 1)
        if not Path(p).exists():
            print(f"chybí {p}, přeskakuji {name}")
            continue
        X, an, seq = ens_load(p)
        if X.shape[1] != L:
            print(f"{name}: {X.shape[1]} reziduí ≠ ATLAS {L}, přeskakuji")
            seq_note[name] = f"délka {X.shape[1]} ≠ {L}"
            continue
        if "seq" in meta:
            mis = sum(ALIAS.get(a, a) != b for a, b in zip(meta["seq"], seq))
            seq_note[name] = f"{len(X)} modelů, neshod v sekvenci {mis}"
        else:
            seq_note[name] = f"{len(X)} modelů"
        rows[f"{name} (všech {len(X)})"] = measure(X, an)
        if len(X) > args.n:
            s = np.sort(rng.choice(len(X), args.n, replace=False))
            rows[f"{name}, {args.n}"] = measure(X[s], an[s])
    for spec in args.runs:
        name, d = spec.split("=", 1)
        if ca_sol is None or not Path(d, "hybrid_summary.json").exists():
            print(f"chybí {d} nebo --sys, přeskakuji {name}")
            continue
        from kinetics import load_hyb
        H = load_hyb(d)
        t = H["times"].astype(float)
        keep = np.nonzero(t >= args.skip_ps)[0]
        X = np.asarray(H["pos"][0][keep][:, ca_sol], np.float64)
        an = H["angs"]
        an = an[0] if an.ndim == 4 else an
        an = an[keep]
        sel = even(len(X), args.n)
        rows[f"{name}, {args.n} rovnoměrně"] = measure(X[sel], an[sel])
        rows[f"{name}, všechny"] = measure(X, an)
    res = dict(name=args.name, L=L, core=[int(core[0]) + 1, int(core[-1]) + 1], ref=ref_src, n=args.n, bins=args.bins,
               spaces={k: s.info for k, s in spaces.items()}, ens=seq_note, rows=rows)
    Path(args.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"{args.name}: L={L}, jádro {res['core']}, ref {ref_src}, biny {args.bins}; {seq_note}")
    print(f"{'zdroj':<44} {'n':>5} {'PCA':>6} {'jádro':>6} {'TICA':>6} {'mimo j.':>7} {'Rama':>6} {'χ1':>6} {'RMSF r':>6} {'RMSF×':>6}  jádro R1/R2/R3")
    for lab, r in rows.items():
        sd = f"±{r['core']['JS_sd']:.3f}" if "JS_sd" in r["core"] else ""
        print(f"{lab:<44} {r['n']:>5} {r['full']['JS']:>6.3f} {r['core']['JS']:>6.3f} {r['tica']['JS']:>6.3f} "
              f"{r['core']['outside']:>7.3f} {r['rama'] if r['rama'] is not None else float('nan'):>6.3f} "
              f"{r['chi1'] if r['chi1'] is not None else float('nan'):>6.3f} {r['rmsf_r']:>6.3f} {r['rmsf_ratio']:>6.3f}  "
              + "/".join(f"{x:.3f}" for x in r["core"]["JS_rep"]) + f" {sd}")
    print("Uloženo", args.out)


if __name__ == "__main__":
    main()
