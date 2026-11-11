"""Analýza pilotu: rozdělení pohybů reziduí z replik.

Počítá pro vnitřní rezidua (mají φ i ψ):
  1. Ramachandranovy histogramy a populace stavů (αR, β, PPII, αL),
  2. přechodové matice mezi stavy pro více časových kroků τ a implikované časy,
  3. Chapman-Kolmogorovův test (T(2τ) vs. T(τ)²),
  4. podmíněnou vzájemnou informaci I(s_n(t+τ); s_m(t) | s_n(t)) → velikost okna r,
  5. konvergenci (JS divergence mezi polovinami replik).

    python analyze.py --sys ala5 --lags 1 10 100
"""
import argparse
import json
from pathlib import Path

import mdtraj as md
import numpy as np

STATES = ["aR", "beta", "PPII", "aL"]


def assign(phi, psi):
    """Hrubé Ramachandranovy stavy, úhly v radiánech."""
    phi, psi = np.degrees(phi), np.degrees(psi)
    s = np.empty(phi.shape, np.int8)
    left = phi > 0
    helix = (psi > -120) & (psi <= 50)
    s[~left & helix] = 0
    s[~left & ~helix & (phi < -90)] = 1
    s[~left & ~helix & (phi >= -90)] = 2
    s[left] = 3
    return s


def load_replicas(sysdir, runs, skip_ps):
    top = md.load(str(Path(sysdir) / "solute.pdb")).topology
    files = sorted(Path(runs).glob("rep_*.npz"))
    if not files:
        raise SystemExit(f"Žádné repliky v {runs}")
    phis, psis, save_ps = [], [], None
    res_ids = None
    for f in files:
        d = np.load(f)
        save_ps = float(d["save_ps"])
        skip = int(round(skip_ps / save_ps))
        traj = md.Trajectory(d["pos"][skip:], top)
        iphi, phi = md.compute_phi(traj)
        ipsi, psi = md.compute_psi(traj)
        rphi = [top.atom(a[1]).residue.index for a in iphi]  # N atom reziduua
        rpsi = [top.atom(a[1]).residue.index for a in ipsi]  # CA atom reziduua
        common = sorted(set(rphi) & set(rpsi))
        res_ids = common
        phis.append(phi[:, [rphi.index(r) for r in common]])
        psis.append(psi[:, [rpsi.index(r) for r in common]])
    names = [str(top.residue(r)) for r in res_ids]
    return names, phis, psis, save_ps, len(files)


def transition_matrix(states_list, lag):
    C = np.zeros((4, 4))
    for s in states_list:
        if len(s) > lag:
            np.add.at(C, (s[:-lag], s[lag:]), 1)
    rows = C.sum(1, keepdims=True)
    T = np.divide(C, rows, out=np.zeros_like(C), where=rows > 0)
    return T, C


def implied_timescale(T, lag_ps):
    ev = np.sort(np.abs(np.linalg.eigvals(T)))[::-1]
    if len(ev) < 2 or ev[1] <= 0 or ev[1] >= 1:
        return None
    return float(-lag_ps / np.log(ev[1]))


def cond_mi(fut, other, cur, n_perm=20, rng=None):
    """I(fut; other | cur) v bitech, plug-in odhad + permutační základ uvnitř vrstev cur."""
    rng = rng or np.random.default_rng(0)

    def mi(o):
        val = 0.0
        n = len(cur)
        for c in np.unique(cur):
            m = cur == c
            pc = m.sum() / n
            j = np.zeros((4, 4))
            np.add.at(j, (fut[m], o[m]), 1)
            j /= j.sum()
            pf, po = j.sum(1, keepdims=True), j.sum(0, keepdims=True)
            nz = j > 0
            val += pc * np.sum(j[nz] * np.log2(j[nz] / (pf @ po)[nz]))
        return val

    obs = mi(other)
    base = []
    for _ in range(n_perm):
        o = other.copy()
        for c in np.unique(cur):
            idx = np.where(cur == c)[0]
            o[idx] = o[rng.permutation(idx)]
        base.append(mi(o))
    return obs, float(np.mean(base)), float(np.std(base))


def js_div(h1, h2):
    p, q = h1 / h1.sum(), h2 / h2.sum()
    m = 0.5 * (p + q)
    kl = lambda a, b: np.sum(a[a > 0] * np.log2(a[a > 0] / b[a > 0]))
    return float(0.5 * kl(p, m) + 0.5 * kl(q, m))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sys", default="ala5")
    ap.add_argument("--runs", default=None)
    ap.add_argument("--lags", type=float, nargs="+", default=[1, 10, 100])
    ap.add_argument("--skip-ps", type=float, default=20.0,
                    help="zahodit začátek každé repliky (paměť společného startu)")
    ap.add_argument("--plots", action="store_true")
    args = ap.parse_args()
    runs = args.runs or str(Path(args.sys) / "runs")

    names, phis, psis, save_ps, n_rep = load_replicas(args.sys, runs, args.skip_ps)
    states = [assign(p, q) for p, q in zip(phis, psis)]  # seznam (frames, n_res)
    n_res = len(names)
    out = dict(residues=names, n_replicas=n_rep, save_ps=save_ps,
               frames_per_replica=int(states[0].shape[0]), skip_ps=args.skip_ps)

    allst = np.concatenate(states)
    half = max(1, n_rep // 2)
    out["populations"] = {}
    out["convergence_JS_bits"] = {}
    bins = np.linspace(-np.pi, np.pi, 37)
    for k, nm in enumerate(names):
        pop = np.bincount(allst[:, k], minlength=4) / len(allst)
        a = np.concatenate([s[:, k] for s in states[:half]])
        b = np.concatenate([s[:, k] for s in states[half:]]) if n_rep > 1 else a
        pa, pb = (np.bincount(x, minlength=4) / len(x) for x in (a, b))
        out["populations"][nm] = {st: dict(p=round(float(pop[i]), 4),
                                           half_diff=round(float(abs(pa[i] - pb[i])), 4))
                                  for i, st in enumerate(STATES)}
        h = lambda sel: np.histogram2d(np.concatenate([p[:, k] for p in phis[sel]]),
                                       np.concatenate([q[:, k] for q in psis[sel]]),
                                       bins=[bins, bins])[0] + 1e-9
        if n_rep > 1:
            out["convergence_JS_bits"][nm] = round(js_div(h(slice(0, half)), h(slice(half, None))), 5)

    out["kinetics"] = {}
    for k, nm in enumerate(names):
        per = {}
        for lag_ps in args.lags:
            lag = int(round(lag_ps / save_ps))
            T, C = transition_matrix([s[:, k] for s in states], lag)
            T2, _ = transition_matrix([s[:, k] for s in states], 2 * lag)
            per[f"{lag_ps:g}ps"] = dict(
                T=np.round(T, 4).tolist(), counts=int(C.sum()),
                implied_timescale_ps=implied_timescale(T, lag_ps),
                CK_max_abs_err=round(float(np.max(np.abs(T2 - T @ T))), 4))
        out["kinetics"][nm] = per

    rng = np.random.default_rng(0)
    out["conditional_MI_bits"] = {}
    for lag_ps in args.lags:
        lag = int(round(lag_ps / save_ps))
        rows = []
        for n in range(n_res):
            for m in range(n_res):
                if m == n:
                    continue
                cur = np.concatenate([s[:-lag, n] for s in states])
                fut = np.concatenate([s[lag:, n] for s in states])
                oth = np.concatenate([s[:-lag, m] for s in states])
                obs, base, sd = cond_mi(fut, oth, cur, rng=rng)
                rows.append(dict(n=names[n], m=names[m], dist=abs(m - n),
                                 MI=round(obs, 5), perm_mean=round(base, 5), perm_sd=round(sd, 5)))
        by_dist = {}
        for r in rows:
            by_dist.setdefault(r["dist"], []).append(r["MI"] - r["perm_mean"])
        out["conditional_MI_bits"][f"{lag_ps:g}ps"] = dict(
            pairs=rows, excess_by_seq_distance={d: round(float(np.mean(v)), 5)
                                                for d, v in sorted(by_dist.items())})

    Path(runs, "summary.json").write_text(json.dumps(out, indent=2))
    print(f"Repliky: {n_rep}, snímků/repliku: {out['frames_per_replica']} po {save_ps} ps")
    for nm in names:
        pops = ", ".join(f"{s} {v['p']:.2f}" for s, v in out["populations"][nm].items())
        js = out["convergence_JS_bits"].get(nm)
        print(f"{nm}: {pops} | JS polovin {js}")
    for lag_key, v in out["conditional_MI_bits"].items():
        print(f"τ={lag_key}: nadbytek podmíněné MI podle vzdálenosti v sekvenci {v['excess_by_seq_distance']}")
    for nm in names:
        print(nm, {k: (v["implied_timescale_ps"], v["CK_max_abs_err"]) for k, v in out["kinetics"][nm].items()})

    if args.plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, n_res, figsize=(3.2 * n_res, 3.2), squeeze=False)
        for k, nm in enumerate(names):
            ax = axes[0, k]
            ax.hist2d(np.degrees(np.concatenate([p[:, k] for p in phis])),
                      np.degrees(np.concatenate([q[:, k] for q in psis])),
                      bins=72, range=[[-180, 180], [-180, 180]], cmap="viridis", cmin=1)
            ax.set_title(nm)
            ax.set_xlabel("φ")
            ax.set_ylabel("ψ")
        fig.tight_layout()
        fig.savefig(Path(runs, "ramachandran.png"), dpi=120)
        print("Uloženo", Path(runs, "ramachandran.png"))


if __name__ == "__main__":
    main()
