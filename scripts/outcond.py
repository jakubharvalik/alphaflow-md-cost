"""Test Model D mimo podmínky tréninku: naše repliky (300 K, TIP3P, 5 reziduí) místo MDGen (350 K, TIP3P-FB, 4).

Pro každý systém (ala5, kgpwe, eagwk):
  1. bity φψ testovacích přechodů z replik 50–99 vs. nativní tabulka 18×18 z replik 0–49,
  2. rollout modelu ze společného startu (stejný start jako MD, 100 trajektorií) vs. 100 replik MD:
     JS Ramachandranova rozdělení a χ1, populace hrubých stavů v časových oknech (relaxace ze startu),
     rychlost přechodů α↔β; šum = totéž mezi polovinami replik.

    python outcond.py --model model_d_full --systems ala5 kgpwe eagwk --lags 1 10
"""
import argparse
import json
from pathlib import Path

import mdtraj as md
import numpy as np
import torch

from analyze import assign
from model_c import AA, ALIAS, PAD, js, rama_hist
from model_d import NetD, ab_rate, chi_hist, loss_and_ll, rollout

WINDOWS = [0, 10, 30, 100, 300, 1000, 3000]  # ps


def torsion_quads(top):
    """(L, 3, 4) indexy atomů pro φ, ψ, χ1 (−1 = nedefinováno); reziduum = reziduum 2. atomu jako v MDGen."""
    traj = md.Trajectory(np.zeros((1, top.n_atoms, 3), np.float32), top)
    Q = np.full((top.n_residues, 3, 4), -1, np.int64)
    for k, fn in enumerate((md.compute_phi, md.compute_psi, md.compute_chi1)):
        idx, _ = fn(traj)
        for quad in idx:
            Q[top.atom(int(quad[1])).residue.index, k] = quad
    return Q


def dihedral(p):
    """p (..., 4, 3) → úhel jako mdtraj."""
    b1, b2, b3 = p[..., 1, :] - p[..., 0, :], p[..., 2, :] - p[..., 1, :], p[..., 3, :] - p[..., 2, :]
    c1, c2 = np.cross(b2, b3), np.cross(b1, b2)
    return np.arctan2((b1 * c1).sum(-1) * np.linalg.norm(b2, axis=-1), (c1 * c2).sum(-1))


def torsions(X, Q):
    """X (T, N, 3) nebo (N, 3) → (T, L, 3) / (L, 3), NaN kde úhel není."""
    single = X.ndim == 2
    X = X[None] if single else X
    out = np.full((X.shape[0],) + Q.shape[:2], np.nan, np.float32)
    ok = Q[..., 0] >= 0
    out[:, ok] = dihedral(X[:, Q[ok]])
    return out[0] if single else out


def residue_types(top):
    names = [ALIAS.get(r.name, r.name) for r in top.residues]
    return np.array([AA.index(s) if s in AA else PAD for s in names], np.int64)


def load_model(model_dir, dev):
    cfg = json.loads((Path(model_dir) / "model_d_summary.json").read_text())["config"]
    net = NetD(r=cfg["r"], hist=cfg["hist"], ar=cfg["ar"], hidden=cfg["hidden"], layers=cfg["layers"]).to(dev)
    net.load_state_dict(torch.load(Path(model_dir) / "model_d.pt", map_location=dev))
    net.eval()
    return net, cfg


def load_system(sysdir):
    sysdir = Path(sysdir)
    top = md.load(str(sysdir / "solute.pdb")).topology
    Q = torsion_quads(top)
    reps = [torsions(np.load(f)["pos"], Q) for f in sorted((sysdir / "runs").glob("rep_*.npz"))]
    n_sol = top.n_atoms
    start = md.load(str(sysdir / "start.pdb")).xyz[0, :n_sol]
    return dict(top=top, Q=Q, types=residue_types(top), reps=reps, start=torsions(start, Q))


def native18_bits(train, test_pairs, n, lag):
    b18 = lambda v: np.clip(((v + np.pi) / (2 * np.pi) * 18).astype(int), 0, 17)
    x0 = np.concatenate([r[:-lag, n, :2] for r in train])
    x1 = np.concatenate([r[lag:, n, :2] for r in train])
    k0, k1 = b18(x0[:, 0]) * 18 + b18(x0[:, 1]), b18(x1[:, 0]) * 18 + b18(x1[:, 1])
    m = (np.bincount(k1, minlength=324) + 1.0) / (len(k1) + 324)
    C = np.zeros((324, 324))
    np.add.at(C, (k0, k1), 1)
    tab = (C + 5 * m) / (C.sum(1, keepdims=True) + 5)
    cur, nxt = test_pairs
    return float(np.mean(-np.log2(tab[b18(cur[:, 0]) * 18 + b18(cur[:, 1]),
                                      b18(nxt[:, 0]) * 18 + b18(nxt[:, 1])] / 4)))


def pop_curve(trajs, times, n, edges):
    """Populace 4 hrubých stavů rezidua n v časových oknech; trajs (K, T, L, ≥2), times (T,) v ps."""
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (times >= a) & (times < b)
        if not sel.any():
            break
        s = assign(trajs[:, sel, n, 0].ravel(), trajs[:, sel, n, 1].ravel())
        out.append(np.bincount(s, minlength=4) / len(s))
    return np.array(out)


def evaluate(sysname, S, net, cfg, dev, lags, rng, skip_ps=20):
    reps = S["reps"]
    L = len(S["types"])
    T = min(r.shape[0] for r in reps)
    reps = np.stack([r[:T] for r in reps])  # (R, T, L, 3), snímek i = čas i+1 ps
    R = len(reps)
    tr, te = reps[:R // 2], reps[R // 2:]
    inner = [n for n in range(L) if not np.isnan(reps[0, 0, n, :2]).any()]
    md_times = np.arange(1, T + 1, dtype=float)
    res = dict(residues=[str(r) for r in S["top"].residues], replicas=R, ps_per_replica=T, lags={})
    H = cfg["hist"]
    for lag in lags:
        if T < (H + 1) * lag + 50:
            continue
        # 1) bity
        ri = rng.integers(R // 2, R, size=4000)
        ti = rng.integers((H - 1) * lag, T - lag, size=4000)
        frames = [reps[ri, ti - k * lag] for k in range(H)]
        nxt = reps[ri, ti + lag]
        types = np.repeat(S["types"][None], len(ri), 0)
        bits_m, bits_n = [], []
        for n in inner:
            with torch.no_grad():
                _, pp, _ = loss_and_ll(net, dev, types, frames, nxt, np.full(len(ri), n),
                                       np.full(len(ri), lag), cfg["pred_chi"])
            bits_m.append(float(pp.mean()))
            bits_n.append(native18_bits(tr, (reps[ri, ti, n, :2], nxt[:, n, :2]), n, lag))
        # 2) rollout ze společného startu
        start_hist = np.repeat(S["start"][None], (H - 1) * lag + 1, 0)
        n_steps = T // lag + 1
        traj = rollout(net, dev, [dict(types=S["types"], ang=start_hist)], lag, n_steps, R,
                       cfg["pred_chi"], rng)[0]  # (R, n_steps, L, 3), krok s = čas s·lag
        m_times = np.arange(n_steps, dtype=float) * lag
        msel, dsel = m_times >= skip_ps, md_times >= skip_ps
        per = {}
        for n in range(L):
            nm = res["residues"][n]
            d = {}
            if n in inner:
                d["JS_rama"] = js(rama_hist(traj[:, msel, n, 0].ravel(), traj[:, msel, n, 1].ravel()),
                                  rama_hist(reps[:, dsel, n, 0].ravel(), reps[:, dsel, n, 1].ravel()))
                d["JS_rama_noise"] = js(rama_hist(tr[:, dsel, n, 0].ravel(), tr[:, dsel, n, 1].ravel()),
                                        rama_hist(te[:, dsel, n, 0].ravel(), te[:, dsel, n, 1].ravel()))
                pm = pop_curve(traj, m_times, n, WINDOWS)
                pd = pop_curve(reps, md_times, n, WINDOWS)
                k = min(len(pm), len(pd))
                d["pop_err"] = float(np.abs(pm[:k] - pd[:k]).mean())
                d["pop_err_noise"] = float(np.abs(pop_curve(tr, md_times, n, WINDOWS)[:k]
                                                  - pop_curve(te, md_times, n, WINDOWS)[:k]).mean())
                d["pop_model"] = np.round(pm[:k], 3).tolist()
                d["pop_MD"] = np.round(pd[:k], 3).tolist()
                r_md = ab_rate(reps[:, dsel, n, 1][:, ::lag], lag)
                d["ab_rate_ratio"] = float(ab_rate(traj[:, msel, n, 1], lag) / r_md) if r_md > 0 else None
            if not np.isnan(reps[0, 0, n, 2]):
                d["JS_chi"] = js(chi_hist(traj[:, msel, n, 2]), chi_hist(reps[:, dsel, n, 2]))
                d["JS_chi_noise"] = js(chi_hist(tr[:, dsel, n, 2]), chi_hist(te[:, dsel, n, 2]))
            per[nm] = d
        res["lags"][f"{lag}ps"] = dict(bits_model=float(np.mean(bits_m)), bits_native18=float(np.mean(bits_n)),
                                       per_residue=per)
        print(f"{sysname} τ={lag:>3} ps: bity φψ model {np.mean(bits_m):.2f} | nativní 18 {np.mean(bits_n):.2f}")
        for nm, d in per.items():
            parts = []
            if "JS_rama" in d:
                rr = d["ab_rate_ratio"]
                parts.append(f"JS Rama {d['JS_rama']:.4f} (šum {d['JS_rama_noise']:.4f}), populace v čase "
                             f"chyba {d['pop_err']:.3f} (šum {d['pop_err_noise']:.3f}), α↔β "
                             f"{'—' if rr is None else f'{rr:.2f}×'}")
            if "JS_chi" in d:
                parts.append(f"JS χ1 {d['JS_chi']:.4f} (šum {d['JS_chi_noise']:.4f})")
            if parts:
                print(f"   {nm}: " + "; ".join(parts), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model_d_full")
    ap.add_argument("--systems", nargs="+", default=["ala5", "kgpwe", "eagwk"])
    ap.add_argument("--lags", type=int, nargs="+", default=[1, 10, 100])
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outcond_summary.json")
    args = ap.parse_args()
    dev = torch.device(args.device or ("mps" if torch.backends.mps.is_available() else "cpu"))
    rng = np.random.default_rng(args.seed)
    net, cfg = load_model(args.model, dev)
    print(f"Model {args.model} {cfg}, zařízení {dev}")
    out = {s: evaluate(s, load_system(s), net, cfg, dev, args.lags, rng) for s in args.systems}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("Uloženo", args.out)


if __name__ == "__main__":
    main()
