"""core36: core35 + JS proti každé replice ATLAS zvlášť (hodnocení proti vynechané replice, Gemini 36b, 6. 11. 2026).
Báze PCA/TICA a okraje histogramu zůstávají ze všech replik; druhá tabulka dává JS proti R1, R2, R3 (celé, bez 100 ps).

core35: JS PCA bez volných konců a JS v TICA (podle oponentury Gemini 35c, 6. 11. 2026).

Na 7jfl_C nese PC1 z 58 % C-konec 45–47, takže JS PCA měří hlavně konec řetězce. Tady se pro stejné běhy
spočte totéž ve třech variantách:
  full  = všechny Cα (kontrola: má souhlasit s pca17)
  core  = Cα bez --trim reziduí na každém konci (přeložení i PCA jen na jádru)
  tica  = jádro, 2 nejpomalejší složky TICA z ATLAS (prvních --tica-dim PCA složek, lag --tica-lag-ps v každé replice)
JS 2D histogramu prvních dvou složek proti celému ATLAS (30 binů, okraj 25 %, bez prvních 100 ps) a podíl snímků
v binech, kam ATLAS nedošel. Měřítka: každá replika ATLAS v okně 0–--t-short-ps a celá. Jen čte, nic nesimuluje.

    python core35.py --sys p2_7jfl_C --atlas public/atlas/cache/7jfl_C --runs orakulum=hyb27_… MD=hyb34_… --out core35.json
"""
import argparse
import json
from pathlib import Path

import mdtraj as md
import numpy as np

from atlas_compare import load_atlas
from kinetics import fold_setup, load_hyb


def js(h1, h2):
    p, q = h1 / h1.sum(), h2 / h2.sum()
    m = 0.5 * (p + q)
    kl = lambda a, b: float(np.sum(a[a > 0] * np.log2(a[a > 0] / b[a > 0])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def align(X, ref):
    Xc = X - X.mean(1, keepdims=True)
    Rc = ref - ref.mean(0)
    H = np.einsum("kni,nj->kij", Xc, Rc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.ones((len(X), 3))
    D[:, 2] = d
    R = np.einsum("kij,kj,kjl->kil", U, D, Vt)
    return np.einsum("kni,kij->knj", Xc, R) + ref.mean(0)


class Space:
    """Projekce do 2D (PCA nebo TICA) naučená na ATLAS a histogramy jako v pca17."""

    def __init__(self, XA_list, ref, kind, n_pca, lag_frames, bins):
        self.ref = ref
        flat = [align(X, ref).reshape(len(X), -1) for X in XA_list]
        allA = np.concatenate(flat)
        self.mu = allA.mean(0)
        _, S, Vt = np.linalg.svd(allA - self.mu, full_matrices=False)
        self.var = (S ** 2 / (S ** 2).sum())
        if kind == "pca":
            self.W = Vt[:2].T
            self.info = dict(var=self.var[:2].round(3).tolist())
        else:
            V = Vt[:n_pca].T
            Y = [(F - self.mu) @ V for F in flat]
            C0 = np.zeros((n_pca, n_pca))
            Ct = np.zeros((n_pca, n_pca))
            n = 0
            for y in Y:
                if len(y) <= lag_frames:
                    continue
                a, b = y[:-lag_frames], y[lag_frames:]
                C0 += a.T @ a + b.T @ b
                Ct += a.T @ b + b.T @ a
                n += 2 * len(a)
            C0 /= n
            Ct /= n
            ev, U = np.linalg.eigh(C0)
            keep = ev > 1e-10 * ev.max()
            Wh = U[:, keep] / np.sqrt(ev[keep])
            lam, Q = np.linalg.eigh(Wh.T @ Ct @ Wh)
            order = np.argsort(-lam)[:2]
            self.W = V @ (Wh @ Q[:, order])
            self.info = dict(eig=lam[order].round(4).tolist(), pca_var_used=float(self.var[:n_pca].sum().round(3)))
        PA = [(F - self.mu) @ self.W for F in flat]
        Pall = np.concatenate(PA)
        lo, hi = Pall.min(0), Pall.max(0)
        pad = 0.25 * (hi - lo)
        self.ed = [np.linspace(lo[i] - pad[i], hi[i] + pad[i], bins + 1) for i in (0, 1)]
        self.hA = self.hist(Pall)
        self.PA = PA
        self.hR = [self.hist(P) for P in PA]

    def proj(self, X):
        return (align(X, self.ref).reshape(len(X), -1) - self.mu) @ self.W

    def hist(self, P):
        return np.histogram2d(np.clip(P[:, 0], self.ed[0][0], self.ed[0][-1]),
                              np.clip(P[:, 1], self.ed[1][0], self.ed[1][-1]), bins=self.ed)[0].ravel()

    def describe(self, P):
        h = self.hist(P)
        return dict(JS=round(js(h, self.hA), 3), outside=round(float(h[self.hA == 0].sum() / h.sum()), 3),
                    JS_rep=[round(js(h, hr), 3) for hr in self.hR])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sys", required=True)
    ap.add_argument("--atlas", required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="jméno=adresář hybridu (formát hybrid.py)")
    ap.add_argument("--trim", type=int, default=4, help="kolik reziduí vynechat na každém konci (jádro)")
    ap.add_argument("--tica-lag-ps", type=float, default=5000.0)
    ap.add_argument("--tica-dim", type=int, default=10)
    ap.add_argument("--t-short-ps", type=float, default=10900.0)
    ap.add_argument("--skip-ps", type=float, default=100.0)
    ap.add_argument("--bins", type=int, default=30)
    ap.add_argument("--out", default="core35.json")
    args = ap.parse_args()
    top = md.load(str(Path(args.sys) / "solute.pdb")).topology
    F = fold_setup(args.sys, top, [])
    ref, ca_sol = F["ref"][F["ca"]], F["sol_idx"][F["ca"]]
    L = len(ref)
    core = np.arange(args.trim, L - args.trim)
    meta, A = load_atlas(args.atlas)
    dt = float(meta["dt_ps"])
    XA = [np.asarray(R["ca"], np.float64) for R in A]
    lag = int(round(args.tica_lag_ps / dt))
    spaces = {"full": Space(XA, ref, "pca", 0, 0, args.bins),
              "core": Space([X[:, core] for X in XA], ref[core], "pca", 0, 0, args.bins),
              "tica": Space([X[:, core] for X in XA], ref[core], "tica", args.tica_dim, lag, args.bins)}
    sel_atoms = {"full": slice(None), "core": core, "tica": core}
    res = dict(sys=args.sys, L=L, core_residues=[int(core[0]) + 1, int(core[-1]) + 1], tica_lag_ps=args.tica_lag_ps,
               spaces={k: s.info for k, s in spaces.items()}, rows={})
    # ATLAS měřítka
    for k, R in enumerate(A, 1):
        t = R["t"]
        for lab, m in ((f"ATLAS R{k} do {args.t_short_ps / 1000:g} ns", (t >= args.skip_ps) & (t <= args.t_short_ps)),
                       (f"ATLAS R{k} celá", t >= args.skip_ps)):
            res["rows"][lab] = {n: s.describe(s.PA[k - 1][m]) for n, s in spaces.items()}
    for spec in args.runs:
        name, d = spec.split("=", 1)
        H = load_hyb(d)
        t = H["times"].astype(float)
        keep = t >= args.skip_ps
        X = np.asarray(H["pos"][0][keep][:, ca_sol], np.float64)
        res["rows"][name] = {n: s.describe(s.proj(X[:, sel_atoms[n]])) for n, s in spaces.items()}
    print(f"{args.sys}: jádro = rezidua {res['core_residues'][0]}–{res['core_residues'][1]} z {L}; "
          f"PCA full {spaces['full'].info}, core {spaces['core'].info}; TICA lag {args.tica_lag_ps / 1000:g} ns "
          f"{spaces['tica'].info}")
    print(f"{'zdroj':<28} {'JS PCA celé':>12} {'JS PCA jádro':>13} {'JS TICA jádro':>14}   mimo ATLAS (celé/jádro/TICA)")
    for lab, r in res["rows"].items():
        print(f"{lab:<28} {r['full']['JS']:>12.3f} {r['core']['JS']:>13.3f} {r['tica']['JS']:>14.3f}   "
              f"{r['full']['outside']:.3f}/{r['core']['outside']:.3f}/{r['tica']['outside']:.3f}")
    nR = len(A)
    print(f"\nJS proti jednotlivým replikám (celé, bez {args.skip_ps:g} ps):")
    print(f"{'zdroj':<28} " + " ".join(f"{'jádro R' + str(k):>10}" for k in range(1, nR + 1)) + "  "
          + " ".join(f"{'TICA R' + str(k):>9}" for k in range(1, nR + 1)))
    for lab, r in res["rows"].items():
        print(f"{lab:<28} " + " ".join(f"{x:>10.3f}" for x in r['core']['JS_rep']) + "  "
              + " ".join(f"{x:>9.3f}" for x in r['tica']['JS_rep']))
    Path(args.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print("Uloženo", args.out)


if __name__ == "__main__":
    main()
