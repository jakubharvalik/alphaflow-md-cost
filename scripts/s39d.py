"""Linie 39, změna Z9: s39.py (SHA 170b418d…, beze změny pro fázi B) rozšířený o --clash-min; s18.py beze změny (SHA ec3ed937…).
--clash-min A (Å, > 0): vzorky, v nichž je některá dvojice těžkých atomů blíž než A, se z výběru cílů ED vynechají
(fyzikálně nemožná geometrie: destilovaný AlphaFlow ji měl u 18 % vzorků gen39b, zveřejněný plný u 0,5 %; clash39.py).
Pojistka (Gemini 39k): zbude-li méně než --min-pool (25) vzorků, vezme se --min-pool vzorků s největší d_min.
Cíle ED se pak vyberou stejně jako v s18 (oracle38 v13b, start v medoidu) ze zbylých vzorků, indexy v číslování celého souboru.
--pool-half-seed S (> 0) jako v s39.py (náhodná polovina, numpy default_rng(S)); s oběma volbami se nejdřív bere polovina.
Bez obou voleb se volá s18 beze změny. Ostatní argumenty jdou do s18.main. Výběr a vzdálenosti se zapíší do <work>/pool.json.
    python s39d.py --clash-min 0.5 --seed 0 --sys p2_7jfl_C --ens-pdb afT_dist50/7jfl_C.pdb --work w --out hyb39_7jfl_C_S18d50_s0
"""
import argparse
import json
import sys
from pathlib import Path

import mdtraj as md
import numpy as np

import s18


def min_pair_nm(x):
    """Nejmenší vzdálenost dvou různých atomů (nm)."""
    try:
        from scipy.spatial import cKDTree
        return float(cKDTree(x).query(x, k=2)[0][:, 1].min())
    except ImportError:
        m = np.inf
        for i in range(0, len(x), 256):
            d = np.linalg.norm(x[i:i + 256, None, :] - x[None, :, :], axis=2)
            for j in range(d.shape[0]):
                d[j, i + j] = np.inf
            m = min(m, float(d.min()))
        return m


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--pool-half-seed", type=int, default=0)
    ap.add_argument("--clash-min", type=float, default=0.0)
    ap.add_argument("--min-pool", type=int, default=25)
    ap.add_argument("--work", required=True)
    a, rest = ap.parse_known_args()
    sys.argv = [sys.argv[0], "--work", a.work] + rest
    if a.pool_half_seed > 0 or a.clash_min > 0:
        def ed_pool(ens_pdb, K, trim, pcs):
            import oracle38
            tr = md.load(str(ens_pdb))
            Y = tr.xyz[:, tr.topology.select("name CA")].astype(np.float64)
            sub = np.arange(len(Y))
            info = dict(n_all=len(Y))
            if a.pool_half_seed > 0:
                sub = np.sort(np.random.default_rng(a.pool_half_seed).choice(len(Y), len(Y) // 2, replace=False))
            if a.clash_min > 0:
                heavy = tr.topology.select("not element H")
                dmin = np.array([min_pair_nm(tr.xyz[i, heavy].astype(np.float64)) for i in range(len(Y))]) * 10.0
                excl = [int(i) for i in np.nonzero(dmin < a.clash_min)[0]]
                keep = np.array([i for i in sub if dmin[i] >= a.clash_min], dtype=int)
                if len(keep) < min(a.min_pool, len(sub)):
                    keep = np.sort(sub[np.argsort(-dmin[sub], kind="stable")[:min(a.min_pool, len(sub))]])
                    info["min_pool_used"] = a.min_pool
                    print(f"s39d: po filtru by zbylo < {a.min_pool}, beru {len(keep)} vzorků s největší d_min", flush=True)
                sub = keep
                info.update(clash_min_A=a.clash_min, excluded=excl, dmin_A=[round(float(x), 3) for x in dmin])
                print(f"s39d: vynecháno {len(excl)} z {len(Y)} vzorků s dvojicí těžkých atomů < {a.clash_min} Å: {excl}", flush=True)
            if len(sub) < K:
                raise SystemExit(f"s39d: ve výběru zbylo {len(sub)} vzorků < {K} cílů")
            ns = argparse.Namespace(ens_trim=trim, ens_m=0, seed=0, ens_lam=1.0, ens_cap=0.0, ens_noise="ou",
                                    ens_ends="drift", ens_once=False, ens_once_target="medoid", ens_strat=K,
                                    ens_select="ed", ens_ed_pcs=pcs)
            t = [int(sub[i]) for i in oracle38.ens_setup(Y[sub], ns)["strat"]]
            Path(a.work).mkdir(parents=True, exist_ok=True)
            Path(a.work, "pool.json").write_text(json.dumps(dict(pool_half_seed=a.pool_half_seed, n_pool=len(sub),
                                                                 pool=[int(i) for i in sub], targets=t, **info)))
            print(f"s39d: ED z {len(sub)} z {len(Y)} vzorků (polovina seed {a.pool_half_seed}, clash-min {a.clash_min} Å): {t}",
                  flush=True)
            return t
        s18.ed_targets = ed_pool
    s18.main()


if __name__ == "__main__":
    main()
