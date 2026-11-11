"""Linie 39: obal s18.py (beze změny, SHA ec3ed937…) pro seed 1 podle prereg39 Z5 a kontrolu K18.
--pool-half-seed S (> 0): cíle ED se vyberou stejně jako v s18 (oracle38 v13b, start v medoidu), ale jen z náhodné
poloviny vzorků AF (numpy default_rng(S), bez opakování), indexy se vrátí do číslování celého souboru.
Ostatní argumenty jdou do s18.main beze změny. Výběr se zapíše do <work>/pool.json.
    python s39.py --pool-half-seed 1 --seed 1 --sys p2_7jfl_C --ens-pdb af/7jfl_C.pdb --work w --out hyb39_7jfl_C_S18_s1
"""
import argparse
import json
import sys
from pathlib import Path

import mdtraj as md
import numpy as np

import s18


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--pool-half-seed", type=int, default=0)
    ap.add_argument("--work", required=True)
    a, rest = ap.parse_known_args()
    sys.argv = [sys.argv[0], "--work", a.work] + rest
    if a.pool_half_seed > 0:
        def ed_half(ens_pdb, K, trim, pcs):
            import oracle38
            tr = md.load(str(ens_pdb))
            Y = tr.xyz[:, tr.topology.select("name CA")].astype(np.float64)
            sub = np.sort(np.random.default_rng(a.pool_half_seed).choice(len(Y), len(Y) // 2, replace=False))
            ns = argparse.Namespace(ens_trim=trim, ens_m=0, seed=0, ens_lam=1.0, ens_cap=0.0, ens_noise="ou",
                                    ens_ends="drift", ens_once=False, ens_once_target="medoid", ens_strat=K,
                                    ens_select="ed", ens_ed_pcs=pcs)
            t = [int(sub[i]) for i in oracle38.ens_setup(Y[sub], ns)["strat"]]
            Path(a.work).mkdir(parents=True, exist_ok=True)
            Path(a.work, "pool.json").write_text(json.dumps(dict(pool_half_seed=a.pool_half_seed, n_pool=len(sub),
                                                                 pool=sub.tolist(), targets=t)))
            print(f"s39: ED z poloviny vzorků (seed {a.pool_half_seed}, {len(sub)} z {len(Y)}): {t}", flush=True)
            return t
        s18.ed_targets = ed_half
    s18.main()


if __name__ == "__main__":
    main()
