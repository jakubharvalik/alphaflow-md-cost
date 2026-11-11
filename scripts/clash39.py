"""Linie 39, levná kontrola rizika Z9 (Gemini 39j bod 4): mají vzorky destilovaného AlphaFlow víc těžkých kolizí atomů než plné?
Boxy S18, které ve fázi B selhaly, padaly hned na začátku minimalizace (nekonečná energie) = vada počáteční geometrie.
Pro každý model vícemodelového PDB: počet dvojic těžkých atomů blíž než 0,5 Å a 1,0 Å (kovalentní vazby těžkých atomů jsou
≥ 1,2 Å, takže se nic nevylučuje) a nejmenší vzdálenost. Sady: zveřejněné plné (250), naše plné afT_base a destilované
afT_dist z gen39b (6 a 11 na protein sady B). Ověření ukazatele: selhané boxy S18_s0/S18_s1 (hybrid_summary.json „failed“,
ens_idx = index zveřejněného vzorku) proti ostatním vzorkům téhož souboru. Jen čte soubory, CPU, sekundy až minuty.
    python clash39.py --list l39srv/listB.txt --pub l39srv/af/alphaflow_md_templates_base_202402 \
        --set afT_base=l39srv/gen/run_25413654/samples/afT_base --set afT_dist=l39srv/gen/run_25413654/samples/afT_dist \
        --runs l39srv/runs --out clash39.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def models(pdb):
    out, cur = [], []
    for ln in open(pdb):
        if ln.startswith("MODEL"):
            cur = []
        elif ln.startswith(("ATOM", "HETATM")):
            el = (ln[76:78].strip() or ln[12:16].strip()[0]).upper()
            if el != "H":
                cur.append((float(ln[30:38]), float(ln[38:46]), float(ln[46:54])))
        elif ln.startswith("ENDMDL"):
            out.append(np.array(cur))
            cur = []
    if cur:
        out.append(np.array(cur))
    return out


def clash(x):
    t = cKDTree(x)
    p10 = t.query_pairs(1.0, output_type="ndarray")
    d = np.linalg.norm(x[p10[:, 0]] - x[p10[:, 1]], axis=1) if len(p10) else np.zeros(0)
    dmin = float(t.query(x, k=2)[0][:, 1].min())
    return int((d < 0.5).sum()), int(len(d)), dmin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--pub", required=True)
    ap.add_argument("--set", action="append", default=[], help="jméno=adresář s <N>.pdb")
    ap.add_argument("--runs", default=None)
    ap.add_argument("--tags", default="S18_s0,S18_s1")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    names = [ln.split()[0] for ln in open(a.list) if ln.strip()]
    sets = {"pub_plny": Path(a.pub)} | {s.split("=", 1)[0]: Path(s.split("=", 1)[1]) for s in a.set}
    res = dict(per_protein={}, failed=[], summary={})
    allv = {k: [] for k in sets}
    for N in names:
        row = {}
        for k, d in sets.items():
            f = d / f"{N}.pdb"
            if not f.exists():
                continue
            v = [clash(x) for x in models(f)]
            allv[k] += v
            row[k] = dict(n=len(v), frac_lt05=round(float(np.mean([c05 > 0 for c05, _, _ in v])), 4),
                          frac_lt10=round(float(np.mean([c10 > 0 for _, c10, _ in v])), 4),
                          dmin_med=round(float(np.median([m for _, _, m in v])), 3), per_model=v)
        res["per_protein"][N] = row
        if a.runs and "pub_plny" in row:
            for tag in a.tags.split(","):
                h = Path(a.runs) / f"hyb39_{N}_{tag}" / "hybrid_summary.json"
                if not h.exists():
                    continue
                for fl in json.loads(h.read_text()).get("failed", []):
                    i = fl["ens_idx"]
                    c05, c10, dm = row["pub_plny"]["per_model"][i]
                    res["failed"].append(dict(protein=N, tag=tag, ens_idx=i, lt05=c05, lt10=c10, dmin=round(dm, 3),
                                              error=fl.get("error", "")[:80]))
    print(f"{'sada':<10} {'modelů':>7} {'s páry < 0,5 Å':>15} {'s páry < 1,0 Å':>15} {'medián d_min Å':>15}")
    for k, v in allv.items():
        if not v:
            continue
        s = dict(n=len(v), frac_lt05=round(float(np.mean([c > 0 for c, _, _ in v])), 4),
                 frac_lt10=round(float(np.mean([c > 0 for _, c, _ in v])), 4), dmin_med=round(float(np.median([m for _, _, m in v])), 3),
                 dmin_q05=round(float(np.percentile([m for _, _, m in v], 5)), 3))
        res["summary"][k] = s
        print(f"{k:<10} {s['n']:>7} {s['frac_lt05']:>15.2%} {s['frac_lt10']:>15.2%} {s['dmin_med']:>15.3f}  (5. percentil d_min {s['dmin_q05']:.3f})")
    if res["failed"]:
        print(f"\nselhané boxy S18 ({len(res['failed'])}), kolize jejich zveřejněného vzorku:")
        for f in res["failed"]:
            print(f"  {f['protein']:<8} {f['tag']:<7} vzorek {f['ens_idx']:>3}: páry < 0,5 Å {f['lt05']}, < 1,0 Å {f['lt10']}, "
                  f"d_min {f['dmin']:.3f} Å  {f['error']}")
        hit = sum(f["lt10"] > 0 for f in res["failed"])
        print(f"  selhaných s párem < 1,0 Å: {hit} z {len(res['failed'])}; ve zveřejněných celkem "
              f"{res['summary']['pub_plny']['frac_lt10']:.2%} modelů")
    for N, row in res["per_protein"].items():
        for k in row.values():
            k.pop("per_model", None)
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print("Uloženo", a.out)


if __name__ == "__main__":
    main()
