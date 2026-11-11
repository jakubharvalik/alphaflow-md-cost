"""sumA39: souhrn fáze A (af38 na testovacích proteinech ATLAS, bez MD), podle prereg39 „Fáze A“. Jen čte JSON af38.
Pro každý soubor AI (AF_MD, AF_MD_templ, ESMFlow_MD) při n = 250: medián a kvartily JS jádra PCA a TICA, podíl mimo ATLAS,
Rama, χ1, korelace RMSF; srovnání s ATLAS „replika celá, 250“ a „replika do 10,9 ns, 250“ (průměr 3 replik);
podíl dobrého AF (TICA < 0,45), Spearman s délkou, kontrola biny 15. Chybějící proteiny se vypíšou s důvodem (chybí JSON / řádek).
    python sumA39.py --root l39srv --out l39srv/sumA39.json > l39srv/sumA39.txt
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

ENS = ("AF_MD_templ", "AF_MD", "ESMFlow_MD")
KEYS = ("core", "tica", "outside", "rama", "chi1", "rmsf_r")


def val(r, k):
    if k in ("core", "tica"):
        return r[k]["JS"]
    if k == "outside":
        return r["core"]["outside"]
    return r.get(k)


def ens_row(rows, e):
    for lab, r in rows.items():  # 250 modelů → „e (všech 250)“, víc → podvzorek „e, 250“
        if lab == f"{e}, 250":
            return r
    for lab, r in rows.items():
        if lab.startswith(f"{e} (všech "):
            return r
    return None


def rep_mean(rows, pat, k):
    v = [val(r, k) for lab, r in rows.items() if lab.startswith("ATLAS R") and pat in lab]
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else None


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx, ry = x.argsort().argsort(), y.argsort().argsort()
    return float(np.corrcoef(rx, ry)[0, 1]) if len(x) > 2 else None


def q(v):
    v = np.asarray([x for x in v if x is not None], float)
    if not len(v):
        return None
    return [round(float(np.median(v)), 3), round(float(np.percentile(v, 25)), 3), round(float(np.percentile(v, 75)), 3), int(len(v))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--good", type=float, default=0.45)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    R = Path(a.root)
    names = [r["name"] for r in csv.DictReader(open(R / "atlas_test.csv"))]
    data, missing = {}, {}
    for N in names:
        f = R / "af38" / f"af38_{N}_b30.json"
        if not f.exists():
            missing[N] = "chybí af38 b30 JSON"
            continue
        d = json.loads(f.read_text())
        f15 = R / "af38" / f"af38_{N}_b15.json"
        d15 = json.loads(f15.read_text()) if f15.exists() else None
        data[N] = (d, d15)
    res = dict(n_list=len(names), n_json=len(data), missing=missing, ens={}, per_protein={})
    for N, (d, d15) in data.items():
        p = dict(L=d["L"])
        for e in ENS:
            r = ens_row(d["rows"], e)
            if r is None:
                p[e] = None
                continue
            p[e] = {k: val(r, k) for k in KEYS}
            if d15:
                r15 = ens_row(d15["rows"], e)
                p[e]["tica_b15"] = val(r15, "tica") if r15 else None
        for pat, nm in (("celá, 250", "rep100"), ("do 10.9 ns, 250", "rep10_9")):
            p[nm] = {k: rep_mean(d["rows"], pat, k) for k in KEYS}
        res["per_protein"][N] = p
    P = res["per_protein"]
    print(f"Fáze A: {len(data)} z {len(names)} proteinů s af38 b30 JSON; chybí: {missing if missing else 'nic'}")
    hdr = f"{'soubor':<14} {'n':>3}  " + "  ".join(f"{k:>19}" for k in KEYS) + "   dobrý(T<%.2f)  <rep10,9 T  <rep100 T  ρ(L,T)" % a.good
    print("medián [kvartily] při n = 250, biny 30")
    print(hdr)
    for src in ENS + ("rep10_9", "rep100"):
        rows = [(N, p[src]) for N, p in P.items() if p.get(src)]
        if not rows:
            print(f"{src:<14}   0  (žádná data)")
            continue
        st = {k: q([r[k] for _, r in rows]) for k in KEYS}
        line = f"{src:<14} {len(rows):>3}  " + "  ".join(
            (f"{s[0]:.3f} [{s[1]:.3f}–{s[2]:.3f}]" if s else f"{'–':>19}") for s in st.values())
        e = dict(n=len(rows), stats=st)
        T = [r["tica"] for _, r in rows]
        e["good"] = int(sum(t < a.good for t in T))
        e["rho_L_tica"] = spearman([P[N]["L"] for N, _ in rows], T)
        if src in ENS:
            b10 = [r["tica"] < P[N]["rep10_9"]["tica"] for N, r in rows if P[N]["rep10_9"]["tica"] is not None]
            b100 = [r["tica"] < P[N]["rep100"]["tica"] for N, r in rows if P[N]["rep100"]["tica"] is not None]
            e["better_rep10_9_tica"], e["better_rep100_tica"] = [int(sum(b10)), len(b10)], [int(sum(b100)), len(b100)]
            t15 = [r.get("tica_b15") for _, r in rows if r.get("tica_b15") is not None]
            e["tica_b15"] = q(t15)
            g15 = [(r["tica"] < a.good) == (r["tica_b15"] < a.good) for _, r in rows if r.get("tica_b15") is not None]
            e["good_agree_b15"] = [int(sum(g15)), len(g15)]
            line += f"   {e['good']:>3}/{len(rows):<3}      {b10.count(True) if b10 else 0:>3}/{len(b10):<3}    {b100.count(True) if b100 else 0:>3}/{len(b100):<3}"
        else:
            line += f"   {e['good']:>3}/{len(rows):<3}      {'':>7}    {'':>7}"
        line += f"   {e['rho_L_tica']:+.2f}" if e["rho_L_tica"] is not None else ""
        res["ens"][src] = e
        print(line)
    for e in ENS:
        x = res["ens"].get(e)
        if x and x.get("tica_b15"):
            print(f"kontrola biny 15, {e}: TICA medián {x['tica_b15'][0]:.3f} [{x['tica_b15'][1]:.3f}–{x['tica_b15'][2]:.3f}], "
                  f"shoda dobrý/špatný s biny 30 {x['good_agree_b15'][0]}/{x['good_agree_b15'][1]}")
    # L ≤ 120 (fáze B) zvlášť pro AF_MD_templ
    sm = [(N, p) for N, p in P.items() if p["L"] <= 120 and p.get("AF_MD_templ")]
    if sm:
        T = [p["AF_MD_templ"]["tica"] for _, p in sm]
        print(f"AF_MD_templ, L ≤ 120: {len(sm)} proteinů, TICA medián {np.median(T):.3f}, dobrý {sum(t < a.good for t in T)}/{len(sm)}")
        res["phaseB_subset"] = dict(n=len(sm), tica_median=round(float(np.median(T)), 3), good=int(sum(t < a.good for t in T)))
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print("Uloženo", a.out)


if __name__ == "__main__":
    main()
