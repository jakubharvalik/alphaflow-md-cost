"""Linie 39, změna Z8/Z8a: souhrn měření ceny AlphaFlow/ESMFlow (úloha gen39). Jen čte.
Vstup = adresář l39/gen z MetaCentra: rt/rt_<model>_<sada>.json (predict.py --runtime_json, {protein: [s na vzorek]}),
out_<J>/gpu.txt a calls.tsv, bench_gen_<J>_*.json (bench39 na téže GPU), msa_counts.txt, templates/<N>.pdb,
samples/afT_base/<N>.pdb (naše vzorky); zveřejněné vzorky AF-MD+templates z --pub (l39/af/alphaflow_md_templates_base_202402).
Výstup: t_gen(N) = medián vzorků bez prvního (zahřátí), fit t = a + b·L^p (a, b ≥ 0, relativní rezidua), odhad průměru
přes 82 testovacích délek a poměr k publikaci (38 s / 3,8 s na A100), kontrola instalace (Z8a: medián poměru průměrného
RMSD Cα ke šabloně naše/zveřejněné přes sadu B v 0,75–1,33 a Spearman ≥ 0,70), počty MSA, režie volání.
    python gen39sum.py --gen l39/gen --list l39/gen39_list.txt --atlas-csv atlas_test.csv \
        --pub l39/af/alphaflow_md_templates_base_202402 --out gen39_sum.json
--run l39/gen/run_<JOB> (úloha gen39b): časy, GPU, volání, bench a naše vzorky jen z této úlohy (rt/, gpu.txt, calls.tsv,
bench_gen_*.json, samples/), šablony a MSA dál z --gen. Bez --run staré rozložení (rt/, out_*/, bench_gen_* v --gen).
"""
import argparse
import csv
import glob
import json
import re
from pathlib import Path

import numpy as np

PUB = {"afT_base": 38.0, "afT_dist": 3.8}  # AlphaFlow obr. 13 / příl. B.3: průměr přes testovací proteiny ATLAS, A100
MODELS = ["afT_base", "afT_dist", "afT12_base", "afT12_dist", "esT_base", "esT_dist"]


def fit_power(L, t):
    from scipy.optimize import curve_fit
    L, t = np.asarray(L, float), np.asarray(t, float)
    f = lambda x, a, b, p: a + b * x ** p
    best = None
    for p0 in (1.0, 2.0, 3.0):
        try:
            par, _ = curve_fit(f, L, t, p0=(0.0, t.max() / L.max() ** p0, p0), sigma=t,
                               bounds=([0, 0, 0.3], [np.inf, np.inf, 4.0]), maxfev=20000)
        except Exception:
            continue
        r = float(np.sqrt(np.mean(((f(L, *par) - t) / t) ** 2)))
        if best is None or r < best[1]:
            best = (par, r)
    if best is None:
        return None
    (a, b, p), r = best
    return dict(a=float(a), b=float(b), p=float(p), rel_rms=round(r, 4))


def mean_rmsd(pdb, ref_ca):
    import mdtraj as md
    t = md.load(str(pdb))
    t = t.atom_slice(t.topology.select("name CA"))
    if t.n_atoms != ref_ca.n_atoms:
        return None, t.n_frames
    return float(np.mean(md.rmsd(t, ref_ca)) * 10.0), t.n_frames  # Å


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True)
    ap.add_argument("--list", required=True)
    ap.add_argument("--atlas-csv", required=True)
    ap.add_argument("--pub", default=None)
    ap.add_argument("--no-rmsd", action="store_true")
    ap.add_argument("--run", default=None, help="adresář jedné úlohy gen39b (run_<JOB>)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    G = Path(args.gen)
    R = Path(args.run) if args.run else None
    D = R or G  # odkud časy, bench a naše vzorky
    Lall = {r["name"]: len(r["seqres"]) for r in csv.DictReader(open(args.atlas_csv))}
    lst = [ln.split() for ln in open(args.list) if ln.strip()]
    sada = {n: s for n, _, s in lst}
    res = dict(models={}, notes=[])
    gfiles = [R / "gpu.txt"] if R else list(G.glob("out_*/gpu.txt"))
    gpus = sorted({p.read_text().strip() for p in gfiles if p.exists()})
    res["gpu"] = gpus
    res["run"] = str(R) if R else None
    if len(gpus) > 1:
        res["notes"].append("měření z více GPU smíchána (staré rozložení bez --run)")
    # časy
    for m in MODELS:
        per = {}
        for f in sorted(D.glob(f"rt/rt_{m}_*.json")):
            for n, ts in json.loads(f.read_text()).items():
                ts = [float(x) for x in ts]
                use = ts[1:] if len(ts) > 1 else ts
                per[n] = dict(L=Lall.get(n), sada=sada.get(n, "?"), n=len(ts), t_first=round(ts[0], 3),
                              t_med=round(float(np.median(use)), 4), t_min=round(min(use), 4), t_max=round(max(use), 4),
                              cv=round(float(np.std(use) / np.mean(use)), 4) if len(use) > 1 else None)
        if not per:
            continue
        d = dict(per_protein=per)
        Ls = [v["L"] for v in per.values()]
        ts = [v["t_med"] for v in per.values()]
        if len(per) >= 4:
            fp = fit_power(Ls, ts)
            if fp:
                d["fit"] = fp
                pred = [fp["a"] + fp["b"] * L ** fp["p"] for L in Lall.values()]
                d["mean82_pred_s"] = round(float(np.mean(pred)), 3)
                if m in PUB:
                    d["pub_A100_s"] = PUB[m]
                    d["ratio_ours_to_pub"] = round(d["mean82_pred_s"] / PUB[m], 3)
        missB = [n for n, s in sada.items() if s == "B" and n not in per]
        d["missing_B"] = missB
        res["models"][m] = d
    # bench na GPU měření
    bg = {}
    for f in D.glob("bench_gen_*.json"):
        j = json.loads(f.read_text())
        key = re.sub(r"^bench_gen_\d+_", "", f.stem)
        key = key.replace("_anchor", "")
        nm = key[3:] if key.startswith("p2_") else key
        bg[nm + ("@anchor" if "anchor" in f.stem else "")] = dict(ns_per_day=j["ns_per_day"], n_atoms=j["n_atoms"])
    res["bench_gen"] = bg
    # MSA, režie volání
    mc = G / "msa_counts.txt"
    if mc.exists():
        res["msa_n_seq"] = {ln.split()[0]: int(ln.split()[1]) for ln in mc.read_text().split("\n") if ln.strip()}
    calls = []
    for f in ([R / "calls.tsv"] if R else G.glob("out_*/calls.tsv")):
        if not f.exists():
            continue
        for r in csv.DictReader(open(f), delimiter="\t"):
            calls.append(r)
    ovh = {}
    for r in calls:
        m, s = r["model"], r["sada"]
        f = D / "rt" / f"rt_{m}_{s}.json"
        if r["kod"] != "0" or not f.exists():
            ovh[f"{m} {s}"] = dict(kod=r["kod"])
            continue
        tot = sum(sum(v) for v in json.loads(f.read_text()).values())
        ovh[f"{m} {s}"] = dict(wall_s=round(float(r["wall_s"]), 1), inference_s=round(tot, 1),
                               overhead_s=round(float(r["wall_s"]) - tot, 1))
    res["calls"] = ovh
    # kontrola instalace (Z8a)
    if args.pub and not args.no_rmsd:
        import mdtraj as md
        from scipy.stats import spearmanr
        chk = {}
        for n, s in sada.items():
            if s != "B":
                continue
            ours, pub, tp = D / "samples" / "afT_base" / f"{n}.pdb", Path(args.pub) / f"{n}.pdb", G / "templates" / f"{n}.pdb"
            if not (ours.exists() and pub.exists() and tp.exists()):
                chk[n] = dict(chyba="chybí soubor")
                continue
            ref = md.load(str(tp))
            ref = ref.atom_slice(ref.topology.select("name CA"))
            o, no = mean_rmsd(ours, ref)
            p, npub = mean_rmsd(pub, ref)
            chk[n] = dict(ours_A=None if o is None else round(o, 3), pub_A=None if p is None else round(p, 3), n_ours=no, n_pub=npub)
            if o is not None and p is not None and p > 0:
                chk[n]["ratio"] = round(o / p, 3)
        ok = [v for v in chk.values() if "ratio" in v]
        if ok:
            med = float(np.median([v["ratio"] for v in ok]))
            rs = float(spearmanr([v["ours_A"] for v in ok], [v["pub_A"] for v in ok])[0]) if len(ok) >= 3 else None
            rs = None if rs is None or np.isnan(rs) else rs
            res["install_check"] = dict(n=len(ok), median_ratio=round(med, 3), spearman=None if rs is None else round(rs, 3),
                                        passed=bool(0.75 <= med <= 1.33 and rs is not None and rs >= 0.70), per_protein=chk)
        else:
            res["install_check"] = dict(n=0, passed=False, per_protein=chk)
    # výpis
    print("GPU:", "; ".join(gpus) or "?", f"(úloha {R.name})" if R else "")
    for n in res["notes"]:
        print("POZOR:", n)
    for m, d in res["models"].items():
        f = d.get("fit")
        print(f"\n{m}: {len(d['per_protein'])} proteinů" + (f", chybí v sadě B: {d['missing_B']}" if d["missing_B"] else ""))
        if f:
            print(f"  fit t = {f['a']:.3f} + {f['b']:.3g}·L^{f['p']:.2f} (rel. RMS {f['rel_rms']:.3f}); průměr přes 82 délek "
                  f"{d['mean82_pred_s']:.2f} s" + (f", publikace A100 {d['pub_A100_s']} s, poměr {d['ratio_ours_to_pub']:.2f}" if "pub_A100_s" in d else ""))
        for n, v in sorted(d["per_protein"].items(), key=lambda x: x[1]["L"] or 0):
            print(f"  {n:<8} L={v['L']:>4} {v['sada']:<2} n={v['n']:>2} medián {v['t_med']:>8.3f} s (min {v['t_min']:.3f}, max {v['t_max']:.3f}, "
                  f"první {v['t_first']:.2f})")
    if "install_check" in res:
        ic = res["install_check"]
        print(f"\nKontrola instalace (Z8a): {ic['n']} proteinů, medián poměru RMSD {ic.get('median_ratio')}, Spearman {ic.get('spearman')} → "
              + ("PROŠLA" if ic["passed"] else "NEPROŠLA"))
    Path(args.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print("Uloženo", args.out)


if __name__ == "__main__":
    main()
