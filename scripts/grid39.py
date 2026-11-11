"""Linie 39, doplňky k Z8 bez GPU (Gemini 39i, body 4a, 4b; preprint 3.4 a 3.7). Jen čte JSON sum39 (--gen) a gen39sum.
1) Účetní mřížka {plný afT_base, destil. afT_dist} × {250, 50, 18 vzorků}: zrychlení = MD / (S18 + n · t_gen · v_gen / v_B),
   po proteinech, medián a jednostranné 95% meze bootstrapu mediánu (10 000, rng 39). Mění se jen cena; kvalita S18 je
   ve všech buňkách ta z 250 plných vzorků (předpoklad, kromě buňky plný 250).
2) Wall-clock řádek: MD z krystalu včetně stavby jednoho boxu na CPU (md_cost_build), S18 včetně stavby 18 boxů (build_s)
   a generátor včetně režie volání (načtení modelu, příprava vstupů: overhead_s volání sady B / počet proteinů v něm).
   Vše sériově na jedné GPU (horní odhad ceny CPU částí). MSA (server ColabFold) ani zde.
3) Bod zvratu (preprint 3.7): nejvyšší t_gen na vzorek (s na GPU měření), při kterém zrychlení = 1,5× (tgen_max_gen_s ze sum39),
   medián přes proteiny a počet proteinů, kde změřený t_gen je pod ním.
    python grid39.py --sum sum39_S18_s0_gen.json [--sum sum39_S18_s1_gen.json] --gensum gen39b_sum.json --out grid39.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

MOD = {"plný": "afT_base", "destil.": "afT_dist"}


def stats(v, rng, B=10000):
    v = np.asarray(v, float)
    m = np.median(v[rng.integers(0, len(v), (B, len(v)))], axis=1)
    return dict(median=round(float(np.median(v)), 3), lower95=round(float(np.percentile(m, 5)), 3),
                upper95=round(float(np.percentile(m, 95)), 3), frac_ge_1_5=round(float(np.mean(v >= 1.5)), 3), n=len(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sum", action="append", required=True)
    ap.add_argument("--gensum", required=True)
    ap.add_argument("--seed", type=int, default=39)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    g = json.loads(Path(a.gensum).read_text())
    nB = {m: sum(1 for v in g["models"][m]["per_protein"].values() if v["sada"] == "B") for m in MOD.values()}
    ovh = {m: g["calls"][f"{m} B"]["overhead_s"] / nB[m] for m in MOD.values()}
    print("režie volání sady B na protein (s, GPU měření):", {m: round(v, 1) for m, v in ovh.items()})
    out = dict(overhead_per_protein_s=ovh, runs={})
    for sp in a.sum:
        S = json.loads(Path(sp).read_text())
        rng = np.random.default_rng(a.seed)
        rows = [r for r in S["rows"] if "speedup" in r and r.get("v_gen") and all(m in r.get("tgen", {}) for m in MOD.values())]
        res = dict(tag=S["tag"], v_src=S["v_src"], tstar=S["tstar"], n=len(rows), grid={}, wall={}, breakeven={})
        print(f"\n=== {S['tag']} (v {S['v_src']}, t* {S['tstar']}), {len(rows)} proteinů")
        for lab, m in MOD.items():
            for n in (250, 50, 18):
                d = {}
                for k in ("jádro", "TICA"):
                    v = [r[f"md_cost_{k}"] / (r["gpu_s18_s"] + n * r["tgen"][m]["t_s"] * r["v_gen"] / r["v_B"]) for r in rows]
                    d[k] = stats(v, rng)
                d["nevyplaci_se"] = d["jádro"]["upper95"] < 1.5 and d["TICA"]["upper95"] < 1.5
                res["grid"][f"{lab}, {n}"] = d
                print(f"  mřížka {lab:<7} {n:>3}: jádro {d['jádro']['median']:.2f} ({d['jádro']['upper95']:.2f}), "
                      f"TICA {d['TICA']['median']:.2f} ({d['TICA']['upper95']:.2f})" + ("  nevyplácí se" if d["nevyplaci_se"] else ""))
        for lab, m, n in (("plný", "afT_base", 250), ("destil.", "afT_dist", 50)):
            d = {}
            for k in ("jádro", "TICA"):
                v = [(r[f"md_cost_{k}"] + r["build_s"] / r["boxes"]) /
                     (r["gpu_s18_s"] + r["build_s"] + (n * r["tgen"][m]["t_s"] + ovh[m]) * r["v_gen"] / r["v_B"]) for r in rows]
                d[k] = stats(v, rng)
            res["wall"][f"{lab}, {n}"] = d
            print(f"  wall-clock {lab:<7} {n:>3}: jádro {d['jádro']['median']:.2f} ({d['jádro']['upper95']:.2f}), "
                  f"TICA {d['TICA']['median']:.2f} ({d['TICA']['upper95']:.2f})")
        bs = [r["build_s"] for r in rows]
        res["build_s"] = dict(median=round(float(np.median(bs)), 1), min=round(min(bs), 1), max=round(max(bs), 1))
        print(f"  stavba 18 boxů na CPU: medián {res['build_s']['median']} s ({res['build_s']['min']}–{res['build_s']['max']})")
        for key in ("jádro, 250 vzorků", "TICA, 250 vzorků", "jádro, 50 vzorků", "TICA, 50 vzorků"):
            n = int(key.split(", ")[1].split()[0])
            m = "afT_base" if n == 250 else "afT_dist"
            tm = [r["tgen_max_gen_s"][key] for r in rows if "tgen_max_gen_s" in r]
            below = sum(r["tgen"][m]["t_s"] < r["tgen_max_gen_s"][key] for r in rows if "tgen_max_gen_s" in r)
            res["breakeven"][key] = dict(median_s=round(float(np.median(tm)), 3), q25=round(float(np.percentile(tm, 25)), 3),
                                         q75=round(float(np.percentile(tm, 75)), 3), n_neg=int(sum(x <= 0 for x in tm)),
                                         model=m, n_measured_below=int(below), n=len(tm))
            b = res["breakeven"][key]
            print(f"  bod zvratu {key:<18}: medián {b['median_s']:.2f} s/vzorek [{b['q25']:.2f}–{b['q75']:.2f}], ≤ 0 u {b['n_neg']}; "
                  f"změřený {m} pod ním u {b['n_measured_below']} z {b['n']}")
        out["runs"][S["tag"] + "|" + S["v_src"] + "|" + S["tstar"]] = res
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("Uloženo", a.out)


if __name__ == "__main__":
    main()
