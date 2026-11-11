"""Linie 39, změna Z9 (prereg39): vyhodnocení S18d50 (S18 z 50 vzorků destilovaného AlphaFlow) proti S18_s0. Jen čte JSON.
Z9-i  (primární): řádek účtování KEY (výchozí „Z8 i-b změřeno: destil., 50“) ze souhrnu sum39 pro S18d50_s0
      (--gen gen39b_sum.json --v-src xfer --tstar first). Medián zrychlení, horní a dolní mez jednostranného 95% bootstrap CI
      mediánu (10 000 výběrů přes proteiny, rng 39 jako sum39). Nevyplácí se: horní mez < 1,5 v jádru i TICA.
      Vyplácí se v prostoru (jádro, TICA zvlášť): medián ≥ 1,5 a dolní mez > 1,0. Jinak nerozhodnuto.
Z9-ii (sekundární): proteiny s dobrým AF podle fáze A (af38, plný model, AF_T < 0,45; stejné jako S18_s0): S18d50 porazí
      v LOO TICA 100ns repliku (eval39, shodné n) u ≥ 75 % a Rama ≤ 0,08 i χ1 ≤ 0,08 (rel38, 50–520 ps) u ≥ 75 %.
      Rozklad: LOO TICA, LOO jádro, Rama, χ1, průnik (původní (ii)), pro S18d50 i S18_s0.
Z9-iii (sekundární): párový ΔJS = JS(S18d50) − JS(S18_s0) přes proteiny s oběma; destilace kvalitu nezhorší, když medián
      ΔJS ≤ +0,03 v jádru i TICA. Primárně (Gemini 39j) JS všech snímků okna 50–520 ps proti sjednocení replik ze sub38
      (ana/sub39_<N>_<TAG>.json, js_union, n ≈ 8 460), sekundárně af38 (ana/af39_<N>_<TAG>_b30.json, běh 250 rovnoměrně).
      Měřítko šumu: týž rozdíl S18_s1 − S18_s0 (popisně).
Fyzikální selhání (prereg Z9 po Gemini 39k): protein, jehož běh skončil s hybrid_summary_partial.json (> 2 boxy selhaly
      nekonečnou energií nebo NaN, < 16 platných), se počítá jako zrychlení 0× (Z9-i) a jako neúspěch (Z9-ii); v Z9-iii chybí
      a hlásí se. Chybějící běh bez partial (technický výpadek) se má před vyhodnocením spustit znovu; jinak se hlásí jako chybějící.
      Z9-i je nerozhodnuto, jen když platných + fyzikálně selhaných < 20.
Popisně: AF_dist50 (50) a AF_MD_templ (250) proti ATLAS z af39 téhož běhu (n se liší, JS závisí na n, jen orientačně).
    python cmp39d.py --root l39 --sum sum39_S18d50_s0_gen.json --tag S18d50_s0 --base S18_s0 --ref2 S18_s1 --out cmp39d.json
"""
import argparse
import json
from pathlib import Path

import numpy as np


def jload(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def boot(v, rng, B=10000):
    v = np.asarray(v, float)
    m = np.median(v[rng.integers(0, len(v), (B, len(v)))], axis=1)
    return float(np.percentile(m, 5)), float(np.percentile(m, 95))


def run_js(R, N, tag):
    a = jload(R / "ana" / f"af39_{N}_{tag}_b30.json")
    if not a:
        return None
    rn = [v for k, v in a["rows"].items() if k.endswith(", 250 rovnoměrně") and not k.startswith("ATLAS")]
    return {"jádro": rn[0]["core"]["JS"], "TICA": rn[0]["tica"]["JS"]} if rn else None


def sub_js(R, N, tag):
    s = jload(R / "ana" / f"sub39_{N}_{tag}.json")
    if not s or not s.get("runs"):
        return None
    r = next(iter(s["runs"].values()))
    return {k: r[k]["js_union"] for k in ("jádro", "TICA")}


def comps(R, N, tag):
    """Složky původního kritéria (ii) pro jeden běh: LOO TICA, LOO jádro, Rama ≤ 0,08, χ1 ≤ 0,08."""
    e = jload(R / "ana" / f"eval39_{N}_{tag}.json")
    rl = jload(R / "ana" / f"rel39_{N}_{tag}.json")
    if not (e and rl and "50–520" in rl.get("windows", {})):
        return None
    er = next(iter(e["runs"].values()))
    w = rl["windows"]["50–520"]
    d = dict(loo_tica=bool(er["TICA"]["run_beats_rep100_loo"]), loo_core=bool(er["jádro"]["run_beats_rep100_loo"]),
             rama=w["rama"] is not None and w["rama"] <= 0.08, chi1=w["chi1"] is not None and w["chi1"] <= 0.08)
    d["vse"] = all(d.values())
    d["z9ii"] = d["loo_tica"] and d["rama"] and d["chi1"]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--sum", required=True, help="JSON sum39 pro --tag (S18d50_s0)")
    ap.add_argument("--tag", default="S18d50_s0")
    ap.add_argument("--base", default="S18_s0")
    ap.add_argument("--ref2", default="S18_s1")
    ap.add_argument("--key", default="Z8 i-b změřeno: destil., 50")
    ap.add_argument("--good", type=float, default=0.45)
    ap.add_argument("--seed", type=int, default=39)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    R = Path(a.root)
    S = json.loads(Path(a.sum).read_text())
    assert S["tag"] == a.tag, f"souhrn je pro {S['tag']}, ne {a.tag}"
    rng = np.random.default_rng(a.seed)
    res = dict(tag=a.tag, base=a.base, key=a.key, v_src=S.get("v_src"), tstar=S.get("tstar"), n_list=S.get("n_list"))
    ok = [r for r in S["rows"] if a.key in r.get("speedup", {})]
    phys, miss = [], []
    for r in S["rows"]:
        if a.key in r.get("speedup", {}):
            continue
        d = R / "runs" / f"hyb39_{r['name']}_{a.tag}"
        (phys if (d / "hybrid_summary_partial.json").exists() and not (d / "hybrid_summary.json").exists() else miss).append(r["name"])
    res.update(n_ok=len(ok), phys_fail=phys, missing=miss)
    print(f"  fyzikálně selhalo (< 16 boxů, počítá se 0×): {phys or 'žádný'}; chybí (technicky, spustit znovu): {miss or 'žádný'}")
    print(f"=== Z9 {a.tag}: {len(ok)} z {S.get('n_list')} proteinů s účtováním „{a.key}“ (v {S.get('v_src')}, t* {S.get('tstar')})")
    if S.get("v_src") != "xfer" or S.get("tstar") != "first":
        print("  POZOR: Z9-i předepisuje sum39 --v-src xfer --tstar first")
    # Z9-i
    z1 = {}
    for k in ("jádro", "TICA"):
        v = [r["speedup"][a.key][k] for r in ok] + [0.0] * len(phys)
        lo, up = boot(v, rng) if v else (None, None)
        med = float(np.median(v)) if v else None
        z1[k] = dict(median=med, lower95=lo, upper95=up, frac_ge_1_5=float(np.mean(np.array(v) >= 1.5)) if v else None,
                     vyplaci_se=bool(v) and med >= 1.5 and lo > 1.0)
    z1["nevyplaci_se"] = bool(ok) and z1["jádro"]["upper95"] < 1.5 and z1["TICA"]["upper95"] < 1.5
    z1["n_eval"] = len(ok) + len(phys)
    z1["verdikt"] = ("nerozhodnuto (< 20 proteinů)" if z1["n_eval"] < 20 else "NEVYPLÁCÍ SE" if z1["nevyplaci_se"] else
                     "VYPLÁCÍ SE v " + " i ".join(k for k in ("jádro", "TICA") if z1[k]["vyplaci_se"])
                     if any(z1[k]["vyplaci_se"] for k in ("jádro", "TICA")) else "nerozhodnuto")
    res["Z9_i"] = z1
    for k in ("jádro", "TICA"):
        d = z1[k]
        if d["median"] is not None:
            print(f"  Z9-i {k:<5} medián {d['median']:.2f} (dolní mez {d['lower95']:.2f}, horní mez {d['upper95']:.2f}, "
                  f"≥ 1,5× {d['frac_ge_1_5']:.0%})")
    print(f"  Z9-i verdikt: {z1['verdikt']}")
    # Z9-ii
    good = [r["name"] for r in S["rows"] if r.get("af_tica") is not None and r["af_tica"] < a.good]
    z2 = {}
    for tag in (a.tag, a.base):
        c = {N: comps(R, N, tag) for N in good}
        if tag == a.tag:
            for N in good:
                if N in phys:
                    c[N] = dict(loo_tica=False, loo_core=False, rama=False, chi1=False, vse=False, z9ii=False, phys_fail=True)
        c = {N: v for N, v in c.items() if v}
        n = len(c)
        cnt = {f: sum(v[f] for v in c.values()) for f in ("loo_tica", "loo_core", "rama", "chi1", "vse", "z9ii")}
        z2[tag] = dict(n_good=n, counts=cnt, per_protein=c,
                       met=n >= 4 and cnt["loo_tica"] / n >= 0.75 and sum(v["rama"] and v["chi1"] for v in c.values()) / n >= 0.75)
        print(f"  Z9-ii {tag}: dobrý AF {n}, LOO TICA {cnt['loo_tica']}, LOO jádro {cnt['loo_core']}, Rama {cnt['rama']}, "
              f"χ1 {cnt['chi1']}, průnik 4 {cnt['vse']}, LOO TICA + Rama + χ1 {cnt['z9ii']}"
              + (f" → {'SPLNĚNO' if z2[tag]['met'] else 'NESPLNĚNO'}" if tag == a.tag else " (srovnání)"))
    res["Z9_ii"] = z2
    # Z9-iii
    names = [r["name"] for r in S["rows"]]
    z3 = {}
    for src, fn in (("sub38 plné n", sub_js), ("af38 n=250", run_js)):
        for other in (a.tag, a.ref2):
            d = {N: (fn(R, N, other), fn(R, N, a.base)) for N in names}
            d = {N: {k: x[k] - y[k] for k in ("jádro", "TICA")} for N, (x, y) in d.items() if x and y}
            z = dict(n=len(d), per_protein=d, src=src)
            for k in ("jádro", "TICA"):
                v = [x[k] for x in d.values()]
                if v:
                    lo, up = boot(v, rng)
                    z[k] = dict(median=float(np.median(v)), lower95=lo, upper95=up)
            z["met"] = bool(d) and all(z[k]["median"] <= 0.03 for k in ("jádro", "TICA"))
            z3[f"{other} | {src}"] = z
            if d:
                print(f"  Z9-iii [{src}{', PRIMÁRNÍ' if src.startswith('sub38') else ''}] ΔJS {other} − {a.base} (n = {len(d)}): jádro medián {z['jádro']['median']:+.3f} "
                      f"(CI {z['jádro']['lower95']:+.3f} až {z['jádro']['upper95']:+.3f}), TICA {z['TICA']['median']:+.3f} "
                      f"({z['TICA']['lower95']:+.3f} až {z['TICA']['upper95']:+.3f})"
                      + (f" → {'SPLNĚNO' if z['met'] else 'NESPLNĚNO'} (≤ +0,03)" if other == a.tag else " (šum seedu, popisně)"))
    res["Z9_iii"] = z3
    # popisně: soubory AF proti ATLAS
    desc = {}
    for N in names:
        x = jload(R / "ana" / f"af39_{N}_{a.tag}_b30.json")
        if not x:
            continue
        row = {}
        for k, v in x["rows"].items():
            if k.startswith(("AF_MD_templ (", "AF_dist50 (")):
                row[k] = {"jádro": v["core"]["JS"], "TICA": v["tica"]["JS"]}
        desc[N] = row
    res["af_soubory"] = desc
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print("Uloženo", a.out)


if __name__ == "__main__":
    main()
