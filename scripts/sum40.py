"""sum40 (prereg39 Z10, opravený přepočet po externím posudku 11. 11. 2026) = sum39 (SHA 539504c3…) + patch40.py.
t* ze sub40 (oddělená reference po foldech, shodné n, projekce a hrany jen z R_-k), LOO / (iii) / kontrola 2 × 5,4 ns
ze sub40, čas selhaných boxů v ceně (fail40.json), fyzikální selhání 0×, rozhodnutí (i) třícestně jako Z9-i.
    python sum40.py --root l39 --tag S18_s0 --gen gen39b_sum.json --v-src xfer --fail fail40.json --out sum40_S18_s0.json
Původní popis sum39:
Linie 39: souhrn fáze B podle prereg39 (kritéria (i), (ii), (iii), kontroly Z7). Jen čte JSON.
Kořen --root má strukturu jako l39 na MetaCentru: listB.txt, atlas_test.csv, af38/af38_<N>_b30.json (fáze A, bez --sys),
ana/{sub39,eval39,rel39}_<N>_<TAG>.json, ana/af39_<N>_<TAG>_b30.json, bench/bench_p2_<N>.json,
runs/hyb39_<N>_<TAG>/hybrid_summary.json.
Cena (prereg39 + Z1–Z3): GPU sekundy na téže GPU. MD z krystalu = t* (sub38, proti sjednocení, shodné n; nedosaženo →
100 ns jako dolní mez) / v_p2 + průměrná (minimalizace + ekvilibrace) boxu S18. S18 = součet min + ekv + MD 18 boxů.
Generátor = T_pub · L^p / průměr_82(L^p) · n vzorků (A100 = A4000). Zrychlení = MD / (S18 + generátor).
Rozhodnutí (i): horní mez jednostranného 95% bootstrap CI mediánu (10 000 výběrů přes proteiny) < 1,5× v jádru i TICA.
Změna Z8/Z8a (--gen gen39_sum.json z gen39sum.py): naměřená cena generátoru na GPU měření převedená na GPU úlohy B,
C_gen = n · t_gen(N) · v_gen(N) / v_B(N) (bench39 téhož p2_<N>; chybí-li na GPU měření, kotva p2_7jfl_C škálovaná počtem
atomů, označeno). Rozhodující je změřené účtování, když kontrola instalace prošla a plný model má ≥ 20 proteinů sady B;
jinak publikovaná cena (Z1–Z3). Citlivost: jen proteiny, jejichž úloha B běžela na stejném modelu GPU (bez přepočtu).
Oprava 9. 11. 21:50/22:20 (chyba implementace, ne změna metody): b39.pbs měřil bench_p2_<N> jen v první úloze proteinu,
druhý seed a K18 mohly běžet na jiné GPU. --v-src xfer (hlavní): úloha, která bench měřila (--bench-src, výchozí S18_s0),
nebo úloha na stejném modelu GPU → bench; jinak bench × poměr propustnosti MD obou úloh téhož proteinu v atom·ns/den
(MD boxů S18/K18: týž integrátor, čtení poloh po 1 ps jako bench39; boxy mají jiný počet atomů než p2, proto
normalizace atomy). --v-src s18md = rychlost MD boxů úlohy bez normalizace (jen pro srovnání), bench = původní výpočet.
--tstar: first = prereg (první t, kde replika ≤ běh při shodném n), ostatní jen citlivost (post hoc, neregistrováno):
last = t, od kterého replika zůstává ≤ běh, union/loo = bez shodného n (sub38 t_equiv_union_ns / t_equiv_loo_ns).
    python sum39.py --root l39 --tag S18_s0 [--gen gen39_sum.json] [--v-src s18md] [--tstar first] --out sum39_S18_s0.json
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

T_PUB = {"plný": 38.0, "destil.": 3.8}
ACCT = [("i-a plný, 250", "plný", 250), ("i-b destil., 50", "destil.", 50), ("destil., 250", "destil.", 250),
        ("nula (vzorky existují)", None, 0)]
PS = [2.0, 1.5, 1.0, 0.0]
GEN_ACCT = [("Z8 i-a změřeno: plný, 250", "afT_base", 250), ("Z8 i-b změřeno: destil., 50", "afT_dist", 50),
            ("změřeno: destil., 250", "afT_dist", 250), ("změřeno 12l: plný, 250", "afT12_base", 250),
            ("změřeno 12l: destil., 50", "afT12_dist", 50), ("změřeno ESMFlow: plný, 250", "esT_base", 250),
            ("změřeno ESMFlow: destil., 50", "esT_dist", 50)]


def gpu_name(s):
    s = s.strip()
    s = s[5:] if s.startswith("GPU: ") else s
    return s.split(",")[0].strip()


def gen_time(gen, m, N, L):
    d = gen["models"].get(m)
    if not d:
        return None, None
    if N in d["per_protein"]:
        return d["per_protein"][N]["t_med"], "změřeno"
    f = d.get("fit")
    return (f["a"] + f["b"] * L ** f["p"], "z fitu") if f else (None, None)


def t_last(rep, run):
    """t (ns), od kterého je průměr replik trvale ≤ běh (interpolace v posledním přechodu); None, když ani na konci ne."""
    d = [(a[0], a[1] - b[1]) for a, b in zip(rep, run)]
    if not d or d[-1][1] > 0:
        return None
    pos = [i for i, (_, x) in enumerate(d) if x > 0]
    if not pos:
        return float(d[0][0])
    i = pos[-1] + 1
    f = d[i - 1][1] / (d[i - 1][1] - d[i][1])
    return float(d[i - 1][0] + f * (d[i][0] - d[i - 1][0]))


def jload(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def boot_q(v, rng, qs, B=10000):
    """Percentily bootstrapového mediánu (výběry přes proteiny)."""
    v = np.asarray(v, float)
    if len(v) == 0:
        return [float("nan")] * len(qs)
    m = np.median(v[rng.integers(0, len(v), (B, len(v)))], axis=1)
    return [float(np.percentile(m, q)) for q in qs]


def boot_upper(v, rng, B=10000, q=95):
    v = np.asarray(v, float)
    if len(v) == 0:
        return None
    m = np.median(v[rng.integers(0, len(v), (B, len(v)))], axis=1)
    return float(np.percentile(m, q))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--tag", default="S18_s0")
    ap.add_argument("--good", type=float, default=0.45)
    ap.add_argument("--seed", type=int, default=39)
    ap.add_argument("--gen", default=None)
    ap.add_argument("--v-src", choices=["bench", "s18md", "xfer"], default="bench")
    ap.add_argument("--bench-src", default=None, help="JSON {protein: TAG úlohy, která měřila bench}; výchozí S18_s0")
    ap.add_argument("--tstar", choices=["first", "last"], default="first")
    ap.add_argument("--metric", choices=["js30", "js15", "ed"], default="js30")
    ap.add_argument("--fail", default=None, help="fail40.json: čas selhaných boxů z logů")
    ap.add_argument("--fail-mode", choices=["none", "log", "upper"], default="log")
    ap.add_argument("--exclude", default="", help="čárkou oddělené proteiny vynechané z vyhodnocení (podskupina)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    R = Path(args.root)
    gen = jload(args.gen) if args.gen else None
    gen_gpu = gpu_name(gen["gpu"][0]) if gen and gen.get("gpu") else None
    anchor = gen["bench_gen"].get("7jfl_C@anchor") if gen else None
    rng = np.random.default_rng(args.seed)
    Lall = {r["name"]: len(r["seqres"]) for r in csv.DictReader(open(R / "atlas_test.csv"))}
    norm = {p: float(np.mean([L ** p for L in Lall.values()])) for p in PS}
    names = [ln.split()[0] for ln in (R / "listB.txt").read_text().split("\n") if ln.strip()]
    excl = {x for x in args.exclude.split(",") if x}
    names = [n for n in names if n not in excl]
    fail = jload(args.fail) if args.fail else {}
    bsrc = jload(args.bench_src) if args.bench_src else {}

    def job_info(N, tag):
        hh = jload(R / "runs" / f"hyb39_{N}_{tag}" / "hybrid_summary.json")
        g = sorted(R.glob(f"logs/gpu_{N}_{tag}_*.txt"))
        gpu = gpu_name(g[-1].read_text().split("\n")[0]) if g else None
        if not hh or not hh["wall_breakdown_s"].get("md"):
            return gpu, None, None
        vs = hh.get("md_free_ns", 0) / hh["wall_breakdown_s"]["md"] * 86400.0
        na = [j["n_atoms"] for j in hh.get("jump_records", []) if j.get("n_atoms")]
        return gpu, vs, (float(np.mean(na)) if na else None)
    rows, notes = [], []
    for N in names:
        row = dict(name=N, L=Lall[N])
        a = jload(R / "af38" / f"af38_{N}_b30.json")
        if a:
            af = [v for k, v in a["rows"].items() if k.startswith("AF_MD_templ (")]
            reps = [v["tica"]["JS"] for k, v in a["rows"].items() if k.startswith("ATLAS R") and "celá, 250" in k]
            if af:
                row["af_tica"], row["af_core"] = af[0]["tica"]["JS"], af[0]["core"]["JS"]
                row["good"] = row["af_tica"] < args.good
                row["good2"] = row["af_tica"] <= float(np.mean(reps)) + 0.10 if reps else None
        h = jload(R / "runs" / f"hyb39_{N}_{args.tag}" / "hybrid_summary.json")
        s40 = jload(R / "ana40" / f"sub40_{N}.json")
        s = (s40 or {}).get("runs", {}).get(args.tag)
        b = jload(R / "bench" / f"bench_p2_{N}.json")
        e = s
        rl = jload(R / "ana" / f"rel39_{N}_{args.tag}.json")
        a9 = jload(R / "ana" / f"af39_{N}_{args.tag}_b30.json")
        if not h and (R / "runs" / f"hyb39_{N}_{args.tag}" / "hybrid_summary_partial.json").exists():
            row.update(phys_fail=True, loo_beats=False, loo_beats_tica=False)
            rows.append(row)
            continue
        if not (h and s and b):
            row["missing"] = [k for k, v in (("run", h), ("sub40", s), ("bench", b)) if not v]
            rows.append(row)
            continue
        wb = h["wall_breakdown_s"]
        nb = h["jumps"]
        row["boxes"] = nb
        row["gpu_s18_s"] = wb.get("min", 0) + wb.get("equil", 0) + wb.get("md", 0)
        row["build_s"] = wb.get("build", 0)
        eq1 = (wb.get("min", 0) + wb.get("equil", 0)) / nb
        fi = fail.get(N, {}).get(args.tag, {})
        nf = max(len(h.get("failed", [])), int(fi.get("n", 0)))
        up = nf * row["gpu_s18_s"] / nb
        fs = 0.0 if args.fail_mode == "none" else (up if args.fail_mode == "upper" or fi.get("log_s") is None
                                                   else float(fi["log_s"]))
        row.update(fail_n=nf, fail_log_s=fi.get("log_s"), fail_upper_s=round(up, 1), fail_added_s=round(fs, 1),
                   gpu_s18_ok_s=row["gpu_s18_s"])
        row["gpu_s18_s"] += fs
        row["gpu_B"], v_s18, nat = job_info(N, args.tag)
        row["v_bench_nsd"], row["v_s18md_nsd"] = b["ns_per_day"], None if v_s18 is None else round(v_s18, 1)
        row["v_ratio_s18md_bench"] = None if v_s18 is None else round(v_s18 / b["ns_per_day"], 3)
        row["n_atoms_box"], row["n_atoms_p2"] = nat, b.get("n_atoms")
        row["thr_ratio_box_bench"] = (round(v_s18 * nat / (b["ns_per_day"] * b["n_atoms"]), 3)
                                      if v_s18 and nat and b.get("n_atoms") else None)
        src = bsrc.get(N, "S18_s0")
        row["bench_src"] = src
        vB, row["v_rule"] = b["ns_per_day"], "bench"
        if args.v_src == "s18md" and v_s18:
            vB, row["v_rule"] = v_s18, "s18md"
        elif args.v_src == "xfer" and args.tag != src:
            g0, v0, n0 = job_info(N, src)
            if row["gpu_B"] and g0 and row["gpu_B"] == g0:
                row["v_rule"] = f"bench (stejná GPU jako {src})"
            elif v_s18 and nat and v0 and n0:
                vB = b["ns_per_day"] * (v_s18 * nat) / (v0 * n0)
                row["v_rule"] = f"bench × propustnost {args.tag}/{src} = {(v_s18 * nat) / (v0 * n0):.2f}"
            else:
                row["v_rule"] = "bench (převod nelze, chybí data)"
        row["v_B_nsd"] = round(vB, 1)
        v = vB / 86400.0  # ns za sekundu
        run = s
        for k in ("jádro", "TICA"):
            t = run[k][args.metric]["t_first" if args.tstar == "first" else "t_last"]
            cens = t is None
            t = 100.0 if cens else float(t)
            row[f"t_{k}"], row[f"cens_{k}"] = t, cens
            row[f"md_cost_{k}"] = t / v + eq1
            row[f"md_cost_build_{k}"] = row[f"md_cost_{k}"] + row["build_s"] / nb
        if e:
            er = e
            row["loo_beats"] = bool(er["jádro"]["loo100"]["run_beats"] and er["TICA"]["loo100"]["run_beats"])
            row["loo_beats_tica"] = bool(er["TICA"]["loo100"]["run_beats"])
            row["loo"] = {k: [round(er[k]["loo100"]["run"], 3), round(er[k]["loo100"]["md"], 3)] for k in ("jádro", "TICA")}
            row["lose_10_9"] = {k: bool(er[k]["at10_9"]["run_worse"]) for k in ("jádro", "TICA")}
            row["ctrl_2x5_4"] = {k: [round(er[k]["ctrl_2x5_4"]["run"], 4), round(er[k]["ctrl_2x5_4"]["ctrl"], 4)]
                                 for k in ("jádro", "TICA")}
        if rl and "50–520" in rl["windows"]:
            w = rl["windows"]["50–520"]
            row["rama"], row["chi1"] = w["rama"], w["chi1"]
        if a9:
            af = [v for k, v in a9["rows"].items() if k.startswith("AF_MD_templ (")]
            rn = [v for k, v in a9["rows"].items() if k.endswith(", 250 rovnoměrně") and not k.startswith("ATLAS")]
            if af and rn:
                row["dJS_run_minus_af"] = {k: round(rn[0][kk]["JS"] - af[0][kk]["JS"], 3) for k, kk in (("jádro", "core"), ("TICA", "tica"))}
                row["dJS_run_minus_af"].update({k: round(rn[0][k] - af[0][k], 3) for k in ("rama", "chi1")
                                                if rn[0][k] is not None and af[0][k] is not None})
        sp = {}
        for lab, tp, n in ACCT:
            for p in PS:
                g = 0.0 if tp is None else T_PUB[tp] * Lall[N] ** p / norm[p] * n
                sp[f"{lab} | p={p:g}"] = {k: row[f"md_cost_{k}"] / (row["gpu_s18_s"] + g) for k in ("jádro", "TICA")}
                if tp is None:
                    break
        if gen:
            bg = gen["bench_gen"].get(N)
            if bg:
                v_gen, vsrc = bg["ns_per_day"], "změřeno"
            elif anchor:
                v_gen, vsrc = anchor["ns_per_day"] * anchor["n_atoms"] / b["n_atoms"], "kotva 7jfl_C škálovaná atomy"
            else:
                v_gen, vsrc = None, None
            row["v_gen"], row["v_gen_src"], row["v_B"] = v_gen, vsrc, vB
            row["tgen"] = {}
            for lab, m, n in GEN_ACCT:
                t, src = gen_time(gen, m, N, Lall[N])
                if t is None or v_gen is None:
                    continue
                row["tgen"][m] = dict(t_s=round(t, 4), src=src)
                c = n * t * v_gen / vB
                sp[lab] = {k: row[f"md_cost_{k}"] / (row["gpu_s18_s"] + c) for k in ("jádro", "TICA")}
                if row["gpu_B"] and row["gpu_B"] == gen_gpu:
                    sp["stejná GPU | " + lab] = {k: row[f"md_cost_{k}"] / (row["gpu_s18_s"] + n * t) for k in ("jádro", "TICA")}
            if v_gen:
                row["tgen_max_gen_s"] = {kk: round(v * vB / v_gen, 3) for kk, v in
                                         {f"{k}, {n} vzorků": (row[f"md_cost_{k}"] / 1.5 - row["gpu_s18_s"]) / n
                                          for k in ("jádro", "TICA") for n in (50, 250)}.items()}
        row["speedup"] = sp
        row["speedup_serial_build_i-a_p2"] = {k: row[f"md_cost_build_{k}"] / (row["gpu_s18_s"] + row["build_s"] +
                                              T_PUB["plný"] * Lall[N] ** 2 / norm[2.0] * 250) for k in ("jádro", "TICA")}
        row["tgen_max_s"] = {f"{k}, {n} vzorků": round((row[f"md_cost_{k}"] / 1.5 - row["gpu_s18_s"]) / n, 2)
                             for k in ("jádro", "TICA") for n in (50, 250)}
        rows.append(row)
    keys0 = []
    for r in rows:
        keys0 += [k for k in r.get("speedup", {}) if k not in keys0]
    for r in rows:
        if r.get("phys_fail"):
            r["speedup"] = {key: {"jádro": 0.0, "TICA": 0.0} for key in keys0}
    ok = [r for r in rows if "speedup" in r]
    okf = [r for r in ok if not r.get("phys_fail")]
    res = dict(tag=args.tag, v_src=args.v_src, tstar=args.tstar, metric=args.metric, fail_mode=args.fail_mode,
               exclude=sorted(excl), n_list=len(names), n_ok=len(ok), n_phys_fail=len(ok) - len(okf), rows=rows, criteria={})
    print(f"=== Z10 (opravený přepočet po externím posudku) {args.tag}: {len(ok)} z {len(names)} proteinů "
          f"(fyzikální selhání 0×: {len(ok) - len(okf)}); rychlost MD z {args.v_src}, t* {args.tstar}, metrika {args.metric}, "
          f"selhané boxy {args.fail_mode}" + (f", bez {sorted(excl)}" if excl else ""))
    print(f"  selhané boxy: {sum(r.get('fail_n', 0) for r in okf)} (přičteno {sum(r.get('fail_added_s', 0) for r in okf):.0f} s, "
          f"z logu u {sum(r.get('fail_log_s') is not None for r in okf if r.get('fail_n'))} běhů)")
    for r in rows:
        if "missing" in r:
            print(f"  {r['name']}: chybí {r['missing']}")
    # (i)
    crit = {}
    keys = []
    for r in ok:
        keys += [k for k in r["speedup"] if k not in keys]
    for key in keys:
        d = {}
        sub = [r for r in ok if key in r["speedup"]]
        d["n"] = len(sub)
        for k in ("jádro", "TICA"):
            v = [r["speedup"][key][k] for r in sub]
            lo, hi = boot_q(v, rng, (5, 95))
            d[k] = dict(median=round(float(np.median(v)), 3), upper95=round(hi, 3), lower95=round(lo, 3),
                        frac_ge_1_5=round(float(np.mean(np.array(v) >= 1.5)), 3),
                        vyplácí_se=bool(np.median(v) >= 1.5 and lo > 1.0))
        d["nevyplácí_se"] = bool(d["jádro"]["upper95"] < 1.5 and d["TICA"]["upper95"] < 1.5)
        yes = [k for k in ("jádro", "TICA") if d[k]["vyplácí_se"]]
        d["verdikt"] = "NEVYPLÁCÍ SE" if d["nevyplácí_se"] else (f"VYPLÁCÍ SE ({', '.join(yes)})" if yes else "NEROZHODNUTO")
        crit[key] = d
        print(f"  (i) {key:<46} n={d['n']:>2} jádro medián {d['jádro']['median']:.2f} (90% CI {d['jádro']['lower95']:.2f}–"
              f"{d['jádro']['upper95']:.2f}, ≥1,5× {d['jádro']['frac_ge_1_5']:.0%}), TICA {d['TICA']['median']:.2f} "
              f"({d['TICA']['lower95']:.2f}–{d['TICA']['upper95']:.2f}, {d['TICA']['frac_ge_1_5']:.0%}) → {d['verdikt']}")
    res["criteria"]["i"] = crit
    if gen:
        ic = gen.get("install_check", {})
        nb = len([r for r in ok if "afT_base" in r.get("tgen", {}) and r["tgen"]["afT_base"]["src"] == "změřeno"])
        dec = bool(ic.get("passed")) and nb >= 20
        res["criteria"]["decisive"] = dict(install_passed=bool(ic.get("passed")), n_measured_base=nb,
                                           accounting="změřené (Z8)" if dec else "publikované (Z1–Z3)", gpu_gen=gen_gpu)
        print(f"  ROZHODUJÍCÍ účtování: {'změřené Z8 (i-a, i-b změřeno)' if dec else 'publikované Z1–Z3 (p = 2, citlivost p)'}; "
              f"kontrola instalace {'prošla' if ic.get('passed') else 'NEPROŠLA/chybí'}, změřeno plné na {nb} proteinech, GPU měření {gen_gpu}")
    cens = {k: sum(r[f"cens_{k}"] for r in okf) for k in ("jádro", "TICA")}
    res["criteria"]["censored"] = cens
    print(f"  cenzurováno (t* ≥ 100 ns): jádro {cens['jádro']}, TICA {cens['TICA']}")
    # (ii)
    for gk in ("good", "good2"):
        good = [r for r in ok if r.get(gk) and "loo_beats" in r]
        hit = [r for r in good if r["loo_beats"] and r.get("rama", 9) <= 0.08 and r.get("chi1", 9) <= 0.08]
        d = dict(n_good=len(good), n_hit=len(hit), names=[r["name"] for r in hit],
                 evaluated=len(good) >= 4, met=(len(good) >= 4 and len(hit) / len(good) >= 0.75))
        res["criteria"]["ii_" + gk] = d
        print(f"  (ii) {'primární' if gk == 'good' else 'sekundární'} dobrý AF: {len(good)} proteinů, S18 lepší než 100ns "
              f"replika (LOO, shodné n) + Rama, χ1 ≤ 0,08: {len(hit)} → "
              + ("nehodnotí se (< 4)" if not d["evaluated"] else ("SPLNĚNO" if d["met"] else "NESPLNĚNO")))
    gt = [r for r in ok if r.get("good")]
    res["criteria"]["loo_tica_good"] = dict(n_good=len(gt), n_beats=sum(bool(r.get("loo_beats_tica")) for r in gt),
                                            n_all=len(ok), n_beats_all=sum(bool(r.get("loo_beats_tica")) for r in ok))
    print(f"  LOO TICA (100 ns, shodné n, exploratorně): S18 lepší u {res['criteria']['loo_tica_good']['n_beats']}/{len(gt)} "
          f"dobrých AF, celkem {res['criteria']['loo_tica_good']['n_beats_all']}/{len(ok)}")
    bad = [r for r in ok if r.get("good") is False and "lose_10_9" in r]
    res["criteria"]["iii"] = dict(n_bad=len(bad), lose_both=sum(r["lose_10_9"]["jádro"] and r["lose_10_9"]["TICA"] for r in bad),
                                  lose_any=sum(r["lose_10_9"]["jádro"] or r["lose_10_9"]["TICA"] for r in bad))
    print(f"  (iii) špatný AF: {len(bad)}, S18 horší než MD 10,9 ns (LOO, shodné n) v obou {res['criteria']['iii']['lose_both']}, "
          f"aspoň v jednom {res['criteria']['iii']['lose_any']}")
    cc = [r for r in okf if "ctrl_2x5_4" in r]
    res["criteria"]["ctrl_2x5_4"] = {}
    for k in ("jádro", "TICA"):
        dif = [r["ctrl_2x5_4"][k][0] - r["ctrl_2x5_4"][k][1] for r in cc]
        lo, hi = boot_q(dif, rng, (2.5, 97.5))
        dd = dict(n=len(dif), run_closer=int(sum(x < 0 for x in dif)), median_diff_run_minus_ctrl=round(float(np.median(dif)), 4)
                  if dif else None, ci95=[round(lo, 4), round(hi, 4)])
        res["criteria"]["ctrl_2x5_4"][k] = dd
        print(f"  kontrola 2 × 5,4 ns proti R_k ({k}): S18 blíž u {dd['run_closer']}/{dd['n']}, medián JS(S18) − JS(kontrola) "
              f"{dd['median_diff_run_minus_ctrl']} (95% CI {dd['ci95'][0]}–{dd['ci95'][1]})")
    print(f"\n{'protein':<8} {'L':>4} {'AF TICA':>7} {'t* j/T ns':>11} {'GPU S18 s':>9} {'zrychl. i-a p2 j/T':>18} "
          f"{'nula j/T':>10} {'LOO S18/rep j':>14} {'TICA':>12} {'Rama':>5} {'χ1':>5}")
    for r in okf:
        a = r["speedup"]["i-a plný, 250 | p=2"]
        z = r["speedup"]["nula (vzorky existují) | p=2"]
        lo = r.get("loo", {})
        print(f"{r['name']:<8} {r['L']:>4} {r.get('af_tica', float('nan')):>7.3f} "
              f"{('≥' if r['cens_jádro'] else '') + format(r['t_jádro'], '.1f'):>5}/{('≥' if r['cens_TICA'] else '') + format(r['t_TICA'], '.1f'):<5} "
              f"{r['gpu_s18_s']:>9.0f} {a['jádro']:>8.2f}/{a['TICA']:<8.2f} {z['jádro']:>5.2f}/{z['TICA']:<4.2f} "
              f"{lo.get('jádro', ['', ''])[0]!s:>6}/{lo.get('jádro', ['', ''])[1]!s:<7} {lo.get('TICA', ['', ''])[0]!s:>5}/{lo.get('TICA', ['', ''])[1]!s:<6}"
              f" {r.get('rama', float('nan')):>5.3f} {r.get('chi1', float('nan')):>5.3f}")
    print(f"\n{'protein':<8} {'GPU úlohy':<28} {'bench ns/d':>10} {'box MD ns/d':>11} {'atomy box/p2':>13} {'propust. box/bench':>18} "
          f"{'v_B ns/d':>8}  pravidlo")
    for r in okf:
        print(f"{r['name']:<8} {str(r.get('gpu_B'))[:28]:<28} {r['v_bench_nsd']:>10.0f} {r['v_s18md_nsd'] or float('nan'):>11.0f} "
              f"{(r['n_atoms_box'] or float('nan')):>7.0f}/{(r['n_atoms_p2'] or float('nan')):<5.0f} "
              f"{r['thr_ratio_box_bench'] or float('nan'):>18.2f} {r['v_B_nsd']:>8.0f}  {r['v_rule']}")
    Path(args.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print("Uloženo", args.out)


if __name__ == "__main__":
    main()
