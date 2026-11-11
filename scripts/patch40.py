"""patch40: sum40.py = sum39.py (SHA 539504c3…) upravený pro opravený přepočet prereg39 Z10 (po externím posudku 11. 11.).
Změny (cena, rychlost MD v_B, účtování generátoru a výpis zůstávají jako v sum39):
- t* ze sub40 (ana40/sub40_<N>.json, běh podle --tag): --metric js30 (rozhoduje) | js15 | ed, --tstar first (rozhoduje) | last.
- LOO (100 ns), (iii) (10,9 ns) a kontrola 2 × 5,4 ns proti R_k ze sub40 místo eval39 (shodné n, projekce po foldech).
- Čas selhaných boxů (--fail fail40.json, --fail-mode log | upper | none): přičte se ke GPU času S18; log = čas z logu,
  kde chybí, horní odhad = počet selhání × průměrný čas úspěšného boxu (min + ekv + MD) téhož běhu.
- Fyzikální selhání (jen hybrid_summary_partial.json, Z9): zrychlení 0× ve všech účtováních, v (ii) neúspěch, v (iii) chybí.
- Rozhodnutí (i) třícestně jako Z9-i: nevyplácí se (horní mez jednostranného 95% CI mediánu < 1,5× v jádru i TICA),
  vyplácí se v prostoru (medián ≥ 1,5× a dolní mez > 1,0×), jinak nerozhodnuto; bootstrap 10 000 přes proteiny.
- --exclude: podskupina (19 proteinů bez pilotních šesti); výpis kontroly 2 × 5,4 ns (párový bootstrap mediánu rozdílu).
    python3 patch40.py   (v adresáři se sum39.py)
"""
import hashlib
from pathlib import Path

s = Path("sum39.py").read_text()
h = hashlib.sha256(s.encode()).hexdigest()
assert h.startswith("539504c3ed6e9c60"), f"sum39.py má jiné SHA {h[:16]}, nic neměním"
R = [
    ('"""Linie 39: souhrn fáze B podle prereg39 (kritéria (i), (ii), (iii), kontroly Z7). Jen čte JSON.\n',
     '"""sum40 (prereg39 Z10, opravený přepočet po externím posudku 11. 11. 2026) = sum39 (SHA 539504c3…) + patch40.py.\n'
     't* ze sub40 (oddělená reference po foldech, shodné n, projekce a hrany jen z R_-k), LOO / (iii) / kontrola 2 × 5,4 ns\n'
     'ze sub40, čas selhaných boxů v ceně (fail40.json), fyzikální selhání 0×, rozhodnutí (i) třícestně jako Z9-i.\n'
     '    python sum40.py --root l39 --tag S18_s0 --gen gen39b_sum.json --v-src xfer --fail fail40.json --out sum40_S18_s0.json\n'
     'Původní popis sum39:\n'
     'Linie 39: souhrn fáze B podle prereg39 (kritéria (i), (ii), (iii), kontroly Z7). Jen čte JSON.\n'),
    ('def boot_upper(v, rng, B=10000, q=95):\n',
     'def boot_q(v, rng, qs, B=10000):\n'
     '    """Percentily bootstrapového mediánu (výběry přes proteiny)."""\n'
     '    v = np.asarray(v, float)\n'
     '    if len(v) == 0:\n'
     '        return [float("nan")] * len(qs)\n'
     '    m = np.median(v[rng.integers(0, len(v), (B, len(v)))], axis=1)\n'
     '    return [float(np.percentile(m, q)) for q in qs]\n'
     '\n\n'
     'def boot_upper(v, rng, B=10000, q=95):\n'),
    ('    ap.add_argument("--tstar", choices=["first", "last", "union", "loo"], default="first")\n',
     '    ap.add_argument("--tstar", choices=["first", "last"], default="first")\n'
     '    ap.add_argument("--metric", choices=["js30", "js15", "ed"], default="js30")\n'
     '    ap.add_argument("--fail", default=None, help="fail40.json: čas selhaných boxů z logů")\n'
     '    ap.add_argument("--fail-mode", choices=["none", "log", "upper"], default="log")\n'
     '    ap.add_argument("--exclude", default="", help="čárkou oddělené proteiny vynechané z vyhodnocení (podskupina)")\n'),
    ('    names = [ln.split()[0] for ln in (R / "listB.txt").read_text().split("\\n") if ln.strip()]\n',
     '    names = [ln.split()[0] for ln in (R / "listB.txt").read_text().split("\\n") if ln.strip()]\n'
     '    excl = {x for x in args.exclude.split(",") if x}\n'
     '    names = [n for n in names if n not in excl]\n'
     '    fail = jload(args.fail) if args.fail else {}\n'),
    ('        s = jload(R / "ana" / f"sub39_{N}_{args.tag}.json")\n',
     '        s40 = jload(R / "ana40" / f"sub40_{N}.json")\n'
     '        s = (s40 or {}).get("runs", {}).get(args.tag)\n'),
    ('        e = jload(R / "ana" / f"eval39_{N}_{args.tag}.json")\n',
     '        e = s\n'),
    ('        if not (h and s and b):\n'
     '            row["missing"] = [k for k, v in (("run", h), ("sub38", s), ("bench", b)) if not v]\n',
     '        if not h and (R / "runs" / f"hyb39_{N}_{args.tag}" / "hybrid_summary_partial.json").exists():\n'
     '            row.update(phys_fail=True, loo_beats=False, loo_beats_tica=False)\n'
     '            rows.append(row)\n'
     '            continue\n'
     '        if not (h and s and b):\n'
     '            row["missing"] = [k for k, v in (("run", h), ("sub40", s), ("bench", b)) if not v]\n'),
    ('        eq1 = (wb.get("min", 0) + wb.get("equil", 0)) / nb\n',
     '        eq1 = (wb.get("min", 0) + wb.get("equil", 0)) / nb\n'
     '        fi = fail.get(N, {}).get(args.tag, {})\n'
     '        nf = max(len(h.get("failed", [])), int(fi.get("n", 0)))\n'
     '        up = nf * row["gpu_s18_s"] / nb\n'
     '        fs = 0.0 if args.fail_mode == "none" else (up if args.fail_mode == "upper" or fi.get("log_s") is None\n'
     '                                                   else float(fi["log_s"]))\n'
     '        row.update(fail_n=nf, fail_log_s=fi.get("log_s"), fail_upper_s=round(up, 1), fail_added_s=round(fs, 1),\n'
     '                   gpu_s18_ok_s=row["gpu_s18_s"])\n'
     '        row["gpu_s18_s"] += fs\n'),
    ('        run = next(iter(s["runs"].values()))\n'
     '        for k in ("jádro", "TICA"):\n'
     '            if args.tstar == "first":\n'
     '                t = run[k + "_nmatched"]["t_equiv_ns"]\n'
     '            elif args.tstar == "last":\n'
     '                t = t_last(s["curves"][k]["union"], run[k + "_nmatched"]["curve"])\n'
     '            else:\n'
     '                t = run[k]["t_equiv_union_ns" if args.tstar == "union" else "t_equiv_loo_ns"]\n',
     '        run = s\n'
     '        for k in ("jádro", "TICA"):\n'
     '            t = run[k][args.metric]["t_first" if args.tstar == "first" else "t_last"]\n'),
    ('            er = next(iter(e["runs"].values()))\n'
     '            row["loo_beats"] = bool(er["jádro"]["run_beats_rep100_loo"] and er["TICA"]["run_beats_rep100_loo"])\n'
     '            row["loo"] = {k: [round(er[k]["loo_run"], 3), round(er[k]["loo_rep100_matched"], 3)] for k in ("jádro", "TICA")}\n'
     '            row["lose_10_9"] = {k: bool(er[k]["loo_run_sub_to_10_9"] > er[k]["loo_rep10_9"]) for k in ("jádro", "TICA")}\n'
     '            row["ctrl_3x3_6"] = {k: [round(er[k]["union_run_sub_to_3x3_6"], 3), round(er[k]["union_atlas_3x3_6"], 3)]\n'
     '                                 for k in ("jádro", "TICA")}\n',
     '            er = e\n'
     '            row["loo_beats"] = bool(er["jádro"]["loo100"]["run_beats"] and er["TICA"]["loo100"]["run_beats"])\n'
     '            row["loo_beats_tica"] = bool(er["TICA"]["loo100"]["run_beats"])\n'
     '            row["loo"] = {k: [round(er[k]["loo100"]["run"], 3), round(er[k]["loo100"]["md"], 3)] for k in ("jádro", "TICA")}\n'
     '            row["lose_10_9"] = {k: bool(er[k]["at10_9"]["run_worse"]) for k in ("jádro", "TICA")}\n'
     '            row["ctrl_2x5_4"] = {k: [round(er[k]["ctrl_2x5_4"]["run"], 4), round(er[k]["ctrl_2x5_4"]["ctrl"], 4)]\n'
     '                                 for k in ("jádro", "TICA")}\n'),
    ('    ok = [r for r in rows if "speedup" in r]\n'
     '    res = dict(tag=args.tag, v_src=args.v_src, tstar=args.tstar, n_list=len(names), n_ok=len(ok), rows=rows, criteria={})\n'
     '    print(f"=== {args.tag}: {len(ok)} z {len(names)} proteinů s úplnými daty; rychlost MD z {args.v_src}, t* {args.tstar}")\n',
     '    keys0 = []\n'
     '    for r in rows:\n'
     '        keys0 += [k for k in r.get("speedup", {}) if k not in keys0]\n'
     '    for r in rows:\n'
     '        if r.get("phys_fail"):\n'
     '            r["speedup"] = {key: {"jádro": 0.0, "TICA": 0.0} for key in keys0}\n'
     '    ok = [r for r in rows if "speedup" in r]\n'
     '    okf = [r for r in ok if not r.get("phys_fail")]\n'
     '    res = dict(tag=args.tag, v_src=args.v_src, tstar=args.tstar, metric=args.metric, fail_mode=args.fail_mode,\n'
     '               exclude=sorted(excl), n_list=len(names), n_ok=len(ok), n_phys_fail=len(ok) - len(okf), rows=rows, criteria={})\n'
     '    print(f"=== Z10 (opravený přepočet po externím posudku) {args.tag}: {len(ok)} z {len(names)} proteinů "\n'
     '          f"(fyzikální selhání 0×: {len(ok) - len(okf)}); rychlost MD z {args.v_src}, t* {args.tstar}, metrika {args.metric}, "\n'
     '          f"selhané boxy {args.fail_mode}" + (f", bez {sorted(excl)}" if excl else ""))\n'
     '    print(f"  selhané boxy: {sum(r.get(\'fail_n\', 0) for r in okf)} (přičteno {sum(r.get(\'fail_added_s\', 0) for r in okf):.0f} s, "\n'
     '          f"z logu u {sum(r.get(\'fail_log_s\') is not None for r in okf if r.get(\'fail_n\'))} běhů)")\n'),
    ('            d[k] = dict(median=round(float(np.median(v)), 3), upper95=round(boot_upper(v, rng), 3),\n'
     '                        frac_ge_1_5=round(float(np.mean(np.array(v) >= 1.5)), 3))\n'
     '        d["nevyplácí_se"] = bool(d["jádro"]["upper95"] < 1.5 and d["TICA"]["upper95"] < 1.5)\n',
     '            lo, hi = boot_q(v, rng, (5, 95))\n'
     '            d[k] = dict(median=round(float(np.median(v)), 3), upper95=round(hi, 3), lower95=round(lo, 3),\n'
     '                        frac_ge_1_5=round(float(np.mean(np.array(v) >= 1.5)), 3),\n'
     '                        vyplácí_se=bool(np.median(v) >= 1.5 and lo > 1.0))\n'
     '        d["nevyplácí_se"] = bool(d["jádro"]["upper95"] < 1.5 and d["TICA"]["upper95"] < 1.5)\n'
     '        yes = [k for k in ("jádro", "TICA") if d[k]["vyplácí_se"]]\n'
     '        d["verdikt"] = "NEVYPLÁCÍ SE" if d["nevyplácí_se"] else (f"VYPLÁCÍ SE ({\', \'.join(yes)})" if yes else "NEROZHODNUTO")\n'),
    ('        print(f"  (i) {key:<46} n={d[\'n\']:>2} jádro medián {d[\'jádro\'][\'median\']:.2f} (horní mez {d[\'jádro\'][\'upper95\']:.2f}, ≥1,5× "\n'
     '              f"{d[\'jádro\'][\'frac_ge_1_5\']:.0%}), TICA {d[\'TICA\'][\'median\']:.2f} ({d[\'TICA\'][\'upper95\']:.2f}, "\n'
     '              f"{d[\'TICA\'][\'frac_ge_1_5\']:.0%}) → {\'NEVYPLÁCÍ SE\' if d[\'nevyplácí_se\'] else \'nelze vyloučit ≥ 1,5×\'}")\n',
     '        print(f"  (i) {key:<46} n={d[\'n\']:>2} jádro medián {d[\'jádro\'][\'median\']:.2f} (90% CI {d[\'jádro\'][\'lower95\']:.2f}–"\n'
     '              f"{d[\'jádro\'][\'upper95\']:.2f}, ≥1,5× {d[\'jádro\'][\'frac_ge_1_5\']:.0%}), TICA {d[\'TICA\'][\'median\']:.2f} "\n'
     '              f"({d[\'TICA\'][\'lower95\']:.2f}–{d[\'TICA\'][\'upper95\']:.2f}, {d[\'TICA\'][\'frac_ge_1_5\']:.0%}) → {d[\'verdikt\']}")\n'),
    ('    cens = {k: sum(r[f"cens_{k}"] for r in ok) for k in ("jádro", "TICA")}\n',
     '    cens = {k: sum(r[f"cens_{k}"] for r in okf) for k in ("jádro", "TICA")}\n'),
    ('    bad = [r for r in ok if r.get("good") is False and "lose_10_9" in r]\n',
     '    gt = [r for r in ok if r.get("good")]\n'
     '    res["criteria"]["loo_tica_good"] = dict(n_good=len(gt), n_beats=sum(bool(r.get("loo_beats_tica")) for r in gt),\n'
     '                                            n_all=len(ok), n_beats_all=sum(bool(r.get("loo_beats_tica")) for r in ok))\n'
     '    print(f"  LOO TICA (100 ns, shodné n, exploratorně): S18 lepší u {res[\'criteria\'][\'loo_tica_good\'][\'n_beats\']}/{len(gt)} "\n'
     '          f"dobrých AF, celkem {res[\'criteria\'][\'loo_tica_good\'][\'n_beats_all\']}/{len(ok)}")\n'
     '    bad = [r for r in ok if r.get("good") is False and "lose_10_9" in r]\n'),
    ('    print(f"\\n{\'protein\':<8} {\'L\':>4} {\'AF TICA\':>7}',
     '    cc = [r for r in okf if "ctrl_2x5_4" in r]\n'
     '    res["criteria"]["ctrl_2x5_4"] = {}\n'
     '    for k in ("jádro", "TICA"):\n'
     '        dif = [r["ctrl_2x5_4"][k][0] - r["ctrl_2x5_4"][k][1] for r in cc]\n'
     '        lo, hi = boot_q(dif, rng, (2.5, 97.5))\n'
     '        dd = dict(n=len(dif), run_closer=int(sum(x < 0 for x in dif)), median_diff_run_minus_ctrl=round(float(np.median(dif)), 4)\n'
     '                  if dif else None, ci95=[round(lo, 4), round(hi, 4)])\n'
     '        res["criteria"]["ctrl_2x5_4"][k] = dd\n'
     '        print(f"  kontrola 2 × 5,4 ns proti R_k ({k}): S18 blíž u {dd[\'run_closer\']}/{dd[\'n\']}, medián JS(S18) − JS(kontrola) "\n'
     '              f"{dd[\'median_diff_run_minus_ctrl\']} (95% CI {dd[\'ci95\'][0]}–{dd[\'ci95\'][1]})")\n'
     '    print(f"\\n{\'protein\':<8} {\'L\':>4} {\'AF TICA\':>7}'),
    ('    for r in ok:\n        a = r["speedup"]["i-a plný, 250 | p=2"]\n',
     '    for r in okf:\n        a = r["speedup"]["i-a plný, 250 | p=2"]\n'),
    ('    for r in ok:\n        print(f"{r[\'name\']:<8} {str(r.get(\'gpu_B\'))[:28]:<28}',
     '    for r in okf:\n        print(f"{r[\'name\']:<8} {str(r.get(\'gpu_B\'))[:28]:<28}'),
]
for old, new in R:
    assert s.count(old) == 1, f"úsek nenalezen právě jednou ({s.count(old)}×): {old[:70]!r}"
    s = s.replace(old, new)
Path("sum40.py").write_text(s)
print("sum40.py", hashlib.sha256(s.encode()).hexdigest())
