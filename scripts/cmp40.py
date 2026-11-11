#!/usr/bin/env python3
"""cmp40: srovnání původního vyhodnocení (sum39 JSON) a opraveného přepočtu Z10 (sum40 JSON) po proteinech.

Jen čte dva JSON výstupy a píše tabulku Markdown (do SI). Nic nepočítá znovu kromě mediánů a počtů.
Použití: python cmp40.py --old sum39_S18_s0.json --new sum40_S18_s0.json --tag S18_s0 --out cmp40_S18_s0.md
"""
import argparse
import json
from pathlib import Path

import numpy as np

KEYS = [("Z8 i-b změřeno: destil., 50", "i-b"), ("Z8 i-a změřeno: plný, 250", "i-a"),
        ("nula (vzorky existují) | p=2", "nula")]


def f(x, d=2):
    return "–" if x is None else f"{x:.{d}f}"


def tstr(r, k):
    if r is None or f"t_{k}" not in r:
        return "–"
    return ("≥" if r.get(f"cens_{k}") else "") + f"{r[f't_{k}']:.1f}"


def sp(r, key, k):
    try:
        return r["speedup"][key][k]
    except (KeyError, TypeError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    old = {r["name"]: r for r in json.loads(Path(a.old).read_text())["rows"]}
    newj = json.loads(Path(a.new).read_text())
    new = {r["name"]: r for r in newj["rows"]}
    names = sorted(set(old) | set(new))
    L = [f"## {a.tag}: původní vyhodnocení × opravený přepočet Z10", "",
         f"Z10: t* {newj.get('tstar')}, metrika {newj.get('metric')}, selhané boxy {newj.get('fail_mode')}, "
         f"rychlost MD {newj.get('v_src')}. Fyzikální selhání = 0×.", "",
         "| protein | t* jádro ns (původně → Z10) | t* TICA ns | i-b jádro × | i-b TICA × | nula jádro × | nula TICA × "
         "| LOO jádro S18/rep (Z10) | LOO TICA S18/rep (Z10) | selhané boxy (s) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    diffs = {k: [] for k in ("jádro", "TICA")}
    for n in names:
        o, w = old.get(n), new.get(n)
        cells = [n]
        for k in ("jádro", "TICA"):
            cells.append(f"{tstr(o, k)} → {tstr(w, k)}")
            if o and w and f"t_{k}" in o and f"t_{k}" in w and not (w.get("phys_fail") or o.get("phys_fail")):
                diffs[k].append(w[f"t_{k}"] - o[f"t_{k}"])
        for key, lab in (KEYS[0], KEYS[2]):
            for k in ("jádro", "TICA"):
                cells.append(f"{f(sp(o, key, k))} → {f(sp(w, key, k))}")
        lo = (w or {}).get("loo", {})
        for k in ("jádro", "TICA"):
            v = lo.get(k)
            cells.append(f"{v[0]:.3f}/{v[1]:.3f}" if v else "–")
        fn = (w or {}).get("fail_n", 0)
        cells.append(f"{fn} ({(w or {}).get('fail_added_s', 0):.0f})" if fn else "0")
        if w and w.get("phys_fail"):
            cells[0] += " (fyz. selhání, 0×)"
        L.append("| " + " | ".join(cells) + " |")
    L.append("")
    for key, lab in KEYS:
        for k in ("jádro", "TICA"):
            vo = [sp(old[n], key, k) for n in names if n in old and sp(old[n], key, k) is not None]
            vn = [sp(new[n], key, k) for n in names if n in new and sp(new[n], key, k) is not None]
            if vo or vn:
                L.append(f"- {lab} {k}: medián původně {f(float(np.median(vo)) if vo else None)} (n = {len(vo)}), "
                         f"Z10 {f(float(np.median(vn)) if vn else None)} (n = {len(vn)})")
    for k in ("jádro", "TICA"):
        d = np.array(diffs[k])
        if d.size:
            L.append(f"- změna t* {k} (Z10 − původně): medián {np.median(d):+.2f} ns, delší u {int((d > 0).sum())}, "
                     f"kratší u {int((d < 0).sum())}, stejné u {int((d == 0).sum())} z {d.size}")
    crit = newj.get("criteria", {}).get("i", {})
    for key, lab in KEYS[:2]:
        if key in crit:
            c = crit[key]
            L.append(f"- Z10 verdikt {lab}: {c['verdikt']} (jádro {c['jádro']['median']:.2f}, 90% CI "
                     f"{c['jádro']['lower95']:.2f}–{c['jádro']['upper95']:.2f}; TICA {c['TICA']['median']:.2f}, "
                     f"{c['TICA']['lower95']:.2f}–{c['TICA']['upper95']:.2f}; n = {c['n']})")
    Path(a.out).write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
