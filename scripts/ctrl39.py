import json, statistics as st
from pathlib import Path
R = Path("l39srv")
F = {"S18_s0": R / "out_gen/sum39_S18_s0_gen.json", "S18_s1": R / "out_gen/sum39_S18_s1_gen.json",
     "S18d50_s0": R / "out_z9/sum39_S18d50_s0_gen.json"}
def summ(rows, lab):
    for k in ("jádro", "TICA"):
        v = [r["ctrl_3x3_6"][k] for r in rows if r.get("ctrl_3x3_6") and r["ctrl_3x3_6"].get(k)]
        if not v:
            print(f"  {lab} {k}: žádná data"); continue
        s = [a for a, b in v]; a3 = [b for a, b in v]
        print(f"  {lab} {k:<5} n={len(v)}: S18 < ATLAS 3×3,6 u {sum(x < y for x, y in v)}/{len(v)}; medián S18 {st.median(s):.3f}, "
              f"ATLAS 3×3,6 {st.median(a3):.3f}, medián rozdílu S18 − ATLAS {st.median([x - y for x, y in v]):+.3f}")
for tag, f in F.items():
    S = json.loads(f.read_text())
    rows = S["rows"]
    have = [r for r in rows if r.get("ctrl_3x3_6")]
    print(f"=== {tag} ({f}): proteinů s ctrl_3x3_6 {len(have)} z {len(rows)}")
    summ(have, "všechny")
    summ([r for r in have if r.get("good")], "good=True")
print("\nPo proteinech S18_s0 (JS proti sjednocení 3 replik; S18 náhodně podvzorkovaný na n ATLAS 3×3,6):")
S = json.loads(F["S18_s0"].read_text())
print(f"{'N':8} {'good':5} | {'jádro S18':>9} {'ATLAS':>6} | {'TICA S18':>8} {'ATLAS':>6} | n ATLAS 3×3,6 (n_split), snímků S18 (eval39)")
ns = []
for r in S["rows"]:
    c = r.get("ctrl_3x3_6") or {}
    e = R / f"ana/eval39_{r['name']}_S18_s0.json"
    nsplit = nfr = None
    if e.exists():
        E = json.loads(e.read_text()); nsplit = E["n_split"]; nfr = next(iter(E["runs"].values()))["frames"]; ns.append((nsplit, nfr))
    j = c.get("jádro") or [None, None]; t = c.get("TICA") or [None, None]
    f3 = lambda x: "-" if x is None else f"{x:.3f}"
    print(f"{r['name']:8} {str(r.get('good')):5} | {f3(j[0]):>9} {f3(j[1]):>6} | {f3(t[0]):>8} {f3(t[1]):>6} | {nsplit}, {nfr}")
print(f"\nn ATLAS 3×3,6 (n_split) přes proteiny: min {min(a for a, b in ns)}, max {max(a for a, b in ns)}; snímků S18 v okně: min {min(b for a, b in ns)}, max {max(b for a, b in ns)}")
print("eval39: S18 se pro toto srovnání náhodně podvzorkuje na min(n_split, snímků S18) snímků (--draws losů, průměr JS); ATLAS 3×3,6 = všechny snímky 3 replik od --skip-ps do --t-split-ps (n_split). Obojí proti sjednocení všech 3 replik (in-sample: ATLAS snímky jsou součástí reference).")
