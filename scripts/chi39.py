import json, re, glob, statistics as st
from pathlib import Path
import numpy as np
R = Path("l39srv")
def sp(x, y):
    from scipy.stats import spearmanr
    r, p = spearmanr(x, y); return f"rho {r:+.3f} (p {p:.3f}, n {len(x)})"
def val(v):
    return v["JS"] if isinstance(v, dict) and "JS" in v else v
files = sorted(R.glob("ana/af39_*_S18d50_s0_b30.json"))
first = json.loads(files[0].read_text())
print(f"(a) klíče rows u {first['name']}:")
for k, v in first["rows"].items():
    print(f"   {k!r}: n={v.get('n')}, rama={v.get('rama')}, chi1={v.get('chi1')}")
sub50 = [k for k in first["rows"] if k.startswith("AF_MD_templ") and re.search(r"\b50\b", k)]
print(f"   řádek plného modelu podvzorkovaného na 50: {sub50 or 'NENÍ (porovnání je templ n=250 vs dist50 n=50)'}")
rows = []
for f in files:
    a = json.loads(f.read_text()); N = a["name"]; rw = a["rows"]
    t = next(v for k, v in rw.items() if k.startswith("AF_MD_templ ("))
    d = next(v for k, v in rw.items() if k.startswith("AF_dist50 ("))
    run = next(v for k, v in rw.items() if k.endswith(", 250 rovnoměrně") and not k.startswith("ATLAS"))
    r0 = json.loads((R / f"ana/rel39_{N}_S18_s0.json").read_text())["windows"]["50–520"]
    r5 = json.loads((R / f"ana/rel39_{N}_S18d50_s0.json").read_text())["windows"]["50–520"]
    vyl = None
    for lg in glob.glob(str(R / f"logs/b39_{N}_S18d50_s0_*.log")):
        for ln in open(lg, errors="replace"):
            m = re.search(r"s39d: vynecháno (\d+) z (\d+)", ln)
            if m: vyl = int(m.group(1))
    rows.append(dict(N=N, vyl=vyl, c_t=val(t["chi1"]), c_d=val(d["chi1"]), c_run=val(run["chi1"]),
                     r_t=val(t["rama"]), r_d=val(d["rama"]), r_run=val(run["rama"]),
                     c0=r0["chi1"], c5=r5["chi1"], ra0=r0["rama"], ra5=r5["rama"]))
for r in rows:
    r["dc_raw"] = r["c_d"] - r["c_t"]; r["dc_run"] = r["c5"] - r["c0"]
    r["dr_raw"] = r["r_d"] - r["r_t"]; r["dr_run"] = r["ra5"] - r["ra0"]
print("\n(a–c) tabulka (chi1 a Rama = JS; AF_* = surové vzorky před MD z af39; run s0/d50 = rel39 okno 50–520; chi1_af39run = řádek běhu d50 v af39, 250 rovnoměrně)")
print(f"{'N':8} {'vyl':>3} | {'χ1 templ':>8} {'χ1 d50':>7} {'Δχ1raw':>7} | {'χ1 s0':>6} {'χ1 d50':>6} {'Δχ1run':>7} {'χ1 af39run':>10} | {'Ra templ':>8} {'Ra d50':>6} {'ΔRaraw':>7} | {'Ra s0':>6} {'Ra d50':>6} {'ΔRarun':>7}")
for r in rows:
    print(f"{r['N']:8} {r['vyl']:>3} | {r['c_t']:8.3f} {r['c_d']:7.3f} {r['dc_raw']:+7.3f} | {r['c0']:6.3f} {r['c5']:6.3f} {r['dc_run']:+7.3f} {r['c_run']:10.3f} | {r['r_t']:8.3f} {r['r_d']:6.3f} {r['dr_raw']:+7.3f} | {r['ra0']:6.3f} {r['ra5']:6.3f} {r['dr_run']:+7.3f}")
g = lambda k: [r[k] for r in rows]
print(f"\nSouhrn (n = {len(rows)} proteinů):")
print(f"  medián Δχ1_raw (dist50 − templ) {st.median(g('dc_raw')):+.3f}; kladné u {sum(x > 0 for x in g('dc_raw'))}/{len(rows)}")
print(f"  medián Δχ1_run (d50 − s0)      {st.median(g('dc_run')):+.3f}; kladné u {sum(x > 0 for x in g('dc_run'))}/{len(rows)}")
print(f"  medián ΔRama_raw {st.median(g('dr_raw')):+.3f}; medián ΔRama_run {st.median(g('dr_run')):+.3f}")
print(f"  medián χ1 templ {st.median(g('c_t')):.3f}, dist50 {st.median(g('c_d')):.3f}; run s0 {st.median(g('c0')):.3f}, run d50 {st.median(g('c5')):.3f}")
print(f"  Spearman (vyl, Δχ1_run):     {sp(g('vyl'), g('dc_run'))}")
print(f"  Spearman (Δχ1_raw, Δχ1_run): {sp(g('dc_raw'), g('dc_run'))}")
print(f"  Spearman (vyl, Δχ1_raw):     {sp(g('vyl'), g('dc_raw'))}")
print(f"  Spearman (vyl, ΔRama_run):   {sp(g('vyl'), g('dr_run'))}")
