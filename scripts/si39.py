"""Podklady SI preprintu (linie 39): jeden řádek na protein, jen čte hotové JSON/logy v l39srv, píše out_fig/si39.csv."""
import csv, glob, json, re
from pathlib import Path
R = Path("l39srv")
J = lambda p: json.loads(Path(p).read_text()) if Path(p).exists() else None
s0 = J(R / "out_gen/sum39_S18_s0_gen.json"); s0l = J(R / "out_gen/sum39_S18_s0_gen_last.json")
d50 = {r["name"]: r for r in J(R / "out_z9/sum39_S18d50_s0_gen.json")["rows"]}
last = {r["name"]: r for r in s0l["rows"]}
cmp = J(R / "out_z9/cmp39d_gen.json")
z2 = cmp["Z9_ii"]["S18d50_s0"]["per_protein"]
def nfail(N, tag):
    d = R / "runs" / f"hyb39_{N}_{tag}"
    for f in ("hybrid_summary.json", "hybrid_summary_partial.json"):
        h = J(d / f)
        if h is not None:
            return len(h.get("failed") or []), f
    return "", ""
def t(r, k):
    return "" if r is None or f"t_{k}" not in r else (f"≥100" if r[f"cens_{k}"] else f"{r[f't_{k}']:.2f}")
def excl(N):
    for lg in glob.glob(str(R / f"logs/b39_{N}_S18d50_s0_*.log")):
        for ln in open(lg, errors="replace"):
            m = re.search(r"s39d: vynecháno (\d+) z (\d+)", ln)
            if m: return int(m.group(1))
    return ""
cols = ["N", "L", "good", "gpu_s0", "atoms_box", "v_p2_nsday", "tstar_first_core", "tstar_first_tica", "tstar_last_core",
        "tstar_last_tica", "js_core", "js_tica", "rama", "chi1", "loo_core_win", "loo_tica_win", "boxes_failed_s0",
        "boxes_failed_s1", "boxes_failed_d50", "filter_excluded_d50", "chi1_d50", "rama_d50", "loo_tica_win_d50", "note"]
with open(R / "out_fig/si39.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(cols)
    for r in sorted(s0["rows"], key=lambda r: (r["L"], r["name"])):
        N = r["name"]; note = []
        sub = J(R / f"ana/sub39_{N}_S18_s0.json"); run = next(iter(sub["runs"].values())) if sub else None
        ev = J(R / f"ana/eval39_{N}_S18_s0.json"); er = next(iter(ev["runs"].values())) if ev else None
        rl5 = J(R / f"ana/rel39_{N}_S18d50_s0.json")
        w5 = (rl5 or {}).get("windows", {}).get("50–520") or {}
        f0, f1, f5 = nfail(N, "S18_s0"), nfail(N, "S18_s1"), nfail(N, "S18d50_s0")
        if f5[1].endswith("partial.json"): note.append("d50 partial (fyz. selhání)")
        lt = z2[N]["loo_tica"] if N in z2 and not z2[N].get("phys_fail") else None
        if lt is None:
            lt5 = J(R / f"ana/eval39_{N}_S18d50_s0.json")
            lt = next(iter(lt5["runs"].values()))["TICA"]["run_beats_rep100_loo"] if lt5 else None
        w.writerow([N, r["L"], r.get("good"), r.get("gpu_B", ""), round(r["n_atoms_box"]) if r.get("n_atoms_box") else "",
                    r.get("v_bench_nsd", ""), t(r, "jádro"), t(r, "TICA"), t(last.get(N), "jádro"), t(last.get(N), "TICA"),
                    f"{run['jádro']['js_union']:.3f}" if run else "", f"{run['TICA']['js_union']:.3f}" if run else "",
                    r.get("rama", ""), r.get("chi1", ""),
                    int(er["jádro"]["run_beats_rep100_loo"]) if er else "", int(er["TICA"]["run_beats_rep100_loo"]) if er else "",
                    f0[0], f1[0], f5[0], excl(N), w5.get("chi1", ""), w5.get("rama", ""),
                    "" if lt is None else int(bool(lt)), "; ".join(note)])
print("Uloženo", R / "out_fig/si39.csv")
