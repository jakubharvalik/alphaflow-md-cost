"""Podklady pro grafy preprintu (linie 39). Jen čte JSON, píše speedup39.csv a tgen39.csv vedle sebe."""
import csv, json
from pathlib import Path
R = Path("l39srv"); O = R / "out_fig"
L = lambda p: json.loads((R / p).read_text())
s0, s1, d50 = L("out_gen/sum39_S18_s0_gen.json"), L("out_gen/sum39_S18_s1_gen.json"), L("out_z9/sum39_S18d50_s0_gen.json")
KA, KB = "Z8 i-a změřeno: plný, 250", "Z8 i-b změřeno: destil., 50"
zk = [k for k in s0["rows"][0]["speedup"] if k.startswith("nula")]
KZ = "nula (vzorky existují) | p=2" if "nula (vzorky existují) | p=2" in zk else zk[0]
print("klíč nulové ceny:", KZ, "| všechny nula*:", zk)
idx = lambda S: {r["name"]: r for r in S["rows"]}
i1, id50 = idx(s1), idx(d50)
def sp(r, k):
    v = (r or {}).get("speedup", {}).get(k)
    return (f"{v['jádro']:.3f}", f"{v['TICA']:.3f}") if v else ("", "")
cols = ["N", "L", "good"]
for c in ("full250_s0", "full250_s1", "zero_s0", "dist50bound_s0", "dist50direct"):
    cols += [f"{c}_core", f"{c}_tica"]
with open(O / "speedup39.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(cols)
    for r in s0["rows"]:
        N = r["name"]
        d = sp(id50.get(N), KB)
        if not d[0]:
            d = ("FAIL", "FAIL")
        w.writerow([N, r["L"], r.get("good"), *sp(r, KA), *sp(i1.get(N), KA), *sp(r, KZ), *sp(r, KB), *d])
gb, ge = L("gen39b_sum.json"), L("gen39e_sum.json")
pp = lambda G, m: G["models"][m]["per_protein"]
with open(O / "tgen39.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["N", "L", "sada", "t_base_A40", "t_dist_A40", "t_base_A100", "t_dist_A100", "nsday_A40", "nsday_A100"])
    names = sorted(pp(gb, "afT_base"), key=lambda n: (pp(gb, "afT_base")[n]["sada"], pp(gb, "afT_base")[n]["L"], n))
    for N in names:
        a = pp(gb, "afT_base")[N]
        g = lambda G, m: (f"{pp(G, m)[N]['t_med']:.3f}" if N in pp(G, m) else "")
        b = lambda G: (str(G["bench_gen"][N]["ns_per_day"]) if N in G.get("bench_gen", {}) else "")
        w.writerow([N, a["L"], a["sada"], g(gb, "afT_base"), g(gb, "afT_dist"), g(ge, "afT_base"), g(ge, "afT_dist"), b(gb), b(ge)])
print("Uloženo", O / "speedup39.csv", O / "tgen39.csv")
