"""Podklady pro obr. 3 a úpravu obr. 1 (linie 39). Jen čte hotové JSON v l39srv, píše out_fig/loo39.csv a out_fig/loc39.csv."""
import csv, json
from pathlib import Path
R = Path("l39srv")
J = lambda p: json.loads(Path(p).read_text()) if Path(p).exists() else None
s0 = J(R / "out_gen/sum39_S18_s0_gen.json")
d50 = {r["name"]: r for r in J(R / "out_z9/sum39_S18d50_s0_gen.json")["rows"]}
rows = sorted(s0["rows"], key=lambda r: (r["L"], r["name"]))
f3 = lambda x: "" if x is None else f"{x:.3f}"
def ev(N, tag):
    e = J(R / f"ana/eval39_{N}_{tag}.json")
    return next(iter(e["runs"].values())) if e else None
def t(r, k):
    return "" if not r or f"t_{k}" not in r else ("≥100" if r[f"cens_{k}"] else f"{r[f't_{k}']:.2f}")
with open(R / "out_fig/loo39.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["N", "L", "good", "loo_tica_js_s18", "loo_tica_js_rep100", "loo_core_js_s18", "loo_core_js_rep100",
                "loo_tica_js_d50", "loo_tica_js_rep100_d50", "tstar_first_core_d50", "tstar_first_tica_d50"])
    for r in rows:
        N = r["name"]; a, b = ev(N, "S18_s0"), ev(N, "S18d50_s0")
        w.writerow([N, r["L"], r.get("good"),
                    f3(a and a["TICA"]["loo_run"]), f3(a and a["TICA"]["loo_rep100_matched"]),
                    f3(a and a["jádro"]["loo_run"]), f3(a and a["jádro"]["loo_rep100_matched"]),
                    f3(b and b["TICA"]["loo_run"]), f3(b and b["TICA"]["loo_rep100_matched"]),
                    t(d50.get(N), "jádro"), t(d50.get(N), "TICA")])
def afrow(N, tag, prefix):
    a = J(R / f"ana/af39_{N}_{tag}_b30.json")
    if not a: return None
    return next((v for k, v in a["rows"].items() if k.startswith(prefix)), None)
def rel(N, tag):
    x = J(R / f"ana/rel39_{N}_{tag}.json")
    return (x or {}).get("windows", {}).get("50–520")
missing = []
with open(R / "out_fig/loc39.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["N", "stav", "rama", "chi1"])
    for r in rows:
        N = r["name"]
        for stav, get in (("AF_raw250", lambda: afrow(N, "S18_s0", "AF_MD_templ (")),
                          ("S18_s0", lambda: rel(N, "S18_s0")),
                          ("AFdist_raw50", lambda: afrow(N, "S18d50_s0", "AF_dist50 (")),
                          ("S18d50", lambda: rel(N, "S18d50_s0")),
                          ("K18", lambda: rel(N, "K18_s0"))):
            v = get()
            if v is None:
                if stav != "K18": missing.append(f"{N} {stav}")
                continue
            w.writerow([N, stav, v["rama"], v["chi1"]])
print("chybí (mimo K18):", missing or "nic")
