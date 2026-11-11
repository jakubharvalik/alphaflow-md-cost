"""fail40 (prereg39 Z10): čas selhaných boxů a pokusů S18 z logů b39 (jen čte).
Pro každý řádek „HH:MM:SS box K (vzorek I) SELHAL: …“ v logs/b39_<N>_<TAG>_<JOB>_<idx>_.log je čas pokusu = razítko SELHAL
minus razítko předchozího řádku s časem (přes půlnoc + 24 h). Interval zahrnuje i stavbu boxu na CPU, je to tedy horní odhad
času pokusu na GPU. Typ (popisně): NaN-coord (souřadnice NaN) | inf-min (nekonečná energie) | jiné.
Výstup JSON {N: {TAG: {n, log_s, items: [{box, sample, s, type, log}]}}}.
    python3 fail40.py --logs l39srv/logs --out l39srv/fail40.json
"""
import argparse
import json
import re
from pathlib import Path

FN = re.compile(r"^b39_([0-9a-z]{4}_[A-Za-z0-9]+?)_((?:S18|S18d50|K18)_s\d+)_(\d+)_(\d+)_\.log$")
TS = re.compile(r"^(\d\d):(\d\d):(\d\d) ")
FAIL = re.compile(r"^\d\d:\d\d:\d\d box (\d+) \(vzorek (\d+)\) SELHAL: (.*)$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    res, nlog = {}, 0
    for f in sorted(Path(args.logs).glob("b39_*.log")):
        m = FN.match(f.name)
        if not m:
            print("neznámé jméno logu, přeskočeno:", f.name)
            continue
        N, tag = m.group(1), m.group(2)
        nlog += 1
        prev = None
        for ln in f.read_text(errors="replace").split("\n"):
            t = TS.match(ln)
            if not t:
                continue
            sec = int(t.group(1)) * 3600 + int(t.group(2)) * 60 + int(t.group(3))
            fm = FAIL.match(ln)
            if fm and prev is not None:
                dt = (sec - prev) % 86400
                err = fm.group(3)
                e = err.lower()
                typ = "NaN-coord" if "coordinate is nan" in e else ("inf-min" if ("infin" in e or "energy" in e) else "jiné")
                d = res.setdefault(N, {}).setdefault(tag, dict(n=0, log_s=0.0, items=[]))
                d["n"] += 1
                d["log_s"] += dt
                d["items"].append(dict(box=int(fm.group(1)), sample=int(fm.group(2)), s=dt, type=typ, log=f.name,
                                       error=err[:120]))
            prev = sec
    Path(args.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    tot = {}
    for N, dd in res.items():
        for tag, d in dd.items():
            a = tot.setdefault(tag, [0, 0.0])
            a[0] += d["n"]
            a[1] += d["log_s"]
    print(f"logů {nlog}; selhání podle tagu: " + ", ".join(f"{t} {a[0]} ({a[1]:.0f} s)" for t, a in sorted(tot.items())))
    for N in sorted(res):
        for tag, d in sorted(res[N].items()):
            print(f"  {N} {tag}: {d['n']} × " + ", ".join(f"box {i['box']} {i['type']} {i['s']} s" for i in d["items"]))
    print("Uloženo", args.out)


if __name__ == "__main__":
    main()
