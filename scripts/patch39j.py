"""patch39j: gen39b.pbs, oprava kotvy bench39 (25405838: p2_7jfl_C z bench1_files.tgz nemá system.xml → FileNotFoundError).
Kotva, zahřátí i kontrola na konci berou box fáze B $L/sys/p2_7jfl_C (má system.xml). Vstup gen39b.pbs SHA 0762e59d….
    python3 patch39j.py   (v adresáři s gen39b.pbs)
"""
import hashlib
from pathlib import Path

p = Path("gen39b.pbs")
s = p.read_text()
h = hashlib.sha256(s.encode()).hexdigest()
assert h.startswith("0762e59d5f940dc5"), f"gen39b.pbs má jiné SHA {h[:16]}, nic neměním"

OLD = "[ -d p2_7jfl_C ] && $OP bench39.py --sys p2_7jfl_C "
NEW = "[ -s $L/sys/p2_7jfl_C/system.xml ] && $OP bench39.py --sys $L/sys/p2_7jfl_C "
assert s.count(OLD) == 3, f"čekám 3 výskyty kotvy, je {s.count(OLD)}"
s = s.replace(OLD, NEW)
R = [
    ("# rychlost MD na této GPU NAPŘED (kdyby generátory nedoběhly do walltime, převod na MD-ns zůstane): p2_7jfl_C (stejný systém\n"
     "# jako bench3, kotva) a všechny už postavené p2_<N> z fáze B; omm13 bez LD_LIBRARY_PATH af39, jako ve fázi B (b39.pbs)\n",
     "# rychlost MD na této GPU NAPŘED (kdyby generátory nedoběhly do walltime, převod na MD-ns zůstane): kotva = box fáze B\n"
     "# $L/sys/p2_7jfl_C (p2_7jfl_C z bench1_files.tgz nemá system.xml, patch39j) a všechny už postavené p2_<N> z fáze B;\n"
     "# omm13 bez LD_LIBRARY_PATH af39, jako ve fázi B (b39.pbs)\n"),
]
for old, new in R:
    assert s.count(old) == 1, f"úsek nenalezen právě jednou: {old[:60]!r}"
    s = s.replace(old, new)
p.write_text(s)
print("gen39b.pbs", hashlib.sha256(s.encode()).hexdigest())
