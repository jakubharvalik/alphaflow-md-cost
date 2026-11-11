"""Linie 39, kontrola K18 (prereg39 Z7b): PDB s 18 stejnými modely (struktura ATLAS = start MD z krystalu), aby s18.py
s --targets 0..17 postavil 18 boxů z krystalu stejným protokolem jako S18 (liší se jen seed rychlostí a šumu).
    python k18_make.py atlas/pdb/7jfl_C.pdb k18_7jfl_C.pdb
"""
import sys
from pathlib import Path

src, dst = sys.argv[1], sys.argv[2]
lines = [ln for ln in Path(src).read_text().splitlines() if ln.startswith(("ATOM", "HETATM"))]
out = []
for m in range(18):
    out.append(f"MODEL     {m:4d}")
    out += lines
    out.append("ENDMDL")
out.append("END")
Path(dst).write_text("\n".join(out) + "\n")
print(f"{dst}: 18 modelů po {len(lines)} atomech z {src}")
