"""patch39i: gen39b.pbs po Gemini 39g. Vstup gen39b.pbs SHA 3721874f….
1) walltime 20 h → 16 h (kratší čekání ve frontě; rozhodující afT_base a afT_dist jdou v sadách B i K první);
2) před kotvou bench39 zahřívací běh 10 ps (GPU z klidového taktu, výsledek se zahodí);
3) kotva p2_7jfl_C znovu na konci úlohy jako kontrola driftu (bench_end_<J>_p2_7jfl_C.json, gen39sum ji do v_gen nebere).
    python3 patch39i.py   (v adresáři s gen39b.pbs)
"""
import hashlib
from pathlib import Path

p = Path("gen39b.pbs")
s = p.read_text()
h = hashlib.sha256(s.encode()).hexdigest()
assert h.startswith("3721874f1b64566b"), f"gen39b.pbs má jiné SHA {h[:16]}, nic neměním"

ANCHOR = "  [ -d p2_7jfl_C ] && $OP bench39.py --sys p2_7jfl_C --platform CUDA --out $RUN/bench_gen_${J}_p2_7jfl_C_anchor.json\n"
WARM = ("  # zahřátí GPU z klidového taktu, výsledek se zahodí (Gemini 39g)\n"
        "  [ -d p2_7jfl_C ] && $OP bench39.py --sys p2_7jfl_C --platform CUDA --ps 10 --out $SCRATCHDIR/bench_warm.json > /dev/null 2>&1\n")
END = ("# kotva znovu na konci = kontrola driftu taktu a teploty (Gemini 39g); jiný název, gen39sum ji do v_gen nebere\n"
       "( unset LD_LIBRARY_PATH; cd $SCRATCHDIR && [ -d p2_7jfl_C ] && $OP bench39.py --sys p2_7jfl_C --platform CUDA --out $RUN/bench_end_${J}_p2_7jfl_C.json )\n")
R = [
    ("#PBS -l walltime=20:00:00\n", "#PBS -l walltime=16:00:00\n"),
    ("Walltime 20 h (patch39h).\n", "Walltime 16 h (patch39h, 39i).\n"),
    (ANCHOR, WARM + ANCHOR),
    ("cat $OUT/calls.tsv\n", END + "cat $OUT/calls.tsv\n"),
]
for old, new in R:
    assert s.count(old) == 1, f"úsek nenalezen právě jednou: {old[:60]!r}"
    s = s.replace(old, new)
p.write_text(s)
print("gen39b.pbs", hashlib.sha256(s.encode()).hexdigest())
