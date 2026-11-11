"""patch39h: gen39b.pbs po prvních časech sady K na L40S (25405838: afT_base 5znj_A 776 s na volání se 4 vzorky).
Na A40/A100 bude generátor pomalejší než na L40S, 8 h nemusí stačit a bench39 byl až na konci. Vstup gen39b.pbs SHA 97b679ed….
1) walltime 8 h → 20 h; 2) bench39 (rychlost MD na této GPU) hned po kontrole GPU, ne na konci, a bez LD_LIBRARY_PATH af39
   (omm13 jako ve fázi B, b39.pbs), aby převod na MD-ns zůstal, i kdyby generátory nedoběhly.
    python3 patch39h.py   (v adresáři s gen39b.pbs)
"""
import hashlib
from pathlib import Path

p = Path("gen39b.pbs")
s = p.read_text()
h = hashlib.sha256(s.encode()).hexdigest()
assert h.startswith("97b679edc5c4ff70"), f"gen39b.pbs má jiné SHA {h[:16]}, nic neměním"

OLD_BENCH = (
    "# rychlost MD na této GPU: p2_7jfl_C (stejný systém jako bench3, kotva) a všechny už postavené p2_<N> z fáze B\n"
    "OP=$W/envs/omm13/bin/python\n"
    "cd $SCRATCHDIR && tar xzf $W/bench/bench1_files.tgz --warning=no-unknown-keyword; tar xzf $L/code39.tgz --warning=no-unknown-keyword\n"
    "cp $L/bench39.py .\n"
    "[ -d p2_7jfl_C ] && $OP bench39.py --sys p2_7jfl_C --platform CUDA --out $RUN/bench_gen_${J}_p2_7jfl_C_anchor.json\n"
    "for S in $L/sys/p2_*; do\n"
    "  [ -s $S/system.xml ] || continue\n"
    "  $OP bench39.py --sys $S --platform CUDA --out $RUN/bench_gen_${J}_$(basename $S).json > /dev/null 2>&1 || echo \"bench $(basename $S) selhal\"\n"
    "done\n"
    "echo \"$(date '+%T') bench na této GPU: $(ls $RUN/bench_gen_${J}_*.json 2>/dev/null | wc -l) systémů\"\n"
)
NEW_BENCH = (
    "# rychlost MD na této GPU NAPŘED (kdyby generátory nedoběhly do walltime, převod na MD-ns zůstane): p2_7jfl_C (stejný systém\n"
    "# jako bench3, kotva) a všechny už postavené p2_<N> z fáze B; omm13 bez LD_LIBRARY_PATH af39, jako ve fázi B (b39.pbs)\n"
    "OP=$W/envs/omm13/bin/python\n"
    "(\n"
    "  unset LD_LIBRARY_PATH\n"
    "  cd $SCRATCHDIR && tar xzf $W/bench/bench1_files.tgz --warning=no-unknown-keyword; tar xzf $L/code39.tgz --warning=no-unknown-keyword\n"
    "  cp $L/bench39.py .\n"
    "  [ -d p2_7jfl_C ] && $OP bench39.py --sys p2_7jfl_C --platform CUDA --out $RUN/bench_gen_${J}_p2_7jfl_C_anchor.json\n"
    "  for S in $L/sys/p2_*; do\n"
    "    [ -s $S/system.xml ] || continue\n"
    "    $OP bench39.py --sys $S --platform CUDA --out $RUN/bench_gen_${J}_$(basename $S).json > /dev/null 2>&1 || echo \"bench $(basename $S) selhal\"\n"
    "  done\n"
    ")\n"
    "echo \"$(date '+%T') bench na této GPU: $(ls $RUN/bench_gen_${J}_*.json 2>/dev/null | wc -l) systémů\"\n"
)
TORCH = "$P -c \"import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))\" || exit 3\n"
R = [
    ("#PBS -l walltime=8:00:00\n", "#PBS -l walltime=20:00:00\n"),
    ("# Pořadí: sada B (25 proteinů ≤ 120, jedno volání predict.py na model) primární modely napřed, pak vedlejší,\n"
     "# pak sada K (6 proteinů 131–724, jedno volání na model a protein, aby OOM jednoho nezastavil ostatní), nakonec bench39.\n",
     "# Pořadí: bench39 napřed, pak sada B (25 proteinů ≤ 120, jedno volání predict.py na model) primární modely napřed, pak vedlejší,\n"
     "# pak sada K (6 proteinů 131–724, jedno volání na model a protein, aby OOM jednoho nezastavil ostatní). Walltime 20 h (patch39h).\n"),
    (OLD_BENCH, ""),
    (TORCH, TORCH + NEW_BENCH),
]
for old, new in R:
    assert s.count(old) == 1, f"úsek nenalezen právě jednou: {old[:60]!r}"
    s = s.replace(old, new)
p.write_text(s)
print("gen39b.pbs", hashlib.sha256(s.encode()).hexdigest())
