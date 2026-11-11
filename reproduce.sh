#!/usr/bin/env bash
# reproduce.sh: recompute the summary tables of the preprint from the per-protein inputs in inputs/l39
# and compare each result with the archived file in tables/. Reads the archive, writes only into ./work.
# Usage, from the archive root:  PY=/path/to/python bash reproduce.sh
# Needs Python 3 with NumPy, SciPy and mdtraj, and inputs/l39/atlas_test.csv
# (splits/atlas_test.csv of the AlphaFlow repository, SHA-256 starting c5029649db0b7ff6).
set -u
PY=${PY:-python3}
A=$(cd "$(dirname "$0")" && pwd)
S=$A/scripts
T=$A/tables
R=$A/work/l39srv
CMP() { "$PY" "$A/tools/jcmp39.py" "$@"; }
SAME() { if cmp -s "$1" "$2"; then echo "IDENTICAL $(basename "$1")"; else echo "DIFFERS $(basename "$1") vs $2"; fi; }
step() { echo; echo "== $*"; }
[ -s "$A/inputs/l39/atlas_test.csv" ] || { echo "Put splits/atlas_test.csv of the AlphaFlow repository into inputs/l39/ first."; exit 1; }
rm -rf "$A/work"
mkdir -p "$R/out_gen" "$R/out_z9" "$R/out_a100" "$R/out_fig"
cp -Rp "$A/inputs/l39/." "$R/"
cd "$A/work" || exit 1

step "Phase A summary (sumA39)"
"$PY" "$S/sumA39.py" --root l39srv --out l39srv/sumA39.json > l39srv/sumA39.txt && CMP l39srv/sumA39.json "$T/sumA39.json"

step "Generator cost (gen39sum, timing part only)"
for J in 25413654:gen39b 25424183:gen39e; do
  "$PY" "$S/gen39sum.py" --gen l39srv/gen --run "l39srv/gen/run_${J%%:*}" --list l39srv/gen39_list.txt \
    --atlas-csv l39srv/atlas_test.csv --no-rmsd --out "l39srv/${J##*:}_sum_norm.json" > /dev/null \
    && CMP "l39srv/${J##*:}_sum_norm.json" "$T/${J##*:}_sum.json" --common
done
# The accounting below uses the archived generator summaries, which also hold the RMSD installation
# check (that check needs the generated samples, which are not archived).
cp -p "$T/gen39b_sum.json" "$T/gen39e_sum.json" l39srv/

s39() {  # tag generator-summary v-src t* output reference
  "$PY" "$S/sum39.py" --root l39srv --tag "$1" --gen "l39srv/$2" --v-src "$3" --tstar "$4" --out "$5" > "${5%.json}.txt" \
    && CMP "$5" "$6"
}
step "Accounting on the A40 (sum39; decisive)"
s39 S18_s0 gen39b_sum.json bench first l39srv/out_gen/sum39_S18_s0_gen.json "$T/sum39_gen/sum39_S18_s0_gen.json"
s39 S18_s0 gen39b_sum.json bench last l39srv/out_gen/sum39_S18_s0_gen_last.json "$T/sum39_gen/sum39_S18_s0_gen_last.json"
s39 S18_s1 gen39b_sum.json xfer first l39srv/out_gen/sum39_S18_s1_gen.json "$T/sum39_gen/sum39_S18_s1_gen.json"
s39 K18_s0 gen39b_sum.json xfer first l39srv/out_gen/sum39_K18_s0_gen.json "$T/sum39_gen/sum39_K18_s0_gen.json"

step "Accounting on the A100 (sensitivity)"
s39 S18_s0 gen39e_sum.json bench first l39srv/out_a100/sum39_S18_s0_a100.json "$T/sum39_a100/sum39_S18_s0_a100.json"
s39 S18_s1 gen39e_sum.json xfer first l39srv/out_a100/sum39_S18_s1_a100.json "$T/sum39_a100/sum39_S18_s1_a100.json"

step "Distilled generator (amendment Z9)"
s39 S18d50_s0 gen39b_sum.json xfer first l39srv/out_z9/sum39_S18d50_s0_gen.json "$T/z9/sum39_S18d50_s0_gen.json"
s39 S18d50_s0 gen39b_sum.json xfer last l39srv/out_z9/sum39_S18d50_s0_last.json "$T/z9/sum39_S18d50_s0_last.json"
s39 S18d50_s0 gen39e_sum.json xfer first l39srv/out_z9/sum39_S18d50_s0_a100.json "$T/z9/sum39_S18d50_s0_a100.json"
for X in gen last a100; do
  "$PY" "$S/cmp39d.py" --root l39srv --sum "l39srv/out_z9/sum39_S18d50_s0_$X.json" --tag S18d50_s0 --base S18_s0 \
    --ref2 S18_s1 --out "l39srv/out_z9/cmp39d_$X.json" > "l39srv/out_z9/cmp39d_$X.txt" \
    && CMP "l39srv/out_z9/cmp39d_$X.json" "$T/z9/cmp39d_$X.json"
done

step "Accounting grid (grid39)"
"$PY" "$S/grid39.py" --sum l39srv/out_gen/sum39_S18_s0_gen.json --sum l39srv/out_gen/sum39_S18_s1_gen.json \
  --gensum l39srv/gen39b_sum.json --out l39srv/out_gen/grid39.json > l39srv/out_gen/grid39.txt \
  && CMP l39srv/out_gen/grid39.json "$T/sum39_gen/grid39.json"

step "Figure and SI data (fig39, fig39c, si39; these scripts read ./l39srv)"
for f in fig39 fig39c si39; do "$PY" "$S/$f.py" > "l39srv/out_fig/$f.log" 2>&1 || echo "FAILED $f"; done
for f in speedup39 tgen39 loo39 loc39 si39; do
  out=$(ls l39srv/out_fig/$f.* 2>/dev/null | grep -v '\.log$' | head -1)
  ref=$(ls "$T"/$f.* 2>/dev/null | head -1)
  if [ -n "$out" ] && [ -n "$ref" ]; then SAME "$out" "$ref"; else echo "MISSING $f (output '$out', reference '$ref')"; fi
done

step "Checks for Z9 and Z7a (chi39, ctrl39; printed output)"
for f in chi39 ctrl39; do
  "$PY" "$S/$f.py" > "l39srv/out_z9/$f.txt" 2>&1 || echo "FAILED $f"
  SAME "l39srv/out_z9/$f.txt" "$T/z9/$f.txt"
done
echo; echo "Done. Outputs are in ./work/l39srv."
