import hashlib
from pathlib import Path
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
need = {"gen39.pbs": "844c056576b74e12", "gen39sum.py": "996060bc1acf9474", "tmpl39.py": "c1ac7eee3bb82e4f"}
for f, h in need.items():
    assert sha(f).startswith(h), f"{f}: SHA {sha(f)[:16]} != {h}"

# 1) gen39b.pbs z gen39.pbs
s = open("gen39.pbs", encoding="utf-8").read()
def rep(o, n):
    global s
    assert s.count(o) == 1, o
    s = s.replace(o, n)
rep("#PBS -N l39_gen\n", "#PBS -N l39_gen2\n")
rep("# Vzorků: sada B plná 6, destilovaná 11; sada K plná 4, destilovaná 6; první vzorek = zahřátí, do mediánu se nepočítá.\n",
"# Vzorků: sada B plná 6, destilovaná 11; sada K plná 4, destilovaná 6; první vzorek = zahřátí, do mediánu se nepočítá.\n"
"# gen39b (opakování po 25405838, 10. 11. 2026): sada B tam spadla celá (šablony z ATLAS PDB bez ID řetězce → openfold KeyError ' '),\n"
"# 12vrstvé váhy 202406 mají 'training' v hyper_parameters (TypeError v predict.py). Oprava: šablony s řetězcem A v $G/templates_A\n"
"# (původní beze změny), v predict.py se 'training' z hyper_parameters vynechá (třetí úprava kódu autorů, v git diff).\n"
"# Vše (sady B a K, všechny modely, bench) znovu v jedné úloze na jedné GPU, výstupy odděleně v $G/run_<JOB> (nic se nemíchá s 25405838).\n")
rep("OUT=$G/out_$J\nmkdir -p $OUT $G/rt $G/samples $L/bench\n", "RUN=$G/run_$J\nOUT=$RUN\nmkdir -p $RUN/rt $RUN/samples\n")
rep("""$P -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))" || exit 3
""", """$P -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))" || exit 3
# šablony s ID řetězce A (openfold protein.from_pdb_string nezná prázdné ID); jen řádky ATOM/HETATM/TER s mezerou ve sloupci 22
mkdir -p $G/templates_A
$P - $G/templates $G/templates_A <<'PY'
import sys, pathlib
src, dst = map(pathlib.Path, sys.argv[1:3])
nf = nfix = nl = 0
for f in sorted(src.glob("*.pdb")):
    out, ch = [], 0
    for ln in f.read_text().splitlines():
        if ln.startswith(("ATOM", "HETATM", "TER")):
            ln = ln.ljust(22)
            if ln[21] == " ":
                ln = ln[:21] + "A" + ln[22:]
                ch += 1
        out.append(ln)
    (dst / f.name).write_text("\\n".join(out) + "\\n")
    nf += 1
    nfix += ch > 0
    nl += ch
print(f"šablony templates_A: {nf} souborů, ID řetězce doplněno v {nfix} ({nl} řádků)")
PY
[ $(ls $G/templates_A/*.pdb 2>/dev/null | wc -l) -ge $(wc -l < $L/gen39_list.txt) ] || { echo "templates_A neúplné"; exit 4; }
# predict.py: 12vrstvé váhy nesou 'training' v hyper_parameters → vynechat (jinak TypeError: multiple values for 'training')
$P - $G/alphaflow/predict.py <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
old = "model_class(**ckpt['hyper_parameters'], training=False)"
new = "model_class(**{k: v for k, v in ckpt['hyper_parameters'].items() if k != 'training'}, training=False)"
if new in s:
    print("predict.py: oprava 12l už je")
elif s.count(old) == 1:
    open(p, "w").write(s.replace(old, new))
    print("predict.py: oprava 12l provedena ('training' z hyper_parameters vynechán)")
else:
    print("predict.py: řádek pro opravu 12l nenalezen, 12vrstvé modely mohou selhat")
PY
(cd $G/alphaflow && $ENV/bin/git rev-parse HEAD && $ENV/bin/git diff | tee $RUN/alphaflow_diff.txt)
""")
rep("  local rtj=$G/rt/rt_${tag}_${sada}.json\n", "  local rtj=$RUN/rt/rt_${tag}_${sada}.json\n")
rep("--templates_dir $G/templates --weights", "--templates_dir $G/templates_A --weights")
rep("cp -r $SCRATCHDIR/pdb/afT_base $SCRATCHDIR/pdb/afT_dist $G/samples/ 2>/dev/null\n", "cp -r $SCRATCHDIR/pdb/afT_base $SCRATCHDIR/pdb/afT_dist $RUN/samples/ 2>/dev/null\n")
rep("--out $G/bench_gen_${J}_p2_7jfl_C_anchor.json", "--out $RUN/bench_gen_${J}_p2_7jfl_C_anchor.json")
rep("--out $G/bench_gen_${J}_$(basename $S).json", "--out $RUN/bench_gen_${J}_$(basename $S).json")
rep("$(ls $G/bench_gen_${J}_*.json 2>/dev/null | wc -l)", "$(ls $RUN/bench_gen_${J}_*.json 2>/dev/null | wc -l)")
open("gen39b.pbs", "w", encoding="utf-8").write(s)

# 2) gen39sum.py: volba --run
s = open("gen39sum.py", encoding="utf-8").read()
rep('''    python gen39sum.py --gen l39/gen --list l39/gen39_list.txt --atlas-csv atlas_test.csv \\
        --pub l39/af/alphaflow_md_templates_base_202402 --out gen39_sum.json
"""''', '''    python gen39sum.py --gen l39/gen --list l39/gen39_list.txt --atlas-csv atlas_test.csv \\
        --pub l39/af/alphaflow_md_templates_base_202402 --out gen39_sum.json
--run l39/gen/run_<JOB> (úloha gen39b): časy, GPU, volání, bench a naše vzorky jen z této úlohy (rt/, gpu.txt, calls.tsv,
bench_gen_*.json, samples/), šablony a MSA dál z --gen. Bez --run staré rozložení (rt/, out_*/, bench_gen_* v --gen).
"""''')
rep('''    ap.add_argument("--no-rmsd", action="store_true")''', '''    ap.add_argument("--no-rmsd", action="store_true")
    ap.add_argument("--run", default=None, help="adresář jedné úlohy gen39b (run_<JOB>)")''')
rep('''    G = Path(args.gen)
''', '''    G = Path(args.gen)
    R = Path(args.run) if args.run else None
    D = R or G  # odkud časy, bench a naše vzorky
''')
rep('''    gpus = sorted({p.read_text().strip() for p in G.glob("out_*/gpu.txt")})
    res["gpu"] = gpus''', '''    gfiles = [R / "gpu.txt"] if R else list(G.glob("out_*/gpu.txt"))
    gpus = sorted({p.read_text().strip() for p in gfiles if p.exists()})
    res["gpu"] = gpus
    res["run"] = str(R) if R else None
    if len(gpus) > 1:
        res["notes"].append("měření z více GPU smíchána (staré rozložení bez --run)")''')
rep('''        for f in sorted(G.glob(f"rt/rt_{m}_*.json")):''', '''        for f in sorted(D.glob(f"rt/rt_{m}_*.json")):''')
rep('''    for f in G.glob("bench_gen_*.json"):''', '''    for f in D.glob("bench_gen_*.json"):''')
rep('''    for f in G.glob("out_*/calls.tsv"):''', '''    for f in ([R / "calls.tsv"] if R else G.glob("out_*/calls.tsv")):
        if not f.exists():
            continue''')
rep('''        f = G / "rt" / f"rt_{m}_{s}.json"''', '''        f = D / "rt" / f"rt_{m}_{s}.json"''')
rep('''            ours, pub, tp = G / "samples" / "afT_base" / f"{n}.pdb", Path(args.pub) / f"{n}.pdb", G / "templates" / f"{n}.pdb"''', '''            ours, pub, tp = D / "samples" / "afT_base" / f"{n}.pdb", Path(args.pub) / f"{n}.pdb", G / "templates" / f"{n}.pdb"''')
rep('''    print("GPU:", "; ".join(gpus) or "?")''', '''    print("GPU:", "; ".join(gpus) or "?", f"(úloha {R.name})" if R else "")
    for n in res["notes"]:
        print("POZOR:", n)''')
open("gen39sum.py", "w", encoding="utf-8").write(s)

# 3) tmpl39.py: ID řetězce A
s = open("tmpl39.py", encoding="utf-8").read()
rep('''        t.save_pdb(str(out / f"{name}.pdb"))
''', '''        t.save_pdb(str(out / f"{name}.pdb"))
        set_chain_a(out / f"{name}.pdb")
''')
rep('''def main():
''', '''def set_chain_a(path):
    """openfold protein.from_pdb_string nezná prázdné ID řetězce (ATLAS PDB ho nemá) → ve sloupci 22 doplnit A (gen39 25405838)."""
    out = []
    for ln in Path(path).read_text().splitlines():
        if ln.startswith(("ATOM", "HETATM", "TER")):
            ln = ln.ljust(22)
            if ln[21] == " ":
                ln = ln[:21] + "A" + ln[22:]
        out.append(ln)
    Path(path).write_text("\\n".join(out) + "\\n")


def main():
''')
open("tmpl39.py", "w", encoding="utf-8").write(s)
for f in ("gen39b.pbs", "gen39sum.py", "tmpl39.py"):
    print(f, sha(f))
