"""patch39l: gen39e.pbs = gen39b.pbs zúžená na měření na A100 (prereg39 Z9b, citlivost, nerozhoduje).
Cluster zia.cerit-sc.cz (4× A100 40 GB, gpu_cap sm_80, fronty q_gpu/q_2h/q_4h; pbsnodes 10. 11.), walltime 4 h.
Jen afT_base a afT_dist (sady B a K, stejný počet vzorků jako gen39b) + bench39 na téže A100 (napřed, zahřátí, kotva, konec).
Šablony s řetězcem A se píší do $SCRATCHDIR/templates_A (ne do sdíleného $G/templates_A, kdyby současně běžela gen39d).
Úloha skončí (kód 8), když GPU není A100. Vstup gen39b.pbs SHA 8ceb3a4d…, výstup gen39e.pbs (gen39b.pbs beze změny).
    python3 patch39l.py   (v adresáři s gen39b.pbs)
"""
import hashlib
from pathlib import Path

s = Path("gen39b.pbs").read_text()
h = hashlib.sha256(s.encode()).hexdigest()
assert h.startswith("8ceb3a4dc7c2d7be"), f"gen39b.pbs má jiné SHA {h[:16]}, nic neměním"
TAGS = "for tag in afT_base afT_dist afT12_base afT12_dist esT_base esT_dist; do\n"
assert s.count(TAGS) == 2, "čekám 2 smyčky přes modely"
s = s.replace(TAGS, "for tag in afT_base afT_dist; do\n")
assert s.count("$G/templates_A") == 6, f"čekám 6 výskytů (1 v komentáři) $G/templates_A, je {s.count('$G/templates_A')}"
s = s.replace("$G/templates_A", "$TA")
R = [
    ("#PBS -N l39_gen2\n", "#PBS -N l39_gA100\n"),
    (":gpu_cap=sm_80:gpu_mem=40gb:cl_fobos=False\n", ":gpu_cap=sm_80:cl_zia=True\n"),
    ("#PBS -l walltime=16:00:00\n", "#PBS -l walltime=4:00:00\n"),
    ("#PBS -j oe\n",
     "#PBS -j oe\n"
     "# gen39e (patch39l, prereg39 Z9b): totéž měření jako gen39b, ale na A100 (cluster zia) a jen afT_base a afT_dist; citlivost\n"
     "# „GPU z publikace“ místo přepočtu ÷ 2,57, primární účtování zůstává z A40 (gen39b). Šablony v $SCRATCHDIR/templates_A.\n"),
    ("RUN=$G/run_$J\n", "RUN=$G/run_$J\nTA=$SCRATCHDIR/templates_A\n"),
    ("[ \"$CAP\" = 8.9 ] && echo \"POZOR: GPU Ada (8.9), ne Ampere; měření platí, ale srovnání s A100 v publikaci je slabší\"\n",
     "echo \"$GPU\" | grep -q A100 || { echo \"GPU není A100, měření pro Z9b neplatí, končím\"; exit 8; }\n"),
    ("afT12_base:alphaflow_12l_md_templates_base_202406:alphafold:\n"
     "afT12_dist:alphaflow_12l_md_templates_distilled_202406:alphafold:--noisy_first --no_diffusion\n"
     "esT_base:esmflow_md_templates_base_202402:esmfold:\n"
     "esT_dist:esmflow_md_templates_distilled_202402:esmfold:--noisy_first --no_diffusion\"\n",
     "\"\n"),
]
for old, new in R:
    assert s.count(old) == 1, f"úsek nenalezen právě jednou: {old[:60]!r}"
    s = s.replace(old, new)
Path("gen39e.pbs").write_text(s)
print("gen39e.pbs", hashlib.sha256(s.encode()).hexdigest())
