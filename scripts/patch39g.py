"""patch39g: úprava gen39b.pbs po Gemini 39f (10. 11. 2026). Vstup gen39b.pbs SHA 73e1b5b5…, výstup se vypíše.
1) GPU: gpu_cap=sm_80 (8.0–8.9, ne Hopper) + gpu_mem=40gb (velké proteiny sady K bez OOM) + cl_fobos=False (L40S, kde běžela 25405838;
   rozhodující měření na Ampere jako A100 v publikaci, 25405838 zůstane jako srovnání sady K na jiné GPU).
2) Kontrola šablon v úloze funkcí, kterou volá AlphaFlow (alphaflow.utils.protein.from_pdb_string): šablona, která se nenačte nebo má
   jiný počet reziduí než seqres, se z volání vynechá a vypíše s důvodem (jedna vadná šablona jinak shodí celé volání sady B).
    python3 patch39g.py   (v adresáři s gen39b.pbs)
"""
import hashlib
from pathlib import Path

p = Path("gen39b.pbs")
s = p.read_text()
h = hashlib.sha256(s.encode()).hexdigest()
assert h.startswith("73e1b5b56f8ce3ec"), f"gen39b.pbs má jiné SHA {h[:16]}, nic neměním"

R = [
    ("#PBS -l select=1:ncpus=4:ngpus=1:mem=32gb:scratch_local=60gb:gpu_cap=sm_86\n",
     "#PBS -l select=1:ncpus=4:ngpus=1:mem=32gb:scratch_local=60gb:gpu_cap=sm_80:gpu_mem=40gb:cl_fobos=False\n"),
    ("# GPU: gpu_cap=sm_86 = architektura 8.x od 8.6 (RTX A4000 jako bench3, A40, L40), ne Hopper (torch 1.12 cu113 nemá sm_90).\n",
     "# GPU: gpu_cap=sm_80 = 8.0–8.9 (ne Hopper, torch 1.12 cu113 nemá sm_90), gpu_mem=40gb (sada K do 724 reziduí bez OOM),\n"
     "# cl_fobos=False (L40S, Ada; gen39b po Gemini 39f na Ampere jako A100 v publikaci, 25405838 na L40S = srovnání sady K mezi GPU).\n"),
    ("[ \"${CAPM:-0}\" -ge 9 ] && { echo \"compute capability $CAP: torch 1.12 cu113 nemá jádra pro sm_90, končím (znovu odeslat s jinou GPU)\"; exit 9; }\n",
     "[ \"${CAPM:-0}\" -ge 9 ] && { echo \"compute capability $CAP: torch 1.12 cu113 nemá jádra pro sm_90, končím (znovu odeslat s jinou GPU)\"; exit 9; }\n"
     "[ \"$CAP\" = 8.9 ] && echo \"POZOR: GPU Ada (8.9), ne Ampere; měření platí, ale srovnání s A100 v publikaci je slabší\"\n"),
    ("[ $(ls $G/templates_A/*.pdb 2>/dev/null | wc -l) -ge $(wc -l < $L/gen39_list.txt) ] || { echo \"templates_A neúplné\"; exit 4; }\n",
     "[ $(ls $G/templates_A/*.pdb 2>/dev/null | wc -l) -ge $(wc -l < $L/gen39_list.txt) ] || { echo \"templates_A neúplné\"; exit 4; }\n"
     "# kontrola šablon funkcí, kterou volá AlphaFlow (alphaflow.utils.protein.from_pdb_string, inference.py): vyřadit jen šablonu, která se\n"
     "# nenačte nebo má jiný počet reziduí než seqres (vypíše se s důvodem, $RUN/tmplcheck.tsv); neshody sekvence a neúplná páteř se jen hlásí\n"
     "touch $SCRATCHDIR/tmpl_bad.txt\n"
     "(cd $G/alphaflow && $P - $G/gen39_in.csv $G/templates_A $L/gen39_list.txt $RUN/tmplcheck.tsv $SCRATCHDIR/tmpl_bad.txt) <<'PY' || echo \"kontrola šablon se nespustila, pokračuji bez ní\"\n"
     "import sys\n"
     "import numpy as np, pandas as pd\n"
     "import alphaflow.utils.protein as protein\n"
     "from openfold.np import residue_constants as rc\n"
     "csv, td, lst, out, bad = sys.argv[1:6]\n"
     "seq = dict(pd.read_csv(csv)[[\"name\", \"seqres\"]].values)\n"
     "rows, nbad = [], []\n"
     "for ln in open(lst):\n"
     "    if not ln.strip():\n"
     "        continue\n"
     "    n, _, s = ln.split()\n"
     "    r = dict(name=n, sada=s, ok=0, L=len(seq.get(n, \"\")), n_res=None, neshod=None, pater=None, duvod=\"\")\n"
     "    try:\n"
     "        if n not in seq:\n"
     "            raise ValueError(\"není v gen39_in.csv\")\n"
     "        pr = protein.from_pdb_string(open(f\"{td}/{n}.pdb\").read())\n"
     "        r[\"n_res\"] = int(pr.atom_positions.shape[0])\n"
     "    except Exception as e:\n"
     "        r[\"duvod\"] = f\"{type(e).__name__}: {e}\"\n"
     "    else:\n"
     "        try:\n"
     "            sq = \"\".join(rc.restypes_with_x[a] for a in pr.aatype)\n"
     "            r[\"neshod\"] = sum(a != b for a, b in zip(sq, seq[n])) + abs(len(sq) - len(seq[n]))\n"
     "            r[\"pater\"] = int((np.asarray(pr.atom_mask)[:, :3].sum(1) == 3).sum())\n"
     "        except Exception as e:\n"
     "            r[\"duvod\"] = f\"jen informativní část kontroly selhala ({type(e).__name__}: {e})\"\n"
     "        if r[\"n_res\"] != r[\"L\"]:\n"
     "            r[\"duvod\"] = f\"počet reziduí {r['n_res']} != seqres {r['L']}\"\n"
     "        else:\n"
     "            r[\"ok\"] = 1\n"
     "    if not r[\"ok\"]:\n"
     "        nbad.append(n)\n"
     "        print(\"šablona VYŘAZENA:\", n, s, r[\"duvod\"])\n"
     "    elif r[\"neshod\"] or (r[\"pater\"] is not None and r[\"pater\"] != r[\"L\"]):\n"
     "        print(\"šablona\", n, s, \"neshod v sekvenci\", r[\"neshod\"], \"úplná páteř\", r[\"pater\"], \"z\", r[\"L\"])\n"
     "    rows.append(r)\n"
     "pd.DataFrame(rows).to_csv(out, sep=\"\\t\", index=False)\n"
     "open(bad, \"w\").write(\"\".join(x + \"\\n\" for x in nbad))\n"
     "ok = [r for r in rows if r[\"ok\"]]\n"
     "print(f\"kontrola šablon: {len(ok)} z {len(rows)} OK (sada B {sum(r['sada'] == 'B' for r in ok)}), \"\n"
     "      f\"neshod v sekvenci celkem {sum(r['neshod'] or 0 for r in ok)}\")\n"
     "PY\n"),
    ("awk '$3==\"B\"{print $1}' $L/gen39_list.txt > $SCRATCHDIR/setB.txt\n"
     "awk '$3==\"K\"{print $1}' $L/gen39_list.txt > $SCRATCHDIR/setK.txt\n",
     "awk '$3==\"B\"{print $1}' $L/gen39_list.txt | grep -vxFf $SCRATCHDIR/tmpl_bad.txt > $SCRATCHDIR/setB.txt\n"
     "awk '$3==\"K\"{print $1}' $L/gen39_list.txt | grep -vxFf $SCRATCHDIR/tmpl_bad.txt > $SCRATCHDIR/setK.txt\n"),
]
for old, new in R:
    assert s.count(old) == 1, f"úsek nenalezen právě jednou: {old[:60]!r}"
    s = s.replace(old, new)
s = s.replace("# gen39b (opakování po 25405838, 10. 11. 2026)", "# gen39b (opakování po 25405838, 10. 11. 2026; GPU a kontrola šablon po Gemini 39f)", 1)
p.write_text(s)
print("gen39b.pbs", hashlib.sha256(s.encode()).hexdigest())
