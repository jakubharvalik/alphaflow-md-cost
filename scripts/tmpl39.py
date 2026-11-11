"""Linie 39, změna Z8: vstupy pro měření ceny AlphaFlow/ESMFlow (predict.py autorů).
1) CSV name,seqres (bez sloupce msa_id, MSA se pak hledá pod jménem) pro proteiny ze seznamu "jméno délka sada",
2) šablony: jeden řetězec, jen těžké atomy, standardní názvy reziduí a atomů (mdtraj: HSD/HSE -> HIS, ILE CD -> CD1),
   počet reziduí musí být roven délce seqres. Sada B (fáze B): struktura ATLAS (stejný vstup jako u autorů).
   Sada K (kalibrace délky): první zveřejněný vzorek AF-MD+templates (pro měření času stačí správná délka, obsah
   šablony čas nemění). Neshody sekvence šablony a seqres se hlásí, nic se tiše nevypouští.
    python tmpl39.py --list gen39_list.txt --atlas-csv atlas_test.csv --atlas-pdb l39/atlas/pdb \
        --af-zip alphaflow_md_templates_base_202402.zip --out-csv gen39_in.csv --out-dir templates --report tmpl39.json
"""
import argparse
import csv
import json
import zipfile
from pathlib import Path

import mdtraj as md


def set_chain_a(path):
    """openfold protein.from_pdb_string nezná prázdné ID řetězce (ATLAS PDB ho nemá) → ve sloupci 22 doplnit A (gen39 25405838)."""
    out = []
    for ln in Path(path).read_text().splitlines():
        if ln.startswith(("ATOM", "HETATM", "TER")):
            ln = ln.ljust(22)
            if ln[21] == " ":
                ln = ln[:21] + "A" + ln[22:]
        out.append(ln)
    Path(path).write_text("\n".join(out) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--atlas-csv", required=True)
    ap.add_argument("--atlas-pdb", required=True)
    ap.add_argument("--af-zip", default=None)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()
    seqres = {r["name"]: r["seqres"] for r in csv.DictReader(open(args.atlas_csv))}
    rows = [ln.split() for ln in open(args.list) if ln.strip()]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    zf = zipfile.ZipFile(args.af_zip) if args.af_zip else None
    rep, ok_rows = {}, []
    for name, length, sada in rows:
        s = seqres.get(name)
        r = dict(sada=sada, L=int(length))
        if s is None:
            r["chyba"] = "není v atlas_test.csv"
            rep[name] = r
            print(name, r["chyba"])
            continue
        r["L_seqres"] = len(s)
        if sada == "B":
            src = Path(args.atlas_pdb) / f"{name}.pdb"
            r["zdroj"] = f"ATLAS {src.name}"
        else:
            src = None
            if zf is not None:
                cand = [f for f in zf.namelist() if f.endswith(f"/{name}.pdb") or f == f"{name}.pdb"]
                if cand:
                    src = out / f"_zip_{name}.pdb"
                    src.write_bytes(zf.read(cand[0]))
                    r["zdroj"] = f"AF-MD+templates vzorek 0 ({cand[0]})"
        if src is None or not Path(src).exists():
            r["chyba"] = f"chybí zdroj šablony ({src})"
            rep[name] = r
            print(name, r["chyba"])
            continue
        t = md.load(str(src))[0]
        if str(src).startswith(str(out / "_zip_")):
            Path(src).unlink()
        prot = t.topology.select("protein and element != H")
        t = t.atom_slice(prot)
        chains = [c for c in t.topology.chains if c.n_residues > 0]
        r["retezcu"] = len(chains)
        if len(chains) != 1:
            t = t.atom_slice(t.topology.select(f"chainid {chains[0].index}"))
        fasta = t.topology.to_fasta()[0]
        r["n_res"] = t.n_residues
        r["neshod"] = sum(a != b for a, b in zip(fasta, s)) if len(fasta) == len(s) else None
        if t.n_residues != len(s):
            r["chyba"] = f"počet reziduí šablony {t.n_residues} != seqres {len(s)}"
            rep[name] = r
            print(name, r["chyba"])
            continue
        t.save_pdb(str(out / f"{name}.pdb"))
        set_chain_a(out / f"{name}.pdb")
        ok_rows.append((name, s))
        rep[name] = r
        print(f"{name} L={len(s)} sada {sada}: {r['zdroj']}, neshod v sekvenci {r['neshod']}")
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "seqres"])
        w.writerows(ok_rows)
    Path(args.report).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(f"šablon {len(ok_rows)} z {len(rows)}")


if __name__ == "__main__":
    main()
