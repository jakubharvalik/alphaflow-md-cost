"""Stáhne vybrané proteiny z databáze ATLAS a převede je na vstupy Modelu E.

ATLAS (www.dsimb.inserm.fr/ATLAS): 3 × 100 ns na protein, CHARMM36m + TIP3P, 300 K, snímek 10 ps,
jen protein. Výběr z rozdělení MDGen (splits/atlas_{train,val,test}.csv, GitHub bjing2016/mdgen)
podle délky řetězce, pořadí je pevně zamíchané (--seed), takže stejné zadání dá stejné proteiny.

Pro každý protein: stáhne <name>_protein.zip, ověří a rozbalí do <out>/raw/<name>/ (surová data
zůstanou, pokud není --no-raw), spočte φ, ψ, χ1 a polohy páteře N, CA, C a uloží do
<out>/cache/<name>/: meta.json, R1_ang.npy (T, L, 3), R1_bb.npy (T, L, 3, 3), … (float16,
páteř posunutá do těžiště snímku). Hotové proteiny přeskočí, lze přerušit a znovu spustit.
Když na disku zbývá méně než --min-free-gb, stahování skončí.

    python atlas_fetch.py --split atlas_test --min-len 40 --max-len 80 --n 4 --out public/atlas
    python atlas_fetch.py --split atlas_train --min-len 40 --max-len 100 --n 60 --out public/atlas
    python atlas_fetch.py --names 6okd_C --out public/atlas
"""
import argparse
import csv
import io
import json
import shutil
import subprocess
import time
import urllib.request
import zipfile
from pathlib import Path

import mdtraj as md
import numpy as np

from model_c import AA, PAD
from model_e import backbone_index
from outcond import torsion_quads, torsions

ATLAS = "https://www.dsimb.inserm.fr/ATLAS/database/ATLAS/{name}/{name}_protein.zip"
GH = "https://raw.githubusercontent.com/bjing2016/mdgen/{branch}/splits/{split}.csv"
# názvy z CHARMM/GROMACS a protonační varianty → standardní reziduum
RES_ALIAS = {"HSD": "HIS", "HSE": "HIS", "HSP": "HIS", "HID": "HIS", "HIE": "HIS", "HIP": "HIS",
             "CYX": "CYS", "CYM": "CYS", "ASH": "ASP", "GLH": "GLU", "LYN": "LYS"}


def read_split(split):
    for branch in ("master", "main"):
        try:
            with urllib.request.urlopen(GH.format(branch=branch, split=split), timeout=60) as r:
                return list(csv.DictReader(io.StringIO(r.read().decode())))
        except Exception:
            continue
    raise SystemExit(f"Split {split} nelze stáhnout")


def select(rows, min_len, max_len, n, offset, seed):
    ok = [r["name"] for r in rows if min_len <= len(r["seqres"]) <= max_len]
    order = np.random.default_rng(seed).permutation(len(ok))
    return [ok[i] for i in order[offset:offset + n]]


def free_gb(path):
    return shutil.disk_usage(path).free / 1e9


def download(url, dest):
    part = dest.with_suffix(dest.suffix + ".part")
    subprocess.run(["curl", "-sSfL", "--retry", "4", "--retry-delay", "5", "-C", "-", "-o", str(part), url],
                   check=True)
    part.rename(dest)


def unpack(zpath, dest):
    with zipfile.ZipFile(zpath) as zf:
        bad = zf.testzip()
        if bad:
            raise RuntimeError(f"poškozený zip ({bad})")
        tmp = dest.with_name(dest.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        zf.extractall(tmp)
    # obsah může být ve vlastní podsložce; vše srovnat do dest
    tmp.mkdir(exist_ok=True)
    dest.mkdir(parents=True, exist_ok=True)
    for f in tmp.rglob("*"):
        if f.is_file():
            f.rename(dest / f.name)
    shutil.rmtree(tmp)


def find_inputs(raw):
    pdbs = sorted(raw.glob("*.pdb"))
    xtcs = sorted(raw.glob("*_R[0-9]*.xtc")) or sorted(raw.glob("*.xtc"))
    if not pdbs or not xtcs:
        raise RuntimeError(f"v {raw} chybí pdb nebo xtc: {[f.name for f in raw.iterdir()]}")
    return pdbs[0], xtcs


def heavy_topology(pdb):
    top = md.load(str(pdb)).topology
    for r in top.residues:
        r.name = RES_ALIAS.get(r.name, r.name)
    keep = top.select("protein and not element H")
    return top.subset(keep), keep


def chain_breaks(xyz0, bbi):
    """Indexy reziduí n, za kterými je přerušená páteř (C(n)–N(n+1) > 0,2 nm)."""
    d = np.linalg.norm(xyz0[bbi[1:, 0]] - xyz0[bbi[:-1, 2]], axis=-1)
    return np.nonzero(d > 0.2)[0].tolist()


def convert(name, raw, cache):
    pdb, xtcs = find_inputs(raw)
    top, keep = heavy_topology(pdb)
    Q, bbi = torsion_quads(top), backbone_index(top)
    seq = [r.name for r in top.residues]
    types = [AA.index(s) if s in AA else PAD for s in seq]
    tmp = cache.with_name(cache.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    reps = []
    for i, x in enumerate(xtcs, 1):
        tr = md.load(str(x), top=str(pdb), atom_indices=keep)
        dt = np.diff(tr.time)
        bb = tr.xyz[:, bbi]
        bb = bb - bb.mean((1, 2), keepdims=True)
        np.save(tmp / f"R{i}_ang.npy", torsions(tr.xyz, Q).astype(np.float16))
        np.save(tmp / f"R{i}_bb.npy", bb.astype(np.float16))
        reps.append(dict(file=x.name, n_frames=tr.n_frames, dt_ps=float(np.median(dt)) if len(dt) else None,
                         dt_uniform=bool(len(dt) and np.allclose(dt, dt[0], atol=1e-3)), t_end_ps=float(tr.time[-1])))
    meta = dict(name=name, source="ATLAS", forcefield="CHARMM36m + TIP3P, 300 K (ATLAS)", pdb=pdb.name,
                L=len(seq), seq=seq, types=types, unknown=[s for s in seq if s not in AA],
                breaks=chain_breaks(tr.xyz[0], bbi), reps=reps, dt_ps=reps[0]["dt_ps"])
    (tmp / "meta.json").write_text(json.dumps(meta, indent=1))
    shutil.rmtree(cache, ignore_errors=True)
    tmp.rename(cache)
    return meta


def process(name, out, zip_dir, keep_raw, min_free_gb):
    cache = out / "cache" / name
    if (cache / "meta.json").exists():
        return "přeskočeno (hotovo)", 0.0, 0.0
    raw = out / "raw" / name
    t0 = time.time()
    if not raw.exists() or not any(raw.glob("*.xtc")):
        if free_gb(out) < min_free_gb:
            raise SystemExit(f"Na disku zbývá {free_gb(out):.0f} GB < {min_free_gb} GB, končím.")
        zpath = (zip_dir / f"{name}_protein.zip") if zip_dir else (out / "raw" / f"{name}_protein.zip")
        if not zip_dir:
            download(ATLAS.format(name=name), zpath)
        size = zpath.stat().st_size / 1e6
        unpack(zpath, raw)
        if not zip_dir:
            zpath.unlink()
    else:
        size = sum(f.stat().st_size for f in raw.iterdir()) / 1e6
    t_dl = time.time() - t0
    t0 = time.time()
    meta = convert(name, raw, cache)
    if not keep_raw:
        shutil.rmtree(raw)
    frames = "+".join(str(r["n_frames"]) for r in meta["reps"])
    note = f", neznámá rezidua {meta['unknown']}" if meta["unknown"] else ""
    note += f", přerušení páteře za {meta['breaks']}" if meta["breaks"] else ""
    note += "" if all(r["dt_uniform"] for r in meta["reps"]) else ", NEROVNOMĚRNÝ krok snímků"
    return f"ok {size:.0f} MB, L {meta['L']}, snímky {frames} po {meta['dt_ps']} ps{note}", t_dl, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="atlas_train")
    ap.add_argument("--min-len", type=int, default=40)
    ap.add_argument("--max-len", type=int, default=100)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--names", nargs="*", help="místo výběru ze splitu konkrétní řetězce, např. 6okd_C")
    ap.add_argument("--out", default="public/atlas")
    ap.add_argument("--zip-dir", default=None, help="vzít už stažené <name>_protein.zip odsud")
    ap.add_argument("--no-raw", action="store_true", help="po převodu smazat surová data")
    ap.add_argument("--min-free-gb", type=float, default=25.0)
    args = ap.parse_args()
    out = Path(args.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    (out / "cache").mkdir(parents=True, exist_ok=True)
    names = args.names or select(read_split(args.split), args.min_len, args.max_len, args.n, args.offset, args.seed)
    tag = "names" if args.names else f"{args.split}_{args.min_len}-{args.max_len}_{args.offset}_{len(names)}"
    (out / f"list_{tag}.json").write_text(json.dumps(names))
    print(f"{len(names)} proteinů: {' '.join(names)} | volno {free_gb(out):.0f} GB", flush=True)
    t_all, n_err = time.time(), 0
    for i, name in enumerate(names):
        try:
            status, t_dl, t_proc = process(name, out, Path(args.zip_dir) if args.zip_dir else None,
                                           not args.no_raw, args.min_free_gb)
        except SystemExit:
            raise
        except Exception as e:
            status, t_dl, t_proc, n_err = f"CHYBA {e}", 0.0, 0.0, n_err + 1
        print(f"[{i + 1}/{len(names)}] {name}: {status} | stažení {t_dl:.0f} s, převod {t_proc:.0f} s "
              f"| celkem {time.time() - t_all:.0f} s, volno {free_gb(out):.0f} GB", flush=True)
    print(f"Hotovo, chyb {n_err}.")


if __name__ == "__main__":
    main()
