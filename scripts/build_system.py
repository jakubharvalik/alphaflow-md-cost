"""Postaví a zekvilibruje malý peptid v explicitní vodě pro pilot.

Výstup do --out: system.xml, state.xml (zekvilibrovaný společný start),
start.pdb (celý systém) a solute.pdb (jen peptid, pro analýzu).

Příklad:
    python build_system.py --seq AAAAA --out ala5
Start ze známé struktury (PDB soubor nebo 4znakové ID, stáhne se z RCSB; bere se 1. řetězec bez vody a H):
    python build_system.py --seq YYDPETGTWY --pdb 5AWL --out cln025f
Protein ze startovní struktury ATLAS v silovém poli ATLAS (CHARMM36m, CHARMM TIP3P, 0,15 M NaCl);
názvy CHARMM/GROMACS (HSD, ILE CD, OT1/OT2) se převedou, sekvence se vezme ze souboru:
    python build_system.py --pdb public/atlas/raw/6okd_C/6okd_C.pdb --ff charmm36m --ionic-molar 0.15 --out p_6okd_C
S --restrain-k 1000 drží ekvilibrace těžké atomy proteinu u krystalu (jako ATLAS), jinak se start může přeskupit.
"""
import argparse
import json
from pathlib import Path

import openmm as mm
import openmm.app as app
import openmm.unit as u
import PeptideBuilder
from Bio.PDB import PDBIO, PDBParser, Select

FORCEFIELDS = {"amber14": (("amber14-all.xml", "amber14/tip3p.xml"), "amber14-all (ff14SB) + TIP3P"),
               "charmm36m": (("charmm36_2024.xml", "charmm36_2024/water.xml"), "CHARMM36m (charmm36_2024) + CHARMM TIP3P")}
RES_ALIAS = {"HSD": "HIS", "HSE": "HIS", "HSP": "HIS", "HID": "HIS", "HIE": "HIS", "HIP": "HIS",
             "CYX": "CYS", "CYM": "CYS", "ASH": "ASP", "GLH": "GLU", "LYN": "LYS"}
ATOM_ALIAS = {("ILE", "CD"): "CD1", ("*", "OT1"): "O", ("*", "OT2"): "OXT", ("*", "O1"): "O", ("*", "O2"): "OXT"}
ONE_TO_THREE = dict(A="ALA", R="ARG", N="ASN", D="ASP", C="CYS", Q="GLN", E="GLU",
                    G="GLY", H="HIS", I="ILE", L="LEU", K="LYS", M="MET", F="PHE",
                    P="PRO", S="SER", T="THR", W="TRP", Y="TYR", V="VAL")


def build_peptide_pdb(seq, path):
    # Natažená konformace; zwitterion (NH3+/COO-), čepičky ACE/NME zatím ne.
    structure = PeptideBuilder.make_extended_structure(seq)
    PeptideBuilder.add_terminal_OXT(structure)
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(path))


def peptide_from_pdb(src, seq, path):
    """1. model a 1. řetězec s aminokyselinami ze souboru/ID PDB, bez vody, H a alternativních poloh kromě A.
    Názvy CHARMM/GROMACS převede na standardní PDB; seq = None → sekvence ze souboru. Vrací sekvenci."""
    src_path = Path(src)
    if not src_path.exists() and len(src) == 4:
        import urllib.request
        src_path = path.parent / f"{src.upper()}.pdb"
        if not src_path.exists():
            urllib.request.urlretrieve(f"https://files.rcsb.org/download/{src.upper()}.pdb", src_path)
    st = PDBParser(QUIET=True).get_structure("x", str(src_path))
    model = st[0]
    chain = next(c for c in model if any(r.id[0] == " " for r in c))
    for r in chain:
        r.resname = RES_ALIAS.get(r.get_resname(), r.get_resname())
        for a in list(r):
            new = ATOM_ALIAS.get((r.get_resname(), a.get_id())) or ATOM_ALIAS.get(("*", a.get_id()))
            if new and new not in r:
                a.id = a.name = a.fullname = new
    three_to_one = {v: k for k, v in ONE_TO_THREE.items()}
    got = "".join(three_to_one.get(r.get_resname(), "X") for r in chain if r.id[0] == " ")
    if "X" in got:
        raise SystemExit(f"Neznámé reziduum v {src}: {got}")
    seq = seq or got
    if got != seq:
        raise SystemExit(f"Sekvence v {src} ({got}) neodpovídá --seq {seq}")

    class Sel(Select):
        def accept_model(self, m):
            return m.id == model.id

        def accept_chain(self, c):
            return c.id == chain.id

        def accept_residue(self, r):
            return r.id[0] == " "

        def accept_atom(self, a):
            return a.element != "H" and a.get_altloc() in (" ", "A")

    io = PDBIO()
    io.set_structure(st)
    io.save(str(path), Sel())
    clean = PDBParser(QUIET=True).get_structure("p", str(path))
    if "OXT" not in list(clean.get_residues())[-1]:
        PeptideBuilder.add_terminal_OXT(clean)
    io.set_structure(clean)
    io.save(str(path))
    return seq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default=None, help="výchozí AAAAA; s --pdb nepovinné (vezme se ze souboru)")
    ap.add_argument("--out", default="ala5")
    ap.add_argument("--pdb", default=None, help="start ze struktury (soubor nebo ID PDB) místo natažené konformace")
    ap.add_argument("--padding-nm", type=float, default=1.0)
    ap.add_argument("--ff", choices=sorted(FORCEFIELDS), default="amber14")
    ap.add_argument("--ionic-molar", type=float, default=0.0, help="NaCl navíc k neutralizaci (ATLAS 0,15)")
    ap.add_argument("--temperature", type=float, default=300.0)
    ap.add_argument("--hmr", type=float, default=1.5, help="hmotnost H v amu (1.0 = bez HMR)")
    ap.add_argument("--equil-ps", type=float, default=500.0)
    ap.add_argument("--restrain-k", type=float, default=0.0,
                    help="kJ/mol/nm²: ekvilibrace s polohovými restrainty těžkých atomů peptidu/proteinu (jako ATLAS), "
                         "start pak zůstane u krystalu; 0 = bez restraintů (dřívější chování)")
    ap.add_argument("--platform", default=None)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "peptide_raw.pdb"
    if args.pdb:
        args.seq = peptide_from_pdb(args.pdb, args.seq, raw)
    else:
        args.seq = args.seq or "AAAAA"
        build_peptide_pdb(args.seq, raw)

    ff_files, ff_name = FORCEFIELDS[args.ff]
    ff = app.ForceField(*ff_files)
    pdb = app.PDBFile(str(raw))
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.addHydrogens(ff, pH=7.0)
    n_solute = modeller.topology.getNumAtoms()
    with open(out / "solute.pdb", "w") as f:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, f)

    modeller.addSolvent(ff, padding=args.padding_nm * u.nanometer,
                        boxShape="dodecahedron", neutralize=True, ionicStrength=args.ionic_molar * u.molar)
    system = ff.createSystem(modeller.topology, nonbondedMethod=app.PME,
                             nonbondedCutoff=0.9 * u.nanometer,
                             constraints=app.HBonds,
                             hydrogenMass=args.hmr * u.amu)
    dt_fs = 4.0 if args.hmr > 1.0 else 2.0
    temp = args.temperature * u.kelvin
    system.addForce(mm.MonteCarloBarostat(1.0 * u.bar, temp))
    pos_force = None
    if args.restrain_k > 0:  # polohové restrainty těžkých atomů solutu jen pro ekvilibraci
        posres = mm.CustomExternalForce("0.5*k_pos*periodicdistance(x, y, z, x0, y0, z0)^2")
        posres.addGlobalParameter("k_pos", args.restrain_k)
        for p in ("x0", "y0", "z0"):
            posres.addPerParticleParameter(p)
        for atom in list(modeller.topology.atoms())[:n_solute]:
            if atom.element is not None and atom.element.symbol != "H":
                posres.addParticle(atom.index, modeller.positions[atom.index].value_in_unit(u.nanometer))
        pos_force = system.addForce(posres)
    integ = mm.LangevinMiddleIntegrator(temp, 1.0 / u.picosecond, dt_fs * u.femtoseconds)
    platform = mm.Platform.getPlatformByName(args.platform) if args.platform else None
    sim = app.Simulation(modeller.topology, system, integ, platform) if platform else \
        app.Simulation(modeller.topology, system, integ)
    sim.context.setPositions(modeller.positions)
    sim.minimizeEnergy()
    sim.context.setVelocitiesToTemperature(temp, 1)
    sim.step(int(args.equil_ps * 1000 / dt_fs))

    state = sim.context.getState(getPositions=True, getVelocities=True,
                                 enforcePeriodicBox=True)
    if pos_force is not None:  # do system.xml pro produkci už bez restraintů
        system.removeForce(pos_force)
    with open(out / "state.xml", "w") as f:
        f.write(mm.XmlSerializer.serialize(state))
    with open(out / "system.xml", "w") as f:
        f.write(mm.XmlSerializer.serialize(system))
    with open(out / "start.pdb", "w") as f:
        app.PDBFile.writeFile(sim.topology, state.getPositions(), f)

    meta = dict(seq=args.seq, start_pdb=args.pdb, n_atoms=system.getNumParticles(), n_solute=n_solute,
                temperature_K=args.temperature, dt_fs=dt_fs, hmr_amu=args.hmr,
                friction_per_ps=1.0, forcefield=ff_name, ionic_molar=args.ionic_molar, n_residues=len(args.seq),
                padding_nm=args.padding_nm, equil_ps=args.equil_ps, restrain_k=args.restrain_k,
                platform=sim.context.getPlatform().getName())
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
