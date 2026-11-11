"""Spustí N replik MD ze společného startu (jiné náhodné rychlosti) a ukládá peptid.

Každá replika: rep_XXX.npz s polohami a rychlostmi peptidu po --save-ps
(nm, nm/ps, float32) a rep_XXX.dcd s celým systémem po --full-ps.
Hotové repliky se přeskočí, takže běh lze přerušit a znovu spustit.

Benchmark (krátký běh, vypíše ns/den a doporučenou délku replik):
    python run_replicas.py --sys ala5 --benchmark --budget-hours 8 --n-replicas 100
Ostrý běh:
    python run_replicas.py --sys ala5 --n-replicas 100 --ns 2 --parallel 2
"""
import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as u


def load(sysdir):
    sysdir = Path(sysdir)
    meta = json.loads((sysdir / "meta.json").read_text())
    system = mm.XmlSerializer.deserialize((sysdir / "system.xml").read_text())
    state = mm.XmlSerializer.deserialize((sysdir / "state.xml").read_text())
    top = app.PDBFile(str(sysdir / "start.pdb")).topology
    return meta, system, state, top


def make_sim(meta, system, state, top, seed, platform_name):
    integ = mm.LangevinMiddleIntegrator(meta["temperature_K"] * u.kelvin,
                                        meta["friction_per_ps"] / u.picosecond,
                                        meta["dt_fs"] * u.femtoseconds)
    integ.setRandomNumberSeed(seed)
    if platform_name:
        sim = app.Simulation(top, system, integ, mm.Platform.getPlatformByName(platform_name))
    else:
        sim = app.Simulation(top, system, integ)
    sim.context.setState(state)
    # Společný start: stejné polohy, nové Maxwell-Boltzmannovy rychlosti.
    sim.context.setVelocitiesToTemperature(meta["temperature_K"] * u.kelvin, seed)
    return sim


def run_one(args_tuple):
    sysdir, outdir, rep, ns, save_ps, full_ps, platform_name = args_tuple
    out = Path(outdir) / f"rep_{rep:03d}.npz"
    if out.exists():
        return rep, "skip", 0.0
    meta, system, state, top = load(sysdir)
    n_sol = meta["n_solute"]
    sim = make_sim(meta, system, state, top, seed=1000 + rep, platform_name=platform_name)
    steps_save = int(round(save_ps * 1000 / meta["dt_fs"]))
    steps_full = int(round(full_ps * 1000 / meta["dt_fs"]))
    n_frames = int(round(ns * 1000 / save_ps))
    sim.reporters.append(app.DCDReporter(str(Path(outdir) / f"rep_{rep:03d}.dcd"), steps_full))
    pos = np.empty((n_frames, n_sol, 3), np.float32)
    vel = np.empty((n_frames, n_sol, 3), np.float32)
    t0 = time.time()
    for i in range(n_frames):
        sim.step(steps_save)
        st = sim.context.getState(getPositions=True, getVelocities=True)
        pos[i] = st.getPositions(asNumpy=True).value_in_unit(u.nanometer)[:n_sol]
        vel[i] = st.getVelocities(asNumpy=True).value_in_unit(u.nanometer / u.picosecond)[:n_sol]
    wall = time.time() - t0
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, pos=pos, vel=vel, save_ps=save_ps, seed=1000 + rep, wall_s=wall)
    tmp.rename(out)
    return rep, "done", wall


def benchmark(sysdir, platform_name, ps=20.0):
    meta, system, state, top = load(sysdir)
    sim = make_sim(meta, system, state, top, seed=1, platform_name=platform_name)
    steps = int(ps * 1000 / meta["dt_fs"])
    sim.step(500)  # zahřátí / kompilace kernelů
    t0 = time.time()
    sim.step(steps)
    wall = time.time() - t0
    return sim.context.getPlatform().getName(), ps / 1000 / wall * 86400


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sys", default="ala5")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-replicas", type=int, default=100)
    ap.add_argument("--ns", type=float, default=2.0)
    ap.add_argument("--save-ps", type=float, default=1.0)
    ap.add_argument("--full-ps", type=float, default=100.0)
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--platform", default=None)
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--budget-hours", type=float, default=8.0)
    args = ap.parse_args()

    if args.benchmark:
        names = [args.platform] if args.platform else \
            [mm.Platform.getPlatform(i).getName() for i in range(mm.Platform.getNumPlatforms())]
        res = {}
        for name in names:
            if name == "Reference":
                continue
            try:
                pname, nsday = benchmark(args.sys, name)
                res[pname] = nsday
                print(f"{pname:10s} {nsday:8.1f} ns/den (1 proces)")
            except Exception as e:  # platforma nemusí být k dispozici
                print(f"{name:10s} nelze: {e}")
        if res:
            best = max(res, key=res.get)
            total_ns = res[best] * args.budget_hours / 24
            print(f"\nNejrychlejší: {best}. Za {args.budget_hours} h ≈ {total_ns:.1f} ns celkem, "
                  f"tj. ≈ {total_ns / args.n_replicas:.2f} ns na repliku při {args.n_replicas} replikách "
                  f"(bez souběhu; s --parallel může být víc, ověř).")
        return

    outdir = Path(args.out or Path(args.sys) / "runs")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "run_args.json").write_text(json.dumps(vars(args), indent=2))
    jobs = [(args.sys, str(outdir), r, args.ns, args.save_ps, args.full_ps, args.platform)
            for r in range(args.n_replicas)]
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.parallel) as pool:
        for rep, status, wall in pool.imap_unordered(run_one, jobs):
            print(f"rep {rep:03d} {status} {wall:7.1f} s | celkem {time.time() - t0:7.0f} s", flush=True)


if __name__ == "__main__":
    main()
