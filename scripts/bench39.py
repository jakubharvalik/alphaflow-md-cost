"""Linie 39: rychlost MD boxu z krystalu p2_<N> (v_p2 pro účtování MD z krystalu, prereg39) na stejné GPU.
Stejný integrátor jako s18 (LangevinMiddle, dt z meta.json), system.xml + state.xml z build_system, 5 ps zahřátí,
pak --ps ps se čtením poloh po 1 ps (jako MD v s18). Výstup JSON: atomy, ns/den, platforma.
    python bench39.py --sys p2_7jfl_C --platform CUDA --out bench_p2_7jfl_C.json
"""
import argparse
import json
import time
from pathlib import Path

import openmm as mm
import openmm.unit as u


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sys", required=True)
    ap.add_argument("--ps", type=float, default=50.0)
    ap.add_argument("--platform", default="CUDA")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    d = Path(args.sys)
    meta = json.loads((d / "meta.json").read_text())
    system = mm.XmlSerializer.deserialize((d / "system.xml").read_text())
    state = mm.XmlSerializer.deserialize((d / "state.xml").read_text())
    dt_fs = float(meta["dt_fs"])
    integ = mm.LangevinMiddleIntegrator(meta["temperature_K"] * u.kelvin, meta.get("friction_per_ps", 1.0) / u.picosecond,
                                        dt_fs * u.femtoseconds)
    ctx = mm.Context(system, integ, mm.Platform.getPlatformByName(args.platform))
    ctx.setState(state)
    spp = int(round(1000 / dt_fs))
    integ.step(5 * spp)
    ctx.getState(getPositions=True)
    t0 = time.time()
    for _ in range(int(round(args.ps))):
        integ.step(spp)
        ctx.getState(getPositions=True)
    w = time.time() - t0
    res = dict(sys=d.name, n_atoms=system.getNumParticles(), dt_fs=dt_fs, ps=args.ps, wall_s=round(w, 2),
               ns_per_day=round(args.ps / 1000 / w * 86400, 1), platform=ctx.getPlatform().getName(),
               barostat=any(isinstance(system.getForce(i), mm.MonteCarloBarostat) for i in range(system.getNumForces())))
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
