# Code and derived data: cost accounting of AlphaFlow-seeded MD on ATLAS test proteins

This repository accompanies the preprint "What does a generative ensemble really save? Honest cost accounting of AlphaFlow-seeded molecular dynamics on ATLAS test proteins".

- Author: Jakub Harvalík, Department of Organic Chemistry, University of Chemistry and Technology, Prague, Czech Republic
- Preprint: [DOI to be added]
- Version: 1.0 (GitHub release v1.0, archived at Zenodo: [DOI to be added])
- Repository: https://github.com/jakubharvalik/alphaflow-md-cost

## Contents

| Folder | What it holds | Licence |
|---|---|---|
| scripts/ | Python scripts for simulation, evaluation, cost accounting and figure data, and the helper modules they import | MIT (LICENSE) |
| jobs/ | PBS job files run on the MetaCentrum cluster; listB.txt (the 25 phase-B proteins and their lengths); gen39_list.txt (proteins of the generator-cost measurement, sets B and K) | MIT (LICENSE) |
| alphaflow_patch/ | Diff of the local changes to AlphaFlow at commit 0408d7c8 (preprint Table S5) | as AlphaFlow (MIT) |
| tables/ | Derived per-protein tables and summaries (CSV, JSON, TXT) behind the figures and tables of the preprint | CC BY 4.0 |
| inputs/l39/ | Per-protein inputs of the summary scripts: phase-A and phase-B analysis outputs (af38, sub39, eval39, rel39, af39 JSON), run summaries with GPU times (runs/), MD speed benchmarks (bench/), GPU names (logs/gpu_*.txt), the S18d50 job logs read by si39.py and chi39.py, and the generator timing records (gen/, without samples, templates or MSAs). Metrics and timings only, no coordinates | CC BY 4.0 |
| protocol/ | prereg39.md, the internal time-stamped protocol log with the decision rules and their amendments (in Czech, authoritative); prereg39_en.md, an English translation made for this archive | CC BY 4.0 |
| reproduce.sh, tools/ | Script that recomputes every summary table from inputs/ and compares it with tables/ (added for this archive); tools/jcmp39.py compares two JSON files numerically | MIT (LICENSE) |

SHA256SUMS.txt lists the SHA-256 checksum of every file except itself; check it with `shasum -a 256 -c SHA256SUMS.txt` (macOS) or `sha256sum -c SHA256SUMS.txt` (Linux). The checksum prefixes quoted in the preprint can be looked up there. Scripts, job files, inputs and tables are unchanged copies of the files that were run or produced, except for the name correction described under Notes; README.md, LICENSE, CITATION.cff, .gitignore, .gitattributes, reproduce.sh, tools/ and protocol/prereg39_en.md were written for this archive. Comments, log messages and the protocol log are mostly in Czech.

## Workflow

1. Phase A, CPU only (jobs/atlas39a.pbs): downloads ATLAS (atlas_fetch.py) and the published AlphaFlow/ESMFlow sample sets for the 82 ATLAS test proteins and compares the samples with ATLAS MD (af38.py); sumA39.py summarises.
2. Phase B, GPU (jobs/b39.pbs, one protein of listB.txt per array task): s18.py runs S18, 18 independent short MD boxes started from AlphaFlow samples; s39.py runs seed 1 (targets from a random half of the samples) and the K18 control from the crystal (input made by k18_make.py); build_system.py builds the solvated systems; bench39.py measures the MD speed from the crystal on the same GPU.
3. Evaluation: sub38.py (MD time t* needed to match S18), eval39.py (leave-one-out evaluation with an equal number of frames; also run by jobs/ana39.pbs), rel38.py (Ramachandran and χ1 local structure), ctrl39.py (control Z7a: 3 × 3.6 ns of ATLAS MD from the crystal compared with S18).
4. Generator cost (protocol amendments Z8 and Z8a): jobs/inst39.pbs installs AlphaFlow; tmpl39.py prepares the inputs; jobs/gen39b.pbs measures on an A40 (decisive); jobs/gen39e.pbs on an A100 (sensitivity); gen39sum.py summarises; grid39.py builds the accounting grid. The patch39*.py files are the scripted edits that produced each job file from its predecessor; each checks the SHA-256 of its input.
5. Distilled generator (amendment Z9): jobs/gen39d.pbs generates 50 distilled samples per protein; jobs/b39d.pbs runs S18 from them (s39d.py, with the clash filter); clash39.py counts atom clashes; cmp39d.py evaluates; chi39.py checks the χ1 change after MD in these runs.
6. Accounting and figures: sum39.py (speed-ups, one-sided 95 % bootstrap bounds, decision criteria); fig39.py, fig39c.py and si39.py extract the data for the figures and SI tables in tables/.

The other .py files in scripts/ (analyze, atlas_compare, atlas_fetch, core36, hybrid, jump28, kinetics, model_c, model_d, model_e, model_q, oracle38, outcond, run_replicas) are modules of the earlier stages of the project. They are included because the scripts above import them, often only for a few helper functions; most of their code is not used in this work. They were copied unchanged from the code archive that every cluster job unpacked (code39.tgz, SHA-256 50651d50…).

## Where the numbers of the preprint come from

| Preprint | Archive file(s) |
|---|---|
| Speed-ups, bounds and decisions (sections 3.4, 3.7, 3.8) | tables/sum39_gen/, tables/sum39_a100/, tables/z9/ (sum39 and cmp39d outputs), tables/sum39_gen/grid39.* |
| Phase A (published AlphaFlow/ESMFlow ensembles against ATLAS) | tables/sumA39.* |
| Generator timing and the installation check | tables/gen39b_sum.* (A40), tables/gen39e_sum.* (A100) |
| Figure 1 and Table S2 | tables/speedup39.csv |
| Figure 2 and Table S3 | tables/tgen39.csv |
| Figure 3 | tables/loo39.csv |
| Figure 4 | tables/loc39.csv |
| Table S4 | tables/si39.csv |
| Atom clashes in generated samples | tables/clash39.* |
| Table S1 (protocol timeline) | protocol/prereg39.md |
| Table S5 (software and checksums) | SHA256SUMS.txt, alphaflow_patch/ |

## Reproducing the tables

1. Download splits/atlas_test.csv from the AlphaFlow repository into inputs/l39/ (SHA-256 starting c5029649db0b7ff6).
2. From the archive root run `PY=/path/to/python bash reproduce.sh` (Python 3 with NumPy, SciPy and mdtraj). The script copies inputs/l39 into ./work/l39srv, runs sumA39, gen39sum, sum39, cmp39d, grid39, fig39, fig39c, si39, chi39 and ctrl39 there, and compares each output with tables/ (JSON numerically with tools/jcmp39.py, CSV and text byte by byte).
3. Expected result: all 23 comparisons EQUAL or IDENTICAL. gen39sum is run with --no-rmsd, because the RMSD installation check needs the generated samples; its timing part is compared, and the accounting then uses the archived generator summaries, which contain that check.

When the archive was assembled (11 November 2026), the script ran from a clean copy of the archive on a laptop CPU in about 5 seconds, and all 23 comparisons were EQUAL or IDENTICAL.

The MD simulations and the generator runs themselves need the cluster, the ATLAS data and the AlphaFlow weights; jobs/ shows how they were run.

## Environment

- MD and analyses: conda environment with OpenMM 8.6.1 and Python 3.12 (CUDA platform), with mdtraj, NumPy, SciPy, Biopython and PeptideBuilder from conda-forge and PyPI (November 2026).
- AlphaFlow: github.com/bjing2016/alphaflow at commit 0408d7c8, installed as in jobs/inst39.pbs (Python 3.9, PyTorch 1.12.1+cu113, OpenFold at 103d037), with the changes in alphaflow_patch/.
- Cluster: MetaCentrum (PBS Pro).

## Not included

- ATLAS trajectories and structures (licence CC BY-NC 4.0): https://www.dsimb.inserm.fr/ATLAS (downloaded by atlas39a.pbs).
- AlphaFlow/ESMFlow sample sets and model weights: AlphaFlow repository and its Hugging Face page (downloaded by the job files).
- The test-split list atlas_test.csv: splits/atlas_test.csv of the AlphaFlow repository.
- Templates, generated samples, MSAs, trajectories, coordinates and PCA/TICA bases derived from ATLAS. The tables and inputs contain only derived metrics and timings.
- Cluster job logs, except the S18d50 job logs that si39.py and chi39.py read.

## Notes

- The job files use the storage path of the author's cluster account (variable W near the top), and the same path appears inside many input files. Set W to your own path to run the jobs.
- The S18 seed 0 and seed 1 runs (jobs 25398494 and 25398495) used an earlier revision of b39.pbs that did not yet copy eval39.py into the run; their eval39 outputs were computed afterwards by ana39.pbs. The included b39.pbs (SHA-256 ed4f0729…) is the revision used for K18 (job 25400146).
- gen39.pbs is the first generator-cost job (25405838). It failed on set B and none of its results are used; it is included because it is the input of patch39f.py. All reported generator costs come from gen39b.pbs (25413654) and gen39e.pbs.
- The author's first name in protocol/prereg39.md (five places) and in one comment of jobs/inst39.pbs was corrected on 11 November 2026, because the working log had recorded the author under a different first name. Nothing else in these two files differs from the versions that were recorded and run; jobs/inst39.pbs therefore no longer matches byte for byte the file of job 25405837.
- Labels such as "Gemini 39j" in comments and in the protocol log refer to rounds of an internal methodological review by an AI assistant (see "Use of AI assistants" in the preprint).

## How to cite

Harvalík J. What does a generative ensemble really save? Honest cost accounting of AlphaFlow-seeded molecular dynamics on ATLAS test proteins. Code and derived data, version 1.0. https://github.com/jakubharvalik/alphaflow-md-cost, archived at Zenodo, [DOI to be added]. CITATION.cff holds the same information in machine-readable form.
