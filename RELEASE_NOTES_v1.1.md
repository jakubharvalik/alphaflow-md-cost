# Version 1.1 — corrected re-analysis (Z10)

- Adds the corrected re-analysis Z10 after an external review (11 November 2026); its rules and script checksums were recorded in protocol/prereg39.md before it was run.
- Fixes, in the re-analysis: the held-out replica is no longer part of its own reference (three folds, reference = the two other replicas); equal numbers of frames at every time point; PCA/TICA basis and histogram edges fitted on the reference replicas only, with a uniform 100 ps burn-in; GPU time of failed boxes included in the cost.
- New scripts: sub40.py, patch40.py → sum40.py, fail40.py, cmp40.py, diag40.py (post hoc diagnostic) and the job file ana40.pbs.
- New inputs (derived metrics only): inputs/l39/ana40/ and inputs/l39/fail40.json. New outputs: tables/z10/ (all reported variants, per-protein comparison, Figure 1 data) and tables/diag40/.
- reproduce.sh now also recomputes Z10 and compares 51 outputs with the archived tables (all EQUAL or IDENTICAL at release).
- protocol/: Z10 entries appended to the Czech log and its English translation. All version 1.0 files are otherwise unchanged; the original pre-specified results remain for comparison.
