# Source-Robust Landmark 110D v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether removing source-sensitive static facial morphology from the frozen Landmark 110D representation improves acquisition-shift stability without sacrificing PalsyNet person-disjoint discrimination.

**Architecture:** Freeze exactly three representations derived from the same four-window `clinical23_v2` cache: the current 110D, an 87D within-video dynamics representation that drops all 23 channel medians, and a 93D representation that restores only the six direction-free clinical asymmetry medians. Use the same fixed mirror-trained L2 Logistic for every arm. Select only on reviewed PalsyNet development groups using both the registered four folds and a deterministic acquisition-PC1-blocked four-fold stress split; MEEI, Mayo, and protected PalsyNet remain unopened.

**Tech Stack:** Python 3.10, NumPy, scikit-learn, existing PalsyNet identity gate, `clinical23_v2`, and fixed mirror-invariant Logistic evaluation utilities.

---

### Task 1: Frozen source-robust feature views

**Files:**
- Create: `src/preprocessing/source_robust_110d.py`
- Test: `tests/test_source_robust_110d.py`

- [x] **Step 1: Write RED tests**

Assert the exact candidate order and dimensions `(110, 87, 93)`, exact feature-name order, finite `(N,D)` outputs, 87D removal of every channel median, 93D restoration of only `fissure_h_absdiff`, `fissure_w_absdiff`, `eye_area_absdiff`, `brow_h_absdiff`, `corner_y_absdiff`, and `commissure_x_absdiff` medians, and exact compatibility with the frozen mirror transform.

- [x] **Step 2: Run the new test and confirm failure**

Run: `/Users/williamqiu/opt/anaconda3/bin/python3 tests/test_source_robust_110d.py`

Expected: import failure because `source_robust_110d.py` does not exist.

- [x] **Step 3: Implement minimum feature slicing**

Derive all candidates from the existing 110D vector and registered 110D feature names. Do not recompute landmarks, use labels, fit a transform, add a hyperparameter, or touch raw videos.

- [x] **Step 4: Run focused and frozen-byte regressions**

Run the new test plus `tests/test_trajectory_features.py`, `tests/test_110d_generalization_features.py`, and `tests/test_mirror_invariant_110d.py`.

### Task 2: PalsyNet-only source-robust evaluation

**Files:**
- Create: `src/evaluation/source_robust_110d_v1.py`
- Create: `scripts/run_source_robust_110d_v1.py`
- Test: `tests/test_source_robust_110d_v1.py`

- [x] **Step 1: Write RED tests for the closed protocol**

Freeze the three candidates, `C=0.01`, threshold `0.5`, original-plus-mirror training, mean original/mirror inference, group-balanced sample weights, four registered identity-disjoint folds, and zero protected cache reads/fits/predictions.

- [x] **Step 2: Implement deterministic acquisition-blocked folds**

Use only seven nonclinical acquisition fields: bitrate proxy, detection rate, luminance, face-scale mean/std, and eye-line-roll mean/std. Aggregate duplicate recordings at the reviewed-group level, standardize using development groups only, compute PCA with deterministic sign canonicalization, sort groups within each binary label by PC1, and divide each label into four contiguous blocks. Reject any group/label/fold overlap or missing class.

- [x] **Step 3: Implement the locked promotion gate**

A reduced candidate may advance only if registered-fold AUROC is no more than `0.02` below baseline, registered-fold balanced accuracy is no more than one affected-group half-error (`1/(2*21)`) below baseline, registered specificity remains `1.0`, and both AUROC and balanced accuracy strictly improve over baseline on acquisition-blocked folds. Ties retain the current 110D; candidate order breaks no tie.

- [x] **Step 4: Generate one aggregate report**

Run against the authenticated 39-recording/38-group PalsyNet development partition. Record both split families, paired bootstrap intervals, implementation hashes, and exact protected-use counters. Export no IDs, paths, labels, nuisance rows, fold assignments, or row probabilities.

### Task 3: Fast H200 reproduction and handoff

**Files:**
- Create: `docs/results/source_robust_110d_v1.md`
- Create: `outputs/dynamic_landmark/benchmarks/development/source-robust-110d-v1/report.json`

- [x] **Step 1: Reproduce the final report on H200**

Upload only implementation files and the already authenticated PalsyNet development caches to an immutable release directory. Require identical decisions and metric agreement within `1e-14`; do not upload Mayo or MEEI data.

- [x] **Step 2: Record the scientific decision**

If no candidate passes, retain the frozen 110D and document the negative result. If one passes, mark it only as a PalsyNet development successor awaiting a new untouched external cohort; do not rescore MEEI or call Mayo confidence accuracy.

- [x] **Step 3: Final verification and commit**

Run all focused/regression tests, `py_compile`, report-contract recomputation, `git diff --check`, privacy/secret scans, and exact staged-diff review. Commit code, tests, plan, and aggregate-safe reports only.
