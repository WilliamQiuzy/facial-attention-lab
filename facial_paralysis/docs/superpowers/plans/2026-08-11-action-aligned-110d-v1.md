# Action-Aligned 110D v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compare the frozen four-time-window Landmark 110D model with the same 110D/logistic model built from seven deterministic motion-aligned windows, using only identity-reviewed PalsyNet development folds for selection.

**Architecture:** A label-free MediaPipe proposal pass samples each video at 6 Hz and locates one brow, two eye, two smile, one pucker, and one lower-face activation peak. Seven 32-frame windows at the audited 30 Hz PalsyNet rate are extracted around those peaks, pooled by the existing 110D landmark statistics, and evaluated with the exact frozen mirror augmentation, L2 logistic model, threshold, and group-disjoint folds. Mayo remains an assumed-positive post-lock challenge and cannot select the representation.

**Tech Stack:** Python 3.10, OpenCV, MediaPipe Face Landmarker, NumPy, scikit-learn, existing identity gate and 110D evaluation utilities.

---

### Task 1: Deterministic action-window representation

**Files:**
- Create: `src/preprocessing/action_aligned_110d.py`
- Modify: `src/preprocessing/trajectory_features.py`
- Test: `tests/test_action_aligned_110d.py`

- [x] **Step 1: Write the failing tests**

Test the exact seven-slot registry, known synthetic peaks, within-signature non-maximum suppression, edge clamping, malformed-input rejection, 110D output dimension, and preservation of the frozen four-window 110D output.

- [x] **Step 2: Run test to verify RED**

Run: `/Users/williamqiu/opt/anaconda3/bin/python3 tests/test_action_aligned_110d.py`

Expected: FAIL because `src.preprocessing.action_aligned_110d` does not exist.

- [x] **Step 3: Implement the minimum representation**

Define fixed action signatures from registered MediaPipe blendshape columns, choose peaks without labels or classifier scores, and pool the resulting `(7, 32, 95)` tensor through the existing Landmark 110D statistics. Generalize only the internal trajectory-array validator from four windows to a positive number of 32-frame windows; retain exact four-window outputs.

- [x] **Step 4: Run focused and regression tests**

Run the new test plus `tests/test_trajectory_features.py`, `tests/test_110d_generalization_features.py`, and `tests/test_mirror_invariant_110d.py` directly. Expected: all pass.

### Task 2: Identity-bound PalsyNet action cache

**Files:**
- Create: `scripts/build_palsynet_action_aligned_v1.py`
- Test: `tests/test_build_palsynet_action_aligned_v1.py`

- [x] **Step 1: Write RED tests**

Test that source identity is joined only by SHA-256, proposal scanning never reads labels or model probabilities, seven complete windows are emitted, model/feature/source provenance is pinned, output is transactional and private, and malformed or incomplete generations fail closed.

- [x] **Step 2: Implement the builder**

Reuse the reviewed identity manifest and existing strict PalsyNet source enumeration. Scan at every fifth 30 Hz frame, extract the seven selected 32-frame windows with clinical23_v2, and write an ignored local cache under `data/external/palsynet/derived/action_aligned_clinical23_v1` without paths or filenames in the manifest.

- [x] **Step 3: Run builder tests and one-video smoke**

Expected: deterministic window starts and byte-stable feature arrays for a repeated smoke source.

### Task 3: Locked PalsyNet development comparison

**Files:**
- Create: `src/evaluation/action_aligned_110d_v1.py`
- Create: `scripts/run_action_aligned_110d_v1.py`
- Test: `tests/test_action_aligned_110d_v1.py`

- [x] **Step 1: Write RED tests**

Freeze exactly two candidates: `four_time_window_110d` and `seven_action_window_110d`. Require the same fixed `C=0.01` mirror-trained logistic model, the reviewed four group-disjoint development folds, one OOF prediction per development recording, and zero protected feature reads/fits/predictions.

- [x] **Step 2: Implement and run the comparison**

Select action-aligned 110D only if AUROC and balanced accuracy are non-inferior to baseline and at least one of balanced accuracy or Brier improves. Do not expose a model, threshold, fold, or candidate tuning surface through the CLI.

- [x] **Step 3: Verify the real report**

Generate an aggregate no-overwrite report, independently recompute metrics, verify group disjointness, check zero protected access, and scan for identifiers, paths, labels, and per-record probabilities.

### Task 4: Post-lock Mayo challenge and handoff

**Files:**
- Create: `scripts/build_mayo_action_aligned_v1.py`
- Create: `scripts/run_mayo_action_aligned_challenge_v1.py`
- Test: `tests/test_mayo_action_aligned_challenge_v1.py`
- Create: `docs/results/action_aligned_110d_v1.md`

- [x] **Step 1: Gate Mayo execution on the PalsyNet decision**

If baseline remains selected, stop without generating new Mayo predictions. If action-aligned 110D is selected, build a content-deduplicated action cache and perform one assumed-positive aggregate challenge.

- [x] **Step 2: Report only valid Mayo quantities**

Report record count, positive-call rate, confidence distribution, action coverage, and low-confidence count. Explicitly set `accuracy_defined=false`, `mayo_used_for_model_selection=false`, and `clinical_validation=false`.

- [x] **Step 3: Final verification and commit**

Run all focused tests, compile changed Python files, `git diff --check`, secret/path scan, and exact diff review. Commit only code, tests, plans, and aggregate-safe documentation; keep raw media, caches, identifiers, and row-level predictions local and ignored.
