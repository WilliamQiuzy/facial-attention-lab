# Scale-Robust Eye Geometry v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether deterministic, gap-safe temporal median filtering improves the frozen 110D landmark representation for small-face recordings while quantifying how little of the eight-action Mayo protocol the current four-window sampler observes.

**Architecture:** Compare exactly three representations with the same fixed standardized L2 Logistic classifier: current raw 110D, eye-channel median-3 110D, and all-clinical-landmark median-3 110D. Select only on authenticated, identity-reviewed PalsyNet development folds with a preregistered overall and low-face-scale gate; apply the locked winner once to the Mayo assumed-positive cohort. Produce only aggregate Mayo diagnostics and never infer action labels from unlabeled frames.

**Tech Stack:** Python 3.9, NumPy, scikit-learn, existing `clinical23_v2` caches, authenticated PalsyNet split registry, script-style tests, and the verified H200 runtime.

---

### Task 1: Freeze scale-robust representation contracts

**Files:**
- Create: `facial_paralysis/src/preprocessing/scale_robust_geometry_v1.py`
- Create: `facial_paralysis/tests/test_scale_robust_geometry_v1.py`

- [x] **Step 1: Write failing tests** for the exact ordered registry `raw_110d`, `eye_median3_110d`, `all_landmark_median3_110d`.
- [x] **Step 2: Verify RED** because the new module is absent.
- [x] **Step 3: Implement minimal window-local median-3 filtering** that changes only a complete valid three-frame neighborhood, never crosses a window/detector gap, preserves invalid canonical zeros, leaves blendshapes unchanged, and commutes with horizontal mirroring.
- [x] **Step 4: Verify GREEN** for identity on constant/linear paths, suppression of a planted one-frame spike, gap isolation, candidate dimensions, and mirror commutation.

### Task 2: Freeze selection and action-coverage audit

**Files:**
- Create: `facial_paralysis/src/evaluation/scale_robust_geometry_v1.py`
- Modify: `facial_paralysis/tests/test_scale_robust_geometry_v1.py`

- [x] **Step 1: Write failing tests** for deterministic label-stratified low-face-scale group selection, exact candidate metric fields, tie retention, fail-closed promotion, and identifier-free action-coverage summaries.
- [x] **Step 2: Verify RED** for missing evaluation behavior.
- [x] **Step 3: Implement the fixed gate:** overall AUROC and balanced accuracy cannot decrease, Brier cannot increase by more than 0.01, low-scale AUROC cannot decrease, and low-scale balanced accuracy or Brier must improve; ties retain raw 110D.
- [x] **Step 4: Implement aggregate four-window coverage:** sampled-frame fraction and start-to-start gap seconds only. State explicitly that four windows are not eight action segments and action accuracy is undefined.
- [x] **Step 5: Verify GREEN** and reject any report containing record/group/source IDs, filenames, or paths.

### Task 3: Run authenticated PalsyNet selection and post-lock Mayo challenge

**Files:**
- Create: `facial_paralysis/scripts/run_scale_robust_geometry_v1.py`
- Modify: `facial_paralysis/tests/test_scale_robust_geometry_v1.py`
- Create: `facial_paralysis/outputs/dynamic_landmark/benchmarks/external/scale-robust-eye-geometry-v1/report.json`

- [x] **Step 1: Write failing runner tests** for no-overwrite output, exact input schemas, four group-disjoint OOF folds, and zero protected cache reads/fits/predictions.
- [x] **Step 2: Verify RED**, then implement the minimal runner by reusing the authenticated development gate and frozen Logistic fitter.
- [x] **Step 3: Derive all three representations only for registered development rows**, fit/predict every OOF row once, and select without reading Mayo.
- [x] **Step 4: Refit the locked winner on all development rows**, then score the existing private Mayo cache once and emit only aggregate positive-call/confidence and sampling-coverage results.
- [x] **Step 5: Verify the report is deterministic, identifier-free, one-class-limited, and records zero protected access.**

### Task 4: Verify, document, and publish

**Files:**
- Create: `facial_paralysis/docs/results/scale_robust_eye_geometry_v1.md`
- Modify only if promotion succeeds: `facial_paralysis/docs/CURRENT_MODEL.md`
- Modify only if promotion succeeds: `facial_paralysis/docs/results/current_development_model.json`

- [x] **Step 1: Run focused and regression tests locally**, compile checks, report recomputation, diff checks, and sensitive-data scans.
- [x] **Step 2: Run the focused tests and real PalsyNet-only comparison on H200** without uploading Mayo videos, private scores, or identity images.
- [x] **Step 3: Document the exact four-window formulas**, observed Mayo frame-use fraction, candidate results, limitations, and the next independent Action-aligned Goal.
- [ ] **Step 4: Commit and push** the exact intended files to `codex/110d-generalization-v1`.
