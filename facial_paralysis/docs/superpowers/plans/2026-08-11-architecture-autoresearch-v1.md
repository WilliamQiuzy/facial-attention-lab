# Architecture Autoresearch v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete a reproducible, development-only architecture search within twelve hours, preserve the protected PalsyNet outer test, and identify one model family worth a three-seed confirmation run.

**Architecture:** Reuse the authenticated PalsyNet identity and split gate from 110D-Generalization v1. Compare classical summary models and compact temporal landmark models on exactly the same four group-disjoint development folds, first with one screening seed and then with three seeds for the winning family only. MediaPipe extraction is already cached; H200 receives code plus the 3.3 MB deidentified dynamic cache, not raw videos.

**Tech Stack:** Python 3.12, PyTorch 2.7/CUDA 12.8 on the Nebius H200; NumPy and scikit-learn for classical candidates and metrics; pytest-style repository test scripts; JSON reports with SHA-256 provenance.

---

### Task 1: Freeze the development-only research contract

**Files:**
- Create: `docs/superpowers/plans/2026-08-11-architecture-autoresearch-v1.md`
- Create: `tests/test_architecture_search_v1.py`
- Create: `src/models/architecture_search_v1.py`
- Create: `src/training/architecture_search_v1.py`
- Create: `scripts/run_architecture_search_v1.py`

- [ ] Write failing tests requiring the authenticated reviewed-identity gate and exact frozen development folds.
- [ ] Test that protected indices cannot enter feature extraction, scaling, fitting, epoch selection, prediction, or reporting.
- [ ] Test one aligned group-level OOF probability per candidate and threshold `0.5` without candidate-specific threshold tuning.
- [ ] Run `/Users/williamqiu/opt/anaconda3/bin/python3 tests/test_architecture_search_v1.py` and confirm the expected missing-module failure.

### Task 2: Register diverse compact architectures

**Files:**
- Modify: `tests/test_architecture_search_v1.py`
- Modify: `src/models/architecture_search_v1.py`

- [ ] RED-test exact candidate registry, deterministic parameter counts, finite logits, mask handling, and mirror-compatible inputs.
- [ ] Implement summary candidates: frozen 110D Logistic, Extra Trees 110D, HistGradientBoosting 110D, and a small 110D MLP.
- [ ] Implement temporal candidates over the 23 clinical landmark channels: depthwise TCN, BiGRU, tiny Transformer, region-factorized TCN, and 110D-plus-TCN hybrid.
- [ ] Keep every neural candidate under 300,000 trainable parameters and reject architecture drift.
- [ ] Run the focused tests until green.

### Task 3: Implement short-budget, group-disjoint screening

**Files:**
- Modify: `tests/test_architecture_search_v1.py`
- Modify: `src/training/architecture_search_v1.py`

- [ ] RED-test train-fold-only standardization, group-balanced weights, mirror train augmentation, mirror-mean inference, deterministic seeds, and early stopping.
- [ ] Screen all candidates on four fixed development folds with seed `0`, at most 40 epochs, patience `6`, and no broad hyperparameter sweep.
- [ ] Rank by AUROC first, then balanced accuracy, Brier score, and simplicity; exact ties retain the simpler model.
- [ ] Confirm only the winning family receives seeds `0,1,2`; no protected prediction API exists.
- [ ] Emit aggregate candidate metrics, fold dispersion, parameter counts, timing, and provenance without record IDs, group IDs, or per-record probabilities.

### Task 4: Build the authenticated runner and local smoke

**Files:**
- Modify: `tests/test_architecture_search_v1.py`
- Modify: `scripts/run_architecture_search_v1.py`

- [ ] Load the exact 49-cache collection and authenticate the reviewed manifest, review ledger, and frozen person split registry before model work.
- [ ] Add `--smoke` for two epochs and the first four candidates while retaining the same data gate.
- [ ] Run focused tests, the existing 110D gate regression tests, `py_compile`, `git diff --check`, and a local smoke.

### Task 5: Deploy the immutable research release to H200

**Files:**
- Generate remotely: `/home/ssh-ziyue/facial-paralysis-h200/releases/architecture-autoresearch-v1/`

- [ ] Record the local Git commit and SHA-256 inventory for code, cache, identity evidence, and split registry.
- [ ] Transfer only the intended code/evidence/cache allowlist; verify remote hashes before execution.
- [ ] Install only missing pinned runtime dependencies in the dedicated H200 environment; do not install MediaPipe or upload raw video because landmarks are already cached.
- [ ] Run unit smoke and confirm CUDA execution on the H200.

### Task 6: Run screening, confirm the winner, and audit results

**Files:**
- Generate: `outputs/dynamic_landmark/benchmarks/development/architecture-autoresearch-v1/report.json`
- Create: `docs/results/architecture_autoresearch_v1.md`
- Modify only if promotion gates pass: `docs/CURRENT_MODEL.md`

- [ ] Run one-seed screening for all registered candidates on H200.
- [ ] Run three-seed confirmation for the winning family only.
- [ ] Compare against the frozen 110D reference using paired group-level predictions and class-stratified group bootstrap intervals.
- [ ] Reject promotion if the gain is explained by one fold, calibration materially worsens, mirror consistency fails, or the result cannot be reproduced from the immutable release.
- [ ] Keep the protected outer test, MEEI outcomes, and Mayo confidence outcomes unopened during architecture selection.
- [ ] Run a final secret scan, provenance audit, regression suite, and working-tree scope check before committing or pushing.

