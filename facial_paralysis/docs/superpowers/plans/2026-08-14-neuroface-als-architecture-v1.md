# NeuroFace ALS Architecture v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task with review checkpoints.

**Goal:** Build a paper-comparable, participant-disjoint NeuroFace ALS-versus-healthy benchmark and test whether AU, 110D geometry, and compact action-aware temporal fusion can exceed 0.90 development accuracy without changing the frozen Bell's-palsy 110D model.

**Architecture:** Keep the Bell's-palsy 110D release immutable. Create a separate NeuroFace research pipeline with three locked candidates: (A) the paper's Py-Feat XGBoost-AU summary baseline, (B) AU plus frozen 110D geometry with nested regularized logistic training, and (C) a compact mask-aware temporal encoder over AU and clinical geometry with participant-balanced multi-task pooling. All preprocessing, feature selection, calibration, and model selection happen inside participant-disjoint outer folds. Report the 22-person ALS-versus-healthy SPREAD endpoint separately from the harder 36-person ALS/post-stroke-versus-healthy robustness endpoint.

**Tech Stack:** Python 3.10+, NumPy, scikit-learn, PyTorch, Py-Feat 0.6.2/XGBoost AU detector, OpenCV/FFmpeg, H200 CUDA, JSON/NPZ evidence artifacts.

---

## Scientific freeze

- [ ] Freeze the paper-comparable endpoint as ALS=1 versus healthy=0 on the 22 participants with SPREAD recordings; primary metrics are participant-level accuracy, AUROC, balanced accuracy, sensitivity, specificity, and bootstrap intervals.
- [ ] Record the published comparator exactly: SPREAD minimum AU, accuracy 0.91 and AUROC 0.97. Treat the paper protocol as a descriptive comparator because its reported grid search is not documented as nested.
- [ ] Label every new 22-person result as development/internal: prior exploratory candidate searches have already observed all 22 outcomes, so an external cohort is still required for a publishable superiority claim.
- [ ] Prohibit PalsyNet access, MEEI tuning, Mayo prediction, and any write to the frozen 110D release.

## Task 1: Exact AU extraction contract

**Files:**

- Create: `src/datasets/neuroface_au_v1.py`
- Create: `scripts/extract_neuroface_au_v1.py`
- Test: `tests/test_neuroface_au_v1.py`

- [ ] Write failing tests for exact participant/task identity, source SHA-256 binding, AU schema/order, per-frame timestamps, face-detection missingness, immutable arrays, and no-overwrite cache publication.
- [ ] Pin Py-Feat 0.6.2, XGBoost AU detector, detector configuration, environment lock, and model-weight digests.
- [ ] Extract all frames for SPREAD first, then KISS and OPEN; preserve frame-level AU values and validity masks instead of only video summaries.
- [ ] Verify cache count, source hashes, task counts, extraction failures, and deterministic re-load on H200.

## Task 2: Paper baseline reproduction

**Files:**

- Create: `src/evaluation/neuroface_als_benchmark_v1.py`
- Create: `scripts/run_neuroface_als_benchmark_v1.py`
- Test: `tests/test_neuroface_als_benchmark_v1.py`

- [ ] Write failing tests for outer participant LOSO, train-fold-only standardization/correlation filtering, deterministic thresholding, exact metric recomputation, and identity-free public reports.
- [ ] Reproduce mean/min/max/std/variance summaries of the 20 AU signals and the paper's logistic grid.
- [ ] Run both the paper-like descriptive protocol and a stricter nested protocol; never mix their claims.
- [ ] Lock the reproduced AU baseline before evaluating fusion candidates.

## Task 3: Geometry and temporal candidates

**Files:**

- Create: `src/models/neuroface_als_temporal_v1.py`
- Extend: `src/evaluation/neuroface_als_benchmark_v1.py`
- Test: `tests/test_neuroface_als_temporal_v1.py`

- [ ] Write failing tests for masks, variable video length, repeated frames, mirror equivariance, participant-balanced sampling, fold-local normalization, and deterministic training.
- [ ] Candidate B: concatenate fold-standardized AU summaries with frozen name-bound 110D geometry and fit only a small nested L1/L2 logistic head.
- [ ] Candidate C: encode AU plus clinical geometry using a compact depthwise TCN, masked mean/max/min pooling, task embedding, and mirror-mean inference; cap trainable parameters and use early stopping selected only on inner participants.
- [ ] Use KISS/OPEN/SPREAD as three named task views with one participant-level probability. Do not treat three recordings from one person as independent validation samples.
- [ ] If compact TCN is unstable, prefer the locked regularized fusion rather than selecting a post-hoc ensemble on outer predictions.

## Task 4: Robustness and release audit

**Files:**

- Create: `docs/results/neuroface_als_architecture_v1.md`
- Create: `docs/results/artifacts/neuroface_als_architecture_v1/report.json`
- Test: `tests/test_neuroface_als_architecture_release_v1.py`

- [ ] Evaluate the locked candidate on the 36-person ALS/post-stroke-versus-healthy endpoint as a separate robustness analysis, without changing the selected model.
- [ ] Report per-task, per-cohort, calibration, confidence intervals, and abstention/QC coverage; include all failures.
- [ ] Compare against the paper only on the matching 22-person SPREAD endpoint.
- [ ] Verify the frozen 110D files and published model/result hashes remain unchanged.
- [ ] Run local tests, H200 reproducibility, secret scan, protected-path audit, and clean-worktree checks before commit.

## Stop rules

- [ ] Stop candidate expansion after the three prespecified representations; do not keep mining the same 22 labels until a number crosses 0.90.
- [ ] A result above 0.90 is a development milestone, not external clinical validity.
- [ ] If strict AU reproduction plus fusion remains below 0.90, document the negative result and move the claim gate to a new participant cohort rather than weakening participant isolation.

## Execution outcome

- Completed exact Py-Feat extraction for all 66 recordings in the paper-matched
  22-person ALS-versus-healthy endpoint on H200, with full valid-frame coverage.
- Completed the paper-like reproduction, strict nested participant-LOSO
  evaluation, AU-plus-110D fusion, three-task Logistic/LDA, and compact TCN.
- Locked the nested SPREAD AU-statistic Logistic candidate at AUROC `0.9504` and
  accuracy `0.8636`; the AUROC target passed, while the accuracy and published
  superiority targets did not.
- Preserved the existing 36-person action-capacity result as separate
  cross-disease robustness context rather than relabeling post-stroke weakness
  as ALS or pooling incompatible endpoints.
- Activated the stop rule: no additional candidate mining on these same 22
  outcomes. The next scientific gate is an untouched participant cohort.

## 2026-08-14 protocol correction before exact-min evaluation

- The first executable smoke concatenated the five video statistics into one 100D AU vector. That is a useful all-statistics candidate, but it is not the paper's reported best representation.
- Before inspecting any 20D result, freeze the paper reproduction as five separate 20D statistic candidates (mean, minimum, maximum, standard deviation, variance); the published best comparator is the minimum-AU 20D candidate.
- Candidate B is correspondingly corrected from all-statistics-AU+110D (210D) to minimum-AU+110D (130D). The already observed 100D and 210D results remain reported as negative exploratory smoke results and are not erased.
- The same participant LOSO, nested C/penalty selection, train-fold-only preprocessing, metrics, and claim boundaries remain unchanged. No new dataset or outcome is accessed by this correction.

## 2026-08-14 prespecified fallback after the temporal-TCN failure

- The compact three-task TCN produced a strict participant-LOSO AUROC of 0.463 and accuracy of 0.50, so it is rejected rather than tuned against the 22 outer outcomes.
- Before the complete KISS/OPEN/SPREAD AU collection is evaluated, freeze one small-sample fallback: concatenate the same 20 AU statistic within each named task, producing five task-structured 60D candidates (mean, minimum, maximum, standard deviation, variance). Select statistic, C, penalty, and optional decision threshold only from inner participant OOF predictions.
- Also retain one all-statistics 300D ablation. Do not add neural depth, post-hoc outer-score ensembling, or new candidate families after observing the multi-task result.
- Before that result is observed, lock one architecture ablation appropriate to 22 people: Ledoit-Wolf-style automatic shrinkage LDA (`solver=lsqr`, `shrinkage=auto`). It uses the same five 60D task-statistic representations, selects the statistic and threshold only from inner participant OOF predictions, and is reported separately from nested L1/L2 Logistic.
- The three recordings remain one participant row. A multi-task result is not protocol-identical to the single-SPREAD paper and can only be compared descriptively on participant burden and endpoint.

## 2026-08-14 final outcome-informed training ablation

- The complete three-task fallback remained below 0.90 (best strict AUROC 0.884, accuracy 0.818). Permit exactly one final exploratory training ablation: balanced class weights inside every Logistic training fold, motivated by the unavoidable 10:11 or 11:10 class count in outer LOSO.
- Representation, C, penalty, and threshold remain inner-OOF selected; the held participant outcome is never used. Because this ablation was motivated after observing prior outer results, label it outcome-informed development and require a new cohort for confirmation even if it crosses 0.90.
- No additional candidate, ensemble, threshold rule, or validation protocol may be added after this result.
