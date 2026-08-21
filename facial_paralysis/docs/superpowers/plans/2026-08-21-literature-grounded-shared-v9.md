# Literature-Grounded Shared V9 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate a small frozen set of paper-supported shared-encoder changes that can improve specificity, AUROC, and accuracy across PalsyNet development, NeuroFace, and MEEI without increasing endpoint-specific capacity.

**Architecture:** Keep the deterministic V8 full-mesh plus 110D shared trunk and its small script heads as the comparator. Add a source-blind bilateral anatomical relation residual and fold-local clinical-kinematic auxiliary targets; change shared optimization only when measured diagnostics authorize CAGrad or GradNorm. Evaluate every authorized candidate with the same participant-disjoint folds and add a universal-head leave-one-source-out stress test.

**Tech Stack:** Python 3.10, NumPy, PyTorch 2.7/CUDA, SciPy constrained optimization for CAGrad, repository direct-test harness, NVIDIA H200.

---

### Task 1: Freeze diagnostics and candidate authorization

**Files:**
- Create: `src/evaluation/literature_grounded_diagnostics_v9.py`
- Create: `tests/test_literature_grounded_diagnostics_v9.py`

- [x] Write RED tests for exact source losses, gradient norms, pairwise cosine,
  relative loss descent, and immutable aggregate-only output.
- [x] Run the direct test and verify it fails because the diagnostic module is
  absent.
- [x] Implement a deterministic fold-train-only audit over exact `RSR8-001`.
- [x] Run the direct test and relevant V8 regressions.
- [x] On H200, run the audit once; authorize GradNorm only for norm ratio at
  least `2.0` or relative-loss-descent ratio at least `2.0`, and authorize
  CAGrad only when a negative shared-gradient pair is reproduced.
- [x] Freeze the diagnostic observation time at five ERM epochs (one quarter
  of the unchanged 20-epoch V8 schedule) before reading any diagnostic value.
- [ ] Commit the diagnostic and frozen authorization record.  The frozen result
  authorizes neither GradNorm nor CAGrad, so Task 4 is skipped rather than
  counted as an experiment.

### Task 2: Add the bilateral anatomical relation residual

**Files:**
- Create: `src/models/anatomical_relational_router_v9.py`
- Create: `tests/test_anatomical_relational_router_v9.py`

- [ ] Write RED tests that bind exact 110D feature names to bilateral
  eye/brow/oral/global regions, reject reordered schemas, and prove the same
  relation weights serve all three sources.
- [ ] Verify RED.
- [ ] Implement region-local projections, one small shared relation block, and
  a zero-initialized local-global residual gate on top of V8 action tokens.
- [ ] Prove the comparator is numerically exact when the gate is zero and that
  endpoint-specific parameters remain below ten percent.
- [ ] Run the new and V8 regression tests.
- [ ] Commit.

### Task 3: Add clinical-kinematic auxiliary supervision

**Files:**
- Create: `src/training/clinical_kinematic_auxiliary_v9.py`
- Create: `tests/test_clinical_kinematic_auxiliary_v9.py`

- [ ] Write RED tests for exact label-free targets: regional excursion,
  regional velocity, bilateral correlation, valid-action masking, and no use
  of disease labels or held rows.
- [ ] Verify RED.
- [ ] Implement a small shared auxiliary head attached to action tokens and a
  robust standardized regression loss.  Fit target scaling on each training
  fold only and discard the head at inference.
- [ ] Run unit and leakage regressions.
- [ ] Commit.

### Task 4: Implement only authorized multi-task optimizers

**Files:**
- Create: `src/training/evidence_authorized_multitask_v9.py`
- Create: `tests/test_evidence_authorized_multitask_v9.py`

- [ ] Write RED tests for authorization rejection, exact shared-parameter
  scope, CAGrad worst-local-improvement construction, and GradNorm weight
  normalization/training-rate targets.
- [ ] Verify RED.
- [ ] Implement CAGrad and, only if Task 1 authorizes it, GradNorm.  Task heads
  retain their ordinary source gradients; only genuinely shared parameters
  receive the multi-task update.
- [ ] Compare numerical results with hand-calculated two-task examples.
- [ ] Run all optimizer and historical PCGrad regressions.
- [ ] Commit.

### Task 5: Freeze participant-disjoint evaluation and LOSO stress test

**Files:**
- Create: `src/evaluation/literature_grounded_shared_search_v9.py`
- Create: `tests/test_literature_grounded_shared_search_v9.py`
- Create: `scripts/run_literature_grounded_shared_search_v9.py`
- Create: `tests/test_run_literature_grounded_shared_search_v9.py`

- [ ] Write RED tests for the exact authorized registry, six participant-
  disjoint folds, deterministic seeds, fold-local scaling, per-source metrics,
  universal-head leave-one-source-out evaluation, and zero protected/Mayo
  reads.
- [ ] Verify RED.
- [ ] Implement comparator, each individually justified mechanism, and only
  non-degrading combinations.  Do not add arbitrary widths, dropout values,
  thresholds, or epoch counts.
- [ ] Run smoke locally, then the complete frozen screen on H200.
- [ ] Confirm the top candidate across seeds `0,1,2`; reject any candidate that
  loses more than `0.01` AUROC or accuracy on a source, has sensitivity below
  `0.85`, or fails to improve worst-source specificity.
- [ ] Commit.

### Task 6: Release the decision, not just the best run

**Files:**
- Create: `docs/results/literature_grounded_shared_v9.md`
- Create: `docs/results/artifacts/literature_grounded_shared_v9/report.json`
- Create: `tests/test_literature_grounded_shared_v9_release.py`
- Modify: `docs/model_registry.json` only if every promotion gate passes.
- Modify: `docs/CURRENT_DEPLOYMENT_MODEL.md` only if every promotion gate passes.

- [ ] Write RED release tests for exact artifact hashes, aggregate-only output,
  no patient/source paths, no Mayo claim, and byte-identical V8 registry files
  when no candidate passes.
- [ ] Verify RED.
- [ ] Publish the complete candidate table, authorization diagnostics,
  deterministic multi-seed metrics, LOSO stress test, and explicit decision.
- [ ] Run all relevant direct tests, `py_compile`, secret/path scan, and
  `git diff --check`.
- [ ] Commit the evidence release.
