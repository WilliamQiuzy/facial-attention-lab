# Mayo Failure Analysis + Robust Inference v1 Implementation Plan

> **Scope:** Explain the two below-threshold records in the local Mayo assumed-positive challenge and test one robust inference change without using Mayo for model or threshold selection.

**Goal:** Produce a privacy-preserving failure analysis, lock any candidate solely from identity-reviewed PalsyNet development evidence, and perform one post-lock Mayo challenge comparison.

**Architecture:** Refit the already frozen mirror-invariant 110D Logistic model on authenticated PalsyNet development records. Decompose its Mayo predictions into geometry-region contributions and compare extraction/nuisance characteristics. In parallel, evaluate a preregistered family of deterministic mirror-view aggregators entirely on the existing PalsyNet patient/group-disjoint folds. Promote only if the PalsyNet acceptance rule is met, then apply the locked aggregator once to the Mayo assumed-positive cohort.

**Safety boundary:** Protected PalsyNet outer records remain unopened. Mayo is an all-positive-assumption challenge, so its output is positive-call consistency and confidence—not accuracy, AUROC, specificity, or model selection. Public artifacts contain aggregates only; per-record Mayo identifiers and scores remain private and untracked.

---

## Task 1: Freeze the diagnostic and aggregation contracts

**Files:**
- Create: `facial_paralysis/src/evaluation/mayo_failure_analysis_v1.py`
- Create: `facial_paralysis/tests/test_mayo_failure_analysis_v1.py`

1. Write failing tests for strict aligned input validation, region assignment for all 110 feature names, identifier-free aggregate output, and deterministic aggregation candidates.
2. Preregister exactly three score aggregators before inspecting the two failures: `mirror_mean` (current), `mirror_logit_mean`, and `mirror_conservative_max`.
3. Define the PalsyNet-only promotion rule: candidate must not reduce pooled out-of-fold AUROC or balanced accuracy versus `mirror_mean`, must not worsen Brier score by more than 0.01, and must improve at least one of balanced accuracy or Brier score. Ties retain the current model.
4. Implement only enough code to pass the contract tests.

## Task 2: Build the fail-closed private runner

**Files:**
- Create: `facial_paralysis/scripts/run_mayo_failure_analysis_v1.py`
- Modify: `facial_paralysis/tests/test_mayo_failure_analysis_v1.py`

1. Write failing tests proving that the runner authenticates the reviewed PalsyNet development gate, never loads protected cache records, rejects Mayo cache schema drift, excludes identifiers from the public report, and refuses overwrite.
2. Reuse the frozen PalsyNet fold registry and existing 110D feature vectors; do not create new random folds or tune Logistic hyperparameters.
3. Fit each fold on original+mirrored development views, score both held-out views, and compare only the three preregistered aggregators.
4. Refit the frozen champion on all development records, compute private Mayo per-record evidence in memory, and emit only aggregate failure cohorts, nuisance effect sizes, region contribution summaries, and the one post-lock Mayo comparison.

## Task 3: Run, audit, and document

**Files:**
- Create: `facial_paralysis/docs/results/mayo_failure_analysis_robust_inference_v1.md`
- Modify only if promotion succeeds: `facial_paralysis/docs/CURRENT_MODEL.md`
- Modify only if promotion succeeds: `facial_paralysis/docs/results/current_development_model.json`

1. Run the targeted test suite locally until green.
2. Run the authenticated PalsyNet development comparison and private Mayo analysis locally; write the aggregate report with mode `0600` and no overwrite.
3. Verify protected-record load count is zero, public artifacts contain no recording/group/source identifiers or raw paths, and the Mayo claim remains explicitly one-class.
4. Run the same lightweight evaluation/test code on H200 to confirm the GPU host is reachable and the release environment remains usable; do not upload Mayo videos or private per-record artifacts.
5. Document the causal diagnosis, the PalsyNet-only selection result, Mayo aggregate behavior after lock, limitations, and next decision.
6. Run focused regression tests, compile checks, diff/secret scan, and Git status before commit/push.
