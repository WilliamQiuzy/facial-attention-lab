# 110D Outer Release v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the already-authorized PalsyNet protected outer evaluation exactly once for the locked mirror-invariant 110D candidate, then freeze one inference-ready PalsyNet artifact for later unchanged external validation.

**Architecture:** A release module validates the reviewed identity, frozen person split, locked development report, exact implementation digests, and one-shot authorization before any protected cache is loaded. A thin outer CLI fits the fixed mirror-invariant L2 Logistic model on development groups and writes one aggregate protected report; a separate thin freezer CLI may run only after that sealed report exists and fits the same model once on every eligible reviewed PalsyNet group. Both outputs are fixed-path, atomic, no-overwrite, identifier-free JSON artifacts.

**Tech Stack:** Python 3.9.12 from `/Users/williamqiu/opt/anaconda3/bin/python3`, NumPy, scikit-learn, SHA-256, JSON, fixed group-weighted L2 Logistic Regression, H200 release execution. Bare `python3` is forbidden for experiment execution.

---

## Frozen scientific contract

- Candidate: `landmark_mi_110d`, exactly 110 dimensions and byte-identical to the locked development extractor.
- Training: original plus horizontal-mirror rows; each reviewed group has equal total weight; train-only `StandardScaler`; L2 Logistic Regression with `C=0.01`, `liblinear`, `max_iter=2000`, random state `0`.
- Inference: mean of original and horizontal-mirror positive-class probabilities; threshold `0.5`.
- Outer fit: all and only frozen development groups. Outer scoring: all and only frozen protected groups, averaged once per reviewed group.
- Report: group-level AUROC primary; average precision, Brier, balanced accuracy, sensitivity, and specificity secondary; fixed 5,000 affected/unaffected-stratified group bootstrap draws at seed `20260805` for descriptive 95% intervals.
- No candidate, threshold, feature, calibration, model, split, seed, or output override is exposed.
- The protected result cannot change the candidate. After it is sealed, the final artifact is trained once on all eligible reviewed PalsyNet groups for unchanged MEEI inference.

## Task 1: Specify and test the authorization gate

**Files:**
- Create: `tests/test_110d_outer_release_v1.py`
- Create: `src/evaluation/outer_release_110d_v1.py`
- Create after code freeze: `docs/registries/110d-generalization-v1-outer-authorization.json`

- [x] Write tests proving missing, malformed, stale, self-consistent-forged, legacy-video, alternate-candidate, alternate-split, alternate-source, alternate-implementation, and alternate-output authorizations fail before protected cache access, fitting, or prediction.
- [x] Run `KMP_DUPLICATE_LIB_OK=TRUE /Users/williamqiu/opt/anaconda3/bin/python3 tests/test_110d_outer_release_v1.py` and require the expected missing-module failure.
- [x] Implement a closed authorization schema bound to the locked development-report SHA-256, reviewed identity SHA-256, review-ledger SHA-256, person-split SHA-256, source-collection SHA-256, release-implementation aggregate SHA-256, candidate, model protocol, fixed protected output path, and `authorized_once=true`.
- [x] Re-run the focused test and require the authorization tests to pass without touching real protected arrays.

## Task 2: Implement the protected one-shot evaluator

**Files:**
- Modify: `tests/test_110d_outer_release_v1.py`
- Modify: `src/evaluation/outer_release_110d_v1.py`
- Create: `scripts/run_110d_outer_release_v1.py`

- [x] Add failing synthetic tests for exact development/protected isolation, group-balanced mirror training, symmetric inference, one probability per protected group, metric recomputation, fixed bootstrap, identifier-free report schema, atomic no-overwrite output, and zero tuning CLI.
- [x] Run the focused test and confirm failures are caused by the absent evaluator/CLI behavior.
- [x] Implement the minimal evaluator and CLI by reusing the authenticated metadata gate, locked 110D feature extractor, mirror transform, group weights, and fixed model constants from the development implementation.
- [x] Independently recompute every report metric/count from in-memory protected OOF results before serialization; reject NaN, identifiers, probabilities, filenames, paths, and malformed leaves.
- [x] Run the focused tests plus `test_110d_generalization_features.py`, `test_110d_generalization_v1.py`, and `test_mirror_invariant_110d.py`; require zero failures.

## Task 3: Implement the final inference artifact freezer

**Files:**
- Create: `tests/test_freeze_110d_generalization_v1_artifact.py`
- Modify: `src/evaluation/outer_release_110d_v1.py`
- Create: `scripts/freeze_110d_generalization_v1_artifact.py`

- [x] Add failing tests that reject an absent/unsealed/mismatched protected report, unlocked candidate, altered cache/identity/split/development/outer/implementation digest, incomplete or duplicate eligible groups, model-setting drift, non-finite parameters, and overwrite.
- [x] Add a known-answer test proving serialized scaler statistics, Logistic coefficients/intercept, mirror inference, and threshold reproduce the fitted sklearn pipeline probability to numerical tolerance.
- [x] Run the focused test and confirm the missing freezer behavior fails.
- [x] Implement one fit on all eligible reviewed groups with the frozen protocol and emit only feature names, scaler statistics, coefficients/intercept, threshold, counts, protocol, and authenticated provenance; include no identifiers or paths.
- [x] Run both new test files and all existing 110D release regression tests; require zero failures.

## Task 4: Freeze the H200 release and execute once

**Files:**
- Generate: `outputs/dynamic_landmark/benchmarks/protected/110d-generalization-v1/report.json`
- Generate: `outputs/dynamic_landmark/artifacts/110d-generalization-v1/final_palsynet_artifact.json`

- [x] Freeze the release code and authorization digests locally; run `py_compile`, `git diff --check`, closed-schema/secret scans, and the complete relevant test suite before any protected access.
- [x] Sync only the exact commit, 49-record authenticated cache, reviewed identity evidence, frozen split registry, locked development report, and authorization to a new immutable H200 release directory.
- [x] On H200, verify GPU/runtime availability, repository/cache/evidence digests, authorization, and focused tests without loading protected NPZ arrays.
- [x] Invoke `run_110d_outer_release_v1.py` exactly once. Require an exit-zero, no-overwrite aggregate protected report and independently verify its digest/schema/counts.
- [x] Invoke `freeze_110d_generalization_v1_artifact.py` exactly once and independently reproduce its probabilities from serialized scaler/model parameters on a synthetic known-answer input.
- [x] Copy only aggregate report/artifact evidence back to the canonical local worktree; raw videos and per-record predictions never leave their governed locations.

## Task 5: Publish the release evidence

**Files:**
- Modify: `docs/CURRENT_MODEL.md`
- Modify: `docs/results/current_development_model.json`
- Create: `docs/results/110d_outer_release_v1.md`

- [x] Report the protected metric separately from development AUROC `0.980`; do not retune or relabel the protected result as Mayo/HB/clinical validation.
- [x] Record the exact release commit, host release path, input/implementation/report/artifact SHA-256 values, counts, and test results.
- [x] Run fresh focused and full relevant verification, `py_compile`, `git diff --check`, secret scan, and worktree-scope audit.
- [ ] Commit only intended code/tests/docs/aggregate artifacts and push `codex/110d-generalization-v1`.
