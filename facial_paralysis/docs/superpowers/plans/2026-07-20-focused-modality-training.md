# Focused Mayo Modality Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a bounded Fusion smoke, select one Mayo input arm under a fixed fair development protocol, train only that arm with three seeds, and perform strict source/privacy audit only after model evidence exists.

**Architecture:** Add a development-only, bridge-local execution path beside the existing formal HMAC workflow; do not weaken or silently reuse the formal namespace. Every phase uses the existing 95-dimensional Mayo bundle, recording split, train-only scaler, common held-out mask, model, optimizer, and raw-feature metrics. Smoke, selection, and winner training publish into separate owner-only atomic namespaces, while the final audit reauthorizes the live sources and either certifies or rejects the development evidence.

**Tech Stack:** Python 3.9, PyTorch 2.2.1 with fixed CUDA-if-available/CPU-fallback execution, NumPy 1.26.4, existing dynamic-landmark SSL modules.

---

### Task 1: Add a fixed, bridge-local focused-study protocol

**Files:**
- Modify: `scripts/pretrain_dynamic_landmarks.py`
- Modify: `src/pretraining/dynamic_landmark_ssl.py` only if an existing helper cannot express the fixed protocol
- Test: `tests/test_dynamic_landmark_ssl.py`

- [ ] Write RED tests proving the new command has no epoch/seed/arm tuning flags and accepts only `smoke`, `select`, or `winner` phases.
- [ ] Write RED tests proving local input validation authenticates the canonical keyed bridge closure/HMAC with the existing canonical keys, then checks bundle hashes, schema, modes, finite tensors, split disjointness, shared scaler/mask, and never live-authorizes the source trees. Self-reported bundle hashes alone are insufficient.
- [ ] Implement `smoke`: Fusion, seed 0, one epoch, finite train loss, checkpoint reload, no held-out metric, atomic owner-only publication. Record monotonic elapsed time and reject the phase unless it completes within 900 seconds; `select` must cryptographically bind and revalidate this exact successful smoke artifact.
- [ ] Implement `select`: fresh seed-0 training for `blendshape_only`, `landmark_only`, and `fusion`, exactly five epochs each, identical initial state/split/scaler/mask/full-95 target, and one held-out evaluation per arm.
- [ ] Select the unique minimum of `common_target_metrics.trained.raw_mae.equal_block_macro`; reject nonfinite values and bind a deterministic arm-order tie breaker in the report.
- [ ] Publish an aggregate path-free selection report while retaining owner-only selection checkpoints and adjacent receipts for final decision-chain audit; omit recording IDs, raw paths, source names, key material, and patient/session identifiers from all reports.
- [ ] Before execution, run only focused contract tests, `py_compile`, and bridge-local checks; defer the full dynamic SSL/regression suites to Task 3.

### Task 2: Train only the selected arm with three fresh seeds

**Files:**
- Modify: `scripts/pretrain_dynamic_landmarks.py`
- Test: `tests/test_dynamic_landmark_ssl.py`

- [ ] Write RED tests proving `winner` requires and revalidates the exact immutable selection report and refuses a caller-supplied arm.
- [ ] Write RED tests proving seeds are exactly `0,1,2`, epochs exactly `30`, and each seed starts from a fresh deterministic initialization rather than the five-epoch selection checkpoint.
- [ ] Implement three-seed training against the same full-95 target and publish exactly three checkpoints, three adjacent receipts, and one aggregate report atomically.
- [ ] Preserve the primary equal-block raw MAE plus raw block MAE, standardized diagnostics, untrained baseline, and train-mean baseline; label the result as recording-held-out development reconstruction evidence, not HB accuracy or patient-held-out confirmation.
- [ ] Run focused winner-contract tests and `py_compile`; defer broad regression tests to Task 3.

### Task 3: Execute and audit

**Generated files outside Git:**
- `outputs/dynamic_landmark/pretraining/development/focused-modality-v1/smoke/`
- `outputs/dynamic_landmark/pretraining/development/focused-modality-v1/selection/`
- `outputs/dynamic_landmark/pretraining/development/focused-modality-v1/winner/`

- [ ] Run smoke and verify one checkpoint/receipt/report, finite loss, exact modes, clean transaction state, and `elapsed_seconds <= 900`; do not start selection if the bound is exceeded.
- [ ] Run selection and report all three primary validation metrics plus the deterministic winner.
- [ ] Run winner training and report seeds 0/1/2 aggregate mean/SD.
- [ ] Run fresh full tests, `py_compile`, `git diff --check`, tracked-file privacy/size checks, and independent spec/code-quality review. Any review-driven code change invalidates or regenerates every affected smoke, selection, or winner artifact before proceeding.
- [ ] As the literal final certification gate, run strict live source, determinism, privacy, and publication checks over the complete transitive decision chain: authenticated bridge, fixed smoke/config, selection checkpoints/receipts/report and metric derivation, selected-arm commitment, and winner checkpoints/receipts/report. If audit fails, mark the entire smoke-to-selection-to-winner chain uncertified rather than changing results.

## Scientific claim boundary

The arm is selected on a recording-held-out Mayo development split and the same split is reused for the three-seed stability estimate. Therefore the result is model-selection evidence, not an independent test-set comparison. Mayo lacks HB labels and reliable patient grouping, so no HB accuracy, clinical agreement, or patient-generalization claim is permitted.
