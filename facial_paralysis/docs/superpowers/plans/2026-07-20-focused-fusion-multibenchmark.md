# Focused Fusion Multi-Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a reproducible, fail-closed multi-benchmark evaluator for the frozen three-seed focused Mayo Fusion checkpoints.

**Architecture:** Keep all perturbation and aggregation logic in a new evaluation module so the three trainer-digest-bound files remain byte-identical. A thin CLI will authenticate the existing focused evidence chain, load tensors and checkpoints through current validators, call the pure evaluator, and atomically write one deidentified development report outside the focused evidence namespace.

**Tech Stack:** Python 3.9.12, PyTorch, NumPy, existing custom `Check` test harness, canonical JSON.

---

### Task 1: Freeze pure benchmark protocols

**Files:**
- Create: `tests/test_focused_fusion_robustness.py`
- Create: `src/evaluation/focused_fusion_robustness.py`

- [ ] **Step 1: Write failing tests for the exact protocol registry**

Assert exact condition names, modality arms, dropout/noise seeds, probabilities, and no caller-defined arbitrary conditions.

- [ ] **Step 2: Run the test and verify RED**

Run: `python3 tests/test_focused_fusion_robustness.py`

Expected: FAIL because `src.evaluation.focused_fusion_robustness` does not exist.

- [ ] **Step 3: Implement the minimal immutable registry**

Define a frozen `BenchmarkCondition` dataclass and an exact `BENCHMARK_CONDITIONS` tuple for clean, modality removal, context dropout, landmark noise, and frame-order shuffle.

- [ ] **Step 4: Run the test and verify GREEN**

Run the same command and require all registry checks to pass.

### Task 2: Implement deterministic context perturbations

**Files:**
- Modify: `tests/test_focused_fusion_robustness.py`
- Modify: `src/evaluation/focused_fusion_robustness.py`

- [ ] **Step 1: Write failing synthetic-tensor tests**

Verify that perturbations are deterministic, never change the clean target tensor or target mask, touch only observed context, preserve invalid storage, preserve timestamps/source indices, use modality zeroing through `input_arm`, and retain at least one observed context position per sample.

- [ ] **Step 2: Run the test and verify RED**

Expected: FAIL because `build_condition_inputs` is missing.

- [ ] **Step 3: Implement `build_condition_inputs`**

Return model input features, model reconstruction mask, and input arm. Dropout adds context positions to the model-only reconstruction mask; landmark noise modifies columns 72:95 only; frame shuffle permutes observed 95D rows only.

- [ ] **Step 4: Run the test and verify GREEN**

Require deterministic byte-equal outputs for repeat calls and explicit invariants for every condition.

### Task 3: Evaluate and aggregate three authenticated seeds

**Files:**
- Modify: `tests/test_focused_fusion_robustness.py`
- Modify: `src/evaluation/focused_fusion_robustness.py`

- [ ] **Step 1: Write failing evaluator tests with real synthetic models**

Assert exact metric fields and the bounded input-metric domain `[0, 1e9]` with at most 64 coefficient digits and exponent `[-100,100]`, unchanged clean scoring target, three required seeds, per-condition mean/sample-SD aggregation, a strictly positive clean mean, signed degradation range `[-100,1e16]`, clean replay precondition, exact Fusion arm, exact accounting of 160 held-out packets/10 recording groups/5,120 scored mask positions, and rejection of any field outside the closed deidentified public-report schema enumerated in the specification.

- [ ] **Step 2: Run the test and verify RED**

Expected: FAIL because `evaluate_fusion_conditions` and `aggregate_condition_metrics` are missing.

- [ ] **Step 3: Implement minimal evaluation and aggregation**

Use `DynamicLandmarkSSLModel`, existing `reconstruction_report`, and per-recording equal-weight metrics. Retain full per-seed aggregate metrics in memory but serialize no per-recording rows. Decimal work must use the specification's private precision-32 context, exact traps and exponent bounds; convert to JSON floats only after exact five-decimal canonicalization.

- [ ] **Step 4: Run the test and verify GREEN**

Require all new tests to pass with no warnings.

### Task 4: Add authenticated CLI and deidentified report

**Files:**
- Create: `scripts/run_focused_fusion_robustness.py`
- Modify: `tests/test_focused_fusion_robustness.py`

- [ ] **Step 1: Write failing CLI contract tests**

Verify import safety, default output namespace, private `0600` atomic write, canonical JSON, checkpoint/report/script commitments, exact three-seed requirement, fail-closed clean metric replay, and every exact top-level/nested allowlisted field enumerated in the specification rather than recursive blacklist filtering.

- [ ] **Step 2: Run the test and verify RED**

Expected: FAIL because the CLI does not exist.

- [ ] **Step 3: Implement the thin CLI**

Authenticate smoke -> selection -> winner with existing internal validators; fail closed unless the selected winner arm is exactly `fusion`, seeds are exactly `[0, 1, 2]`, and the authorized held-out accounting is exactly 160 packets, 10 recording groups, and 5,120 scored mask positions. Load the authorized held-out tensors, invoke the evaluation module, assert clean metrics equal persisted winner metrics after the existing five-decimal canonicalization, and write `outputs/dynamic_landmark/benchmarks/development/focused-fusion-robustness-v1/report.json` without identifiers or paths.

- [ ] **Step 4: Run the test and verify GREEN**

Require CLI contract tests to pass.

### Task 5: Execute, verify, and publish code

**Files:**
- Generated, ignored: `outputs/dynamic_landmark/benchmarks/development/focused-fusion-robustness-v1/report.json`

- [ ] **Step 1: Run the authenticated benchmark**

Run: `python3 scripts/run_focused_fusion_robustness.py`

Expected: 10 conditions x 3 seeds, clean replay exact, aggregate report written, no raw identifiers.

- [ ] **Step 2: Independently recompute summary statistics**

Read the report with a separate process and verify means, sample SDs, condition count, target accounting, and canonical JSON. Define each degradation percentage as `100 * (condition three-seed mean equal-block raw MAE / clean_fusion three-seed mean equal-block raw MAE - 1)`; lower MAE remains better.

- [ ] **Step 3: Run relevant regression tests**

Run the new test plus `tests/test_dynamic_landmark_model.py`, `tests/test_dynamic_landmark_ssl.py`, and `tests/test_dynamic_landmark_benchmark.py`.

- [ ] **Step 4: Verify protected-file digests and git scope**

Confirm `scripts/pretrain_dynamic_landmarks.py`, `src/pretraining/dynamic_landmark_ssl.py`, and `src/models/dynamic_landmark.py` remain identical to pushed commit `6d7fbf2`; stage only the new evaluator, CLI, tests, specification, and plan.

- [ ] **Step 5: Commit and push**

Commit tersely, push `codex/landmark-fusion`, and verify the remote branch SHA. Never stage the existing data deletions or any `outputs/dynamic_landmark/` content.
