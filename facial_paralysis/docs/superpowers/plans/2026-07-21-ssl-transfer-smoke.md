# Focused SSL Transfer Smoke Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and run one fixed, development-only comparison of random Landmark, random Fusion, and authenticated Fusion SSL warm-start initialization.

**Architecture:** A focused training module owns the exact encoder-transfer allowlist and four-fold inner-OOF loop. A thin CLI authenticates the existing focused winner chain, loads the already validated PalsyNet cache, runs the three fixed candidates, and writes a deidentified JSON report. No outer-test prediction API is added or changed.

**Tech Stack:** Python 3.9, PyTorch, NumPy, scikit-learn, repository `_testlib` runner.

---

### Task 1: Exact Fusion encoder transfer and inner-OOF smoke

**Files:**
- Create: `src/training/dynamic_landmark_transfer_smoke.py`
- Create: `tests/test_dynamic_landmark_transfer_smoke.py`

- [ ] **Step 1: Write the failing transfer tests**

Test that a full `DynamicLandmarkSSLModel.state_dict()` copies exactly the four
projection modules, BiGRU, attention, and pool projection into a Fusion
`DynamicLandmarkModel`, while the binary head remains byte-for-byte fresh.
Test that an incomplete or extra source-state schema, incompatible shape or
dtype, nonfinite tensor, or non-Fusion destination fails before any
destination parameter changes. The accepted source must be the exact 22-tensor
focused SSL model state; only 16 encoder tensors are copied.

- [ ] **Step 2: Run the test and verify RED**

Run: `python3 tests/test_dynamic_landmark_transfer_smoke.py`

Expected: FAIL because `src.training.dynamic_landmark_transfer_smoke` does not
exist.

- [ ] **Step 3: Implement the minimal atomic transfer function**

Define `transfer_focused_fusion_encoder(source_state, downstream)` with an
exact prefix allowlist:

```python
TRANSFER_PREFIXES = (
    "proj_bs_x.", "proj_bs_dx.", "proj_lm_x.", "proj_lm_dx.",
    "temporal.", "attention_score.", "pool_projection.",
)
```

Validate the complete focused SSL state schema and every required tensor before
loading a copied destination state, then return the sorted 16-key transfer
tuple. The two RAVDESS projections and four reconstruction-decoder tensors are
validated as source lineage but never transferred.

- [ ] **Step 4: Write and verify RED for the OOF loop**

Add a synthetic grouped fold whose protected outer rows contain nonfinite
sentinels. Require `run_development_inner_oof(...)` to return predictions only
for the four inner validation folds, never inspect or predict the protected
outer rows, and apply transfer only for `fusion_ssl_warmstart`.

- [ ] **Step 5: Implement the fixed inner-OOF loop**

For each inner fold, fit the existing standardizer on its train rows, create a
fresh seed-0 candidate, optionally transfer the authenticated state, train for
the fixed epoch count through the existing benchmark primitive, and fill only
the inner validation positions. Return labels, probabilities, and transfer
audit metadata for outer-train rows.

- [ ] **Step 6: Run focused and regression tests**

Run:

```bash
python3 tests/test_dynamic_landmark_transfer_smoke.py
python3 tests/test_dynamic_landmark_model.py
python3 tests/test_dynamic_landmark_benchmark.py
```

Expected: all pass.

- [ ] **Step 7: Commit Task 1**

```bash
git add facial_paralysis/src/training/dynamic_landmark_transfer_smoke.py \
  facial_paralysis/tests/test_dynamic_landmark_transfer_smoke.py
git commit -m "feat(training): add focused SSL transfer smoke"
```

### Task 2: Authenticated fixed-budget runner and real development smoke

**Files:**
- Create: `scripts/run_dynamic_landmark_transfer_smoke.py`
- Modify: `tests/test_dynamic_landmark_transfer_smoke.py`

- [ ] **Step 1: Write failing CLI/report contract tests**

Require exactly two private input arguments (`--ssl-pretraining-root` and
`--palsynet-cache-root`), a fixed seed 0, outer fold 0, and 12 epochs. Require a
closed report schema with three candidate rows, no identifiers or paths, and
an explicit development-only claim.

- [ ] **Step 2: Run the test and verify RED**

Run: `python3 tests/test_dynamic_landmark_transfer_smoke.py`

Expected: FAIL because the runner/report builder does not exist.

- [ ] **Step 3: Implement the thin runner**

Authenticate bridge, smoke, selection, and winner artifacts through the
existing focused validators. Require selected arm `fusion` and seed 0, load the
validated PalsyNet cache, run the exact three candidates, compute group-level
metrics, and write one owner-private canonical JSON report. Do not serialize
either input path or any row/group identifier.

- [ ] **Step 4: Run tests and the real smoke**

Run the focused/regression tests, then execute the runner against the existing
private focused pretraining tree and canonical PalsyNet derived cache.

Expected: one complete `inner_oof_development_smoke_only` report with 39
development records, 10 protected records receiving no prediction, and three
metric rows.

- [ ] **Step 5: Commit Task 2**

```bash
git add facial_paralysis/scripts/run_dynamic_landmark_transfer_smoke.py \
  facial_paralysis/tests/test_dynamic_landmark_transfer_smoke.py
git commit -m "feat(evaluation): run authenticated SSL transfer smoke"
```

### Task 3: Final audit and decision

**Files:**
- Review: `src/training/dynamic_landmark_transfer_smoke.py`
- Review: `scripts/run_dynamic_landmark_transfer_smoke.py`
- Review: generated development report (not committed)

- [ ] **Step 1: Verify all focused tests, `git diff --check`, and clean source status**
- [ ] **Step 2: Independently recompute the three metric rows from OOF outputs in memory**
- [ ] **Step 3: Confirm the outer evaluator remains byte-identical to base**
- [ ] **Step 4: Apply the preregistered expansion gate and report the next decision**
