# Action-Aligned Clinical Geometry v1 Implementation Plan

**Goal:** Produce a repeatable, development-only test of a fixed clinical landmark-dynamics representation for incremental information beyond recorded nuisance, without extracting or predicting protected outer rows.

**Architecture:** Add a small preprocessing module that converts one validated `(4, 32, 95)` trajectory into 58 direction-free clinical dynamics. Add a dedicated runner that loads the existing validated cache, computes four fixed candidate matrices, and fits a fixed grouped inner-OOF logistic model on outer-fold-0 development rows only.

**Tech stack:** Python, NumPy, scikit-learn, existing dynamic cache loader and nested group split/evaluation utilities.

---

### Task 1: Frozen clinical-dynamics feature extractor

**Files:**
- Create: `src/preprocessing/clinical_dynamics.py`
- Create: `tests/test_clinical_dynamics.py`

1. Write failing tests for exact 58-name ordering, side-swap invariance, known pair dynamics, no cross-window/gap derivatives, and malformed-input rejection. Freeze five-frame, endpoint-valid, within-window lag semantics explicitly.
2. Run the direct test file and confirm the expected RED failure because the module does not exist.
3. Implement the minimum extractor by reusing validated trajectory summary and bilateral-dynamics primitives. Document that intermediate-frame continuity is not enforced for lag.
4. Run the new test plus existing trajectory-feature regression tests and confirm GREEN.

### Task 2: Fixed development-only nuisance challenge

**Files:**
- Create: `scripts/run_action_clinical_geometry_v1.py`
- Create: `tests/test_action_clinical_geometry_v1.py`

1. Write failing tests for the closed candidate registry, fixed parser/training constants, complete inner OOF coverage, zero protected-row extraction/fit/prediction, aggregate report schema, and safe atomic writer.
2. Confirm RED before implementation.
3. Implement fixed-`C` inner OOF prediction, group metrics, the primary paired bootstrap (`clinical_dynamics_plus_nuisance` minus `nuisance`), index-audit aggregation, gate decision, and private no-overwrite report writing. Build candidate matrices only for outer-fold-0 development indices.
4. Run focused tests and the relevant classical-regression tests.

### Task 3: Run and verify the real development screen

**Files:**
- Generate locally (do not commit): `outputs/dynamic_landmark/benchmarks/development/action-clinical-geometry-v1/report.json`

1. Run the dedicated script against the canonical PalsyNet cache using the project Python environment.
2. Independently recompute key metrics from in-memory outputs, confirm no outer predictions, check report permissions and no identifiers/paths.
3. Run all focused tests fresh, `git diff --check`, and review the exact diff.
4. Commit source, tests, and protocol documents only; leave generated report untracked/local.
