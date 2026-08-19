# Dense-Clinical Shared Encoder v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and quickly evaluate one genuinely shared dual-stream encoder trained by PalsyNet development, NeuroFace, and MEEI for eventual Mayo transfer.

**Architecture:** Convert all cohorts into variable-length action bags containing a required frozen 110D clinical token and an optional baseline-centered 478-point temporal mesh. Pair original and true-mirror views, fuse a shared dense spatial-temporal encoder with a shared clinical encoder, aggregate actions with a source-blind Set Transformer, then apply small task heads only after the shared 64D patient embedding. V6 remains the descriptive non-shared benchmark.

**Tech Stack:** Python 3.12, NumPy, scikit-learn, PyTorch CUDA on H200, canonical JSON/NPZ evidence contracts.

---

### Task 1: Common dense-clinical action-token projection

**Files:**
- Create: `src/preprocessing/shared_clinical_tokens_v1.py`
- Create: `tests/test_shared_clinical_tokens_v1.py`

- [x] Write failing tests proving PalsyNet windows and dense action meshes produce exact finite 110D clinical tokens plus bounded `32x478x3` dense tokens under one action-bag schema.
- [x] Test explicit missing-dense masks, closed actions, separate mirrored views, and source-free tokens.
- [x] Implement real-time interpolation, action-minus-rest dense response, dense-to-clinical23-to-110D projection, and the PalsyNet clinical-only adapter.
- [x] Run the new tests plus clinical-landmark, trajectory, dense-cache, and 110D regressions.

### Task 2: Shared encoder and endpoint heads

**Files:**
- Create: `src/models/dense_clinical_shared_encoder_v1.py`
- Create: `tests/test_dense_clinical_shared_encoder_v1.py`

- [x] Test variable masks, permutation-safe pooling, mirror-pair invariance, finite logits, and shared gradients from all three sources.
- [x] Implement the 110D encoder, 478-point spatial bottleneck, TCN, gated fusion, action Set Transformer, and shared 64D embedding.
- [x] Keep missing dense input exactly inert and task heads below five percent of parameters.
- [x] Prove source IDs are absent from the encoder; route endpoints only after the shared embedding.
- [x] Run the model tests on CPU and H200 CUDA.

### Task 3: Participant-disjoint training and transfer evaluation

**Files:**
- Create: `src/evaluation/shared_clinical_encoder_v1.py`
- Create: `tests/test_shared_clinical_encoder_evaluation_v1.py`

- [x] Test six group-disjoint folds, fold-local scaling, equal source-class loss mass, and fixed 0.5 thresholds.
- [x] Implement 110D-only and dense-clinical candidates with seeds 0/1/2.
- [x] Add separate per-source metrics without pooled-score substitution.
- [ ] Leave-one-source-out was not launched because the three-seed candidate failed the V6 no-regression gate.
- [x] Run deterministic synthetic smoke and verify every source updates the shared encoder.

### Task 4: Authenticated H200 experiment runner

**Files:**
- Create: `scripts/run_dense_clinical_shared_encoder_v1.py`
- Create: `tests/test_run_dense_clinical_shared_encoder_v1.py`

- [x] Test exact manifest/cache commitments, PalsyNet development-only filtering, zero protected/Mayo reads, and no-overwrite aggregate output.
- [x] Build participant token bags from authenticated PalsyNet, NeuroFace, and MEEI caches on H200.
- [x] Run a one-seed/20-update CUDA smoke in about 21 seconds.
- [x] Run the retained two-candidate, three-seed loop and stop at the no-promotion boundary.

### Task 5: Evidence and model decision

**Files:**
- Create: `docs/results/dense_clinical_shared_encoder_v1.md`
- Create: `docs/results/artifacts/dense_clinical_shared_encoder_v1/report.json`
- Create: `tests/test_dense_clinical_shared_encoder_release_v1.py`

- [x] Publish only aggregate per-source three-seed metrics, commitments, and the exact decision boundary.
- [x] Compare against V6 descriptively and retain V6 because the shared candidate did not pass.
- [x] Preserve UCR4/current registry and the V6 artifact byte-for-byte.
- [ ] Run final regressions, py_compile, diff-check, secret scan, and identifier/path scan before handoff.
