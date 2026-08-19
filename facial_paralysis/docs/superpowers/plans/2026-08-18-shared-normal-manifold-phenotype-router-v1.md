# Shared Normal-Manifold Phenotype Router v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce cross-source healthy false positives with one shared normal-motion reference while preserving distinct PalsyNet, NeuroFace, and MEEI endpoints.

**Architecture:** Start from the strongest stable genuinely shared v2 encoder. Add a train-fold-only shared healthy anchor and a shared universal normality logit, then retain tiny source-specific endpoint heads only after the shared patient embedding. If that fails, use a predeclared gradient audit to choose either conflict projection or a point-preserving anatomical set encoder.

**Tech Stack:** Python, NumPy, PyTorch 2.7 CUDA, NVIDIA H200, canonical JSON evidence.

---

### Task 1: Freeze the normal-manifold candidate contract

**Files:**
- Create: `src/models/normal_manifold_candidate_registry_v4.py`
- Create: `tests/test_normal_manifold_candidate_registry_v4.py`

- [ ] Write RED tests for exactly six unique candidates and the closed medical rationale.
- [x] Implement the two-axis registry without score-dependent expansion.
- [x] Verify every candidate uses the same shared encoder and differs only in normal-reference strength and shared-normality blend.

### Task 2: Add the shared healthy anchor and phenotype path

**Files:**
- Create: `src/models/shared_normal_manifold_router_v4.py`
- Create: `tests/test_shared_normal_manifold_router_v4.py`

- [x] Write RED tests for one shared anchor, control-only compactness, no affected centroid collapse, source-blind encoding, and endpoint heads after the shared embedding.
- [x] Reuse the locked `MSC2-022` evidence path and expose patient embedding, normal distance, universal logit, and routed endpoint logits.
- [x] Prove PalsyNet missing dense evidence remains masked and no contralateral side is treated as normal.

### Task 3: Build participant-disjoint evaluation and gradient audit

**Files:**
- Create: `src/evaluation/shared_normal_manifold_search_v4.py`
- Create: `tests/test_shared_normal_manifold_search_v4.py`
- Create: `scripts/run_shared_normal_manifold_search_v4.py`
- Create: `tests/test_run_shared_normal_manifold_search_v4.py`

- [x] Write RED tests for six folds, fold-local scaling, control-only manifold loss, exact source-class mass, source gradient cosine, and locked ranking.
- [x] Implement seed-0 six-candidate screen and top-two seeds 1/2 confirmation.
- [x] Emit aggregate metrics and commitments only; Mayo/protected reads and predictions stay zero.

### Task 4: Run v4 on H200

**Files:**
- Create: `docs/results/artifacts/shared_normal_manifold_router_v4/report.json`

- [x] Run a short H200 smoke, then all six candidates for 20 updates and six folds.
- [x] Lock top two by minimum balanced accuracy, specificity, AUROC, and mean balanced accuracy.
- [x] Confirm seeds 1 and 2 and apply the all-source 0.90 balanced-accuracy / 0.85-specificity gate.

### Task 5: Conditional next iteration

**Files:**
- Create only if triggered: `src/models/shared_anatomical_set_router_v5.py`
- Create only if triggered: `src/evaluation/shared_anatomical_set_search_v5.py`
- Create only if triggered: corresponding tests and H200 runner.

- [x] If any audited source-gradient pair is negative, test conflict projection against v4 without changing the representation.
- [x] Otherwise, implement the shared landmark-identity/anatomical-region set encoder and keep the v4 loss/head protocol fixed.
- [x] Stop the preregistered v4/v5 comparison before making any promotion decision.

**Execution trigger:** v4 seed 0 observed negative PalsyNet/NeuroFace and
PalsyNet/MEEI shared-patient gradient cosine, so v5 is authorized. Freeze four
v5 candidates: `{patient_block, all_shared} x {0.5, 1.0}` projection strength,
all using `NMR4-001`; endpoint-head gradients remain untouched.

**User-authorized scope amendment:** after v5 failed, the subsequent request
explicitly authorized another medically justified iteration if the result was
still inadequate. V6–V9 therefore tested, in order, script-aware pooling,
label-free response statistics, rank-limited post-encoder residuals, and
frozen-core endpoint adaptation. Each branch retained a genuinely shared
trainable encoder and was stopped after its locked comparison. None passed the
all-source gate, so no model was promoted.

### Task 6: Freeze evidence and conclusion

**Files:**
- Create: `docs/results/shared_normal_manifold_router_v4.md`
- Create: `tests/test_shared_normal_manifold_release_v4.py`

- [x] Recompute all metrics from OOF probabilities and bind data/code/report hashes.
- [x] Report raw accuracy, balanced accuracy, sensitivity, specificity, AUROC, seed stability, and gradient cosine by source.
- [x] State whether the model passed, remains research-only, or triggered v5; never overwrite UCR4 without a separate promotion decision.
- [x] Run affected tests, py_compile, diff-check, secret/private-path scans, and H200 parity checks.
