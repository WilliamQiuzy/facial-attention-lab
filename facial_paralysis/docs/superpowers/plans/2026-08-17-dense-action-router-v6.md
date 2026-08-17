# Universal Clinical Router v6 Dense-Action Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans
> to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for
> tracking.

**Goal:** Add and validate a dense full-mesh action expert so all three UCR
development profiles reach at least 0.93 participant/group-disjoint accuracy.

**Architecture:** Preserve UCR4 and add a source-blind action-only branch. The
branch extracts normalized 478-point action/rest trajectories, screens compact
model families, then freezes one configuration per evidence profile for a
participant-disjoint development reconstruction and deidentified report.

**Tech Stack:** Python, NumPy, scikit-learn, MediaPipe FaceLandmarker, OpenCV,
PyAV, Docker, Nebius H200.

---

### Task 1: Dense action representation

**Files:**
- Create: `src/preprocessing/dense_bilateral_action_v1.py`
- Create: `tests/test_dense_bilateral_action_v1.py`

- [x] Write failing tests for eye-aligned normalization, invalid support,
  action/rest statistics, deterministic feature names and original/mirror
  independence.
- [x] Run the direct test and confirm the module-missing RED failure.
- [x] Implement the minimal array-only representation.
- [x] Run the direct test and affected preprocessing tests until GREEN.

### Task 2: Private dense-mesh cache contract

**Files:**
- Create: `scripts/extract_dense_action_mesh_v1.py`
- Create: `tests/test_extract_dense_action_mesh_v1.py`

- [x] Write failing tests for exact cache schema, immutable source-frame/action
  binding, masked misses, deterministic serialization and no-overwrite output.
- [x] Implement original and flipped-image FaceLandmarker extraction without
  changing the existing 95D extractor.
- [x] Verify test fixtures, malformed-cache rejection and private-output modes.

### Task 3: Fully nested action evaluator

**Files:**
- Create: `src/evaluation/dense_action_router_v6.py`
- Create: `tests/test_dense_action_router_v6.py`

- [x] Write failing tests for the four-candidate registry, six-by-five group
  separation, train-only ranking/scaling, mirror weighting, inner threshold
  selection and closed aggregate metrics.
- [x] Implement the minimum nested evaluator.
- [x] Verify deterministic synthetic recovery and failure on group leakage.

### Task 4: H200 extraction and rapid screen

**Private output:** `dense-action-router-v6-*`

- [x] Verify H200, exact FaceLandmarker model digest and immutable source
  manifests before extraction.
- [x] Extract MEEI and NeuroFace dense action caches using authenticated frame
  and task evidence; preserve all row-level evidence on H200 only.
- [x] Run the frozen four-family screen and record runtime, candidate metrics,
  fold choices and error complementarity.
- [x] Refine only the representation using the frozen region/statistic
  families; do not tune the gate or read Mayo.

### Task 5: Confirmation and release decision

**Files:**
- Create: `scripts/run_dense_action_router_v6.py`
- Create: `tests/test_run_dense_action_router_v6.py`

- [x] Reconstruct final participant-disjoint predictions from exact private
  cache bytes using the locked configurations and threshold.
- [x] Independently recompute every aggregate metric and UCR4 baseline hash.
- [x] Apply the fixed all-profile accuracy/balanced-accuracy gate.
- [x] Repeat the aggregate-only run and require byte-identical public JSON.

### Task 6: Version maintenance

**Files:**
- Modify: `docs/model_candidates.json`
- Create: `docs/results/universal_clinical_router_v6_candidate.md`
- Create: `docs/results/artifacts/universal_clinical_router_v6_candidate/report.json`
- Modify: `docs/results/README.md`
- Modify: `docs/PIPELINE.md`
- Create: `tests/test_universal_clinical_router_v6_release.py`

- [x] Register the candidate and exact report hash without changing UCR4's
  default registry/import/artifact.
- [x] Run targeted tests, the complete direct-test suite, compilation,
  whitespace, secret/private-data and clean-clone checks.
- [x] Commit the isolated branch. Do not push without an explicit request.
