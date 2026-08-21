# Shared V10 Bilateral Reconstruction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:executing-plans and implement every behavior test-first. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze BLV9-009 as the V9 research baseline and test whether
laterality-safe bilateral reconstruction plus flat-minimum training improves
its three-source robustness.

**Architecture:** Keep the exact V8 shared 478D+110D motor encoder, small
post-shared endpoint heads, and BLV9-009 masked clinical auxiliary. Compare six
frozen candidates: the exact BLV9-009 baseline, baseline+SAM, bilateral
mean/absolute-difference reconstruction with and without SAM, and unordered
twin-view reconstruction with and without SAM. The auxiliary decoder is
training-only; no source identity enters the shared encoder.

**Tech Stack:** Python, PyTorch 2.7/CUDA, NumPy, repository direct-test harness,
NVIDIA H200.

---

## Scientific boundary

- V9 research baseline: `BLV9-009_masked_clinical_reconstruction`.
- V8 remains the deployment model until a new model passes the existing locked
  gate on all three development sources and later receives Mayo validation.
- Use only 38 PalsyNet development, 36 NeuroFace, and 56 MEEI participants.
- Do not read Mayo or the PalsyNet protected outer test.
- Use the same six participant-disjoint folds, three seeds `(0, 1, 2)`, 20
  epochs, fixed threshold `0.5`, fold-local scaler, and source/class weights.
- The bilateral target is `(0.5*(original+mirror), abs(original-mirror))`:
  capacity plus asymmetry magnitude without assuming which side is healthy.
- The unordered twin target reconstructs both views with a swap-invariant set
  loss: preserve unilateral patterns without assigning disease to left/right.
- No threshold, width, learning-rate, masking-fraction, or epoch search.

## Frozen candidates

| ID | Reconstruction | Optimizer | Medical purpose |
|---|---|---|---|
| BRV10-000 | exact BLV9-009 averaged-view target | AdamW | Exact V9 research baseline |
| BRV10-001 | exact BLV9-009 averaged-view target | SAM | Test flatness without changing representation |
| BRV10-002 | bilateral mean + absolute-difference target | AdamW | Preserve magnitude of asymmetry without laterality assumption |
| BRV10-003 | bilateral mean + absolute-difference target | SAM | Combine distributed bilateral evidence with flatness |
| BRV10-004 | unordered original/mirror twin target | AdamW | Preserve unilateral pattern as an unordered pair |
| BRV10-005 | unordered original/mirror twin target | SAM | Test the strongest laterality-safe target with flatness |

## Task 1: Freeze the research baseline and registry

**Files:**
- Create: `src/models/bilateral_reconstruction_candidate_registry_v10.py`
- Test: `tests/test_bilateral_reconstruction_candidate_registry_v10.py`

- [ ] Write a failing test requiring exactly six immutable candidates and an
  exact BRV10-000 mapping to BLV9-009.
- [ ] Run the test and verify RED because the module does not exist.
- [ ] Implement the minimal closed registry and medical rationale fields.
- [ ] Run the test and verify GREEN.

## Task 2: Implement laterality-safe reconstruction losses

**Files:**
- Create: `src/training/bilateral_masked_reconstruction_v10.py`
- Test: `tests/test_bilateral_masked_reconstruction_v10.py`

- [ ] Write failing tests for exact mean/absolute-difference targets, view-swap
  invariance, unordered-twin minimum assignment, masked-index-only loss, finite
  gradients into the shared clinical encoder, and no source argument.
- [ ] Run and verify RED.
- [ ] Implement one training-only decoder for each target mode.
- [ ] Run and verify GREEN plus V9 objective regressions.

## Task 3: Implement the closed participant-disjoint evaluator

**Files:**
- Create: `src/evaluation/bilateral_reconstruction_shared_search_v10.py`
- Test: `tests/test_bilateral_reconstruction_shared_search_v10.py`

- [ ] Write failing tests proving exact BLV9-009 comparator semantics, six-fold
  held-out coverage, three-source shared gradients, SAM only for the three
  declared candidates, and the unchanged promotion gate.
- [ ] Run and verify RED.
- [ ] Implement the minimal evaluator using the frozen V9 tensors and helpers.
- [ ] Run and verify GREEN plus V8/V9 regressions.

## Task 4: Add and run the H200 release runner

**Files:**
- Create: `scripts/run_bilateral_reconstruction_shared_search_v10.py`
- Test: `tests/test_run_bilateral_reconstruction_shared_search_v10.py`

- [ ] Write failing tests requiring H200, exact source/cache commitments, all
  six candidates and three seeds, aggregate-only output, atomic publication,
  and zero Mayo/protected reads.
- [ ] Run and verify RED.
- [ ] Implement the runner without candidate or tuning CLI knobs.
- [ ] Run all six candidates on H200: `6 x 3 x 9 = 162` fits.

## Task 5: Decide, document, and verify

**Files:**
- Create: `docs/CURRENT_RESEARCH_MODEL.md`
- Create: `docs/results/shared_v10_bilateral_reconstruction.md`
- Create: `docs/results/artifacts/shared_v10_bilateral_reconstruction/report.json`
- Create: `tests/test_shared_v10_bilateral_reconstruction_release.py`

- [ ] Record BLV9-009 as the V9 research baseline before inspecting V10.
- [ ] Copy and checksum the aggregate H200 report.
- [ ] Promote a V10 research candidate only if it passes the locked gate;
  otherwise retain BLV9-009 as the research baseline.
- [ ] Verify V8 deployment hashes are unchanged, all tests pass, no patient rows
  or private paths are public, and the worktree is clean after commit.
