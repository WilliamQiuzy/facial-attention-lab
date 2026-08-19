# Medically-Gated Shared Clinical Encoder v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate a frozen 32-candidate family of medically justified, genuinely shared 478D+110D encoders and seek over 90% participant-disjoint accuracy in PalsyNet, NeuroFace, and MEEI.

**Architecture:** Preserve the v1 authenticated action bags and shared 64D patient embedding. Add frozen clinical-region summaries and action-region routing, then compare only view, regional-evidence, pooling, and fusion choices that pass the written medical-rationale gate. Dataset-specific parameters remain confined to tiny heads after the shared embedding.

**Tech Stack:** Python 3.12, NumPy, PyTorch 2.7 CUDA, H200, canonical JSON evidence.

---

### Task 1: Closed medical component and candidate registry

**Files:**
- Create: `src/models/medical_shared_candidate_registry_v2.py`
- Create: `tests/test_medical_shared_candidate_registry_v2.py`

- [ ] Write a failing test requiring exactly 32 unique candidates and closed component fields: phenomenon, evidence, valid labels, contraindication.
- [ ] Verify RED because the registry module does not exist.
- [ ] Implement the four-axis Cartesian registry and component evidence without score-dependent choices.
- [ ] Verify every candidate remains shared and every component has an interpretation boundary.

### Task 2: Clinically regional shared encoder

**Files:**
- Create: `src/models/medically_gated_shared_encoder_v2.py`
- Create: `tests/test_medically_gated_shared_encoder_v2.py`

- [ ] Write failing tests for frozen brow/eye/mouth contours, neutral-referenced excursion, real-time velocity, action-region routing, and commutative bilateral aggregation.
- [ ] Verify RED because the v2 encoder does not exist.
- [ ] Implement full-mesh raw TCN plus optional regional summaries without removing the 110D branch.
- [ ] Implement `original_only` and strictly swap-invariant bilateral modes.
- [ ] Implement masked-concat/reliability-gate fusion and meanmax/Transformer action pooling.
- [ ] Prove source identity cannot enter the shared encoder and task heads remain under five percent of parameters.
- [ ] Prove all sources update the same clinical/action/patient layers and dense sources update the same dense/region layers.

### Task 3: Frozen shared-only evaluation loop

**Files:**
- Create: `src/evaluation/medically_gated_shared_search_v2.py`
- Create: `tests/test_medically_gated_shared_search_v2.py`
- Create: `scripts/run_medically_gated_shared_search_v2.py`
- Create: `tests/test_run_medically_gated_shared_search_v2.py`

- [ ] Write failing tests for the exact 32 candidates, six participant-disjoint folds, fold-local scaling, equal source-class mass, seed-0 screen, and locked top-four ranking.
- [ ] Implement shared-only evaluation using the existing authenticated v1 dataset loader.
- [ ] Emit aggregate metrics only; forbid identifiers, private paths, Mayo reads, protected PalsyNet reads, and overwrite.
- [ ] Run a two-fold/two-update local and H200 probe before the full screen.

### Task 4: H200 candidate screen and stability run

**Files:**
- Create: `docs/results/artifacts/medically_gated_shared_encoder_v2/report.json`

- [ ] Run all 32 candidates with seed 0 and 20 updates on the H200.
- [ ] Apply the frozen ranking rule without adding candidates.
- [ ] Run seeds 1 and 2 for the locked top four.
- [ ] Stop early only if a candidate exceeds 90% mean accuracy in all three sources; otherwise record the exhausted registry.

### Task 5: Audit and handoff

**Files:**
- Create: `docs/results/medically_gated_shared_encoder_v2.md`
- Create: `tests/test_medically_gated_shared_encoder_release_v2.py`

- [ ] Recompute aggregate metrics from seed reports and bind implementation/data hashes.
- [ ] State the flip scope, every failed medical hypothesis, the strongest shared result, and Mayo/HB limits.
- [ ] Run all affected tests, py_compile, diff-check, secret/path/identifier scans, and H200 parity tests.

## Execution record

Completed on 2026-08-18. The frozen v2 screen evaluated all 32 candidates;
the locked top four received seeds 1 and 2. A separately documented 16-version
compact v3 falsification family brought the total to 48 shared candidates.
No candidate passed the three-source 90% gate, so no model was promoted and no
Mayo or protected PalsyNet scoring was performed. Aggregate evidence and the
failed hypotheses are frozen in
`docs/results/artifacts/medically_gated_shared_encoder_v2/report.json` and
`docs/results/medically_gated_shared_encoder_v2.md`.
