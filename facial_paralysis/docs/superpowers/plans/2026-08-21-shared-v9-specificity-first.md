# Shared V9 Specificity-First Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and evaluate a genuinely shared V9 facial-motor model that reduces false-positive healthy-control calls across PalsyNet development, NeuroFace, and MEEI while improving or preserving participant-disjoint AUROC and accuracy.

**Architecture:** Keep the exact V8 110D plus optional 478-point shared action encoder and its 64D patient representation. Add only source-blind healthy-reference mechanisms at or after that representation: a shared normal prototype, affected-versus-control distance margin, cross-protocol control alignment, bounded control cost, and a small normality-logit blend. Dataset/protocol identity remains unavailable to the shared encoder and is used only by the existing small post-embedding endpoint adapters/heads and by training-only audit losses.

**Tech Stack:** Python 3.10, NumPy, PyTorch 2.7/CUDA, scikit-learn metrics, repository direct-test harness, NVIDIA H200.

## Execution record

The initial 24-candidate healthy-reference screen completed and failed the
promotion gate, so no top-three confirmation was scientifically warranted. The
investigation then executed four preregistered follow-up families: 16 equal
deep ensembles, 13 nested OOF-distillation candidates, 16 low-dimensional
mechanism encoders, and 25 full-mesh action-phenotype heads. Across all 94
candidates, none passed the locked gate. Repeated pre-determinism V8 evaluations
also differed by one held-out participant; all subsequent screens therefore
enforced deterministic CUDA and reproduced the same comparator twice. The
final decision and exact machine artifacts are recorded in
`docs/results/shared_v9_specificity_first.md`. V8 registry and deployment files
remain unchanged.

---

## Locked scientific boundary

- Use only the 38 exposed PalsyNet development participants, 36 NeuroFace participants, and 56 MEEI participants.
- Do not read the protected PalsyNet partition, Mayo media, Mayo labels, or Mayo predictions.
- Use the existing deterministic six participant-disjoint folds and fold-local 110D scaler.
- Report the exact V8/RSR8-001 comparator under the same evaluator.
- Screen one seed across a frozen registry, then confirm only the top three candidates with seeds 1 and 2.
- Report fixed-0.5 metrics and a separate fold-train-only calibrated operating point. Calibration may not change AUROC and may not use held-out labels.
- Do not replace or overwrite the deployed V8 unless the confirmation gate passes.

## Promotion gate

The V9 candidate must meet all of the following on the participant-disjoint confirmation aggregate:

- minimum source accuracy at least `0.90`;
- minimum source specificity at least `0.80`;
- minimum source AUROC at least `0.92`;
- minimum source sensitivity at least `0.85`;
- no source AUROC or accuracy more than `0.01` below the same-seed V8 comparator;
- all three sources deliver nonzero gradients to the shared clinical encoder and patient projection;
- task-specific parameters remain below ten percent of total trainable parameters.

### Task 1: Freeze the medically justified V9 candidate registry

**Files:**
- Create: `src/models/specificity_aware_candidate_registry_v9.py`
- Test: `tests/test_specificity_aware_candidate_registry_v9.py`

- [ ] **Step 1: Write the failing registry test**

Test exact candidate IDs, 24 unique combinations, the three allowed healthy-reference modes, two control-cost values, two universal blends, and two control-alignment weights. Require a nonempty medical rationale and contraindication for every component.

- [ ] **Step 2: Run the test and verify RED**

Run: `PYTHONPATH=.:tests python3 tests/test_specificity_aware_candidate_registry_v9.py`

Expected: FAIL because the V9 registry module does not exist.

- [ ] **Step 3: Implement the minimal frozen registry**

Define exact dataclasses and only these search axes:

```python
healthy_mode = ("off", "compact", "compact_margin")
control_cost = (1.0, 1.5)
universal_blend = (0.25, 0.50)
control_alignment_weight = (0.0, 0.02)
```

The compact modes use a shared control prototype; `compact_margin` also pushes affected participants beyond a fixed shared distance margin. Control alignment may align control embeddings across sources during training but cannot expose source identity at inference.

- [ ] **Step 4: Run the test and verify GREEN**

- [ ] **Step 5: Commit the registry and test**

### Task 2: Implement the shared specificity-aware V9 model

**Files:**
- Create: `src/models/specificity_aware_shared_router_v9.py`
- Test: `tests/test_specificity_aware_shared_router_v9.py`

- [ ] **Step 1: Write failing model-contract tests**

Require the exact locked RSR8-001 shared trunk, one shared 64D normal prototype, finite normal distances/logits, endpoint residuals only after the shared patient embedding, no dataset/source argument in inference methods, and task-specific parameters below ten percent.

- [ ] **Step 2: Run the test and verify RED**

- [ ] **Step 3: Implement the minimal V9 model**

Expose shared action tokens, endpoint and universal patient embeddings, a shared prototype-derived normality logit, and the existing post-embedding endpoint adapter/head. The healthy-reference score must be source blind and use no laterality assumption.

- [ ] **Step 4: Run V9 and V8 model tests and verify GREEN**

- [ ] **Step 5: Commit model and tests**

### Task 3: Add fold-train-only specificity calibration and multi-objective metrics

**Files:**
- Create: `src/evaluation/specificity_aware_shared_search_v9.py`
- Test: `tests/test_specificity_aware_shared_search_v9.py`

- [ ] **Step 1: Write failing evaluator tests**

Prove that calibrated thresholds are selected from training probabilities only, target maximum training specificity subject to sensitivity at least `0.90`, never inspect held-out labels, and can differ by protocol because endpoint heads differ. Require both fixed-0.5 and calibrated OOF metrics, exact fold coverage, immutable outputs, and shared-gradient audit.

- [ ] **Step 2: Run the test and verify RED**

- [ ] **Step 3: Implement V9 training losses**

Use source/class-balanced participant weights, then apply the frozen control-cost factor and renormalize. Add control compactness, affected distance margin, and cross-source control-centroid alignment only when enabled. Keep the shared universal BCE auxiliary loss and compute endpoint logits with the frozen candidate blend.

- [ ] **Step 4: Implement operating-point selection**

After each outer-fold model fit, select each protocol threshold only from that fold's training predictions. Apply it once to held-out participants and retain the raw probabilities for AUROC/Brier.

- [ ] **Step 5: Implement constrained ranking**

Rank feasible candidates by minimum source specificity, then minimum AUROC, minimum accuracy, minimum balanced accuracy, and mean accuracy. A candidate with sensitivity below `0.85` or a source regression beyond the frozen comparator constraint is infeasible.

- [ ] **Step 6: Run evaluator tests and V8 regressions**

- [ ] **Step 7: Commit evaluator and tests**

### Task 4: Build the authenticated H200 search runner

**Files:**
- Create: `scripts/run_specificity_aware_shared_search_v9.py`
- Test: `tests/test_run_specificity_aware_shared_search_v9.py`

- [ ] **Step 1: Write failing runner tests**

Require exact source counts `38/36/56`, exact screen registry, top-three confirmation only, H200 requirement, aggregate-only reports, full implementation commitments, no overwrite, and zero Mayo/protected-PalsyNet reads or prediction surface.

- [ ] **Step 2: Run the test and verify RED**

- [ ] **Step 3: Implement the runner**

Reuse the authenticated V1/V2 loaders. Screen all 24 candidates at seed 0 for 20 epochs and compare against RSR8-001. Confirmation accepts exactly three locked candidate IDs and seeds 1 or 2.

- [ ] **Step 4: Run runner tests and relevant regressions**

- [ ] **Step 5: Commit runner and tests**

### Task 5: Execute the H200 screen and confirmation

**Files:**
- Generate: `docs/results/artifacts/specificity_aware_shared_router_v9/screen-seed0.json`
- Generate: `docs/results/artifacts/specificity_aware_shared_router_v9/confirm-seed1.json`
- Generate: `docs/results/artifacts/specificity_aware_shared_router_v9/confirm-seed2.json`

- [ ] **Step 1: Sync only the V9 code/tests to a new H200 development directory**

- [ ] **Step 2: Verify `NVIDIA H200`, source commitments, counts, and protected/Mayo audit before training**

- [ ] **Step 3: Run the 24-candidate seed-0 screen**

- [ ] **Step 4: Freeze the top three candidate IDs without editing the registry**

- [ ] **Step 5: Run seeds 1 and 2 confirmation only for those candidates**

- [ ] **Step 6: Recompute the promotion gate from the three machine reports**

### Task 6: Publish an honest V9 decision without disturbing V8

**Files:**
- Create: `docs/results/specificity_aware_shared_router_v9.md`
- Create: `tests/test_specificity_aware_shared_router_v9_release.py`
- Modify only if the gate passes: `docs/model_registry.json`
- Modify only if the gate passes: `docs/CURRENT_DEPLOYMENT_MODEL.md`

- [ ] **Step 1: Write the failing release test**

Require exact report hashes, the same three confirmed candidate IDs, a recomputed gate, aggregate-only content, and an explicit development/not-Mayo boundary. If the gate fails, prove V8 registry/deployment files remain byte-identical.

- [ ] **Step 2: Run the test and verify RED**

- [ ] **Step 3: Write the concise scientific result and decision**

- [ ] **Step 4: Run all V9, V8 deployment, registry, compilation, secret/path, and diff checks**

- [ ] **Step 5: Commit the evidence and final decision**
