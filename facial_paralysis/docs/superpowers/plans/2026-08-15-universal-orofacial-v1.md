# Universal Orofacial Model v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and honestly evaluate one source-blind orofacial-impairment score across PalsyNet and NeuroFace while retaining endpoint-specific auxiliary heads, then lock the candidate before one repeated MEEI diagnostic without changing the released Bell's-palsy 110D or NeuroFace ALS results.

**Architecture:** Every participant receives the same mirror-paired, name-bound Landmark 110D representation aggregated across eligible recordings. Three prespecified candidates share this input: source/class-balanced L2 Logistic, a strongly regularized GroupDRO low-rank MLP, and a low-rank MLP with one source-blind universal head plus PalsyNet/NeuroFace auxiliary heads. Dataset identity is never an input to the universal head; auxiliary heads are reported separately and cannot satisfy the universal gate.

**Tech Stack:** Python 3.9/3.10, NumPy, scikit-learn, PyTorch, existing authenticated `clinical23_v2` caches, Nebius H200, direct repository test harness, JSON aggregate evidence.

---

## Scientific freeze

- [ ] Common target is participant-level affected versus unaffected: PalsyNet facial palsy versus reviewed unaffected; NeuroFace ALS or post-stroke versus healthy control.
- [ ] PalsyNet contributes only the 38 development groups/39 recordings from the authoritative split; protected cache reads, fits, and predictions remain exactly zero.
- [ ] NeuroFace contributes the frozen 36-participant/231-retained-record collection. It is development evidence because its outcomes have already been explored.
- [ ] MEEI is not used for representation, candidate, threshold, or hyperparameter selection. After candidate lock it may be scored once as a repeated, already-exposed external diagnostic, never as untouched validation.
- [ ] Mayo, YFP, PalsyNet protected records, and NeuroFace AU trajectories are not inputs to candidate selection. Optional AU and disease-specific models remain separate released heads, not universal-score evidence.
- [ ] Universal success requires the same universal score to achieve AUROC and balanced accuracy at least `0.90` within both development sources and AUROC at least `0.90` in both leave-one-source-out directions. MEEI must separately achieve AUROC and balanced accuracy at least `0.90` before the bundle may be called cross-institutionally robust.
- [ ] Stop after the three candidates and three fixed seeds. If the gate fails, publish the negative result and keep the existing endpoint-specific models.

## Task 1: Closed participant representation contract

**Files:**

- Create: `src/evaluation/universal_orofacial_v1.py`
- Test: `tests/test_universal_orofacial_v1.py`

- [ ] Write RED tests for exact `(n, 110)` original/mirror arrays, unique opaque participant IDs, binary labels, exact source names, finite values, immutable outputs, and participant rather than recording weighting.
- [ ] Write RED tests proving that repeated recordings are mean-aggregated separately for original and mirror and that recording count is not exposed as a feature.
- [ ] Implement `UniversalDataset`, participant aggregation, source/class-balanced weights, deterministic six-fold source×class stratification, and metric recomputation.
- [ ] Run the new tests GREEN and rerun 110D feature/data regressions.

## Task 2: Three frozen universal candidates

**Files:**

- Create: `src/models/universal_orofacial_v1.py`
- Extend: `src/evaluation/universal_orofacial_v1.py`
- Test: `tests/test_universal_orofacial_models_v1.py`

- [ ] Write RED tests that the universal forward path receives only 110D features and never source IDs, while the auxiliary loss may route labels to a named head.
- [ ] Freeze candidate A as train-fold-only `StandardScaler` plus L2 Logistic (`C=0.01`, `liblinear`, threshold `0.5`) with each source×class group carrying equal total weight.
- [ ] Freeze candidate B as `110→16→1` tanh GroupDRO MLP, full-batch AdamW, 120 epochs, weight decay `0.05`, exponential group step `0.1`, gradient clipping `1.0`, seeds `0/1/2`.
- [ ] Freeze candidate C as the same `110→16` trunk with universal, PalsyNet, and NeuroFace heads; optimize equal-weight universal BCE plus `0.5` auxiliary BCE under the same regularization and seeds.
- [ ] Implement six-fold participant OOF and the two directional leave-one-source-out evaluations with original/mirror probability averaging.
- [ ] Select by worst-source universal AUROC, then worst-source balanced accuracy, then Brier; auxiliary-head metrics are diagnostic only.
- [ ] Run GREEN tests and negative tests for source leakage, held-fold scaler leakage, mirror imbalance, and per-recording weighting.

## Task 3: Authenticated runner and H200 execution

**Files:**

- Create: `scripts/run_universal_orofacial_v1.py`
- Test: `tests/test_run_universal_orofacial_v1.py`

- [ ] Write RED tests for authoritative PalsyNet development-only loading, exact NeuroFace private/cache joins, zero protected reads, closed CLI candidate space, no IDs in the public report, and no MEEI access during selection mode.
- [ ] Build PalsyNet participant 110D values from development caches only and NeuroFace participant 110D values from all retained recordings, using exact cache-byte commitments and the existing mirror transform.
- [ ] Add `--mode develop` and `--mode meei-diagnostic`; diagnostic mode requires an already-locked candidate report and cannot refit, recalibrate, select a threshold, or change preprocessing.
- [ ] Sync only code and deidentified cache/evidence paths already present on H200; run focused tests, then the three-candidate experiment on `NVIDIA H200`.
- [ ] Lock the winner and implementation/data commitments before invoking the MEEI diagnostic exactly once.

## Task 4: Research bundle and release audit

**Files:**

- Create: `docs/results/universal_orofacial_v1.md`
- Create: `docs/results/artifacts/universal_orofacial_v1/report.json`
- Create: `tests/test_universal_orofacial_release_v1.py`

- [ ] Report overall, per-source, worst-source, source-held-out, and repeated-MEEI metrics for the universal head; keep auxiliary endpoint heads in a separate table.
- [ ] Include the unchanged frozen 110D, NeuroFace ALS AU, and Mayo positive-only results as references, not as universal-head evidence.
- [ ] Set `universal_gate_passed=false` unless every frozen gate is met; do not replace the canonical model on a partial pass.
- [ ] Publish only aggregate metrics and hashes; retain identifiers, row labels, probabilities, learned weights, and private manifests outside Git.
- [ ] Run focused and regression tests, py_compile, diff-check, secret/privacy scan, frozen-artifact diff, and clean-worktree verification before commit.

## Evidence interpretation

- GroupDRO is included because it explicitly optimizes worst predefined groups, but strong regularization and early stopping are mandatory in the small-sample regime.
- The multi-head candidate follows dynamic facial-paralysis work that uses auxiliary tasks and shared spatiotemporal structure, but its source-specific heads cannot be reported as proof that one universal score works everywhere.
- MEEI has already been exposed. Any new MEEI result is a repeated diagnostic and the next genuine external gate remains a new cohort such as AFLFP or labeled Mayo with verified controls.
