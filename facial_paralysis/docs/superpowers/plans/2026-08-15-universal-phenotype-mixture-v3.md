# Universal Phenotype Mixture v3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one source-blind facial-weakness system that preserves the locked unilateral-asymmetry signal while adding full-cohort action-unit and temporal-capacity experts, then select it by the worst participant-disjoint dataset result rather than pooled accuracy.

**Architecture:** The system exposes one affected probability but internally models distinct phenotypes with six auditable heads: global Landmark-110, action-specific Landmark-110, common Fusion-398, raw MediaPipe temporal, AU bilateral-capacity, and AU palsy-capacity. Fusion consumes expert probabilities, reliability and availability only, never a source or dataset identifier. Training-only healthy-control alignment reduces acquisition shift, while all learned fusion remains nested inside the outer participant folds.

**Tech Stack:** Python 3.10, NumPy, scikit-learn, PyTorch, Py-Feat 0.6.2, H200 CUDA, canonical JSON/NPZ evidence contracts.

---

## Scientific protocol

- Development sources: identity-reviewed PalsyNet development only, all NeuroFace participants, and the already-exposed MEEI diagnostic only when explicitly labelled as adaptive development evidence.
- Protected PalsyNet outer participants remain unread with exact zero cache, fit and prediction counters.
- Unit of splitting, weighting, bootstrapping and reporting is the participant.
- Primary selection objective is lexicographic: maximize minimum source AUROC, then minimum source balanced accuracy, then minimize worst-source Brier score.
- Search cannot use source identifiers as model inputs, per-source thresholds, held-out labels, or the PalsyNet protected partition.
- Every outer prediction uses one threshold chosen only from the corresponding inner participant-OOF scores; no source-specific threshold is allowed. The final candidate will also be rechecked as a three-seed probability ensemble so seed variance is not mistaken for representation gain.
- A result over 0.90 on reused development cohorts remains development evidence, not clinical or prospective validation.
- The search stops after the frozen candidate registry has been evaluated once; adding candidates after seeing outcomes requires a new version.

### Task 1: Full-cohort NeuroFace AU cache

**Files:**
- Create: `scripts/extract_neuroface_full_au_v2.py`
- Create: `tests/test_extract_neuroface_full_au_v2.py`
- Reuse: `src/datasets/neuroface_au_v1.py`

- [x] Write failing tests for exact 36-participant/261-record membership, all three cohorts, archive safety, immutable cache bytes and no outcome-dependent extractor controls.
- [x] Run the test and confirm failure because the v2 extractor does not exist.
- [x] Implement the minimal manifest-bound stride-3 Py-Feat extractor by generalizing the audited v1 path without changing the frozen 20-AU order or model resources.
- [x] Run unit and v1 extractor regression tests.
- [x] Transfer only the three authenticated video archives to the H200 private input directory and extract all retained recordings.
- [x] Publish only aggregate evidence and frozen report hashes; keep the private collection manifest and participant rows off Git.

### Task 2: Universal participant evidence contract

**Files:**
- Create: `src/datasets/universal_phenotype_v3.py`
- Create: `tests/test_universal_phenotype_dataset_v3.py`

- [x] Write failing tests for participant aggregation, action bags, explicit modality availability, immutable arrays, unique opaque identities and cross-source identity rejection.
- [x] Run RED and confirm the missing module is the failure.
- [x] Implement strict `PhenotypeDataset` validation and participant-level aggregation from Landmark-110, common MediaPipe sequences, AU statistics and fixed-length AU temporal action bags.
- [x] Verify one participant appears in exactly one outer fold and missing AU is represented by availability, never zero-imputation ambiguity.

### Task 3: Phenotype experts and source-blind fusion

**Files:**
- Create: `src/models/universal_phenotype_v3.py`
- Create: `tests/test_universal_phenotype_models_v3.py`

- [x] Write failing tests for unilateral, capacity and common-temporal expert interfaces; forbid dataset/source input; require one universal probability.
- [x] Implement neural candidates: linear/residual MIL, dilated MediaPipe TCN, AU TCN, action attention, a cross-action Set Transformer, a separate per-action 110D expert, clinically scoped AU heads, healthy-control source alignment and action-instance dropout.
- [x] Implement fixed source-blind fusion candidates: max probability, reliability-weighted noisy-OR and confidence weighting using only expert output/reliability/availability.
- [x] Test missing modality inertness, probability bounds and distinct frozen fusion semantics; learned source-blind routing remains available for nested comparison.

### Task 4: Bounded autoresearch loop

**Files:**
- Create: `src/evaluation/universal_phenotype_v3.py`
- Create: `scripts/run_universal_phenotype_v3.py`
- Create: `tests/test_universal_phenotype_evaluation_v3.py`
- Create: `tests/test_run_universal_phenotype_v3.py`

- [x] Write failing tests for the frozen eight-candidate registry, nested participant-disjoint source-phenotype splits and worst-source selection.
- [x] Evaluate the bounded large-architecture registry with fixed seeds and retain private candidate evidence plus public aggregates.
- [x] Stop after the frozen registry; no candidate was added based on protected or outer-test labels.
- [x] Implement a single source-blind inner-OOF balanced-accuracy threshold with sensitivity-first deterministic tie-breaking.
- [x] Run the architecture and representation searches on the verified H200 and retain failures as decision evidence.
- [x] Lock the evidence-routed Universal Clinical Router v4 because no single shared candidate passed all source gates.

### Task 5: Stress tests and release

**Files:**
- Create: `docs/results/universal_phenotype_mixture_v3.md`
- Create: `docs/results/artifacts/universal_phenotype_mixture_v3/report.json`
- Create: `tests/test_universal_phenotype_release_v3.py`

- [x] Evaluate per-cohort AUROC, accuracy, balanced accuracy, sensitivity, specificity and Brier without pooling participants across cohorts.
- [x] Establish a label-independent MEEI action-timing path from audio prompts: Whisper large-v3-turbo recovered all 60 transcripts and the frozen eight-cue parser accepted 57/60 recordings (95% coverage, including 10/10 controls); three ambiguous prompts remain excluded rather than motion-imputed.
- [x] Evaluate cue-aligned MEEI AU distributions and a nested 145-rule Landmark/AU comparison; reject both because they materially underperform Landmark sequence geometry.
- [x] Label NeuroFace and MEEI as exposed/adaptive participant-disjoint development evidence, never untouched external validation.
- [x] Confirm protected PalsyNet reads/fits/predictions are zero.
- [x] Run final affected regressions, static checks, secret scan and public identifier/path scan.
- [ ] Commit explicitly scoped files on `codex/universal-phenotype-v3`; do not merge or push automatically.
