# Landmark Fusion Implementation Plan

> **For Codex:** Execute this plan test-first. Keep the existing 72-dimensional blendshape path backward compatible and do not promote a new deployment configuration without a paired ablation.

**Goal:** Add a production-quality, 2D similarity-normalized 23-dimensional clinical facial-landmark stream to the existing MediaPipe feature extractor, expose an explicit 72-vs-23-vs-95 feature schema, and create a reproducible Mayo landmark-trajectory audit that is ready for later patient-level training.

**Architecture:** MediaPipe remains the single upstream detector. Its 52 blendshapes plus mirrored deltas form the existing 72-dimensional stream. Its face landmarks feed a separate clinical geometry transform containing bilateral eye, brow, and mouth measurements; the fused per-frame vector is 95-dimensional. The temporal encoder remains schema-agnostic and consumes the selected feature vector. Raw 478-point/GNN modeling is deferred until patient-level labels and a defensible validation split exist.

**Tech Stack:** Python, NumPy, MediaPipe Tasks, OpenCV, PyTorch, the repository's lightweight `_testlib` test runner.

**Research basis:**

- Parra-Dominguez et al. use 29 landmark-derived angles, distances, and ratios before an MLP, supporting an interpretable geometry baseline: https://www.mdpi.com/2075-4418/12/7/1528
- Kim et al. use bilateral landmark displacement ratios during brow raise and smile, supporting rest-relative dynamics: https://pmc.ncbi.nlm.nih.gov/articles/PMC4634507/
- Guarin et al. use disease-specific landmark localization and measurements including brow height, palpebral fissure height, and commissure excursion: https://pmc.ncbi.nlm.nih.gov/articles/PMC7362997/
- Rao et al. model normalized landmark trajectories over standardized facial cues, supporting a future dynamic landmark objective: https://pubmed.ncbi.nlm.nih.gov/40333095/
- Heinrich et al. use all 478 MediaPipe landmarks but reduce bilateral pairs to informative facial regions, warning against assuming every coordinate is equally useful: https://pmc.ncbi.nlm.nih.gov/articles/PMC13113261/
- Oo et al. report that structured 478-landmark input is useful and that multimodal fusion outperforms individual modalities in a small video cohort: https://arxiv.org/abs/2503.10371

---

## Task 1: Freeze the clinical landmark contract with failing tests

**Files:**

- Create: `facial_paralysis/tests/test_clinical_landmarks.py`
- Test: `facial_paralysis/tests/test_clinical_landmarks.py`

1. Build a deterministic synthetic 478-point face fixture containing the required eye, brow, midline, and mouth points.
2. Add tests for the exact 23-name schema and `float32` output.
3. Add translation, uniform-scale, and in-plane-roll invariance tests.
4. Add tests that a symmetric face has zero bilateral differences and a perturbed mouth corner creates the expected signed/absolute asymmetry.
5. Add validation tests for malformed arrays, non-positive image dimensions, and non-finite required landmarks.
6. Run the test script and confirm it fails because the production module does not exist.

Command:

```bash
/Users/williamqiu/opt/anaconda3/bin/python3 facial_paralysis/tests/test_clinical_landmarks.py
```

## Task 2: Implement the reusable 23-dimensional landmark transform

**Files:**

- Create: `facial_paralysis/src/preprocessing/clinical_landmarks.py`
- Modify: `facial_paralysis/scripts/clinical_landmark_features.py`
- Test: `facial_paralysis/tests/test_clinical_landmarks.py`

1. Move the landmark index contract and pose normalization into the production preprocessing package.
2. Validate shape, image dimensions, and the finite values of every required landmark; fail explicitly instead of silently turning a broken frame into a plausible all-zero vector.
3. Preserve the existing 23 clinical measurements and their order for compatibility with the completed web ablation.
4. Convert the old script into a thin compatibility/CLI wrapper around the production module.
5. Run the new tests and confirm they pass.

## Task 3: Add explicit landmark modes to MediaPipe extraction

**Files:**

- Create: `facial_paralysis/tests/test_action_bundle_landmarks.py`
- Modify: `facial_paralysis/src/preprocessing/action_bundle.py`
- Test: `facial_paralysis/tests/test_action_bundle_landmarks.py`

1. Write failing unit tests around a detector-independent feature assembly helper.
2. Define explicit modes: `none` (72-dimensional layout), `legacy5` (old 5-dimensional geometry), and `clinical23` (new clinical geometry).
3. Preserve `with_geometry=True` as a deprecated alias for `legacy5`, while rejecting conflicting arguments.
4. Ensure feature names and dimensions are stable and include a schema identifier.
5. Ensure missing landmark output invalidates a frame only when a landmark mode is requested.
6. Run the tests and confirm green.

## Task 4: Make cache generation schema-aware and fail closed

**Files:**

- Modify: `facial_paralysis/src/preprocessing/action_bundle.py`
- Modify: `facial_paralysis/src/datasets/patient_multistream.py`
- Create: `facial_paralysis/tests/test_feature_schema.py`

1. Add `--landmark-features {none,legacy5,clinical23}` to cache generation.
2. Persist `mp_feature_names`, `mp_feature_schema`, and `mp_feat_dim` in each bundle.
3. Validate all loaded bundles in a record share the requested dimension/schema, with backward compatibility for old dimension-only bundles.
4. Add tests that a mismatched schema or dimension is rejected before model training.

## Task 5: Audit the already-extracted Mayo landmark trajectories

**Files:**

- Create: `facial_paralysis/scripts/audit_mayo_landmark_trajectories.py`
- Create: `facial_paralysis/tests/test_mayo_landmark_audit.py`
- Generate: `facial_paralysis/outputs/landmark_fusion/mayo_clinical23_audit.json`

1. Add a streaming CSV reader that groups the 478 rows belonging to each frame without loading multi-gigabyte inputs into memory.
2. Compute clinical23 features for every valid frame and summarize detection coverage, per-feature finite rate, robust range, and temporal variation per video.
3. Detect exact duplicate trajectory sources and very short videos; report them rather than silently treating them as independent patients.
4. Add rest-reference, peak delta, AUC, velocity, time-to-peak, recovery, and bilateral trajectory-correlation summaries.
5. Run against the canonical Mayo `data/mediapipe_out/*/landmarks.csv` directory from the main workspace.
6. Treat this as a data-readiness audit, not an HB performance estimate, because these videos still lack independent patient labels and controls.

## Task 6: Verify and record the ablation boundary

**Files:**

- Modify: `facial_paralysis/autoresearch_fp/FINDINGS.md`
- Modify: `facial_paralysis/docs/model_design.md`

1. Document the three feature contracts: landmark-only 23, blendshape-only 72, and fused 95.
2. Record the existing web result as historical evidence only: 72-dimensional control versus 95-dimensional fusion was effectively flat within seed variation.
3. State the next valid experiment: patient-held-out comparison on labeled Mayo actions with identical seeds and training budgets.
4. Explicitly defer raw 478/GNN and Transformer capacity until the validation set can support it.
5. Run the focused tests plus existing script-style pipeline tests.

Commands:

```bash
/Users/williamqiu/opt/anaconda3/bin/python3 facial_paralysis/tests/test_clinical_landmarks.py
/Users/williamqiu/opt/anaconda3/bin/python3 facial_paralysis/tests/test_action_bundle_landmarks.py
/Users/williamqiu/opt/anaconda3/bin/python3 facial_paralysis/tests/test_feature_schema.py
/Users/williamqiu/opt/anaconda3/bin/python3 facial_paralysis/tests/test_mayo_landmark_audit.py
/Users/williamqiu/opt/anaconda3/bin/python3 facial_paralysis/tests/test_pipeline.py
/Users/williamqiu/opt/anaconda3/bin/python3 facial_paralysis/tests/test_temporal_pool.py
```

## Task 7: Independent review and clean handoff

1. Ask a reviewer agent to check scientific leakage, feature handedness, pose invariance, compatibility, and whether claims exceed the available labels.
2. Apply only evidence-backed corrections and re-run all focused tests.
3. Inspect `git diff --check`, `git status`, and the generated audit artifact.
4. Keep the deployment config unchanged. Commit the isolated branch only after verification.

## Execution outcome — 2026-07-13

- Implemented separate frozen `legacy_clinical23_v1` reproduction and production
  `clinical23_v2` schemas; the V1 function matches the pinned July-10 transform
  exactly on all 1,453 stored web landmark records.
- Added the fixed-width blendshape-only, landmark-only, and fusion ablations,
  including a raw-feature sensitivity run with per-seed results.
- Re-audited all 15 local Mayo exports at stride 1: 87,732 valid stored landmark
  frame groups over an 87,988-frame source timeline, 256 explicit missing rows,
  and zero audit failures.
- Added exact producer/loader schema validation, side/mirror provenance, finite
  value gates, transactional derived-output publishing, and correct packed-GRU
  handling of interior detector gaps.
- Migrated all first-party MediaPipe bundle writers to the common schema-aware
  payload contract. Historical dimension-only caches are rejected on implicit
  reuse and must be explicitly audited or rebuilt.
- Verified 133/133 script-style tests. Deployment configuration remains
  unchanged; no Mayo classification/HB claim is made without labels and controls.
