# Dynamic Landmark Pretraining Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a leak-safe dynamic-window facial-landmark benchmark, pretrain its temporal geometry on actor-disjoint RAVDESS and a separately disclosed Mayo-development stage, then evaluate a frozen candidate set exactly once against a same-data Blendshape-only baseline on group-held-out PalsyNet.

**Architecture:** Each PalsyNet video yields four deterministic, non-overlapping 32-frame windows sampled across the recording. MediaPipe produces one versioned `72 + 23 = 95` stream from the same frames. The neural model always instantiates both branches: `proj_x(x) + proj_dx(dx)` maps the 72-dimensional blendshape block and the 23-dimensional clinical-landmark block independently to 32 dimensions; concatenation gives a fixed 64-dimensional BiGRU input. Masked max and attention pooling are concatenated, then pooled over windows for binary classification. Input blocks are zeroed for equal-shape ablations. A paper-style classical arm uses robust extrema, bilateral synchrony, and train-only healthy-reference Mahalanobis/Wasserstein distances. RAVDESS and MediaPipe use explicit source adapters into one generic semantic order; OpenFace projection/scaling is never assumed numerically interchangeable with MediaPipe.

**Tech Stack:** Python 3.10 in `/Users/williamqiu/.cache/facial-paralysis/mediapipe-py310` with MediaPipe `0.10.35` and OpenCV for extraction; `/Users/williamqiu/opt/anaconda3/bin/python3` with NumPy, SciPy, scikit-learn, and PyTorch for analysis/training; repository script-style tests; locked group-nested cross-validation; class-stratified paired bootstrap.

**Research basis:**

- Rao et al. model bilateral landmark trajectories with correlation, Gaussian distributions, Mahalanobis distance, and Wasserstein distance: https://pmc.ncbi.nlm.nih.gov/articles/PMC12584918/
- Heinrich et al. reduce 478 MediaPipe landmarks to informative bilateral angle regions, supporting derived regional geometry rather than raw-coordinate flattening: https://pmc.ncbi.nlm.nih.gov/articles/PMC13113261/
- Guan et al. use disease-specific eye and mouth contours and extrema with prospective external validation: https://www.nature.com/articles/s41746-025-02063-6
- Official RAVDESS OpenFace tracking supplies 2,452 actor-identified trials with 68-point trajectories for actor-disjoint healthy-motion pretraining: https://zenodo.org/records/3255102

**Locked scientific boundaries:**

- PalsyNet has no standardized facial-action protocol. This is a **dynamic-window**, not action-aware, experiment; true action-aware HB validation remains future Mayo/FACES work.
- Until the identity audit confirms one independent person per clip, the split and claim are **video-held-out**. Hash uniqueness alone is not identity proof.
- PalsyNet contains acquisition/content confounding. Results can show a PalsyNet modality increment, not disease mechanism or Mayo generalization.
- All candidates must be registered before any real outer-test prediction. After outer results are visible, new PalsyNet methods are exploratory and require a new external cohort for confirmation.
- The existing Mayo cohort is already method-development exposed. If used for SSL, it is permanently marked development-only; future independent HB evidence must come from new people. Mayo labels/controls remain unavailable, so no current HB accuracy claim is possible.
- Cross-paper headline numbers are not comparable across datasets, labels, splits, and metrics.

---

## Task 1: Freeze governance, identity groups, caches, and split contracts with failing tests

**Files:**

- Modify: `.gitignore`
- Create: `facial_paralysis/tests/test_dynamic_window_data.py`
- Create: `facial_paralysis/tests/test_nested_group_splits.py`
- Create: `facial_paralysis/src/datasets/dynamic_landmark.py`
- Create: `facial_paralysis/src/evaluation/nested_group_cv.py`
- Create: `facial_paralysis/scripts/audit_palsynet_identity.py`
- Generate outside Git: `facial_paralysis/outputs/palsynet_identity_audit/`

- [ ] Add exact ignore rules for the Python extraction environment, PalsyNet V2 NPZ cache, local identity montages, RAVDESS-derived trajectories, and SSL checkpoints while leaving small de-identified JSON results trackable.
- [ ] Write a failing window test requiring at least 128 source frames and exactly four ordered, contiguous, non-overlapping 32-frame windows. Reject shorter videos rather than repeat frame indices.
- [ ] Write a failing one-recording-per-NPZ cache test for `features (4,32,95) float32`, `valid_mask (4,32) bool`, strictly increasing `timestamps (4,32)` with unit exactly `seconds`, scalar `source_frame_count`, exact deterministic source frame indices, exact schema/name metadata, unique de-identified `recording_id`, possibly shared de-identified `group_id`, label, and one source hash. CV groups by `group_id`; if a person has multiple recordings, probabilities are aggregated to one group before scoring.
- [ ] Write fail-closed tests for a wrong schema, partial metadata, non-finite valid values, non-adjacent frames within a window, less than 90% valid coverage, or delta computation across a detector gap.
- [ ] Audit cross-video identity before locking folds: use the existing frozen video embeddings to rank near-neighbor pairs, generate local multi-timepoint contact sheets, and manually review every video/top pair. Store no face image or raw filename in Git.
- [ ] If a repeated identity is found, merge videos into one de-identified group. Otherwise record that the dataset card's one-video-per-person statement was checked by embedding plus visual audit and upgrade the claim to person-held-out.
- [ ] Freeze one deterministic stratified 5-fold outer group split and a 4-fold inner group split per outer train set. Test zero overlap and exact coverage.
- [ ] Add an API guard that forbids outer test indices in scaler/prototype/hyperparameter/epoch selection or trainer validation. No real outer prediction is permitted before Task 7.
- [ ] Run the two tests and observe the expected import/implementation failures before production code.

Commands:

```bash
/Users/williamqiu/opt/anaconda3/bin/python3 facial_paralysis/tests/test_dynamic_window_data.py
/Users/williamqiu/opt/anaconda3/bin/python3 facial_paralysis/tests/test_nested_group_splits.py
```

## Task 2: Build and audit the PalsyNet `clinical23_v2` window cache

**Files:**

- Create: `facial_paralysis/scripts/build_palsynet_v2_windows.py`
- Modify: `facial_paralysis/src/datasets/dynamic_landmark.py`
- Test: `facial_paralysis/tests/test_dynamic_window_data.py`
- Generate outside Git: `/Users/williamqiu/Desktop/Harvard/Mayo-Clinic/facial_paralysis/data/external/palsynet/derived/clinical23_v2_windows/`

- [ ] Create the dedicated Python 3.10 environment and install pinned `mediapipe==0.10.35`, NumPy, and OpenCV without modifying the Anaconda or system environments. Save a local environment lock.
- [ ] Enumerate exactly 49 canonical MP4s: 27 affected and 22 unaffected. Verify the observed audit values before extraction: 49 unique SHA-256 hashes, 177,511 frames, 30 Hz, 98.617 total minutes, minimum 172 frames.
- [ ] Do not call `action_bundle._read_frames(max_frames=...)`, which reads entire long videos and produces isolated samples. Use `CAP_PROP_FRAME_COUNT`, seek to each locked start, sequentially decode 32 frames, and verify the reported source frame index after every read.
- [ ] Run one `MediaPipeFeatureExtractor(model_path=..., landmark_features="clinical23", capture_mirrored=None)` instance so the 72- and 23-dimensional blocks come from identical frames.
- [ ] Persist one transactional cache per recording, each with a unique de-identified `recording_id` and an identity-audit-derived `group_id`. The manifest contains source hashes, FPS/timestamps, frame indices, producer/schema versions, coverage, and within-window variation. Keep raw public videos unchanged.
- [ ] Store nuisance-only audit fields separately: duration, dimensions, file size/bitrate proxy, detection rate, luminance, face-scale proxy, roll proxy, and sampled frame-difference proxy. Never feed label-derived values into this probe.
- [ ] Verify at least 47/49 usable videos, each retained video at least 90% valid, and at least 95% of retained videos with non-zero landmark variation. Stop before training if gates fail.

Command:

```bash
/Users/williamqiu/.cache/facial-paralysis/mediapipe-py310/bin/python \
  facial_paralysis/scripts/build_palsynet_v2_windows.py \
  --data-root /Users/williamqiu/Desktop/Harvard/Mayo-Clinic/facial_paralysis/data/external/palsynet/data \
  --model-path /Users/williamqiu/Desktop/Harvard/Mayo-Clinic/facial_paralysis/data/mediapipe_out/_models/face_landmarker.task \
  --identity-manifest /Users/williamqiu/.config/superpowers/worktrees/Mayo-Clinic/landmark-fusion/facial_paralysis/outputs/palsynet_identity_audit/identity_manifest.json \
  --output-root /Users/williamqiu/Desktop/Harvard/Mayo-Clinic/facial_paralysis/data/external/palsynet/derived/clinical23_v2_windows
```

## Task 3: Implement the paper-style classical candidates without viewing outer results

**Files:**

- Create: `facial_paralysis/tests/test_trajectory_features.py`
- Create: `facial_paralysis/src/preprocessing/trajectory_features.py`
- Create: `facial_paralysis/scripts/run_dynamic_landmark_classical.py`

- [ ] First write failing tests for masked median, IQR, range, maximum absolute per-second derivative, bilateral correlation, amplitude ratio, and lagged cross-correlation on synthetic trajectories with known values.
- [ ] Use timestamps and compute derivatives only when both frames are valid and adjacent in source time; normalize derivatives per second.
- [ ] Add explicit capture-side eye/brow/mouth pair and region maps from the frozen 23-column schema. Do not infer patient-left/patient-right while mirroring is unknown.
- [ ] Implement a train-only healthy prototype using shrinkage covariance for Mahalanobis distance and region-wise Wasserstein distance. Test that `fit` receives only the current inner/outer training controls.
- [ ] Define the complete classical registry before outer evaluation: nuisance-only, Blendshape summaries, Landmark summaries, Fusion summaries, and Fusion plus Rao-style healthy-reference distances.
- [ ] Fit `StandardScaler` and L2 logistic regression only inside each inner fold. Use a fixed `C` grid. AUROC is primary; threshold metrics use the fixed probability threshold `0.5` and remain secondary.
- [ ] Permit synthetic tests and real-cache inner-fold smoke runs only. The script must refuse outer scoring until supplied the frozen Task 7 experiment-registry hash.

## Task 4: Implement the equal-shape neural candidates without viewing outer results

**Files:**

- Create: `facial_paralysis/tests/test_dynamic_landmark_model.py`
- Create: `facial_paralysis/tests/test_dynamic_landmark_benchmark.py`
- Create: `facial_paralysis/src/models/dynamic_landmark.py`
- Create: `facial_paralysis/src/training/dynamic_landmark_benchmark.py`
- Create: `facial_paralysis/scripts/run_dynamic_landmark_benchmark.py`

- [ ] Write failing tests that every arm has identical parameter names/shapes, zeroed blocks cannot affect output, and active blocks receive gradients.
- [ ] Freeze the tensor contract: for each valid 30-Hz frame, `blend_latent = proj_bs_x(bs) + proj_bs_dx(dbs_dt)` and `landmark_latent = proj_lm_x(lm) + proj_lm_dx(dlm_dt)`, where all four projections are bias-free and output 32 values. Concatenate to exactly 64 values before a one-layer BiGRU with hidden width 32 per direction.
- [ ] Add masked max and learned-attention pools over the BiGRU sequence, concatenate them, project to 32 values, mean only over present windows, then apply a small binary head.
- [ ] Compute first differences only across adjacent valid source frames. Test interior gaps, all-masked windows, schema-correct mirror swapping/signed negation, finite-value gates, and deterministic inference.
- [ ] Train inner models to one fixed maximum epoch, choose the median best epoch from inner folds, reinitialize, and train the complete outer-train set for that epoch count without any outer validation input.
- [ ] Define three random-init neural candidates now: Blendshape-only, Landmark-only, and Fusion. All use seeds `0,1,2`, identical folds, budgets, fold-local downstream scalers, and probability-level seed ensembling.
- [ ] Keep the historical PalsyNet `0.860` pooled AUC only as a non-nested reference. It used outer-test-as-validation and is not the comparison baseline.

## Task 5: Build an explicit cross-topology RAVDESS semantic adapter

**Files:**

- Create: `facial_paralysis/tests/test_openface68_semantic.py`
- Create: `facial_paralysis/src/preprocessing/semantic_landmarks.py`
- Create: `facial_paralysis/src/preprocessing/openface68_semantic.py`
- Create: `facial_paralysis/scripts/prepare_ravdess_semantic23.py`
- Generate outside Git: `facial_paralysis/data/external/ravdess_facial_tracking/derived_semantic23/`

- [ ] Define a third generic `semantic23_v1` order with source-neutral side-A/side-B names, units, signs, and measurement definitions. Add explicit `clinical23_v2 -> semantic23_v1` and `openface68 -> semantic23_v1` adapters.
- [ ] Write failing tests for exact OpenFace 68 topology, translation/scale/roll invariance, symmetry, signed and absolute perturbations, V2-to-common ordering, and malformed coordinates.
- [ ] Match the existing V2 measurement semantics exactly where weights will be shared; do not silently replace V2's height-times-width eye measure with a polygon-area measure.
- [ ] Filter frames using finite required coordinates and OpenFace confidence at least `0.80`; preserve timestamps and detector gaps without interpolation.
- [ ] Process all 2,452 CSVs into actor-identified trial caches. Reproduce the official local audit: 417,163,019-byte archive, MD5 `5753bbc64a9a790f8a8d3e03cba526ee`, 24 actors, 299,854 frames, one header, and no empty trials.
- [ ] Store source-specific scaling and input-adapter metadata. A same-length 23-vector is not evidence of numerical compatibility.

## Task 6: Pretrain with masked spans and explicit source-transfer boundaries

> **2026-07-15 execution amendment:** The source builders and SSL core are complete, but real pretraining now follows the reviewed bridge plan in `2026-07-15-dynamic-landmark-ssl-bridge.md`. That plan supersedes the older extraction/window details below wherever they differ: all 48 retained Mayo videos are homogeneously re-extracted; all 2,452 RAVDESS trials are retained; both stages use canonical 30-Hz indices with step 1 and local window timestamps; Mayo uses 16 fixed uniform packets per recording; and SSL authorization binds a private bridge receipt through manifest/stage-evidence v2.

**Files:**

- Create: `facial_paralysis/tests/test_dynamic_landmark_ssl.py`
- Create: `facial_paralysis/src/pretraining/dynamic_landmark_ssl.py`
- Create: `facial_paralysis/scripts/build_mayo_ssl_cache.py`
- Create: `facial_paralysis/scripts/pretrain_dynamic_landmarks.py`
- Generate outside Git: `facial_paralysis/outputs/dynamic_landmark/pretraining/`
- Generate outside Git: `facial_paralysis/outputs/dynamic_landmark/mayo_exposure_manifest.json`

- [ ] First write failing tests for contiguous span masking, masked-only SmoothL1 reconstruction, actor/recording split isolation, 30-Hz resampling, train-only source scalers, full 64-dimensional GRU input compatibility, checkpoint names/shapes, and permitted weight transfer.
- [ ] Use masked-span reconstruction only. Do not use a BiGRU next-step objective because its representation already sees future frames.
- [ ] RAVDESS stage: actor-disjoint 32-frame windows at 30 Hz, a RAVDESS-only scaler/input adapter, zero 32-dimensional base latent, and landmark latent in the second half of the full 64-dimensional shared GRU input. Transfer only shared GRU/pooling weights downstream unless cross-detector agreement is independently proven.
- [ ] Freeze the expanded aggregate inventory before extraction: 65 sessions total; 50 video-bearing; 15 without video. Of the video sessions, exclude one exact duplicate copy and one 1.13-second QC-only clip, leaving 48 unique long videos. Existing complete V2 exports cover 13 of those for audit only; do not reuse them. Re-extract all 48 retained long videos homogeneously in MediaPipe VIDEO mode without annotated preview video.
- [ ] Save compact 60-Hz normalized landmark/blendshape tensors, masks, timestamps, and transform metadata rather than new multi-gigabyte CSV/preview artifacts; then downsample the SSL view to 30 Hz. Group only what provenance proves and publish an ignored manifest using salted IDs/fingerprints rather than recording names. Mark every exposed current Mayo recording permanently development-only.
- [ ] Keep the 7 usable ARKit-only sessions (8 nonduplicate 52-blendshape trajectories, 58,054 rows) in a separate auxiliary blendshape SSL pool; never fabricate Landmark columns or concatenate them directly with MediaPipe geometry. Keep the remaining 8 index/metadata-only sessions isolated until source video or depth decoding becomes available.
- [ ] Use a Mayo-only scaler for SSL. The compatible MediaPipe landmark projections plus shared GRU/pooling may warm-start downstream, but the pretraining scaler is never substituted for the PalsyNet fold scaler.
- [ ] In every outer fold, random and pretrained candidates share one scaler fitted only to that outer-train PalsyNet fold. The only intended difference is initialization.
- [ ] Report actor-held-out RAVDESS and recording-held-out Mayo masked reconstruction loss against untrained and train-mean baselines. Recording-held-out is not patient-held-out when identity grouping is unavailable.
- [ ] Save two preregisterable checkpoints: RAVDESS-only and RAVDESS-then-Mayo. Reject partial/mismatched state dictionaries.

## Task 7: Register every candidate, then run one unified outer evaluation

**Files:**

- Create: `facial_paralysis/configs/dynamic_landmark_experiment_registry.json`
- Modify: `facial_paralysis/scripts/run_dynamic_landmark_classical.py`
- Modify: `facial_paralysis/scripts/run_dynamic_landmark_benchmark.py`
- Generate: `facial_paralysis/outputs/dynamic_landmark/locked_outer_results.json`

- [ ] Before any outer prediction, freeze and hash the data manifest, identity groups, outer/inner folds, code commit, hyperparameter grids, seeds, metric code, and the complete candidate set: nuisance-only classical; classical BS/LM/Fusion/Rao-Fusion; neural BS/LM/Fusion-random/Fusion-RAVDESS/Fusion-RAVDESS-Mayo.
- [ ] Make the outer evaluator refuse a dirty/mismatched registry, cache hash, split hash, checkpoint hash, missing candidate, or prior partial outer-result file.
- [ ] For every outer fold, perform selection using only its inner folds, refit on the outer train group, and predict the outer test group once. Do not use outer results to alter candidates or rerun a favored configuration.
- [ ] Primary metric: pooled group OOF AUROC. Secondary: PR-AUC, Brier score, and fixed-0.5 balanced accuracy/sensitivity/specificity. Report all fold probabilities and metrics.
- [ ] Compute a class-stratified paired 5,000-repeat group bootstrap for fixed OOF predictions. Label its interval descriptive/exploratory because it does not include training-set/fold uncertainty.
- [ ] Primary comparison: neural Fusion-random minus neural Blendshape-only. Secondary comparisons: Landmark-only, RAVDESS warm-start, and RAVDESS-Mayo warm-start. Report effect size even when negative.
- [ ] Research-candidate gate only: Fusion AUROC at least Blendshape `+0.03`, bootstrap `P(delta>0) >= 0.95`, balanced accuracy non-decreasing, and sensitivity/specificity each no worse than `-0.05`. Formal efficacy still requires an independent external test set.
- [ ] After this file exists, any dense-angle, Transformer, GNN, or hyperparameter revision on PalsyNet is exploratory and cannot replace the locked result.

Command:

```bash
/Users/williamqiu/opt/anaconda3/bin/python3 \
  facial_paralysis/scripts/run_dynamic_landmark_benchmark.py \
  --registry facial_paralysis/configs/dynamic_landmark_experiment_registry.json \
  --cache-root /Users/williamqiu/Desktop/Harvard/Mayo-Clinic/facial_paralysis/data/external/palsynet/derived/clinical23_v2_windows \
  --output facial_paralysis/outputs/dynamic_landmark/locked_outer_results.json
```

## Task 8: Verify, review, and document without expanding claims

**Files:**

- Modify: `facial_paralysis/docs/landmark_research_20260713.md`
- Modify: `facial_paralysis/docs/model_design.md`
- Modify: `facial_paralysis/docs/training_runs.md`
- Modify: `facial_paralysis/autoresearch_fp/FINDINGS.md`

- [ ] Run every focused test, then the full script-style repository test suite with the Anaconda runtime.
- [ ] Validate result JSON fingerprints and rerun extraction/inner-training smoke tests; do not rerun outer evaluation after viewing its results.
- [ ] Run `git diff --check` and inspect `git status`. Confirm no raw dataset, face montage, clinical identifier, secret, environment, large cache, or checkpoint is tracked.
- [ ] Obtain independent review of identity grouping, source confounds, inner/outer isolation, timestamp/delta logic, mirror semantics, cross-topology adapters, SSL exposure, checkpoint transfer, metrics, and claims.
- [ ] Record negative and positive results. Keep deployment unchanged. State that current Mayo is development-exposed and that HB accuracy remains unknown.
- [ ] Commit implementation, configs, tests, documentation, and only small de-identified result manifests after all verification passes.

Focused verification commands:

```bash
/Users/williamqiu/opt/anaconda3/bin/python3 facial_paralysis/tests/test_dynamic_window_data.py
/Users/williamqiu/opt/anaconda3/bin/python3 facial_paralysis/tests/test_nested_group_splits.py
/Users/williamqiu/opt/anaconda3/bin/python3 facial_paralysis/tests/test_trajectory_features.py
/Users/williamqiu/opt/anaconda3/bin/python3 facial_paralysis/tests/test_dynamic_landmark_model.py
/Users/williamqiu/opt/anaconda3/bin/python3 facial_paralysis/tests/test_dynamic_landmark_benchmark.py
/Users/williamqiu/opt/anaconda3/bin/python3 facial_paralysis/tests/test_openface68_semantic.py
/Users/williamqiu/opt/anaconda3/bin/python3 facial_paralysis/tests/test_dynamic_landmark_ssl.py
git diff --check
```
