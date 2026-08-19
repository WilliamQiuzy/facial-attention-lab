# Formal SSL and Modality Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the preregistered three-seed, 30-epoch dynamic-geometry pretraining experiment and add a scientifically fair Mayo-only blendshape/landmark/fusion SSL comparison.

**Architecture:** This document is a formal addendum to the approved 2026-07-15 bridge plan. The scientific schedule for the fusion experiment remains RAVDESS semantic23 pretraining followed by Mayo 95-dimensional adaptation, but the v4 Mayo generation and the arm-bound config schema change producer bytes and commitments, so bridge, frozen inputs, receipts, and formal outputs are all rebuilt and old schema artifacts are rejected. A separate preregistered Mayo-only ablation uses the same complete 95-dimensional reconstruction target for all three arms and changes only which input block is zeroed. Because RAVDESS has no equivalent 72-dimensional MediaPipe blendshape stream, it does not initialize this fairness comparison. Unlabeled Mayo videos cannot yield HB accuracy.

**Tech Stack:** PyTorch 2.2.1 CPU, NumPy 1.26.4, the existing `DynamicLandmarkSSLModel`, HMAC-bound frozen inputs/checkpoints/results, seeds 0/1/2, AdamW, 30 epochs.

---

## Frozen experiment contract

- Formal fusion run: existing `RAVDESS -> Mayo` bridge, seeds `0,1,2`, 30 epochs per stage, exact `deterministic_microbatch_full_partition_64` policy, unchanged held-out actors/recordings and mask policy.
- Mayo ablation reuses `ARM_BLENDSHAPE`, `ARM_LANDMARK`, and `ARM_FUSION` from `src/models/dynamic_landmark.py`, whose values are `blendshape_only`, `landmark_only`, and `fusion`. Every arm receives the same standardized 95-dimensional packet and predicts the same complete 95-dimensional target; only input indices `72:95`, `0:72`, or neither are zeroed after the shared scaler.
- Every arm has identical parameter names, tensor shapes, parameter count, 95-dimensional decoder, scaler, temporal masks, masked evaluation elements, optimizer, epochs, and train/validation recording split. For a given seed, model and optimizer initial states must be byte-identical before the input-arm mask is applied.
- Primary cross-arm metrics are inverse-scaled raw-feature held-out MAE for blendshape72, clinical23, and their equal-weight two-block macro average over the identical 95-dimensional masked target, aggregated per recording and then across recordings. Full95 raw MAE and standardized SmoothL1/MAE remain secondary diagnostics so differing feature units cannot silently dominate the conclusion. Report mean, standard deviation, paired per-seed differences, and all frozen baselines. This is common-target representation evidence, not facial-paralysis classification, HB agreement, or patient-held-out generalization.
- Formal artifacts are immutable and separate from smoke and earlier v3 runs. No smoke checkpoint initializes a formal run.
- Held-out masks are materialized once in the common frozen inputs and reused byte-for-byte across arms. There is no early stopping, epoch selection, hyperparameter retry, or post-heldout configuration change.
- Per-recording metrics and opaque recording identifiers stay only in owner-local mode-`0600` reports. Any tracked summary contains aggregate values only and no receipt mapping or record-level identifier.

## Task 1: Complete the v4 real smoke gate

- [ ] Require the approved v4 source-attestation implementation and rebuilt bridge/frozen inputs.
- [ ] Execute one disposable two-stage smoke run.
- [ ] Verify exactly two checkpoints, two checkpoint receipts, and one execution-only report; independently authorize at publication edge.
- [ ] Do not start formal training if any raw source drift, stale bridge receipt, nonfinite metric, or unexpected file is observed.

## Task 2: Freeze the three-arm SSL API with RED tests

**Files:**

- Modify: `src/pretraining/dynamic_landmark_ssl.py`
- Modify: `tests/test_dynamic_landmark_ssl.py`
- Modify: `scripts/pretrain_dynamic_landmarks.py`
- Modify: `scripts/prepare_dynamic_landmark_ssl_inputs.py`
- Modify: `tests/test_dynamic_landmark_ssl_bridge.py`

- [ ] Freeze a stage-aware arm enum: a RAVDESS stage accepts only `semantic23_only`; a Mayo stage accepts only the three supervised arm constants. Reject every alias or extra value within the corresponding stage, reject every cross-stage value, and test the exact mapping.
- [ ] Prove inactive input columns cannot change the model input while all arms retain the identical common scaler, full 95-dimensional target, temporal mask, masked-element count, loss definition, and evaluation elements.
- [ ] Prove identical parameter names/shapes/counts and byte-identical same-seed initial states before arm masking.
- [ ] Upgrade the exact frozen-config schema from v2 to v3 with stage-exact `input_arm`, `input_active_indices`, and `target_schema`; update `_V2_CONFIG_FIELDS` to a newly named exact v3 field set and reject v2/v3 mixtures. The only legal combinations are RAVDESS stage `semantic23_only / 0:23 / semantic23_v1`, Mayo formal fusion `fusion / 0:95 / mediapipe72_plus_clinical23_full95_v1`, and Mayo ablation with one of the three supervised arm constants plus the same full95 target. Every illegal cross-stage schema/arm combination fails.
- [ ] Bind config v3, arm, common-target policy, producer digest, and v4 Mayo commitment into bridge/frozen receipts, checkpoints, and results. Because the producer digest changes, rebuild the entire bridge/frozen/checkpoint lineage; no old receipt remains valid.
- [ ] Prove an arm mismatch at resume, verification, or publication fails closed.
- [ ] Run tests and observe RED before implementation, then implement the minimum arm-aware path and return to GREEN.

## Task 3: Freeze formal inputs and dry-run all jobs

- [ ] Re-freeze the fusion formal inputs under the new v4 generation.
- [ ] Freeze one common Mayo split/mask receipt and three arm-bound config-v3 receipts for the ablation.
- [ ] Print and inspect a path-free dry-run matrix containing exactly six fusion stage jobs and nine Mayo-ablation jobs.
- [ ] Freeze `inputs/` once under its persistent lock and no-replace input transaction. Separately stage the complete nine-job `results/` tree, then publish that entire results namespace in one no-replace atomic transaction; partial per-arm result publication is forbidden. The canonical namespace is `outputs/dynamic_landmark/pretraining/ablation/mayo-input-arm-v1/` with exact inputs, nine checkpoints plus receipts, one private report, and no extra files.
- [ ] Verify seeds, same-seed initialization digest, epoch count, optimizer, shared mask digest, source generations, split IDs, and output namespaces before starting any job.

## Task 4: Run the preregistered formal fusion experiment

- [ ] Run seeds 0, 1, and 2 for 30 RAVDESS epochs and then 30 Mayo epochs.
- [ ] Require exactly six checkpoints and six matching receipts under the formal namespace.
- [ ] Publish `formal_pretraining_results.json` only after all seeds pass independent authorization and deterministic receipt verification.
- [ ] Preserve the 2026-07-15 Mayo report's `trained`, `prior_ravdess`, `fresh_untrained`, and `train_mean` baselines and report blockwise/raw macro MAE plus standardized optimization diagnostics.

## Task 5: Run the Mayo-only modality ablation

- [ ] Run `blendshape_only`, `landmark_only`, and `fusion` inputs for seeds 0, 1, and 2, 30 epochs each against the identical full-95 target.
- [ ] Produce one arm-bound checkpoint and receipt per job and a single exact-schema ablation report.
- [ ] Compute common-target paired seed comparisons and private per-recording held-out metrics without choosing a winner based on training loss or selecting epochs on held-out data.
- [ ] Record failures and exclusions explicitly; never silently retry with changed hyperparameters.

## Task 6: Verify, interpret, and review

```bash
python3 tests/test_dynamic_landmark_ssl.py
python3 tests/test_dynamic_landmark_ssl_bridge.py
python3 tests/test_build_mayo_ssl_cache.py
git diff --check
```

- [ ] Independently verify checkpoint/receipt counts and every report source commitment.
- [ ] Report whether fusion improves held-out reconstruction consistently across paired seeds, while stating that labels are still required to test HB prediction.
- [ ] Obtain spec and code-quality review; rerun all focused tests after corrections.
