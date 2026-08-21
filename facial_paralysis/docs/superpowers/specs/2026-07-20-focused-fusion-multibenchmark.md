# Focused Fusion Multi-Benchmark Specification

## Objective

Evaluate the already frozen three-seed Mayo Fusion reconstruction checkpoints under several deterministic development benchmarks without changing the trainer-digest-bound files or unlocking the supervised clinical outer test.

## Required evidence

- Authenticate the focused bridge, smoke report, selection report, winner report, checkpoint receipts, and all three checkpoint states through the existing fail-closed validators.
- Reproduce the published clean held-out metric before grading any stress condition.
- Score every condition against the same clean held-out target tensor and the same 5,120 masked temporal positions from 160 packets in 10 recording-held-out groups.
- Cover five benchmark families: clean reconstruction, modality removal, observed-context dropout, landmark noise, and observed-frame order corruption.
- Run every condition for seeds 0, 1, and 2 and aggregate per-recording equal-weight metrics as mean and sample standard deviation across seeds.
- Publish only aggregate, deidentified development diagnostics; never publish recording identifiers, raw paths, private keys, or per-recording metrics.
- Every input error metric must lie in the closed interval `[0, 1e9]`, have at most 64 coefficient digits, and have a Decimal exponent in `[-100, 100]`. This bound is far above any valid reconstruction error while ensuring the five-decimal protocol quantum remains distinguishable at the JSON floating-point boundary and bounding Decimal work.
- The clean three-seed mean primary metric must be strictly positive. Derived degradation is signed and must lie in `[-100, 1e16]`; it is not an input error metric.
- Deidentification must validate a closed report schema assembled from explicitly allowed aggregate fields; a generic recursive blacklist is not an authorization boundary.

## Exact public report schema

The top-level report has exactly these fields: `schema_version`, `status`, `claim_scope`, `source`, `selected_arm`, `seeds`, `metric_policy`, `accounting`, `protocol_registry`, `commitments`, and `conditions`.

- `metric_policy` has exactly: `canonicalization`, `decimal_places`, `input_metric_min`, `input_metric_max`, `input_metric_max_digits`, `input_metric_exponent_min`, `input_metric_exponent_max`, `primary_metric`, `lower_is_better`, `degradation_formula`, and `degradation_range`.
- `accounting` has exactly: `heldout_packets`, `heldout_recording_groups`, `valid_positions`, `scored_target_positions`, `scored_target_scalars`, `observed_context_positions`, and `feature_width`.
- Every `protocol_registry` row has exactly: `name`, `input_arm`, `context_dropout_probability`, `landmark_noise_sd`, and `rng_seed`, in frozen registry order.
- `commitments` has exactly: `benchmark_script_sha256`, `evaluation_module_sha256`, `trainer_sha256`, `bridge_generation_sha256`, `common_contract_sha256`, `winner_report_sha256`, and `checkpoints`; each checkpoint row has exactly `seed`, `checkpoint_fingerprint`, and `checkpoint_receipt_sha256`.
- `conditions` contains exactly ten rows in frozen registry order. Each row has exactly `condition`, `seed_rows`, `aggregates`, and `degradation_percent_vs_clean`. Each seed row has exactly `condition`, `seed`, and `metrics`, in seed order 0, 1, 2.
- A `metrics` object has exactly baselines `trained`, `fresh_untrained`, and `train_mean`. Each baseline has exactly `raw_mae`, `standardized_mae`, and `standardized_smooth_l1`; `raw_mae` has exactly `blendshape72`, `clinical23`, `equal_block_macro`, and `full95`.
- An aggregate baseline mirrors the metric schema, replacing every scalar with exactly `mean` and `sample_sd`.

No other key, nesting, identifier, path, HMAC, private key, or per-recording row is authorized.

## Decimal policy

All metric arithmetic uses a private Decimal context with precision 32, `ROUND_HALF_EVEN`, `Emin=-100`, `Emax=100`, `capitals=1`, and `clamp=0`. Traps for `InvalidOperation`, `DivisionByZero`, and `Overflow` are enabled; traps for `Underflow`, `Subnormal`, `Inexact`, `Rounded`, and `Clamped` are disabled. Ambient Decimal context must not affect acceptance or results.

## Fixed conditions

- `clean_fusion`: unchanged Fusion input.
- `mask_landmarks`: use the trained model with landmark input disabled.
- `mask_blendshapes`: use the trained model with Blendshape input disabled.
- `context_dropout_{10,25,50}pct`: add deterministic masked context positions with RNG seeds 41010, 41025, and 41050; keep the scored target mask unchanged.
- `landmark_noise_{0.10,0.25,0.50}sd`: add deterministic Gaussian noise only to observed landmark context in training-scaler standard-deviation units, with seeds 52010, 52025, and 52050.
- `frame_order_shuffle`: permute complete observed 95D feature rows within each window using seed 63000 while keeping timestamps, source indices, targets, and invalid positions unchanged.

## Claim boundary

This is a same-split, recording-held-out self-supervised stress evaluation. It is not patient-held-out, not HB classification, not an independent clinical dataset result, and not a substitute for the locked PalsyNet supervised outer benchmark.
