# Current development model

This is the canonical source for present-tense model reporting. Older MARLIN,
web-QWK, Blendshape, Fusion, SSL, standard 110D, and video-held-out results are
historical baselines rather than the current validation claim.

## Canonical result

The released model is the **mirror-invariant 110D Landmark trajectory
representation** with a fixed standardized L2 logistic classifier. It was
selected under an identity-reviewed, patient/group-disjoint PalsyNet
development protocol and then evaluated once on the untouched outer partition.

| Frozen development candidate | AUROC | Balanced accuracy | Sensitivity | Specificity | Brier |
|---|---:|---:|---:|---:|---:|
| **110D Landmark** | **0.980** | **0.952** | **0.905** | **1.000** | 0.117 |
| 110D + 58D Action proxy | 0.975 | 0.929 | 0.857 | 1.000 | **0.103** |
| 110D + Action proxy + 36D Phase proxy | 0.969 | 0.929 | 0.857 | 1.000 | 0.104 |

The 168D and 204D candidates improved Brier calibration error but reduced both
AUROC and balanced accuracy. They therefore fail the preregistered promotion
rule, and the simpler 110D candidate stays locked. The 110D AUROC 95% paired
group-bootstrap interval was `[0.933, 1.000]`; the 168D-minus-110D AUROC delta
was `-0.0056` with interval `[-0.0280, 0.0112]`, and the 204D-minus-110D delta
was `-0.0112` with interval `[-0.0420, 0.0056]`.

These values must not be presented as an improvement over the older AUROC
`0.944`: the earlier number came from a different, identity-unreviewed
video-held-out partition. The new result is stronger evidence because its
evaluation unit is a reviewed person group, not because the numerical score is
directly comparable.

The one-shot protected outer test contained 10 reviewed person groups (5
affected and 5 unaffected). It produced AUROC `1.000`, average precision
`1.000`, balanced accuracy `0.900`, sensitivity `0.800`, specificity `1.000`,
and Brier `0.118` at the frozen threshold `0.5`. The perfect ranking on only ten
groups is encouraging but statistically fragile: one affected group fell below
the fixed decision threshold, and this remains an internal PalsyNet outer test,
not cross-institutional or clinical validation.

## MEEI independent external validation

The final PalsyNet artifact was applied unchanged, exactly once, to all 60 MEEI
participants (50 facial palsy, 10 normal). Every participant had one eligible
video; all 60 passed the frozen label-blind extraction gate. No MEEI photo was
decoded, and no scaler, model, calibration, threshold, or feature was fit or
selected on MEEI.

| External participant-level endpoint | Point estimate | Descriptive bootstrap 95% interval |
|---|---:|---:|
| AUROC | 0.776 | 0.622–0.902 |
| Average precision | 0.949 | 0.910–0.980 |
| Balanced accuracy at 0.5 | 0.650 | 0.500–0.810 |
| Sensitivity at 0.5 | 0.900 | 0.820–0.980 |
| Specificity at 0.5 | 0.400 | 0.100–0.700 |
| Brier score | 0.143 | 0.111–0.178 |

At the frozen threshold, 45/50 affected and 4/10 normal participants were
classified correctly, giving an ordinary accuracy of 49/60 = 81.7%. Balanced
accuracy is the safer summary because the external cohort is 83.3% affected.
For the same reason, the apparently high average precision of 94.9% must not be
reported as accuracy. The main external failure is false-positive normal calls:
sensitivity transferred reasonably, while specificity did not.

This is a genuine independent public cross-dataset evaluation and it does not
meet the requested 95% target. It narrows the next research problem to
cross-source normal-reference and acquisition robustness; it is not permission
to tune against the now-exposed MEEI test result.

## Model and representation

MediaPipe remains the upstream detector. Its face landmarks are reduced to the
frozen 23-channel `clinical23_v2` eye, brow, and mouth geometry contract. The
110D vector contains:

- 23 channels × median, IQR, range, and maximum absolute velocity = 92 values;
- six bilateral pairs × correlation, direction-invariant amplitude ratio, and
  best lag = 18 values.

Every candidate used the same original-plus-horizontal-mirror training,
mean original/mirror inference, train-fold-only `StandardScaler`, group-balanced
weights, and L2 logistic regression with `C=0.01`, `liblinear`,
`max_iter=2000`, random state `0`, and threshold `0.5`. There was no tuning,
feature selection, candidate-specific calibration, or neural architecture
change.

The additional 58D and 36D blocks are geometry-based **Action and recording-
position Phase proxies**. PalsyNet has no standardized action or frame-level
phase annotations, so this experiment does not establish a true action- or
phase-specific disease mechanism.

## Architecture and split-stability stress tests

Architecture Autoresearch v1 compared nine fixed candidates on the same 39
development recordings / 38 reviewed groups: Logistic, Extra Trees,
HistGradientBoosting, MLP, TCN, BiGRU, Transformer, region-factorized TCN, and a
110D + TCN hybrid. The 110D Logistic model remained first (AUROC 0.980,
balanced accuracy 0.952). The closest hybrid reached AUROC 0.978 but balanced
accuracy 0.899. Four adaptive ensembles improved Brier calibration but did not
simultaneously improve AUROC and balanced accuracy, so no candidate advanced.

Across 50 additional deterministic, stratified group-disjoint four-fold
partitions, 110D AUROC had median 0.966, 5th–95th percentile range 0.956–0.981,
and 49/50 repeats were at least 0.95. A 500-repeat reviewed-group label
permutation test produced null mean AUROC 0.489 and null 95th percentile 0.686;
the real fixed-fold AUROC 0.980 gave p=0.001996. These are development-only
stability checks, not an opening of the protected outer partition.

## Mayo positive-only challenge

The current 65-session Mayo folder contains 50 MOV files / 49 unique video
contents after exact SHA-256 deduplication. One insufficient-frame file and one
video below the frozen face-coverage gate were excluded, yielding 47 challenge
videos. The same MediaPipe model, rotation-aware face-present sampling,
clinical23_v2 representation, and frozen 110D classifier called 45/47 positive
at threshold 0.5: positive-call rate 0.957, Wilson 95% interval 0.858–0.988.
Confidence mean was 0.665, median 0.649, IQR 0.598–0.716; extraction coverage
median was 1.000 and minimum 0.922. Raw Mayo videos remained local.

Because this cohort contains no verified negatives, 45/47 is **not binary
accuracy** and cannot estimate specificity or AUROC. Mayo was not used for
candidate selection, and the result is not independent clinical validation.

## Evaluation boundary

- Cohort: PalsyNet binary affected versus unaffected; this is not HB grading.
- Identity audit: all 49 recording overviews and all 1,176 unordered recording
  pairs were reviewed label-blind; one pair was the same person, yielding 48
  reviewed groups, with no unresolved or cross-label group.
- Frozen split: 49 eligible recordings / 48 groups total.
- Development evaluation: 39 recordings / 38 groups, comprising 21 affected
  and 17 unaffected groups, with four-fold group-disjoint OOF predictions.
- Protected partition: 10 recordings / 10 groups in outer fold 0, comprising 5
  affected and 5 unaffected groups.
- One-shot outer execution: 39 development recordings loaded for one scaler and
  one model fit; 10 protected recordings loaded, transformed, and predicted
  once. No candidate, threshold, feature, calibration, or split changed after
  the result.
- Bootstrap: 5,000 paired, class-stratified reviewed-group resamples at seed
  `20260805`.
- Claim unit: `person_held_out`; identity status: `reviewed`.

The report is a closed aggregate: it contains no recording/group IDs, labels,
per-record probabilities, filenames, NPZ names, or local paths. Nine execution-
affecting source components are individually hashed and aggregated before the
result can be serialized.

## Decision and remaining gates

`landmark_mi_110d` is now sealed as 110D-Generalization v1. After the protected
result was pinned, the final inference artifact was fit once on all 49 eligible
recordings / 48 reviewed groups (26 affected and 22 unaffected groups). The
artifact contains the 110 feature names, scaler statistics, Logistic
coefficients/intercept, mirror-inference rule, threshold, and authenticated
provenance; it contains no patient identifiers, filenames, paths, or row-level
predictions.

The MEEI gate is complete and exposed a specificity/generalization gap. MEEI
must not now become a tuning set. The next scientific gate is a PalsyNet-only,
source-robust representation update selected without MEEI outcomes, followed by
a new untouched external cohort such as AFLFP after legitimate access. Mayo
remains a positive-only consistency challenge and cannot supply specificity or
AUROC until verified negative labels exist.

No result here is Mayo-65 two-class accuracy, HB accuracy, prospective clinical
validation, or deployment evidence. MEEI is an independent public cross-dataset
test, not HB grading or a clinical-use study. The Mayo result remains a
positive-only consistency challenge. AFLFP access is not established and
requires an eligible non-student institutional recipient to complete its EULA
process.

YFP has a separate static regional-severity audit manifest with 10,838 accepted
anchors, but it remains `training_eligible=false`: no MediaPipe extraction,
ordinal fitting, or prediction is authorized without the license artifact,
independently reviewed subject map, and eligibility authorization.

## Provenance

- Source branch: `codex/meei-external-validation-v1`
- Generalization lock implementation commit: `14a47f9`
- One-shot outer release implementation commit: `ad5b670`
- Architecture/stability/Mayo challenge implementation commit: `036e033`
- Reviewed identity manifest SHA-256:
  `fa756b79f0e1bc9053527de4632216281d9011a1f75e2bf652371dab38d2da9f`
- Review ledger SHA-256:
  `865fe78137d3d97b11da3bf37c6db105e387174b9f115c01824afea6a5368afd`
- Person split registry SHA-256:
  `738980264a698cb8a2d45a12fdc1ff95f349bbb4ac76787296e5314e40981ba0`
- Development implementation aggregate SHA-256:
  `4f0c67c98c5ec6ea4c8d0746504caae407840f2aefee3668adf6d7c60fb15208`
- Development aggregate report SHA-256:
  `e3f7eb6b9c91fbad74a514be8ba6f0c51418d7953155518033fedb1e228a1f43`
- Outer authorization SHA-256:
  `7aca8ddffeca5479a53ee28103b091acf1f1d64badd46d62ce8a601654b0881b`
- Outer release implementation aggregate SHA-256:
  `b5ac56315b161648662d7b3dcf0991b30413b1dffada11b697547ed7db2aff9d`
- Sealed protected report SHA-256:
  `0e44cfaf2fe5bb3e8e9d9ea8629f8ef873a5af2d94a56025b3f6f32f35227df3`
- Final PalsyNet artifact SHA-256:
  `cbc49d0aa54b504915bebd00fdbe005458378e5675b57461ce83d3385f9b60f9`
- MEEI external implementation commit: `81a83df`
- MEEI authorization SHA-256:
  `8bf70dd1b04381a35af2de99b530e744425384a060b90a48c56156ec6b3ae6af`
- MEEI participant manifest SHA-256:
  `7770473fe4479cf11e4c88474ee28105cef7db5fca026918fc89cd6dc023c5ed`
- MEEI dynamic-cache manifest SHA-256:
  `4ae13f6a77ea2ba3095665df6cccc0753b54a49f8cd3d676dccddb73d12f1d0e`
- MEEI cache-byte collection SHA-256:
  `9f9ac765eb0ce0aff317c27099ed9c4e9828f9f4303b51c1a03372774a3c3f6f`
- MEEI aggregate report SHA-256:
  `445fa3c770addbeb70820a3395c13304e602b91445bb218d404cd1dff5f54baf`
- MEEI H200 release:
  `/home/ssh-ziyue/facial-paralysis-h200/releases/meei-external-v1-81a83df`
- H200 release:
  `/home/ssh-ziyue/facial-paralysis-h200/releases/110d-outer-release-v1-ad5b670`
- Architecture Autoresearch report SHA-256:
  `4f5d77c6c50b2f5d313f0d907416c6ac958837b677bf0b7ecf018fe5a949e8e7`
- Mayo positive-challenge report SHA-256:
  `bd05143e6221d3bf3c95a6014d2f67cf757848d340c1c52980e30a7b7a593060`
- Machine-readable public summary:
  `docs/results/current_development_model.json`

The protected outer evaluator was invoked exactly once and created the sealed
report in 17.68 seconds. The separate final-artifact freezer was then invoked
exactly once and completed in 2.36 seconds. The release scripts reject output
overwrite, and the H200 release is retained as the execution evidence.

The separate MEEI runner was also invoked exactly once. Its closed aggregate
report contains no participant/recording identifiers, filenames, paths,
labels, or row probabilities. The execution audit records 60 authenticated
cache loads, 60 feature extractions, 60 frozen-artifact predictions, 120 mirror
transforms, and zero scaler/model/calibration fits.
