# NeuroFace Generalization and Representation Study v1

## Decision

The current mirror-invariant Landmark 110D + standardized L2 Logistic model
remains locked. NeuroFace provided a useful external failure test and two new
supervision sources—five-domain SLP ratings and 3,306 manually annotated
68-point frames—but neither fixed successor improved the identity-reviewed
PalsyNet development result. No further candidate is authorized on this now
repeatedly exposed development partition.

This study does not estimate House-Brackmann accuracy, Mayo accuracy, or
prospective clinical performance. NeuroFace contains ALS, post-stroke, and
healthy participants; it tests neurological oro-facial impairment rather than
Bell's-palsy-specific unilateral weakness.

## Frozen external evaluation

The frozen 110D artifact was scored once, without fitting, calibration, or
threshold selection, on 231 technically retained videos from all 36 NeuroFace
participants. The primary participant score averaged NSM_KISS, NSM_OPEN, and
NSM_SPREAD.

| Evaluation | AUROC | Balanced accuracy | Sensitivity | Specificity | Brier |
|---|---:|---:|---:|---:|---:|
| Frozen 110D on NeuroFace | 0.349 | 0.416 | 0.560 | 0.273 | 0.283 |

Healthy participants received higher mean model scores than the affected
participants. This is a clear external generalization failure, not evidence of
a 95% classifier. The released SLP total, by contrast, descriptively separated
affected from healthy participants with AUROC 0.905, confirming that the cohort
contains meaningful clinical signal. The frozen 110D score associated most
strongly with pooled affected-participant symmetry (Spearman 0.543) and not
with total movement severity, supporting the interpretation that 110D is
primarily a lateral-asymmetry detector.

## SLP motion-quality pretraining

A 7,081-parameter TCN processed each of the four 32-frame windows separately,
using clinical23 values and within-window velocities. Its 32D recording
embedding retained overall motion plus early-to-late change. Six
participant-disjoint NeuroFace folds supervised five SLP domains; a task
embedding was available only to the discarded regression head.

Held-out correlations were modest: symmetry 0.024, range of motion 0.288,
speed 0.221, variability 0.313, and fatigue 0.176. The final encoder was frozen
before PalsyNet was opened.

| PalsyNet development representation | AUROC | Balanced accuracy | Brier |
|---|---:|---:|---:|
| Current 110D | **0.980** | **0.952** | **0.117** |
| Frozen NeuroFace motion 32D | 0.513 | 0.620 | 0.245 |
| 110D + frozen motion 32D | 0.964 | 0.905 | 0.124 |

The motion embedding did not transfer and was not promoted. This indicates
that ALS/stroke global motion quality is not a drop-in replacement for the
lateral weakness signal used by the Bell's-palsy classifier.

## Manual landmark measurement audit

MediaPipe detected a face on all 3,306 manually annotated frames across all 261
videos. Detection was 100% in ALS, post-stroke, and healthy cohorts and in all
nine tasks. The important limitation was measurement agreement, not face
detection.

The annotation `Frame` value was consumed as the direct zero-based OpenCV frame
position. This was checked against a released annotated-frame JPEG: decoded
position 100 matched its corresponding released image with pixel MSE 2.19,
versus 33.25 at position 99 and 30.87 at position 101. The weak bilateral
difference agreement is therefore not explained by a one-frame indexing shift.

Single-side and global oral measurements were strong: mouth width and mouth
opening had cross-frame Spearman correlations of 0.871 and 0.875, and oral
corner height was approximately 0.932. In contrast, several small bilateral
difference channels were unstable across the two landmark topologies:
fissure-height absolute difference 0.049, fissure-width absolute difference
0.089, and eye-measure absolute difference 0.025. Their errors are amplified
when two individually reasonable side measurements are subtracted.

The eight mirror-invariant measurements had median absolute Spearman 0.347 and
median within-recording Pearson 0.099, below the locked 0.70 measurement gate.
Because MediaPipe FaceMesh and the manual 68-point topology use different
anchors, these values measure rank and motion compatibility, not raw coordinate
interchangeability.

## Manual68 teacher calibration

One fixed, label-free multi-output Ridge calibration (`alpha=1`) was trained
with mirror augmentation to map MediaPipe semantic23 values to manual68
semantic23 values. Six participant-disjoint folds showed that the median
mirror-invariant correlation fell from 0.347 to 0.298 rather than improving.

| PalsyNet development representation | AUROC | Balanced accuracy | Brier |
|---|---:|---:|---:|
| Current 110D | **0.980** | **0.952** | **0.117** |
| Manual68-calibrated 110D | 0.966 | 0.905 | 0.137 |

The calibration was not promoted. All ten protected PalsyNet recordings had
zero cache reads and zero predictions in both improvement experiments.

## What this changes

NeuroFace should no longer be treated as an untouched external validation set;
its labels and ratings have now been used for exploratory diagnosis and
pretraining. It remains useful for oral-motion measurement research, especially
mouth opening, mouth width, and commissure trajectories.

For the Bell's-palsy binary model, the next scientifically valid improvement
requires new labeled unilateral-palsy data—preferably Mayo HB/binary labels
with verified controls—or a new untouched external cohort. Additional
PalsyNet-only architectures or NeuroFace-to-PalsyNet adapters would now be
adaptive reuse of the same small development set and are unlikely to produce a
reliable gain.

## Reproducibility

- Branch: `codex/neuroface-external-validation-v1`
- H200 runtime: `facial-paralysis-neuroface:v1.3`
- H200 model-training time: 33.77 seconds
- H200 manual-frame audit time: 111.31 seconds
- Frozen external report SHA-256: `beda5bee5ed3736a90245e98c165777198aee6d6be2bbcd8a13f0fb2b1a11984`
- SLP audit report SHA-256: `bc269696eadcb1850ec819962b7d623703cb1856bbf59deefd1573e590de421d`
- Motion pretraining report SHA-256: `7a5cfcd813e48f7e6b42b9687df3d243651f6d5af5d8a9811de0b4362e7f8e80`
- Manual68 audit report SHA-256: `a58e42be8101deabd6797bcab24e50485713cbfe3d65c55b65c78594b932dddf`
- Geometry calibration report SHA-256: `9d54a58eede9b6b4c12f4635ca213726e1fff8005df424c346808cba88d8b5a8`
- Protected PalsyNet cache reads/predictions: `0/0`
- Clinical validation, HB grading, and deployment authorization: `false`
