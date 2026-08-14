# NeuroFace Motion-Quality Pretraining v1

## Status and estimand

This protocol is locked after the one-shot frozen 110D NeuroFace evaluation and
its exploratory SLP audit. NeuroFace cohort labels, SLP ratings, and frozen
110D scores are therefore already outcome-exposed. The experiment is an
exploratory representation-learning study, not a new external validation.

The question is whether a small encoder supervised by NeuroFace speech-language
pathologist (SLP) ratings can add motion-quality information to the frozen 110D
facial-asymmetry representation. The downstream estimand is affected-versus-
unaffected discrimination on the already reviewed, identity-disjoint PalsyNet
development partition. It is not House-Brackmann grading, Mayo accuracy, or a
clinical deployment claim.

## Locked data boundaries

- NeuroFace supplies 231 technically retained videos from 36 participants and
  the released mean of two raters for symmetry, range of motion, speed,
  variability, and fatigue. All videos from one participant stay in one fold.
- The 30 label-blind technical QC exclusions from the frozen external run stay
  excluded. No QC rule may be changed after seeing SLP targets.
- PalsyNet supplies only the reviewed development rows and their frozen four
  inner folds. Protected outer rows must not be loaded, featurized, fitted, or
  predicted.
- NeuroFace, MEEI, and Mayo results cannot select the downstream classifier,
  threshold, or promotion decision.

## Locked representation

The encoder receives only the 23 MediaPipe clinical landmark channels from each
four-by-32 cache. Each 32-frame window is processed independently; temporal
convolution never crosses the gaps between the four sampled windows. The input
contains standardized landmark values and within-window per-second first
differences. A compact depthwise TCN creates a 24-dimensional vector per
window. Its recording representation concatenates the mean window vector with
the difference between late and early window means, then projects to 32
dimensions. This explicitly retains overall motion and early-to-late change.

The SLP head receives the 32-dimensional representation plus an eight-
dimensional task embedding and predicts the five ratings on their normalized
1-to-5 scale. Task identity is available only to the discarded SLP head, never
to the transferable representation. Original and horizontally mirrored
sequences receive the same targets and equal participant-balanced weight.

Locked training settings are: six participant-disjoint folds stratified by the
three cohorts, seed 20260813, AdamW, learning rate 0.001, weight decay 0.0001,
80 epochs maximum, patience 10, gradient clipping at 5, and Smooth L1 loss. The
median best epoch from the six held-out folds determines the epoch count for one
final fit on all retained NeuroFace videos. There is no architecture or
hyperparameter search.

## Locked evaluation

NeuroFace reports held-out Spearman correlation and mean absolute error for
each SLP domain plus participant-macro mean absolute error. These are internal
pretraining diagnostics because the SLP outcomes are already exposed.

The final encoder and NeuroFace-only standardizer are frozen before PalsyNet is
opened. On each existing PalsyNet inner fold, the exact same StandardScaler,
L2-logistic regression (`C=0.01`), group-balanced weights, mirror augmentation,
mirror probability averaging, and 0.5 threshold compare three fixed inputs:

1. frozen 110D baseline;
2. frozen 32D motion representation alone (diagnostic);
3. 110D plus frozen 32D motion representation.

The current model is promoted only if fusion has strictly higher group-level
AUROC than the reproduced 110D baseline, no more than a 0.02 absolute loss in
balanced accuracy, and no worse Brier score. Otherwise the frozen 110D remains
the current model. No protected or external retest is authorized by this study.

## Outputs and privacy

Raw videos, frame caches, participant mappings, row-level targets, predictions,
and checkpoints remain owner-only on the H200 host. The public repository may
contain code, this protocol, tests, and an aggregate report without participant
or recording identifiers. All source manifests, cache collections, code, and
runtime dependencies are SHA-256 bound in the private receipt.
