# Source-Robust Landmark 110D v1

## Decision

Retain the frozen mirror-invariant Landmark 110D Logistic model. Removing all 23 static channel medians, or restoring only six direction-free clinical asymmetry medians, preserved PalsyNet development discrimination but did not strictly improve it under the preregistered acquisition-blocked stress test. Neither reduced representation is promoted.

## Representations

- `landmark_mi_110d`: current 23 channel medians, IQRs, ranges, maximum velocities, and six bilateral correlation/amplitude/lag triplets.
- `within_video_dynamics_87d`: removes every channel median and keeps only within-video dispersion, excursion, velocity, and bilateral dynamics.
- `asymmetry_dynamics_93d`: adds back only the median absolute left/right differences for fissure height, fissure width, eye area, brow height, oral-corner height, and commissure position.

All candidates used the same original-plus-mirror training, mean original/mirror inference, train-fold-only standardization, group-balanced weights, L2 Logistic `C=0.01`, and threshold `0.5`. There was no candidate-specific tuning.

## PalsyNet development results

| Representation | Registered AUROC | Registered balanced accuracy | Registered specificity | Acquisition-blocked AUROC | Acquisition-blocked balanced accuracy | Acquisition-blocked Brier |
|---|---:|---:|---:|---:|---:|---:|
| Current 110D | 0.9804 | 0.9524 | 1.0000 | 0.9664 | 0.9286 | 0.1195 |
| Dynamics 87D | 0.9804 | 0.9524 | 1.0000 | 0.9664 | 0.9286 | 0.1193 |
| Asymmetry + dynamics 93D | 0.9804 | 0.9524 | 1.0000 | 0.9664 | 0.9286 | 0.1163 |

The 93D arm improved acquisition-blocked Brier score by `-0.00319`; its paired bootstrap 95% interval was `[-0.00613, -0.00045]`. AUROC and balanced-accuracy changes were exactly zero, so the locked gate retained 110D. The result supports a smaller calibration-oriented representation, not a more accurate classifier.

## Acquisition-blocked stress split

Only seven nonclinical acquisition measurements were used: bitrate proxy, detection rate, luminance, face-scale mean/std, and eye-line-roll mean/std. Measurements were averaged per reviewed identity group, standardized on the 38 development groups, reduced to a deterministic first principal component, and divided into four contiguous blocks within each binary label. This is a PalsyNet development stress test, not an external institution.

## H200 reproduction

The aggregate evaluation was independently rerun on the Nebius NVIDIA H200 in an immutable release directory. It completed in `44.35` seconds. Counts, decisions, audit counters, implementation SHA-256, and all non-floating fields matched exactly; the largest cross-platform floating-point difference was `5.56e-17`, below the preregistered `1e-14` tolerance. No Mayo or MEEI data were uploaded.

## Interpretation and boundary

Static channel medians are not required to reproduce the current PalsyNet development ranking, but removing them does not solve the observed cross-institution specificity gap. MEEI was not reopened or rescored, Mayo was not read or used for selection, and all 10 protected PalsyNet recordings remained unopened. No HB, clinical-validation, or deployment claim is authorized.

The next useful data step is a new untouched external cohort with verified normal controls, such as AFLFP after legitimate access. Without such a cohort, further PalsyNet-only representation changes cannot establish that external normal false positives have improved.
