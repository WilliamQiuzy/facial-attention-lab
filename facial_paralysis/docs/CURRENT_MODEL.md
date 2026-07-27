# Current development model

This file is the canonical source for current-model reporting. If an older
document calls a MARLIN, web-QWK, Blendshape, Fusion, or SSL model the
"champion", treat that wording as historical.

## Canonical result

The current development-set champion is the **110-dimensional Landmark
trajectory model**.

| Development candidate | AUROC | Balanced accuracy | Sensitivity | Specificity |
|---|---:|---:|---:|---:|
| Nuisance-only | 0.768 | 0.716 | 0.667 | 0.765 |
| 58D Clinical Dynamics | 0.922 | 0.857 | 0.714 | 1.000 |
| Clinical Dynamics + Nuisance | 0.913 | 0.881 | 0.762 | 1.000 |
| **110D Landmark** | **0.938** | **0.905** | **0.810** | **1.000** |

The 110D model was a fixed reference comparator and was numerically highest
among the four development candidates. Calling it the current development
champion records that rank; it does not turn it into the preregistered primary
comparison or authorize outer evaluation.

The classifier is a standardized L2 logistic regression with fixed
`C=0.01`, `liblinear`, threshold `0.5`, group-balanced sample weights, and no
hyperparameter search. The performance gain comes from the representation,
not a deeper neural network.

## What 110D Landmark means

MediaPipe remains the upstream detector. Its 478 face landmarks are reduced to
the frozen 23-channel `clinical23_v2` eye, brow, and mouth geometry contract.
Each recording is then summarized as:

- 23 channels × four statistics (median, IQR, range, maximum absolute velocity)
  = 92 features;
- six bilateral pairs × three dynamics (correlation, invariant amplitude ratio,
  and best lag) = 18 features.

Together these form the 110-dimensional recording-level Landmark vector.

## Evaluation boundary

- Dataset: PalsyNet.
- Target: binary affected versus unaffected, **not** House-Brackmann grade.
- Full registry: 49 recordings / 48 groups.
- Development evaluation: 39 recordings / 38 groups, comprising 21 affected
  and 17 unaffected groups.
- Protocol: outer fold 0 was reserved first; four-fold grouped inner
  out-of-fold evaluation used only the remaining development groups.
- Protected partition: 10 recordings / 10 groups; zero protected feature
  extractions, model fits, or predictions.
- Claim unit: video-held-out; identity status remains unreviewed.

These numbers are not Mayo-65 performance, HB accuracy, patient-held-out
clinical validation, or deployment evidence.

## Promotion state

The 110D Landmark model is the only current **development champion**. It
supersedes older models for present-tense status reporting, while the older
experiments remain preserved as historical baselines.

The preregistered primary comparison was Clinical Dynamics + Nuisance versus
Nuisance-only, not selection of the 110D reference. Its direction gate did not
fully pass. Clinical Dynamics +
Nuisance improved over Nuisance-only by AUROC `0.1457`, but the descriptive
paired-bootstrap 95% interval was `[0.0000, 0.2941]`; its lower bound was not
strictly above zero. Therefore:

- protected outer evaluation is not authorized;
- HB claims are not authorized;
- clinical use or deployment is not authorized.

The next valid step is to finish identity/action-coverage review, freeze a
successor protocol, and evaluate the untouched outer partition exactly once.

## Provenance

- Source branch: `codex/action-clinical-geometry`
- Source commit: `e3b069cfef3634b8f210b4315dd574d8b9fa46a6`
- Experiment: `action-clinical-geometry-v1`
- Private source report SHA-256:
  `2c50940b61f633eb4b664b364a7767f0fca215fda92ef85beeb92bb364529649`
- Machine-readable tracked summary:
  `docs/results/current_development_model.json`

The source report remains local and private by protocol. Only its
deidentified aggregate summary and hash are tracked.
