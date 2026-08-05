# Current development model

This file is the canonical source for current-model reporting. If an older
document calls a MARLIN, web-QWK, Blendshape, Fusion, or SSL model the
"champion", treat that wording as historical.

## Canonical result

The current development-set champion is the **mirror-invariant
110-dimensional Landmark trajectory model**.

| Development candidate | AUROC | Balanced accuracy | Sensitivity | Specificity |
|---|---:|---:|---:|---:|
| Nuisance-only | 0.768 | 0.716 | 0.667 | 0.765 |
| 58D Clinical Dynamics | 0.922 | 0.857 | 0.714 | 1.000 |
| Clinical Dynamics + Nuisance | 0.913 | 0.881 | 0.762 | 1.000 |
| 110D Landmark | 0.938 | **0.905** | **0.810** | **1.000** |
| **Mirror-invariant 110D Landmark** | **0.944** | **0.905** | **0.810** | **1.000** |

The successor uses the same fixed 110D representation and classifier, but
trains on each trajectory and its horizontal mirror and averages the two
probabilities at inference. Its maximum observed mirror probability error was
`0.0`. It passed the fixed robustness/non-inferiority gates, so it replaces the
standard 110D model for development reporting; this does not authorize outer
evaluation.

The classifier remains a standardized L2 logistic regression with fixed
`C=0.01`, `liblinear`, threshold `0.5`, group-balanced sample weights, and no
hyperparameter search. The robustness change comes from fixed symmetry
handling, not a deeper neural network or parameter search.

## Preregistered successor experiment

**110D-Generalization v1** is the next experiment; it does not replace the
current champion or add a new result. It compares only these frozen candidates:

- `landmark_mi_110d` (110D);
- `landmark_mi_110d_action_proxy_168d` (110D + 58D Action proxy);
- `landmark_mi_110d_action_phase_proxy_204d` (168D + 36D Phase proxy).

All three retain the same mirror handling, train-fold-only standardization,
group-balanced fixed L2 logistic regression (`C=0.01`, `liblinear`, threshold
`0.5`, `max_iter=2000`, random state `0`), split logic, and evaluation budget,
with no tuning. Promotion is hierarchical: 168D must strictly improve AUROC
over 110D without reducing balanced accuracy or increasing Brier score; 204D
must meet those gates against both simpler candidates, and any exact tie
retains the simpler model.

PalsyNet has no standardized action or frame-level phase labels, so the added
blocks are explicitly geometry-based Action and Phase **proxies**, not true
action/phase supervision. YFP is a separate static regional-severity ordinal
transfer task, while MEEI is the intended standardized-action/HB external
cohort but does not itself provide frame-level Phase truth. PalsyNet comparison
cannot start until the label-blinded identity review and group-disjoint split
registry are complete. The protected outer partition remains sealed and may be
used once only after one candidate and all provenance digests are locked.

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
- Protected partition: 10 recordings / 10 groups; zero protected candidate-110D
  feature construction, model fits, or predictions. The shared loader still
  performs schema and quality validation over all registered cache records
  before the split is applied.
- Claim unit: video-held-out; identity status remains unreviewed.

These numbers are not Mayo-65 performance, HB accuracy, patient-held-out
clinical validation, or deployment evidence.

## Promotion state

The mirror-invariant 110D Landmark model is the current **development
champion**. It
supersedes older models for present-tense status reporting, while the older
experiments remain preserved as historical baselines.

Against the standard 110D baseline, the fixed mirror successor improved AUROC
by `0.0056`; the descriptive paired-bootstrap 95% interval was
`[0.0000, 0.0224]`. The result establishes exact mirror robustness and no
observed loss on the fixed development metrics, but does **not** establish a
statistically reliable superiority claim. Therefore:

- protected outer evaluation is not authorized;
- HB claims are not authorized;
- clinical use or deployment is not authorized.

The next valid step is to finish identity/action-coverage review, freeze a
successor protocol, and evaluate the untouched outer partition exactly once.

## Provenance

- Source branch: `codex/mirror-invariant-110d`
- Base protocol commit: `632bf993a8d38a7426fc52b23923e1d8d14dd857`
- Implementation commit: `7c64c26005895083766dac7760ce498b253741e8`
- Experiment: `mirror-invariant-110d-v1`
- Runner SHA-256:
  `ea41d076230665b55bcd9f2b0b9e047c3d67558ddd48d3271cb20d06e4f03c12`
- Protocol-test SHA-256:
  `e2c0855b5afce630160e83db9bdbec08299f707686226a7e3beb2d40bdbee10c`
- Aggregate source report SHA-256:
  `f84fbd1605dea3515cde524946b1d3e4c1003aac77ece2bcef8b3e413f75cb29`
- Machine-readable tracked summary:
  `docs/results/current_development_model.json`

The owner-private source report remains ignored by Git. The tracked summary is
closed-schema, deidentified aggregate evidence and contains no per-record
predictions, IDs, or local paths.
