# NeuroFace ALS Architecture v1

## Decision

Keep the frozen Bell's-palsy Landmark 110D model unchanged. For the separate NeuroFace ALS-versus-healthy endpoint, lock the single-SPREAD nested Py-Feat statistic selector as the strongest development candidate. It exceeds 0.90 in strict AUROC (`0.9504`) but not in strict accuracy (`0.8636`), and it does not exceed the published descriptive comparator (`0.91` accuracy, `0.97` AUROC).

The requested “above 0.90 on NeuroFace” goal is therefore met only for ranking discrimination, not for correctly classified participants. We do not relabel AUROC as accuracy or use the paper-like same-OOF search as strict evidence.

## Data and extraction

- Endpoint: 11 ALS participants versus 11 healthy controls.
- Recordings: 66 total, with one KISS, OPEN, and SPREAD recording per participant.
- Py-Feat: version `0.6.2`, XGBoost AU model, exact 20-AU order, full-frame extraction.
- H200 result: all `66/66` caches completed; minimum, median, and maximum valid-frame coverage were all `100%`.
- AU collection SHA-256: `6910d02efb9d161b4909408b15b63636f904e6ce3aff5f91b12536ae6e798cc5`.

## Paper reproduction

The published NeuroFace classifier reported its best SPREAD result from minimum AU values: accuracy `0.91` and AUROC `0.97`. Its documented candidate search is not described as nested.

Our paper-like search over five 20D AU statistics, penalties, and C values selected SPREAD AU standard deviation with accuracy `0.9091` and AUROC `0.9669`. Rounded to two decimals, this reproduces the published `0.91/0.97` performance. Because representation and hyperparameters were selected on the same OOF predictions, this result is descriptive and is not used as our strict model claim.

## Strict participant-disjoint result

Every held participant was excluded from feature standardization, correlation filtering, statistic selection, regularization selection, fitting, and threshold selection.

| Pipeline | Accuracy | AUROC | Balanced accuracy | Sensitivity | Specificity |
|---|---:|---:|---:|---:|---:|
| **SPREAD nested AU-statistic Logistic** | **0.8636** | **0.9504** | **0.8636** | **0.8182** | **0.9091** |
| All-statistics AU 100D | 0.8182 | 0.8595 | 0.8182 | 0.7273 | 0.9091 |
| Landmark 110D | 0.5000 | 0.4959 | 0.5000 | 0.4545 | 0.5455 |
| Minimum AU + 110D, 130D | 0.8182 | 0.8347 | 0.8182 | 0.7273 | 0.9091 |
| All-statistics AU + 110D, 210D | 0.7727 | 0.8843 | 0.7727 | 0.7273 | 0.8182 |
| Three-task AU 300D Logistic | 0.8182 | 0.8843 | 0.8182 | 0.7273 | 0.9091 |
| Three-task shrinkage LDA | 0.7727 | 0.7355 | 0.7727 | 0.7273 | 0.8182 |
| Three-task compact TCN | 0.5000 | 0.4628 | 0.5000 | 0.2727 | 0.7273 |

For the strict best candidate, the 5,000-replicate class-stratified participant bootstrap 95% intervals were `0.682–1.000` for accuracy and `0.843–1.000` for AUROC. The interval width reflects the small 22-person cohort.

## What the negative ablations mean

The TCN did not fail because H200 was too slow; it failed because 22 participants are insufficient to estimate a temporal neural model reliably. Adding KISS and OPEN also reduced performance, showing that more actions are not automatically more informative for ALS. Class-balanced training, inner threshold calibration, shrinkage LDA, and AU+110D fusion did not rescue strict accuracy.

The NeuroFace result does not prove that PalsyNet 110D has identity leakage. PalsyNet used participant-disjoint development evaluation and the frozen model still has registered development AUROC `0.9804` and balanced accuracy `0.9524`. Instead, NeuroFace shows that unilateral Bell's-palsy asymmetry geometry is not a universal ALS feature: the frozen 110D representation alone was at chance on ALS versus healthy controls.

## Broader robustness context

The `0.9504` AUROC is intentionally restricted to the paper-matched 11 ALS
versus 11 healthy-control endpoint. It is not a claim about all NeuroFace
diagnoses. The separately preregistered 36-person ALS/post-stroke-versus-control
action-capacity experiment reached AUROC `0.753` and balanced accuracy `0.744`.
That result used a different cross-disease endpoint and is not pooled into the
ALS result. Together, the two experiments show that endpoint-specific fitting
can recover an ALS signal, but a single representation is not yet robust across
Bell's palsy, ALS, and post-stroke weakness.

## Model policy

- Bell's palsy / Mayo path: keep frozen 110D; evaluate it only after participant-level Mayo binary/HB labels and verified controls arrive.
- ALS / NeuroFace path: lock the SPREAD AU-statistic selector; do not keep tuning on these same 22 outcomes.
- A future fused clinical model should have dataset-appropriate heads or a newly collected multi-disease training cohort. One binary head should not silently reinterpret Bell's palsy, ALS, and stroke as the same target.

## Next gate

The next valid way to improve accuracy beyond `0.90` is new participant evidence, not another NeuroFace hyperparameter search. Freeze the current pipeline and validate it on an untouched ALS-versus-healthy cohort. For Mayo, collect verified negative controls plus participant-level binary and HB labels, then compare 110D, action-specific capacity, and any preregistered fusion on participant-disjoint splits before one untouched external test.

Machine-readable aggregate: `docs/results/artifacts/neuroface_als_architecture_v1/report.json`.
