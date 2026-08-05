# Mirror-Invariant Landmark 110D v1

## Goal

Remove horizontal-mirror direction as a nuisance from the existing Landmark
110D classifier without changing its feature dimension, classifier family, or
regularization. This is a development-only robustness test, not model tuning.

## Frozen protocol

- Dataset: validated PalsyNet `clinical23_v2_windows` cache.
- Target: binary affected versus unaffected, not House-Brackmann grade.
- Isolation: outer fold 0 receives no candidate-110D feature construction,
  scaler fitting, model fitting, or prediction. The shared cache loader still
  validates schema and quality over all records before splitting.
- Baseline: the existing 110D vector, standardized L2 logistic regression,
  fixed `C=0.01`, grouped sample weights, and threshold `0.5`.
- Candidate: the same model trained on original plus horizontally mirrored
  trajectories. Validation probability is the mean of the original and
  mirrored probabilities.
- Comparison: four-fold grouped inner OOF predictions and 5,000 paired,
  class-stratified group bootstraps. No hyperparameter search is allowed.

The candidate passes only if mirror probability error is at most `1e-12`,
AUROC and balanced accuracy do not decrease, Brier score does not increase,
and all protected-use audit counts remain zero.

## Result

| Candidate | AUROC | Balanced accuracy | Sensitivity | Specificity | Brier |
|---|---:|---:|---:|---:|---:|
| Standard 110D | 0.9384 | 0.9048 | 0.8095 | 1.0000 | 0.1352 |
| Mirror-invariant 110D | 0.9440 | 0.9048 | 0.8095 | 1.0000 | 0.1328 |

All fixed gates passed and the maximum mirror probability error was `0.0`.
The AUROC difference was `+0.0056`; its descriptive paired-bootstrap 95%
interval was `[0.0000, 0.0224]`. The candidate is therefore the more robust
development successor, but the interval does not support a statistically
reliable superiority claim.

The 10-recording protected partition had zero candidate-110D feature
construction, model fits, and predictions. Identity remains unreviewed, so
this is video-held-out PalsyNet development evidence only: it is not Mayo
performance, HB accuracy, patient-held-out validation, clinical validation, or
deployment evidence.

## Reproduction

```bash
KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/run_mirror_invariant_110d.py \
  --palsynet-cache-root data/external/palsynet/derived/clinical23_v2_windows
```

The script writes a closed-schema aggregate report under
`outputs/dynamic_landmark/benchmarks/development/mirror-invariant-110d-v1/`.
Run it from a clean checkout or archive an earlier local report first; it
refuses to overwrite an existing report.
