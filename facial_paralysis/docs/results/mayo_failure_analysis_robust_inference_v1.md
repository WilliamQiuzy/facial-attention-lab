# Mayo Failure Analysis + Robust Inference v1

## Outcome

The frozen `110D Landmark + mirror-mean Logistic` remains the current model.
Three preregistered mirror-view aggregation rules were compared using only the
identity-reviewed PalsyNet development folds.  All three produced the same
group-level metrics to reported precision, so the tie-retention rule blocked
promotion.  No protected PalsyNet cache record was loaded, fitted, or predicted.

## PalsyNet-only model decision

| Aggregation | AUROC | Balanced accuracy | Brier | Decision |
|---|---:|---:|---:|---|
| Mirror probability mean | 0.980392 | 0.952381 | 0.117367 | Retained |
| Mirror logit mean | 0.980392 | 0.952381 | 0.117367 | No improvement |
| Mirror positive-class maximum | 0.980392 | 0.952381 | 0.117367 | No improvement |

The comparison used 39 development recordings from 38 reviewed identity
groups.  Each validation prediction came from a model fitted on the other
registered groups.  Logistic regularization remained fixed at `C=0.01`, and
the classification threshold remained fixed at 0.5.

## Mayo aggregate challenge

The Mayo cohort remains an assumed-positive, single-class confidence
challenge.  It cannot measure ordinary accuracy, specificity, balanced
accuracy, or AUROC.

- 47 content-deduplicated videos were scoreable.
- 45/47 were called positive at the fixed 0.5 threshold: positive-call rate
  0.957447, Wilson 95% interval 0.857515–0.988252.
- The two below-threshold scores were 0.402286 and 0.426225.
- Their mean extraction coverage was 0.992188, versus 0.994618 in the
  positive-call records; failed face detection therefore does not explain the
  low scores.

## What differs in the two low-confidence videos

This is an aggregate association analysis with only two below-threshold
records, not a causal claim.

- Mean face scale was 1.02 cohort standard deviations lower than in the
  positive-call records.  Smaller faces are the strongest measured capture
  difference and can reduce landmark precision even when face detection
  coverage is high.
- Their eye-region contribution to the fitted Logistic logit averaged −0.573,
  compared with +0.123 in positive-call records (shift −0.696).
- Brow and mouth contributions were also lower (shifts −0.350 and −0.189), but
  the eye region had the largest adverse shift.
- Duration was higher by 0.82 cohort standard deviations, while luminance,
  motion, roll, detection rate, and bitrate showed smaller shifts.  Duration
  alone is not a model input and does not establish a failure mechanism.

## Technical interpretation and next representation target

Changing how two mirrored scores are combined cannot recover information that
is absent or noisy in both views.  The next fast, preregistered experiment
should therefore operate before the classifier: a **scale-robust eye geometry
v1** representation.  It should use the same small fixed Logistic model and
the same PalsyNet identity-disjoint folds, while comparing the current 110D
features with landmark trajectories re-extracted from a deterministic enlarged
face crop and with eye-region reliability summaries.  Promotion must be based
only on PalsyNet development non-inferiority plus improved low-face-scale
stability; Mayo may be rescored once only after that candidate is locked.

## Safety and provenance

The machine-readable aggregate report is
`outputs/dynamic_landmark/benchmarks/external/mayo-failure-analysis-robust-inference-v1/report.json`.
It contains no per-video identifier, content hash, filename, or path.  Mayo was
not used to select an aggregator or tune a threshold.  The report records zero
protected cache loads, feature extractions, model fits, and predictions.
