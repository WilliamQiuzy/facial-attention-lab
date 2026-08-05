# Model card: Landmark 110D

## Summary

Landmark 110D is a development-stage binary facial-paralysis research model.
MediaPipe is the upstream landmark detector. The model reduces each recording
to dynamic, clinically interpretable eye, brow, and mouth geometry and applies
a standardized L2 logistic regression with fixed parameters.

## Intended use

- research on dynamic facial asymmetry;
- reproducible representation comparisons;
- collaboration on identity-disjoint and externally validated studies;
- future ordinal HB, Sunnybrook, and eFACE work after trustworthy labels exist.

It is not intended for diagnosis, treatment, clinical decision support,
severity grading, or deployment.

## Input and output

The feature builder accepts the four deterministic, evenly spaced,
non-overlapping 32-frame windows spanning a recording under the frozen
`clinical23_v2` contract, plus validity masks, timestamps in seconds, and
adjacent source-frame indices together with the original recording frame count.
At least 90% of sampled frames must be valid; masked rows must be canonical
zero. It emits one ordered vector of 110 finite floating-point values.

The fixed classifier uses standardized L2 logistic regression with `C=0.01`,
`liblinear`, `max_iter=2000`, random state `0`, threshold `0.5`, and equal total
training weight per group. The output is an exploratory affected-versus-
unaffected score, not a calibrated clinical probability or HB grade.

## Development evaluation

- Dataset: PalsyNet.
- Target: affected versus unaffected.
- Development: 39 recordings / 38 provisional groups.
- Class groups: 21 affected / 17 unaffected.
- Evaluation: four-fold grouped inner out-of-fold predictions.
- Protected partition: 10 recordings / 10 groups, never extracted, fit, or
  predicted for the reported result.
- Claim unit: video-held-out; identity status remains unreviewed.

The 110D candidate achieved AUROC `0.938375350140056`, balanced accuracy
`0.9047619047619048`, sensitivity `0.8095238095238095`, and specificity `1.0`.
Small counts and unresolved identity mean these figures have wide uncertainty
and are not patient-level or clinical validation.

## Known limitations

- PalsyNet identity and action coverage have not completed human review.
- The protected outer partition has not been evaluated.
- PalsyNet is web-derived and is not an independent clinical cohort.
- Mayo currently lacks row-level labels, verified patient mapping, and healthy
  controls; its all-positive screening run cannot estimate specificity, AUROC,
  balanced accuracy, or negative predictive value.
- Mirror provenance is unresolved, so capture-side geometry cannot be called
  anatomical patient left/right.
- MediaPipe may localize landmarks less accurately on severe palsy faces.
- No HB, Sunnybrook, or eFACE claim is authorized.

## Release contents

This release contains source, tests, documentation, and deidentified aggregate
PalsyNet results. It deliberately excludes media, patient-derived features,
identifiers, per-record predictions, private reports, and fitted weights.
