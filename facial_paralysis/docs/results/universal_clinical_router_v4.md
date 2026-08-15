# Universal Clinical Router v4

## Outcome

We froze one research artifact that selects a clinical expert from authenticated
task and modality evidence, never from a dataset or institution name. It keeps
the existing 110D asymmetry model for free recordings, uses Landmark + Py-Feat
AU + frozen MARLIN phenotype heads for the three NeuroFace tasks, and uses the
cue-aligned Landmark sequence ensemble for the seven-action MEEI/Mayo-style
protocol. Missing required evidence fails closed.

This is a stronger multi-protocol development model, but it is not a claim of
clinical robustness. NeuroFace and MEEI were repeatedly exposed during model
development; Mayo still lacks participant-level control labels and HB grades.

## Participant-disjoint results

Accuracy is the fraction classified correctly at the frozen decision rule.
Balanced accuracy averages sensitivity and specificity so the majority class
cannot dominate. AUROC measures affected-versus-unaffected ranking across all
possible thresholds.

| Evidence profile and cohort | People | AUROC | Accuracy | Balanced accuracy | Sensitivity | Specificity | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| Free asymmetry — PalsyNet development | 38 | 0.980 | 0.947 | 0.952 | 0.905 | 1.000 | 0.117 |
| Free asymmetry — sealed PalsyNet outer | 10 | 1.000 | 0.900 | 0.900 | 0.800 | 1.000 | 0.118 |
| Scripted multimechanism — NeuroFace | 36 | 0.931 | 0.917 | 0.889 | 0.960 | 0.818 | 0.086 |
| Cue-aligned upper/action geometry — MEEI | 56 | 0.911 | 0.875 | 0.885 | 0.870 | 0.900 | 0.191 |

The primary development routes therefore exceed 0.90 AUROC on all three data
sources. They do **not** yet exceed 0.90 accuracy on MEEI, and the small cohort
confidence intervals remain wide. The NeuroFace fixed-ensemble participant
bootstrap 95% intervals are 0.811–1.000 for AUROC and 0.833–1.000 for accuracy.

## What the architecture search established

- A single shared 110D or Fusion-398 classifier did not transfer to NeuroFace;
  it mainly learned unilateral palsy asymmetry and missed bilateral ALS weakness.
- Small end-to-end residual, TCN, MIL and Set-Transformer models overfit the
  38/36-person cohorts. The strongest durable change was to freeze the large
  video encoder and train small phenotype-specific clinical heads.
- On NeuroFace, two endpoint heads first detect post-stroke asymmetry and ALS
  oromotor reduction. The maximum clinical score is retained when confident;
  only scores from 0.2 to 0.8 are replaced by the median of 18 fixed MARLIN
  phenotype heads. This fixed ensemble, not a fold-specific best model, produces
  the reported 0.931 AUROC and 0.917 accuracy.
- On MEEI, adding 36 cue-aligned MARLIN variants failed: MARLIN alone reached
  only 0.589–0.648 AUROC, while nested fusion selection fell to 0.859 AUROC and
  0.839 accuracy. MARLIN was therefore rejected for this route; the 110D-derived
  action sequence ensemble remains frozen.
- We also extracted Py-Feat action units at four authenticated cue times for all
  56 usable MEEI participants. Across 32 AU-only candidates, the exploratory
  best reached only 0.591 AUROC and 0.607 accuracy. A stricter nested comparison
  of 145 Landmark/AU rules fell to 0.600 AUROC and 0.786 accuracy, so this large
  representation change was rejected rather than being allowed to dilute the
  stronger Landmark sequence expert.

These negative results are part of the model decision. A larger network was not
promoted merely because it had more capacity.

## Model and evidence boundary

- The release contains no participant identifiers, row probabilities, raw
  media paths or credentials.
- The universal work read 39 identity-reviewed PalsyNet development recordings;
  protected PalsyNet reads, fits and predictions were all zero. The sealed outer
  result above is inherited unchanged from its earlier one-shot release.
- Mayo data and labels were not used for candidate selection. The current Mayo
  positive-call rate is not accuracy because there is no negative/control class.
- This artifact does not grade House-Brackmann severity and is not authorized
  for diagnosis, deployment or a clinical performance claim.

The executable model is
`docs/results/artifacts/universal_clinical_router_v4/model.json`. The public
runtime is `src/models/universal_clinical_router_v4.py`.
