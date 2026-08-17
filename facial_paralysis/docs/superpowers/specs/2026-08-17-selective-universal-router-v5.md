# Selective Universal Clinical Router v5 candidate

## Decision

Universal Clinical Router v4 remains the sole current model.  This experiment
may add a source-blind selective-decision layer, but it must not refit, replace,
or change any v4 probability.  A rejected case is reported as `abstain`; its
absence must never be counted as a correct classification.

The candidate addresses the next practical failure mode: v4 ranks affected
versus unaffected participants well on three exposed development cohorts, but
its full-cohort accuracy is below 0.95 on NeuroFace and MEEI.  Prior shared
models, deep temporal architectures, global/action fusion, GroupDRO, MIL and
Set Transformer searches did not improve all three routes.  The next bounded
structural attempt is therefore a confidence-and-consensus layer over the
already-frozen clinical experts.

## Claim boundary

- PalsyNet development, NeuroFace and MEEI are adaptive development evidence.
- The sealed PalsyNet outer result is inherited unchanged and is not opened,
  rescored, or used to choose v5.
- Mayo has no usable control/HB labels and is not used to fit or select v5.
- Selective accuracy is always accompanied by coverage and abstention count;
  it is never described as full-cohort accuracy or clinical accuracy.
- Passing this experiment creates a development candidate only.  Promotion of
  the current model still requires an untouched participant-disjoint cohort.

## Frozen inputs

For every participant, the private H200 evaluator reconstructs the exact v4
out-of-fold final probability and its already-defined component-head
probabilities.  Dataset or institution identity is not a model input.  The
router selects the evidence profile from authenticated task/modality evidence:

1. `free_asymmetry`: original-view and mirrored-view 110D probabilities;
2. `scripted_multimechanism`: clinical probability and the 18 frozen MARLIN
   phenotype probabilities;
3. `cue_aligned_upper`: the two frozen sequence-head probabilities.

The final probability, fold-specific OOF decision threshold, component
probabilities, binary reference and anonymous participant group are bound in a
private NPZ. PalsyNet and NeuroFace use constant 0.5 vectors. MEEI retains the
six nested outer-training thresholds so its published development decisions
remain reproducible; the final artifact's single aggregate threshold is a
separate runtime estimand.
Public output contains aggregate counts/metrics and cryptographic commitments,
never row probabilities, identifiers, paths, raw media or labels.

## Fixed candidate registry

The registry is closed before private participant-level feasibility is read.
All scores are deterministic and label-free at inference.

1. `probability_margin`: absolute distance between the frozen final probability
   and its frozen decision threshold.
2. `range_penalized_margin`: final margin minus half the component-probability
   range; strong expert disagreement lowers confidence.
3. `unanimous_min_margin`: minimum component distance from the decision
   threshold when every component predicts the final class, otherwise a
   negative confidence score.
4. `dispersion_normalized_margin`: final margin divided by one plus four times
   the component standard deviation.

No candidate may use source name, cohort name, reference label, participant
identifier, file metadata, background pixels or raw video.

## Evaluation

Each evidence profile is evaluated at fixed top-confidence coverages of 0.60,
0.70, 0.75, 0.80, 0.90 and 1.00.  Selection count is
`ceil(coverage * participants)` with deterministic score/index tie breaking;
labels cannot change which rows are retained.  Report for every point:

- retained participants and coverage;
- selective accuracy;
- selective balanced accuracy, sensitivity and specificity when both classes
  remain present;
- class-specific coverage;
- abstention count and errors among retained cases.

Primary gate is fixed at 0.70 requested coverage.  A candidate passes only if,
for all three evidence profiles:

- realized coverage is at least 0.70;
- both reference classes remain represented, with at least five retained cases
  from each class;
- selective accuracy and selective balanced accuracy are both at least 0.95;
- no profile performs worse than v4 full-cohort accuracy on its retained set.

Among passing candidates, select the highest minimum selective balanced
accuracy, then highest minimum selective accuracy, then highest mean 0.60–1.00
risk-coverage utility, then registry order.  If none passes, v5 is rejected and
v4 remains current.  Secondary 0.60/0.75/0.80/0.90/full results are descriptive
and cannot rescue a primary-gate failure.

## Robustness checks

- exact input schemas, finite float64 probabilities and unique anonymous groups;
- final probability and component count aligned one-to-one;
- decisions recomputed from frozen thresholds;
- permutation invariance of component columns where the score is symmetric;
- rejection of labels/source fields as score inputs;
- aggregate report recomputed from the private evidence rather than trusted;
- no protected PalsyNet reads and no Mayo predictions;
- v4 model/report SHA-256 values unchanged before and after the experiment.

## Version maintenance

The experiment is named `universal_clinical_router_v5_candidate`.  It receives
its own artifact directory and candidate registry entry.  `docs/model_registry.json`,
`src/models/current.py`, `docs/CURRENT_MODEL.md` and the v4 artifact remain
unchanged unless a later untouched validation explicitly authorizes promotion.
