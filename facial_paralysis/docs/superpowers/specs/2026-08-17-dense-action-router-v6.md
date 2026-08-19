# Universal Clinical Router v6 dense-action candidate

## Objective

Build one additive, source-blind action-geometry expert that raises the
participant/group-disjoint development accuracy of all three UCR profiles to at
least `0.93`. The frozen UCR4 free-recording 110D expert remains unchanged.
The new expert is eligible only for recordings with authenticated action/task
evidence; it must not infer a dataset or institution from the input.

The primary development gate is:

- PalsyNet development accuracy `>= 0.93`;
- NeuroFace development accuracy `>= 0.93`;
- MEEI development accuracy `>= 0.93`;
- balanced accuracy `>= 0.90` for every profile;
- Mayo reads/predictions and PalsyNet sealed-outer reads remain zero.

These are exposed development cohorts. Passing creates a development candidate,
not clinical validation or authorization to overwrite UCR4.

## Failure structure being addressed

UCR5 established that confidence rejection cannot solve the remaining errors.
The frozen scripted NeuroFace route has three errors among 36 participants, and
the cue-aligned MEEI route has seven errors among 56 participants. MEEI errors
include high-confidence affected participants, so threshold adjustment and
expert-disagreement rules are insufficient. The next representation must expose
subtle action-specific geometry that the 23-channel clinical summary discards.

## Representation

The new representation is named `dense_bilateral_action_v1`.

For every sampled video frame, MediaPipe FaceLandmarker returns the complete
478-point mesh. Coordinates are transformed in pixel space by:

1. translating to the midpoint of mesh anchors 33 and 263;
2. rotating the eye line to horizontal;
3. dividing by the inter-outer-eye distance;
4. retaining normalized `x`, `y`, and normalized depth `z` only when finite.

The extractor processes both the original frame and a horizontal image flip;
mirror equivalence therefore comes from actual FaceLandmarker inference rather
than an incomplete hand-written 478-index swap table.

For each prompted action, the action frame sequence is compared with an
exogenous rest/baseline segment. Per coordinate, the frozen statistics are:

- median action geometry;
- median action-minus-rest response;
- 10th and 90th response quantiles;
- response range;
- maximum absolute adjacent response change.

MEEI uses its authenticated prompt intervals and REST hold. NeuroFace uses its
authenticated task-labelled recordings and the first/last label-blind edge
frames as baseline. Missing detections remain masked; an action requires at
least six valid observations and a baseline requires at least four. No zero
imputation may represent a missed face.

## Candidate architectures

The initial H200 development screen compared four representation-level families:

1. `dense_sparse_logistic`: train-fold univariate rank, standardization and L2
   Logistic over the concatenated action tensor;
2. `dense_action_experts`: one shared feature transform with task-specific L2
   Logistic heads and equal participant-level probability aggregation;
3. `dense_rbf`: train-fold rank followed by a compact RBF SVM;
4. `dense_ucr4_fusion`: a convex combination of a dense expert and the frozen
   cross-fitted UCR4 route probability.

This screen was exploratory and used the exposed development cohorts to choose
the final profile configurations. It therefore does not provide an untouched
or fully nested estimate of architecture-selection performance. After the
screen, the exact rank sizes, regularization values, aggregations, fusion
weights and standard `0.5` decision threshold were frozen in
`scripts/run_dense_action_router_v6.py`.

## Evaluation and stopping rule

- Six deterministic participant/group-disjoint folds per action cohort for the
  locked configuration.
- Feature rank, scaling and model fit are learned only from each fold's training
  participants. The fixed UCR4 input remains its previously cross-fitted OOF
  probability, and the decision threshold is always `0.5`.
- Report accuracy, balanced accuracy, sensitivity, specificity, AUROC, Brier,
  error count and participant count for every profile.
- A candidate passes only when all primary gates pass on one frozen aggregate
  report. No selective abstention may be counted as a correct prediction.

The development gate can register a candidate but cannot authorize promotion,
because the architecture was selected after viewing these cohorts. A later
untouched participant- or institution-disjoint study must confirm it. The gate
and threshold may not be relaxed after seeing results.

## Data and release boundaries

- Raw video, dense mesh caches, labels, participant identifiers and row-level
  probabilities remain owner-private on H200.
- Git contains only code, tests, the frozen registry, aggregate metrics and
  cryptographic commitments.
- Mayo is neither training nor validation data in this experiment.
- The PalsyNet sealed outer split is not opened.
- `docs/model_registry.json`, `src/models/current.py` and the UCR4 artifact
  remain byte-identical unless a later untouched study authorizes promotion;
  `docs/CURRENT_MODEL.md` may only add an explicit non-current candidate note.
