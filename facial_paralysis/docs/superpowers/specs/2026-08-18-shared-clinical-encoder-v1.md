# Dense-Clinical Shared Encoder v1

## Objective

Build one trainable clinical representation whose parameters receive supervised
gradients from PalsyNet development, NeuroFace, and MEEI. The purpose is transfer
to future participant-disjoint Mayo evaluation, not three unrelated models.
Because the three cohorts encode different binary endpoints, their small output
heads separate only after one shared patient embedding; a low-weight universal
auxiliary head keeps that embedding aligned.

Universal Clinical Router v6 remains a frozen exposed-development comparator.
Its three high-performing routes are not evidence of a shared representation and
must not be renamed or promoted by this experiment.

## Common representation

Every recording is converted to a variable-length bag of action tokens. Each
token has a required name-bound 110D clinical branch and, when the authenticated
cache contains it, a dense `frames x 478 x 3` landmark branch:

- PalsyNet: four fixed 32-frame recording windows become four `FREE_WINDOW`
  tokens;
- NeuroFace: KISS, OPEN, and SPREAD recordings become three prompted-action
  tokens per participant;
- MEEI: the seven authenticated prompted actions become seven action tokens per
  participant.

The dense branch is normalized for translation, roll, and inter-eye scale and
then converted to action-minus-rest response before the same spatial layer and
lightweight temporal layer encode every dataset. Original and true-mirror views
are paired explicitly so bilateral disagreement is available to the model. The
110D branch remains the stable interpretable signal and is derived from absolute
dense geometry through the same clinical23 transform. Missing
dense evidence is represented by an explicit modality mask, never a zero-valued
normal face. PalsyNet v1 caches currently train the clinical and shared patient
layers; NeuroFace and MEEI additionally train the dense encoder. A later
authenticated PalsyNet full-mesh extraction can fill that modality without
changing the model interface.

## Model

The primary model has fold-local clinical scaling, one shared dense spatial and
temporal encoder, one shared clinical encoder, gated fusion, one shared masked
cross-action encoder, and one shared 64D patient embedding. Three linear binary
heads represent the different cohort endpoints only after that embedding. A
fourth universal head contributes 0.25 of the training loss but is not the
reported within-cohort endpoint. Source identity selects a final head and never
enters the shared encoder. Action-name embeddings describe prompted movement,
not the dataset. All sources must produce non-zero gradients in the shared
clinical and cross-action layers; dense-enabled sources must additionally
update the same dense encoder.

The completed bounded smoke compared:

1. 110D-only shared two-layer set Transformer with task heads;
2. baseline-centered dense-clinical shared Transformer with task heads.

The initial single-head design was rejected because cohort label semantics
differ. No task head may bypass the shared patient embedding, and task heads
together must remain under five percent of trainable parameters.

## Evaluation

- Six participant-disjoint folds, stratified by source and binary label.
- Fold-local scaling, fitting, early stopping, and any candidate selection.
- Source-balanced loss so PalsyNet, NeuroFace, and MEEI each contribute one
  third of supervised training mass.
- Report accuracy, balanced accuracy, AUROC, sensitivity, specificity, and
  Brier separately for all three sources.
- A future promoted candidate must report three leave-one-source-out tests; the
  current smoke stopped before this stage because it remained below V6.
- Keep the decision threshold at 0.5 for the first smoke; any later calibrated
  threshold must be learned inside training folds and shared across sources.

The development objective is lexicographic: maximize the minimum source AUROC,
then minimum source balanced accuracy, then minimize worst-source Brier. V6
within-source results are a no-regression reference, not the sole objective.

## Mayo boundary

Mayo has no reliable binary controls or HB labels yet. It is not used for
supervised fitting, model selection, or an accuracy claim. After the shared
candidate is locked, unlabeled Mayo may be used only in a separately
preregistered mirror-consistency or masked-reconstruction stage. Once Mayo HB
labels arrive, the shared encoder is initialized from this checkpoint and an
ordinal HB head is trained with participant-disjoint splits; 110D, shared
binary, and ordinal results are reported separately before any fusion.

## Stopping rule and claim boundary

The first loop is intentionally short: two retained candidates, seeds 0/1/2,
20 updates, plus bounded diagnostic probes at 40 and 100 updates. Stop after the
three-seed comparison if the shared candidate remains below V6; do not promote
or keep adding configurations on the same exposed development participants.

Passing this experiment creates an exposed-development shared-transfer
candidate only. It is not Mayo accuracy, HB validation, clinical validation, or
evidence that a 0.94 result will generalize to a new institution.
