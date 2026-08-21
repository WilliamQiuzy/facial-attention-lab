# Shared Normal-Manifold Phenotype Router v1

## Objective and boundary

Build a genuinely shared facial-motor representation for PalsyNet development,
NeuroFace, and MEEI. The shared representation must improve the weakest cohort,
especially healthy-control specificity, without using Mayo, protected PalsyNet,
participant identifiers, or a dataset-specific encoder. Dataset identity may
select only the endpoint head after the shared patient representation.

The experiment is adaptive development evidence. It is not Mayo validation,
HB grading, clinical validation, or proof that one representation is causally
superior. The frozen 110D UCR4 remains the current model unless this experiment
passes its release gate.

## Medical and methodological rationale

Dynamic facial-palsy studies show that facial weakness alters regional movement
magnitude and velocity, and that the clinically opposite side is not guaranteed
to be normal. The model therefore retains the full observed bilateral signal and
does not normalize one side against the other as a presumed healthy reference.
It uses only laterality-commutative view aggregation for binary affected/control
endpoints; it makes no affected-side claim.

NeuroFace includes bilateral neurological motor impairment, whereas PalsyNet and
MEEI emphasize facial palsy. Affected phenotypes should therefore not be forced
to share one disease centroid. Healthy controls, however, provide the common
cross-source reference. Training may compact only healthy participant embeddings
around one shared normal anchor while supervised endpoint losses are free to
separate distinct affected phenotypes.

The shared representation is organized into evidence families rather than an
opaque single bottleneck:

1. 110D clinical geometry and its bilateral difference;
2. full 478-landmark neutral-referenced movement;
3. brow, eye, oral, and whole-face excursion;
4. real-time movement velocity;
5. cross-action coordination proxy.

The coordination feature is a predictive kinematic proxy, not a synkinesis
score. PalsyNet has no authenticated prompt-aligned dense mesh in this cache, so
it trains the clinical/shared-patient path but never fabricates dense action
evidence.

Primary rationale:

- dynamic 3D facial movement and contralateral abnormalities, PMID 30534499;
- maximum displacement/velocity versus regional clinical grading, PMID 27480299;
- lip and jaw kinematics in ALS, PMID 29800359;
- video-based neurological speech/motor assessment, PMID 36367528.

## Frozen v4 screen

All candidates keep the strongest stable v2 evidence path (`MSC2-022`) and add
one shared normal anchor. The screen crosses only:

- normal-manifold weight: `0.00`, `0.05`, `0.20`;
- shared universal-normality blend into each endpoint: `0.25`, `0.50`.

The task head remains small and begins only after the shared patient embedding.
Healthy compactness is computed only from training-fold controls. Scaling,
optimization, and selection remain inside six participant-disjoint folds.
Seed 0 screens all six candidates; the locked top two receive seeds 1 and 2.

Primary selection is minimum source balanced accuracy, followed by minimum
source specificity, minimum source AUROC, mean balanced accuracy, then candidate
ID. A useful result requires three-seed mean balanced accuracy at least 0.90 and
specificity at least 0.85 in every source. Raw accuracy remains reported.

## Conditional v5 iteration

If v4 does not pass, compute pairwise cosine similarity of the three source
gradients on the shared patient layer before changing optimization. Only if a
negative pair is observed may v5 project conflicting shared gradients. If no
negative conflict is observed, v5 instead replaces the flattened full-mesh
projection with a shared landmark-identity plus anatomical-region set encoder,
preserving point-level information before brow/eye/oral/global pooling. This
conditional rule prevents adding PCGrad or a larger encoder without evidence.

No random flip, color/crop augmentation, presumed-normal contralateral
normalization, dataset-specific trunk, score-driven landmark selection, or Mayo
selection is allowed.

## Frozen v5 conflict-aware follow-up

The completed v4 seed-0 screen did not pass. Its strongest candidate was
`NMR4-001`; shared patient-layer gradient cosine was negative for
PalsyNet/NeuroFace and PalsyNet/MEEI, while NeuroFace/MEEI was positive. This
triggers the predeclared conflict branch without changing representation,
labels, folds, heads, thresholds, or input features.

V5 locks `NMR4-001` and crosses two transparent optimizer settings:

- projection scope: shared patient block only, or every shared parameter except
  endpoint heads;
- removal strength: one half or all of each pairwise negative component.

Projection uses only training-fold source losses. Positive-gradient components
are unchanged. Task-head gradients are never projected. Seed 0 screens all four
settings and the top two receive seeds 1 and 2 under the unchanged v4 ranking
and pass gate.

## Frozen v6 script-aware shared follow-up

V5 seed 0 did not improve the weakest source because its aggregate projection
scope showed nonnegative combined gradients and therefore made almost no
optimization change. V6 addresses the remaining architectural mismatch: the
three cohorts use different action scripts, but v2-v5 forced one identical
action pooling rule before a linear endpoint head.

V6 keeps the entire 110D/full-478 per-action encoder and cross-action encoder
shared. A tiny endpoint query then weights the already shared action tokens for
the relevant script; the weighted token and masked maximum pass through the
same shared patient projection. Only the query and final small binary head are
endpoint-specific. This is equivalent to allowing different clinical forms to
combine common motor measurements differently, not learning three encoders.

The four locked candidates cross final head `{linear, 64-16-1 MLP}` and shared
universal-normality blend `{0.25, 0.50}`. No input, fold, label, threshold,
augmentation, or shared width changes. Seed 0 screens all four and the top two
receive seeds 1 and 2.

## Frozen v7 response-statistic shared follow-up

V6 three-seed confirmation remained unstable, and even the seed-0 oracle
threshold could not raise NeuroFace accuracy to 0.90. V7 therefore changes the
shared input representation, not the decision threshold. For every dense action
and every MediaPipe coordinate it computes neutral-referenced median response,
10th and 90th percentiles, range, and maximum absolute real-time velocity.
Original/mirror views are combined by mean and absolute difference, without
calling either side normal.

Within each outer training fold only, all available action rows from NeuroFace
and MEEI fit a label-free StandardScaler and PCA. PalsyNet never enters the dense
fit because its dense evidence is unavailable. The PCA action vector and 110D
clinical vector then enter one trainable action encoder, Transformer, patient
projection, and universal head shared by all sources. Endpoint-specific
parameters remain only a script query and final binary head.

The four locked candidates cross PCA width `{64, 128}` and head
`{linear, 64-16-1 MLP}`. Universal blend is fixed at 0.25 from the strongest v6
ranking-preserving candidate. Seed 0 screens all four; the top two receive seeds
1 and 2 under the unchanged worst-source gate.

## Frozen v8 shared-core residual follow-up

V7 improved PalsyNet but materially reduced NeuroFace, showing that label-free
variance compression removed low-amplitude discriminative neurological motion.
V8 returns to the full 478D+110D script-aware shared encoder (`SAR6-002`) and
adds a small endpoint residual only after the common patient embedding. The
residual represents disease-specific deviation from common facial-motor
physiology; it cannot read raw landmarks or bypass the shared action encoder.

Every candidate shares the clinical, dense, regional, cross-action, and patient
projection weights. Endpoint-specific parameters are limited to the existing
script query, final head, and a bottleneck residual adapter. The six locked
candidates cross adapter rank `{8, 16, 32}` and residual scale `{0.25, 0.50}`.
No dataset-specific trunk, input feature, threshold, or fold changes. Task-
specific parameters must remain below ten percent of total trainable parameters.

## Frozen v9 staged endpoint adaptation

V8 improved NeuroFace ranking but the small residual adapters remained close to
their near-zero initialization after only 20 joint updates. V9 locks the best
seed-0 v8 candidate (`RSR8-001`) and keeps its 20 joint updates, then freezes
every shared parameter. Only the script queries, rank-8 residual adapters, and
small binary heads receive a second stage. The two frozen durations are 40 and
80 updates at learning rate 0.001. This cannot change the shared core or create
dataset-specific raw-feature encoders.
