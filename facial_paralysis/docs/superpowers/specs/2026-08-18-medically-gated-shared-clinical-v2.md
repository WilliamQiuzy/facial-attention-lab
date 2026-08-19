# Medically-Gated Shared Clinical Encoder v2

## Non-negotiable objective

Every candidate keeps one trainable clinical/action/patient encoder shared by
PalsyNet, NeuroFace, and MEEI. Dataset identity may select only the final small
endpoint head after the shared patient embedding. The optimization target is
the worst of the three participant-disjoint accuracies among shared candidates;
Universal Clinical Router v6 is historical context, not the comparator.

## Medical-rationale gate

A component enters the frozen candidate registry only when it has all four:

1. a named facial-function phenomenon;
2. primary clinical or technical evidence;
3. the labels for which the operation is valid;
4. an explicit contraindication or interpretation limit.

The gate admits these components:

- **Neutral-referenced movement.** Sunnybrook separates resting symmetry from
  voluntary excursion, and dynamic facial-motion studies compare movement with
  rest. Dense trajectories therefore remain action-minus-rest while the 110D
  branch retains absolute clinical geometry.
- **Regional brow, eye, and oral evidence.** Sunnybrook/eFACE and automated
  facial-palsy studies score facial regions and standard expressions separately.
  Region summaries use frozen MediaPipe contour indices, never labels or
  data-driven landmark selection.
- **Excursion and velocity.** Dynamic facial-paralysis studies report impaired
  magnitude, velocity, and zone-specific asymmetry. Candidate summaries may use
  excursion and adjacent-frame velocity on real timestamps.
- **Laterality-invariant bilateral magnitude.** A binary weakness label is
  unchanged when palsy occurs on the opposite side. A true-mirror pair may be
  combined only through commutative mean and absolute-difference operators.
  Signed affected-side prediction, regional HB subscores, or laterality claims
  may not use this invariant representation.
- **Action-to-region routing.** Brow raise maps to brow, eye closure to the eye
  region, and smile/pucker/open-mouth tasks to the oral region. Free-video
  PalsyNet windows use the global face because no prompted action is known.
- **Shared multi-task trunk with endpoint heads.** PalsyNet, NeuroFace, and MEEI
  have different binary endpoint semantics. Their heads may differ only after a
  common 64D patient embedding; all sources must update the same trunk.

Primary evidence:

- Ross et al., Sunnybrook system, PMID 8649870,
  https://pubmed.ncbi.nlm.nih.gov/8649870/
- Banks et al., eFACE validation, PMID 26218397,
  https://pubmed.ncbi.nlm.nih.gov/26218397/
- Trotman et al., dynamic 3D movement, PMID 30534499,
  https://pubmed.ncbi.nlm.nih.gov/30534499/
- Machine Learning Methods to Track Dynamic Facial Function in Facial Palsy,
  PMID 40333095, https://pubmed.ncbi.nlm.nih.gov/40333095/
- MediaPipe canonical facial contours,
  https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/python/vision/face_landmarker.py

## Frozen 32-candidate registry

All candidates use the same 110D clinical branch, optional full
`32 x 478 x 3` dense branch, source-class-balanced loss, six
participant-disjoint folds, task heads, and 20 updates. The registry is the
Cartesian product of:

- view: `original_only`, `bilateral_invariant`;
- regional evidence: `none`, `all_excursion`, `matched_excursion`,
  `matched_excursion_velocity`;
- action pooling: `meanmax_set`, `cross_action_transformer`;
- fusion: `masked_concat`, `reliability_gate`.

No candidate changes hidden width, optimizer, learning rate, folds, threshold,
or seed. Seed 0 screens all 32 candidates. The locked selection rule ranks by
minimum source accuracy, then minimum source AUROC, then mean accuracy, then
candidate ID. The top four receive seeds 1 and 2. Stop when one three-seed
candidate has mean accuracy above 0.90 in every source, or after all medically
admissible registered candidates are exhausted.

## Evaluation and Mayo boundary

Report each source separately; never substitute pooled accuracy. Scaling and
any threshold calibration are fitted inside training folds. PalsyNet protected
test data and Mayo are unreadable during candidate selection. Mayo accuracy and
HB performance remain unknown until Mayo labels and participant-disjoint splits
exist. A future Mayo ordinal head attaches after the same shared patient
embedding.

## Post-registry compact falsification family

After the frozen 32-candidate screen and top-four confirmation failed the
three-source gate, a separately named 16-candidate v3 family tested the
specific overfitting hypothesis without changing the protected boundary. It
removed flip, replaced the high-capacity raw-mesh projection with fixed
brow/eye/oral/global excursion and velocity summaries computed from all 478
points, and compared mean/max versus shared action weighting and flexible
endpoint heads versus monotone calibration of one shared severity axis. These
choices preserved the same clinical phenomena and shared-training objective.
The family was stopped after its complete seed-0 screen because its best
NeuroFace and MEEI accuracies remained below the v2 screen; it was not promoted
or used to redefine the v2 selection rule.
