# Script-Conditioned Action Capacity v1

## Decision question

Can the current unilateral-asymmetry model be complemented by a separately
validated action-capacity branch that retains low or absent bilateral motion
after a verified instruction, without pretending that unscripted PalsyNet
videos contain the FACES protocol?

This is an exploratory representation study. It cannot establish Mayo
accuracy, House-Brackmann performance, Bell's-palsy transfer, clinical
validity, or deployment readiness because Mayo still lacks verified labels and
matched controls.

## Corrected architecture

The system has two deliberately different branches:

1. `landmark_110d_asymmetry`: the current frozen four-window 110D model for
   free-form video. It remains the Bell's-palsy development model.
2. `scripted_action_capacity`: one small model per externally identified
   action. It measures absolute bilateral range and speed, so a prompted action
   with little motion remains a low-capacity observation rather than becoming
   missing.

The branches are not fused in v1. Their targets and evidence differ: the first
was trained on PalsyNet affected/unaffected labels; the second is trained only
as an exploratory neurological oro-facial impairment model on NeuroFace. A
Bell's-palsy fusion requires labeled scripted Mayo patients and controls.

This corrects the archived Action-Aligned 110D experiment, which selected seven
independent visual peaks and pooled them back into one vector. Its slots
followed FACES script order in 0/39 PalsyNet development recordings and only
4/47 Mayo recordings, so it was not true script alignment.

## Data and action identity

- The FACES script contains Neutral/Repose, Eyebrow Raise, Gentle Eye Closure,
  Tight Eye Squeeze, Relaxed Smile, Lip Pucker, Lower Teeth Show, and optional
  Reanimated Smile. Every hold is three seconds.
- NeuroFace provides separate, filename-identified task recordings. The three
  fixed primary tasks map to FACES mouth actions: `NSM_KISS` to lip pucker,
  `NSM_OPEN` to lower-teeth/mouth opening, and `NSM_SPREAD` to smile spread.
  These recording-level task identities are exogenous; visible motion is not
  used to decide whether the action was attempted.
- PalsyNet is unscripted public video. It is not used to train or validate the
  capacity branch. Missing visible action in PalsyNet is unknown, not zero
  capacity.
- Historical Mayo has 53 media files; 51 contain an audio stream and two do
  not. No recording-level event log has been found. Audio alignment is a
  feasibility gate, not assumed ground truth.

## Authoritative segment contract

Every scripted recording must have a recording-relative sidecar with:

- schema and script version;
- SHA-256 of the exact recording;
- timing source: `capture_event_log`, `audio_forced_alignment`,
  `blinded_manual`, or the NeuroFace-only `recording_task_label`;
- for every action: status, prompt start, hold start, hold end, and optional
  completion time in integer milliseconds;
- monotone, non-overlapping intervals bounded by the decoded recording
  duration.

`capture_event_log` is preferred. Before audio decoding/transcription, create
an owner-private audit registry containing the 12 lexicographically smallest
source SHA-256 values among the content-deduplicated, audio-bearing Mayo cohort;
publish only the registry SHA-256. Two blinded annotators mark the six required
prompt/hold intervals and adjudicate any label disagreement or boundary
difference above 500 ms. Match predictions to references one-to-one by action
label and maximum IoU. Pooled precision/recall use all 72 required events:
missing detections are false negatives, extra detections are false positives,
and wrong labels contribute both. Unmatched reference events receive IoU zero;
the gate uses the median across all 72 values. Audio alignment is eligible only
if pooled precision and recall are each at least 0.95, median IoU is at least
0.80, and the locked audit set contains at least two manually verified prompted
but visually flat responses. If the hash-selected set lacks two such attempts,
the bilateral-weakness timing gate fails; records may not be substituted. The
two audio-free Mayo recordings require blinded manual timestamps or must
abstain.

The segmenter emits separate boolean fields:

- `prompted`: an exogenous source proves the instruction occurred;
- `observed_motion`: action-like geometry is visible;
- `tracking_adequate`: at least 80% of the 32 uniformly sampled hold positions
  have valid landmarks and every feature channel has sufficient finite support;
- `eligible = prompted AND tracking_adequate`.

Low motion never changes `prompted` or `eligible`. Invalid landmark positions
remain masked and cannot be converted to zeros or interpreted as flat motion;
feature extraction uses only valid timestamps and fails if fewer than 26 of 32
positions remain. The three-second hold is
represented by exactly 32 positions sampled uniformly in recording time;
nearest decoded frames may be reused only when the source frame rate is below
10.34 Hz. Timestamp error must be no more than half one decoded frame. A visual
curve may select an apex for secondary reporting inside the hold, but may not
move or delete the anchored interval.

An unanchored video is ineligible for scripted training or scoring. A visual
order proposal may support manual review, but cannot establish attempted
actions.

NeuroFace uses a separate recording-level contract because each released file
contains one named task rather than the multi-action FACES script. Its
authenticated filename task is the exogenous `recording_task_label`; the
attempt interval is the complete decoded recording. The representation reuses
the already frozen cache exactly: four non-overlapping contiguous 32-frame
windows, with starts evenly spaced from frame zero to `frame_count - 32` by
`deterministic_window_starts`. No visual peak, hold boundary, or new window is
selected. The existing `candidate_feature_vector(LANDMARK_MI_110D, ...)`
summarizes all four windows jointly into one 110D recording vector without
cross-window derivatives; the fixed name-bound projection then selects one
18D capacity vector. The mirrored four-window cache is summarized in the same
way, so each participant/task contributes exactly two 18D training rows
(original and mirror), never `4 x 18D` rows.

## Frozen capacity representation

Each known-action NeuroFace recording is summarized independently. The branch
uses exactly 18 absolute mouth-capacity features from the existing Landmark
110D names:

- IQR, range, and maximum absolute velocity for left and right oral-corner
  vertical displacement;
- the same three summaries for left and right oral-corner horizontal
  displacement;
- IQR, range, and maximum absolute velocity for global mouth width and mouth
  opening.

Medians, signed side differences, absolute side differences, correlations,
amplitude ratios, and lag features are excluded from this branch. This makes
the estimand bilateral movement capacity rather than lateral asymmetry or
static pose. Horizontal mirroring swaps the two side blocks and leaves the
global block unchanged.

## Frozen NeuroFace experiment

Use the 36 participants and 231 technically retained videos from the prior
frozen inventory; no QC rule changes are allowed. The primary endpoint uses
only the three universally retained primary tasks above.

Use the already frozen six participant-disjoint NeuroFace folds. Within every
fold, fit three separate standardized L2 Logistic models (`C=0.01`, threshold
0.5), one per task. Inside each task-specific fit, every participant's
original-plus-mirror rows have total sample weight one, split equally between
the two rows; this preserves the intended effective regularization. At
inference, average each held-out original/mirror probability first. A held-out
participant then receives exactly one probability per task and the unweighted
mean of the three is the primary capacity score. Any missing primary task
causes participant-level abstention; no fallback or complete-case substitution
is allowed.

The binary target is frozen as `healthy_control=0` and both `als=1` and
`post_stroke=1`. Bootstrap draws are stratified by the three original cohorts:
sample 11 healthy-control, 11 ALS, and 14 post-stroke participants with
replacement inside their own cohort for every draw, preserving the released
cohort sizes and then recomputing the binary participant metrics.

Report participant-level AUROC, average precision, Brier, balanced accuracy,
sensitivity, specificity, each task's AUROC/coverage, and 5,000 cohort-
stratified participant bootstrap intervals. Report a mask-only diagnostic;
the expected primary mask is identical for all 36 participants. The known
frozen 110D NeuroFace result is a descriptive comparator, not rerun or used for
selection.

Because NeuroFace outcomes are already exposed, no metric promotes this branch
into the Bell's-palsy model. A lower 95% AUROC bootstrap bound above 0.50 is
only a feasibility signal that action-conditioned absolute capacity contains
neurological impairment information.

## Protected and clinical boundaries

- PalsyNet protected files and manifests must not be opened, hashed, stat-ed,
  globbed, cached, fitted, or predicted. This goal does not read PalsyNet at
  all. The H200 experiment runs in a container that mounts only the frozen
  NeuroFace input root and the new output root; no PalsyNet directory is
  mounted. Host audit records the exact mount list and process file accesses.
- Mayo is not used to fit or select a classifier. Until action timestamps are
  audited, only audio/timeline feasibility counts may be reported.
- Raw videos, audio, timelines, identifiers, row predictions, and model caches
  remain owner-private and outside Git.
- Future labeled Mayo evaluation must be participant-disjoint and report both
  branches separately before any fusion is preregistered.

## Stop criteria

Retain the frozen 110D as the only Bell's-palsy model if any of the following
occurs:

- exact NeuroFace task identity or participant-disjoint folds cannot be
  reproduced;
- any primary participant/task is missing after the frozen QC;
- action-capacity metrics cannot be recomputed independently;
- historical Mayo audio alignment fails the fixed blinded timing gate;
- any protected PalsyNet path is accessed.
