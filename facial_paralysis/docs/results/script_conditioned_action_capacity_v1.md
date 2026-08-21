# Script-Conditioned Action Capacity v1

## Outcome

This goal adds a separate research-only capacity branch without changing the
frozen 110D Bell's-palsy candidate. The new branch splits known scripted
movements into action-specific observations and fits three small experts, one
each for KISS, OPEN, and SPREAD. Each expert receives an 18D representation of
bilateral mouth movement capacity: IQR, range, and maximum absolute velocity
for the left/right mouth corners, mouth width, and mouth opening. The models
remain standardized L2 logistic regression (`C=0.01`), keeping this feasibility
test focused on action-aligned landmark capacity without an architecture search.

On NeuroFace, six-fold participant-disjoint evaluation included 36 people and
108 primary action recordings. Participant-level AUROC was **0.753** (95%
cohort-stratified bootstrap CI **0.578–0.902**), average precision was 0.891,
balanced accuracy was 0.744, sensitivity was 0.760, and specificity was 0.727.
Action AUROCs were 0.629 for KISS, 0.822 for OPEN, and 0.465 for SPREAD. The
preregistered feasibility gate passed because the lower AUROC confidence bound
was above 0.50. Independent read-only recomputation reproduced all point
metrics and all 5,000 bootstrap intervals exactly.

The frozen 110D result and this fitted NeuroFace capacity result are not a
causal or head-to-head representation comparison: their fitting and transfer
protocols differ, so the observed metrics cannot isolate representation from
endpoint- or domain-specific fitting.

This is a cross-disease orofacial-capacity result—ALS or post-stroke versus
healthy control—not Bell's palsy validation, House–Brackmann grading, Mayo
accuracy, or clinical validation. It does not authorize fusion with or
replacement of the frozen 110D model.

## Why externally anchored actions matter

Selecting a window from the largest observed motion would systematically miss
a correctly prompted but nearly motionless attempt, which is exactly the
pattern expected in severe bilateral weakness. The implemented contract
therefore separates *the action was prompted* from *visible motion occurred*.
FACES requires prompt, event-log, audio-alignment, or blinded manual timing;
NeuroFace uses recording-level task identity bound to its authenticated
manifest. Flat but adequately tracked responses remain eligible low-capacity
observations, while missing or poorly tracked actions fail closed.

## Mayo feasibility audit

The historical Mayo folder contains 53 media files representing 52 unique
contents, including one exact duplicate. Fifty-one source files contain audio,
two do not, and no unique content failed probing. The 52 unique recordings
total 5,357.5 seconds; the median duration is 112.17 seconds.

No capture event logs, audio forced-alignment files, or locked blinded-manual
timing annotations were found. Therefore the Mayo timing gate is not eligible:
the action experts made **zero Mayo predictions**, and Mayo accuracy remains
undefined. A private owner-only registry has frozen the 12 lexicographically
smallest content-deduplicated audio-bearing recordings before transcription;
the public report contains only aggregate counts and its registry commitment.

## Next gate

For the fixed 12 recordings, two blinded annotators should mark six prompted
actions per recording (72 events total), with adjudication for disagreements
over 500 ms. Timing becomes eligible only if pooled precision and recall are at
least 0.95, median IoU across all 72 references is at least 0.80, and at least
two prompted-but-flat attempts are manually verified. Only after that timing
gate—and after Mayo supplies Bell's-palsy labels plus healthy controls—should
the action experts and frozen 110D asymmetry be evaluated participant-disjoint
and reported separately before any fusion. Fusion must then be preregistered
and validated on an untouched split or cohort; it must not be selected and
claimed on the same Mayo participants.

## Evidence

- NeuroFace public report SHA-256: `be246d848e78598c47d0470ff5c4175dbe1a4b6b012fadedbbf0f7494f34c290`.
- Mayo timing-audit public report SHA-256: `b59b1f5fc727858ac6c490ddf0b35f6092070cc45ee398de30faa35f0a8977e6`.
- H200 image: `sha256:25940c2e52d566bbe241b78d8cee2ae72349fb92ed461eb05f2cef540d857c27`.
- Implementation commitment: `d75e865d6a701caca753dbb54f1e79c08845abb5c80300e7f4ea14e920e5904f`.
- Formal runtime: 9.55 seconds; protected PalsyNet reads and predictions: 0.
- The private NeuroFace OOF file remains H200-only; the private Mayo registry is
  ignored by Git and remains mode 0600.
