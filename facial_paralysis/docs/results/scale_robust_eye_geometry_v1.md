# Scale-Robust Eye Geometry v1

## Outcome

The current raw mirror-invariant 110D Landmark Logistic model remains locked.
Window-local median filtering slightly improved probability calibration but
reduced balanced accuracy overall and in the low-face-scale stress subset.  It
therefore failed the preregistered PalsyNet-only promotion gate.

| PalsyNet development representation | AUROC | Balanced accuracy | Brier | Low-scale AUROC | Low-scale balanced accuracy | Low-scale Brier |
|---|---:|---:|---:|---:|---:|---:|
| **Raw 110D** | **0.980392** | **0.952381** | 0.117367 | **0.959596** | **0.954545** | 0.132597 |
| Eye median-3 110D | 0.977591 | 0.904762 | 0.117036 | **0.959596** | 0.909091 | 0.130405 |
| All-landmark median-3 110D | 0.977591 | 0.904762 | **0.115393** | **0.959596** | 0.909091 | **0.128699** |

The low-scale subset was fixed before scoring as the lowest mean-face-scale
half of reviewed groups within each binary label: 20 groups in total.  All
models used the same four identity-disjoint PalsyNet folds, original-plus-
mirror training, `StandardScaler`, L2 Logistic `C=0.01`, and threshold 0.5.

## Exactly how the four windows are selected

Each window is 32 consecutive frames.  For a PalsyNet video with `N` frames,
the four start frames are

`floor(i × (N - 32) / 3)`, for `i = 0, 1, 2, 3`.

This places one window at the beginning, approximately one-third and two-thirds
through the available start range, and one window ending at the final frame.
At frame rate `f`, adjacent start times are therefore approximately
`(N - 32) / (3f)` seconds apart.  The interval is video-dependent, not fixed.

Mayo uses a related but not identical rule.  It first probes time-spread frames
for successful face detection, then places four equally spaced target starts
between the earliest and latest usable face anchor.  Each target is moved to
the nearest usable start while enforcing non-overlapping 32-frame windows.
Face presence is the only selection signal; labels, model scores, and action
names are not used.

The number four is an engineering cache contract that spans a recording with a
small fixed 128-frame budget.  It is not derived from the FACES clinical action
protocol and has no one-to-one correspondence with the eight actions.

## How much of the Mayo action script is currently used

Across the 47 content-deduplicated scoreable Mayo videos:

- exactly 128 frames per video enter the 110D representation;
- sampled-frame fraction has median 1.8997%, range 1.5008%–3.1833%;
- window-start gaps have median 36.80 seconds, range 21.68–47.70 seconds.

The four windows may incidentally capture several prompted expressions, so the
motion is not necessarily neutral.  However, the current cache does not know
which action appears in which window, can miss short actions, and pools all
four windows into one 110D vector.  Per-action coverage, per-action accuracy,
and action order are therefore undefined.  The eight-action protocol is
substantially underused by the current classifier.

## Scale-robust filtering tested

The two candidates changed only the representation, not the classifier:

- eye median-3 filtered the 10 eye-related `clinical23_v2` channels;
- all-landmark median-3 filtered all 23 eye/brow/mouth channels;
- a centre frame was filtered only when its previous, current, and next frames
  were all valid in the same window;
- filtering never crossed a window or detector gap, never touched Blendshape
  channels, and commuted with the horizontal-mirror transform.

Because smoothing reduced balanced accuracy, neither candidate was applied as
a replacement.  The post-lock Mayo result is therefore unchanged at 45/47
positive calls (95.74% positive-call rate, not binary accuracy).

## Next experiment

The next independent Goal should be **Action-aligned Clinical Dynamics v1**:
recover or build one clip per prompted action (repose, brow raise, gentle eye
closure, forced eye closure, smile, pucker, lower-teeth, reanimated smile),
extract the same clinical geometry within every action, preserve an action
mask, and compare:

1. current four-window raw 110D;
2. fixed per-action 110D pooling with the same Logistic classifier;
3. a small shared per-action encoder with masked action attention.

Model selection must remain PalsyNet-development-only unless a different
labeled training cohort with identity-disjoint subjects is explicitly frozen.
Because current Mayo labels are all assumed positive, Mayo can test confidence
consistency after lock but cannot choose the action model or establish
specificity/accuracy.

## Audit

The machine-readable aggregate report is
`outputs/dynamic_landmark/benchmarks/external/scale-robust-eye-geometry-v1/report.json`.
It contains no record/group identifier, source digest, filename, or raw path.
Protected PalsyNet cache loads, feature extractions, fits, and predictions are
all zero.
