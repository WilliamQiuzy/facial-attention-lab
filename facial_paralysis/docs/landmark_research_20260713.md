# Landmark Research and Implementation Decision — 2026-07-13

> **Historical decision record:** The static Landmark/Fusion ablation below is
> superseded for current-model reporting by the 110D Landmark trajectory result
> in `CURRENT_MODEL.md`.

## Decision

Landmarks should be added to the current blendshape stream, but the evidence
does **not** support flattening all 478 raw coordinates and assuming that more
dimensions will improve generalization. The supported progression is:

1. retain all raw MediaPipe landmarks and validity/provenance;
2. compute clinically interpretable bilateral geometry per frame;
3. model rest-relative, action-specific bilateral trajectories;
4. fuse a small landmark encoder with the existing blendshape encoder;
5. consider a dense fixed-topology graph only after external pretraining and a
   patient-held-out labeled validation set exist.

The production `clinical23_v2` is therefore a 23-dimensional clinical landmark baseline,
available alongside the existing 72-dimensional blendshape stream as a fused
95-dimensional schema. It removes translation, uniform scale, and in-plane
roll through a 2D similarity transform. It does **not** claim to remove yaw,
pitch, or all 3D pose effects.

## Primary evidence

| Study | Landmark representation | Evidence relevant to this project |
|---|---|---|
| [Guarin et al., 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7362997/) | Disease-specific 68-point detector; brow height, palpebral fissure height, commissure excursion | 200 facial-palsy patients and 10 controls across eight expressions. Supports clinical geometry and warns that detectors trained on healthy faces localize palsy faces less accurately. |
| [Kim et al., 2015](https://www.mdpi.com/1424-8220/15/10/26756) | 49 points; bilateral displacement ratios during brow raise and smile | 36 subjects, 88.9% binary accuracy under LOOCV. Supports rest-relative movement and bilateral ratios, while the small private cohort limits the performance claim. |
| [Parra-Dominguez et al., 2022](https://www.mdpi.com/2075-4418/12/7/1528) | 51 effective points transformed into 29 distances, angles, slopes, perimeters, and ratios | Supports a compact, interpretable landmark baseline. Reported performance is likely optimistic because patient/control sources differ and frame leakage is a concern; it is not a target estimate for Mayo. |
| [ten Harkel et al., 2024/2025](https://pubmed.ncbi.nlm.nih.gov/39117479/) | 13 clinical landmarks represented as heatmaps and fused with images | In 116 unilateral facial-palsy patients and 9 controls, adding landmarks improved Sunnybrook composite ICC from 0.87 to 0.91, directly supporting multimodal fusion. |
| [Rao, Greene, and Coleman, 2025](https://pubmed.ncbi.nlm.nih.gov/40333095/) | 19 bilateral landmark pairs over standardized facial cues; rest centering, left/right correlation, Gaussian distributions, Wasserstein and Mahalanobis distances | In 65 facial-palsy patients and 50 controls, dynamic trajectories separated groups and correlated with clinical grading. This is the strongest evidence for action-aware landmark dynamics rather than static coordinate flattening. |
| [Guan et al., 2025](https://www.nature.com/articles/s41746-025-02063-6) | 975 dense, palsy-specific landmarks compressed into eyelid and mouth area-asymmetry coefficients | A derivation cohort and independent prospective validation cohort each contained 274 patients. Dense localization helped, but the clinical prediction used compact regional motion measures rather than a raw coordinate classifier. |
| [Oo et al., 2025](https://arxiv.org/abs/2503.10371) | All 478×2 raw coordinates, 52 expression/blendshape features, 29 handcrafted features, and image fusion compared directly | In 20 palsy patients and 20 controls, raw coordinates were materially weaker than blendshape-like or handcrafted inputs; the best result came from fusion. This is the clearest warning that raw dimension count is not signal quality. |
| [Heinrich et al., 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13113261/) | MediaPipe 478 points converted to 225 bilateral angle pairs, then compared at 225/140/91/50/21-pair resolutions | 198 patients and 405 acquisitions over nine expressions. The informative subset concentrated on eyes, nose, and mouth. Any point selection in our work must occur inside the training fold; we will not label a self-selected subset as their 91-pair reproduction. |

## Current feature contracts

| Arm | Per-frame input | Purpose |
|---|---:|---|
| `blendshape_only` | 52 MediaPipe blendshapes + mirrored signed deltas (runtime count normally 72 total) | Current transferable baseline |
| `landmark_only` | 23 clinical features | Interpretable landmark baseline |
| `fusion` | 72 + 23 = 95 | First landmark-enhanced candidate |

The 23 landmark measurements cover bilateral palpebral fissure height/width,
eye area, brow height, mouth-corner height and excursion, plus mouth width and
opening. Absolute and signed bilateral differences are both retained. Feature
names, order, dimension, and schema version are persisted with every new bundle.
The two topology sides are named by their MediaPipe anchors (`mesh33` versus
`mesh263`, and `mesh61` versus `mesh291`) rather than asserted to be the
patient's right or left. New bundles also persist whether capture pixels were
mirrored. Until that value is known, signed columns are valid model inputs but
must not be interpreted as the patient's affected side.
The producer binds the first detector frame to MediaPipe's exact registered
52-category order, and all first-party MediaPipe bundle writers use the same
schema-aware payload validator. Valid rows and MARLIN embeddings must be finite;
masked action-cache rows are serialized as finite zero padding. This is distinct
from the local QC trajectory files, which deliberately preserve missing rows as
NaN plus an explicit validity mask.

For a fixed-parameter-shape cached-data ablation, all three arms instantiate the same 95-input
model. `blendshape_only` zeros columns 72:95, `landmark_only` zeros columns 0:72,
and `fusion` retains both blocks. This keeps input width and parameter count
constant. The deployed `regasym` recipe still gives blendshapes an additional
engineered expansion, so the landmark-only arm is not a modality-symmetric
capacity test. A separate `feat=raw` sensitivity comparison is reported below.
That cache contains the July 10 signed-gap `legacy_clinical23_v1` calculation.
A frozen compatibility function preserves it exactly for reproduction. New
production extraction is explicitly versioned `clinical23_v2`: vertical
clinical dimensions are nonnegative magnitudes, gross finite outliers fail
closed, and names/schema/provenance are persisted. The static experiment is
therefore historical evidence for landmark complementarity; it does not
directly validate the V2 transform or Mayo dynamics.

## Mayo trajectory audit

The existing raw CSV collection contains 15 exports, 87,988 source-video frames,
and 87,732 landmark frame groups. The final audit processes every stored group;
every present CSV group has all 478 finite points. There are 256 timeline frames
without a CSV landmark group: aggregate valid-landmark coverage is 99.71%, and
the lowest per-take coverage is 97.33%. Coverage is computed against each source
video's frame count, not only the first and last CSV identifiers, so missing
leading or trailing detections are included. The audit also confirms:

- one pair of exports is an exact duplicate and must remain in one split;
- one export contains only 68 frame groups and is too short for the
  same dynamic treatment as the other recordings;
- some recordings have gaps in frame identifiers even though every stored
  group is valid, so temporal modeling must retain timestamps/masks;
- the files contain substantial early-reference variation in eye, brow, mouth
  corner, commissure, and mouth-opening measurements, which makes dynamic
  landmark modeling technically feasible. The early valid frames are not yet
  verified rest cues; these summaries are QC, not action-level training features.

The machine-readable record is
`outputs/landmark_fusion/mayo_clinical23_audit.json`. It is a readiness result,
not an HB/binary accuracy result: the Mayo set still lacks healthy controls and
independent patient-level grades. This audit and the per-take trajectory NPZs
remain local-only and git-ignored because they contain Mayo-derived biometric
features, take/date identifiers, and source provenance. Only the aggregate
counts in this document are intended for version control.
Directory auditing is transactional at the validation boundary: all per-take
sequences are staged first and are promoted only when every input succeeds. A
partial failure therefore cannot overwrite or mix with the previous complete
collection.

## Fixed-width static-web ablation

Using the existing July 10 95-dimensional cache, all arms used the same model
width, initialization seeds `(0, 1, 2)`, records, split, and training budget:

| arm | mean QWK | SD | eyes | mouth |
|---|---:|---:|---:|---:|
| blendshape-only | 0.6517 | 0.0060 | 0.4013 | 0.9020 |
| landmark-only | 0.1954 | 0.0092 | 0.0216 | 0.3692 |
| fusion | **0.6685** | 0.0151 | **0.4402** | 0.8967 |

Fusion gained +0.0168 over the fixed-width blendshape control, mainly in the
eye task (+0.0388), while mouth changed by -0.0052. The legacy static
clinical23 block performed poorly alone under this fixed MLP recipe; that does
not establish that all landmark-only representations are weak. The evidence
supports landmark as a complementary stream, not a replacement for
blendshapes. This recipe adds blendshape-specific `regasym` features after the
95-column mask, so the landmark-only result is additionally confounded by
asymmetric feature engineering. Because only three seeds were run, fusion SD
was 2.5× the control, and per-seed paired confidence intervals were not
persisted, the gain is promising rather than conclusive. The best epoch was
selected and scored on the same internal validation set, so these numbers are
selection-biased exploration rather than holdout performance. It is also a
static web-label result, not Mayo validation.

The modality-symmetric `feat=raw` sensitivity comparison removed the
blendshape-only expansion:

| arm | mean QWK | SD | eyes | mouth |
|---|---:|---:|---:|---:|
| raw blendshape-only | 0.5424 | 0.0075 | 0.2500 | 0.8348 |
| raw landmark-only | 0.1968 | 0.0068 | 0.0191 | 0.3744 |
| raw fusion | **0.5673** | 0.0185 | **0.2997** | 0.8349 |

Here fusion gained +0.0249 overall and +0.0496 on eyes, with essentially no
mean mouth change (+0.0001). Eye QWK improved in all three paired seeds, while
the combined metric improved in two of three. With only three seeds, the paired
95% t interval for the combined delta is wide (`-0.0504` to `+0.1001`) and
crosses zero. This is corroborating evidence for an eye-focused complementary
signal, not statistical confirmation.
The raw landmark-only arm also remained weak under this static clinical23 MLP
recipe. That narrows the claim to this representation/data/training setup; it
does not test dense landmarks or rest-relative video trajectories.

## Next landmark representation

The next approved extension is an action-aware trajectory layer over the 23
features, including rest-to-peak displacement, bilateral amplitude ratio,
trajectory correlation, peak asymmetry, AUC, velocity, time-to-peak, recovery,
and cross-region synkinesis correlations. A `bilateral_angles225_v1` candidate
can follow after the exact MediaPipe pairing contract and pose QA are frozen.
These quantities must be computed after cue/action segmentation; integration
and derivatives must remain within contiguous valid intervals.
For the current recurrent baseline, interior detector gaps are stable-compacted
in temporal order before packed-GRU execution, so later valid frames are not
dropped and masked NaNs cannot poison the state. The retained source frame IDs
are still required by future velocity/AUC features because compaction does not
encode elapsed gap duration.

A dense graph remains v2. It should use fixed FaceMesh topology plus explicit
bilateral mirror edges, node features such as canonical coordinates,
rest-relative displacement, velocity, mirror difference, and a validity mask,
then pool into anatomical regions before a small temporal encoder. A deep
478-node Transformer/GCN trained directly on the current Mayo recordings would
be an overfitting experiment, not a reliable clinical model.

## Validation gates

- split by patient/recording before frames or windows are generated;
- keep exact duplicates in only one split and exclude the short take from the
  primary dynamic experiment;
- fit normalization/scalers on training patients only and save them with the
  schema and manifest hash;
- freeze capture mirror/orientation provenance before assigning patient-side labels;
- inspect landmark overlays at rest and peak for severe palsy, eye closure,
  glasses, facial hair, occlusion, yaw, and pitch;
- compare the three arms with identical subject splits, seeds, training budget,
  decoding, and bootstrap confidence intervals;
- do not promote a landmark model from Mayo-positive recall or landmark
  self-consistency alone; specificity requires controls.
