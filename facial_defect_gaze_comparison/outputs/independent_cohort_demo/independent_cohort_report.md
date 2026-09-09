# Independent-cohort Webcam vs professional gaze comparison

> **100% synthetic demonstration; research-only and nonclinical.** These are not observed Mayo data, not observed Prolific data, and not an iMotions result. There are no patient images, recordings, raw gaze exports, or identifiers in this demo.

## The design we are actually analyzing

This demonstration has **500 Webcam participants and 500 professional-camera participants, with different participants in the two groups**. No person-level pairing is performed or implied.

Because participant, recruitment source, setting, device, lighting, and display can all differ together, this design **cannot isolate a pure device effect**. It estimates whether the *complete Webcam/Prolific workflow-and-cohort distribution* is sufficiently similar to the *professional/Mayo workflow-and-cohort distribution* for a prespecified group-level endpoint, provided both cohorts saw the same versioned stimuli under harmonized instructions.

**“Same domain is not one yes/no property.”** Technical acquisition features may be distinguishable while group attention patterns remain close. The decision must therefore name the endpoint.

## Results at a glance

| Priority | Research question | Method | Mock result |
|---:|---|---|---|
| 1 | Were the two collections made comparable? | Protocol and common-stimulus gates | `pass` |
| 2 | Are core technical quality differences acceptably small? | Welch 90% CI plus independent-sample equivalence margins | `all_primary_endpoints_within_mock_margins` |
| 3 | Are group attention maps as close as repeated cohort samples? | Repeated split-half bootstrap and cross-domain map similarity | `group_maps_close_to_sampling_repeatability` |
| 4 | Can a simple model identify the technical acquisition domain? | Repeated cross-validated logistic-regression AUC | `clearly_distinguishable` |
| 5 | Can a simple model identify the attention-pattern domain? | Repeated cross-validated logistic-regression AUC | `low_detectability` |

These results are deliberately illustrative. All tolerances in `config/mock_independent_study.json` are **illustrative margins**, not Mayo acceptance thresholds and not clinical criteria.

## 1. First gate: are the cohorts and tasks comparable?

Use `01_covariate_balance.png` before looking at outcomes. Standardized mean differences (SMDs) place continuous and binary characteristics on one scale. SMD is more useful than a large-sample p-value here: with 500+500 people, a tiny unimportant difference can be statistically significant. The |SMD| < 0.10 band is a review convention, not evidence of randomization.

- Participant characteristics help reveal recruitment differences.
- Display size and lighting are acquisition-context variables; they describe the real workflow difference and should not be silently “adjusted away.”
- If stimuli, task version, timing, transform, or exclusion rules differ, stop the primary comparison or restrict it to the common harmonized subset.

## 2. Technical data quality: estimate differences and test practical similarity

| Endpoint | Why it matters | Webcam | Professional | Difference (90% CI) | Mock decision |
|---|---|---:|---:|---:|---|
| Calibration accuracy error | Checks closeness to known calibration targets. | 1.141 | 0.644 | +0.496 (+0.472, +0.521) | `similar_within_margin` |
| RMS precision | Checks point-to-point stability, separate from accuracy. | 0.505 | 0.268 | +0.237 (+0.224, +0.249) | `similar_within_margin` |
| Data loss | Checks how much expected gaze signal is missing. | 0.093 | 0.041 | +0.052 (+0.048, +0.056) | `similar_within_margin` |
| Valid-trial share | Checks whether complete participants or trials survive QC. | 0.939 | 0.978 | -0.038 (-0.041, -0.036) | `similar_within_margin` |

### How each primary quality metric is used

| Metric | Importance | Calculation | Visualization | Decision rule |
|---|---|---|---|---|
| Calibration accuracy error | A map can look smooth but be spatially wrong. | Mean angular distance from gaze to independently known calibration targets, one value per participant. | `02_quality_equivalence.png`, `03_quality_distributions.png` | Welch 90% CI for Webcam − Professional must fall inside a preregistered ±margin. |
| RMS precision | Separates jitter from systematic offset. | RMS sample-to-sample displacement during stable calibration targets. | Same two figures | Same independent-sample equivalence rule. |
| Data loss | Missingness can alter maps and reduce usable participants. | 1 − valid expected samples / all expected samples. | Same two figures | Same rule; inspect the upper tail as well as the mean. |
| Valid-trial share | A good mean among survivors can hide many failed trials. | Valid trials / expected trials per participant. | Same two figures | Same rule; higher is better. |

Why a usual “no significant difference” test is not enough: failure to reject zero difference does not demonstrate practical equivalence. Equivalence requires a scientifically justified smallest difference of interest chosen before examining the outcomes.

## 3. Group attention: compare against ordinary sampling variation

| Stimulus | Cross-domain SIM | Webcam split-half | Professional split-half | Gap decision |
|---|---:|---:|---:|---|
| SYN-FACE-01 | 0.961 | 0.963 | 0.964 | `similar_to_within_cohort_repeatability` |
| SYN-FACE-02 | 0.956 | 0.964 | 0.964 | `similar_to_within_cohort_repeatability` |
| SYN-FACE-03 | 0.954 | 0.964 | 0.964 | `similar_to_within_cohort_repeatability` |

The group maps in `04_group_attention_maps.png` show Professional, Webcam, and a signed difference map for every common stimulus. The first two panels share a scale within each row; the difference panel uses a zero-centered scale.

The stronger check is `05_map_reproducibility.png`:

1. Randomly split the 500 Webcam participants into two halves and compare their dwell-density maps.
2. Do the same within the 500 professional participants.
3. Compare one random Webcam half with one random professional half.
4. Repeat the procedure 300 times for each stimulus.

Histogram intersection (SIM, 0 to 1) asks how much normalized dwell density the maps share. Cross-domain similarity is interpreted relative to the lower within-cohort split-half similarity, not against an unrealistic perfect score of 1. The mock noninferiority margin is 0.08 SIM units.

`06_aoi_profile.png` then makes the spatial result clinically readable by summarizing dwell share in versioned left-eye, right-eye, nose, mouth, and outside-AOI regions. AOI results are secondary to the full map because a broad AOI can hide within-region shifts.

## 4. “Same domain?” Use an interpretable source classifier

| Feature set | AUC (95% bootstrap interval) | Interpretation |
|---|---:|---|
| Technical Quality | 0.995 (0.991, 0.998) | `clearly_distinguishable` |
| Attention Pattern | 0.552 (0.514, 0.587) | `low_detectability` |

`07_domain_classifier.png` uses repeated stratified cross-validation. A simple logistic model is asked whether it can identify Webcam versus Professional records:

- **Technical-quality model:** calibration accuracy, precision, loss, and valid-trial share.
- **Attention-pattern model:** stimulus-by-AOI dwell shares only.

AUC = 0.50 is chance; higher AUC means the two samples are more separable in that feature space. This is a diagnostic, not a universal same-domain test. Chance-level AUC does not prove equality, and a high technical AUC does not automatically mean group attention maps are unusable.

## What we can and cannot conclude

This independent design can support statements such as: “for the same stimuli and this protocol, group-level Webcam attention maps were within the preregistered loss margin relative to cohort split-half repeatability.” It cannot support person-level agreement, measurement interchangeability for an individual, or a pure causal device effect.

For a later real study:

1. Freeze the common stimulus set, instructions, exposure time, coordinate transform, and exclusions.
2. Define one primary use case: group heatmap, AOI ranking, or individual gaze location. Do not reuse one margin for all three.
3. Set equivalence/noninferiority margins with clinicians and eye-tracking experts before unblinding.
4. Report age/glasses and acquisition context; consider stratification or weighting as sensitivity analyses when participant composition differs.
5. If the goal changes to individual device interchangeability, collect both devices on the same people, preferably simultaneously and in randomized order where possible.

## Reproducibility and evidence boundary

- Project ID: `SYN-INDEPENDENT-GAZE-500X2`
- Random seed: `20260813`
- Design: two independent cohorts, not paired
- Mock cohort size: 500 per workflow
- Map unit: participant-resampled dwell-density field
- Professional tracker role: reference workflow, not error-free truth
- Pure device effect identified: no

The public repository ignores real/raw export directories. Real participant data, videos, images, and linkage keys belong only in approved access-controlled research storage.
