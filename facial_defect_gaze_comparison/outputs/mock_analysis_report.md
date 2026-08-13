# Webcam vs professional gaze comparison

> **100% synthetic demonstration. Research-only and nonclinical.** These are not observed Mayo data, not observed Prolific data, and not an iMotions result. No patient images, recordings, gaze exports, or identifiers are included.

## What this demonstration answers

This mock run shows how a future paired study can compare a webcam gaze estimator with a professional **reference instrument, not error-free ground truth**. It keeps recruitment source, cohort/site, tracker modality and model, and acquisition software/version as separate factors.

It does not answer whether a real webcam workflow is clinically acceptable. The simulated scenarios exist to verify that the pipeline reacts differently to near agreement, systematic spatial bias, temporal lag, and high data loss.

## Synthetic results at a glance

| Scenario | Webcam accuracy error (°) | Reference accuracy error (°) | Map SIM |
|---|---:|---:|---:|
| High Dropout | 1.06 | 0.29 | 0.83 |
| Near Equivalent | 0.48 | 0.28 | 0.93 |
| Systematic Bias | 2.89 | 0.29 | 0.52 |
| Temporal Lag | 0.84 | 0.28 | 0.85 |

The demo produced 12 equivalent, 4 not-equivalent, and 0 inconclusive endpoint/scenario decisions. All use **illustrative margins** from `config/mock_study.json`; they are not acceptance thresholds for a real study. Failure to show equivalence is not proof of inequivalence.

## How to read the figures

1. `01_data_quality.png` keeps calibration error, precision, data loss, and sampling rate separate.
2. `02_accuracy_bland_altman.png` uses one aggregated point per synthetic participant, avoiding pseudoreplication across targets.
3. `03_map_similarity.png` shows two complementary density-map metrics rather than a single universal score.
4. `04_matched_heatmaps.png` uses the same scale within each row and overlays the known synthetic face coordinate frame.
5. `05_aoi_dwell.png` compares versioned eye, nose, mouth, and outside-AOI shares.
6. `06_equivalence_intervals.png` visualizes the exact three-way decision rule.
7. `07_participant_qc_matrix.png` helps locate synthetic comparison units that drive disagreement.
8. `08_temporal_alignment.png` reports estimated lag after common 30 Hz resampling; no silent time shift is applied.

## Evidence gates

Paired map, AOI, Bland–Altman, and equivalence analysis require the same comparison unit and versioned stimulus under both modalities, a common coordinate/time frame, sufficient valid samples, and externally justified preregistered margins for real observations. If the existing Prolific and Mayo exports are separate cohorts rather than paired observations, this pipeline can provide descriptive summaries but **cannot establish interchangeability**.

Vendor-derived fixation events are not compared in this version because algorithms can differ. Reliability ceilings are also deferred until independent repeated sessions are available. The professional instrument is never labeled as truth.

## Recommended next collection

- Record both streams simultaneously for the same consented participant and exact stimulus presentation.
- Preserve display geometry, viewing distance, viewport-to-stimulus transform, timestamps, validity/blink flags, tracker model, acquisition software version, and AOI/stimulus versions.
- Include known-location calibration and validation targets before and after free viewing.
- Preregister primary endpoints, exclusion rules, bootstrap unit, multiplicity family, and clinically/scientifically justified equivalence margins before unblinding results.

## Reproducibility

- Synthetic project ID: `SYN-GAZE-COMPARE-001`
- Random seed: `20260813`
- Analysis mode: paired mock demonstration
- Raw and real export directories are ignored by Git in this public project.

See `docs/metric_spec.md` for formulas and decision rules and `docs/study_protocol.md` for the proposed real acquisition design.
