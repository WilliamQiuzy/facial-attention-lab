# Independent-cohort Webcam vs professional gaze comparison plan

**Goal:** Build a reproducible, research-only analysis for 500 Webcam/Prolific participants and 500 different professional/Mayo participants, using synthetic data until approved real inputs are available.

**Identifiable estimand:** Workflow-and-cohort distribution similarity for prespecified group-level gaze endpoints. This design cannot identify a pure device effect, individual agreement, or device interchangeability.

## Completed implementation

- [x] Keep the project separate from the clinician-facing React apps.
- [x] Generate deterministic, disjoint 500+500 synthetic participant cohorts.
- [x] Preserve participant characteristics, acquisition context, quality endpoints, common stimuli, AOIs, and device-neutral dwell events as separate fields.
- [x] Gate analysis on common stimulus, task, exposure, transform, and QC versions.
- [x] Compare covariate balance with standardized mean differences.
- [x] Compare calibration accuracy, RMS precision, data loss, and valid-trial share with Welch 90% CIs and independent-sample equivalence decisions.
- [x] Compare group heatmaps using repeated within-cohort and cross-domain half-sample SIM distributions.
- [x] Compare versioned facial-AOI dwell profiles.
- [x] Run separate repeated cross-validated source classifiers for technical and attention-pattern features.
- [x] Produce seven question-led figures, ten tables, an evidence-limited Markdown report, and a compact executable Notebook.
- [x] Add a detailed Chinese analysis guide and update the public repository index.
- [x] Preserve the earlier paired code as a secondary future method, while making the independent-cohort workflow the documented default.

## Verification record

- [x] Independent-cohort and CLI tests: 14 passed.
- [x] Earlier paired-method regression tests: 50 passed.
- [x] 500+500 synthetic inputs regenerated from seed `20260813`.
- [x] Clean-kernel Notebook execution succeeded.
- [x] Seven figures visually inspected; overlapping headings were corrected and the map-reliability axis was tightened for readability.
- [x] Output and input manifests include row counts, sizes, and SHA-256 hashes.
- [x] `git diff --check` passed.

## Before real-data use

- [ ] Map the real exports into the documented participant, fixation/dwell, stimulus, AOI, and provenance contracts.
- [ ] Confirm both cohorts share exact stimulus bytes and analysis versions; otherwise restrict to a common subset.
- [ ] Replace every illustrative mock tolerance with a clinician- and methods-justified preregistered margin.
- [ ] Approve the primary use case: group heatmap, AOI summary, or another named endpoint.
- [ ] Keep videos, facial images, gaze exports, and linkage keys in approved access-controlled storage, never this public repository.
- [ ] If the goal becomes individual device interchangeability, design a new same-person paired study rather than repurposing this analysis.
