# Independent-cohort gaze comparison protocol

## Status

This is a research planning template, not an approved human-subjects protocol. IRB, privacy, security, consent, retention, and data-transfer requirements remain the responsibility of the study team.

## Design and estimand

The current design contains two independent samples:

- approximately 500 participants recruited remotely and measured by a Webcam workflow;
- approximately 500 different participants measured by a professional reference workflow.

The estimand is **workflow-and-cohort distribution similarity for a prespecified group-level endpoint**. The design does not identify a pure device effect because participant composition, recruitment source, site, environment, display, camera, and algorithm may all differ together. It cannot estimate individual-level agreement or interchangeability.

The professional instrument is a reference workflow, not error-free ground truth. Prolific is a recruitment source and iMotions is acquisition software; neither should be encoded as the device.

## Non-negotiable comparison gates

Both cohorts must share:

1. identical stimulus bytes, IDs, and versions;
2. identical task instructions and exposure duration;
3. the same randomization logic, or a recorded and modeled ordering difference;
4. one versioned viewport-to-stimulus coordinate transform;
5. the same primary QC and exclusion rules;
6. known-target calibration/validation suitable for the stated accuracy metric;
7. compatible AOI definitions and versions.

If a gate fails, restrict the analysis to a common harmonized subset or treat the result as exploratory. Equal sample size does not repair a protocol mismatch.

## Collection record

Record only justified variables, but preserve enough context to interpret differences:

- study pseudonym and independent cohort ID;
- recruitment source and collection site;
- hardware modality/model and acquisition algorithm/software/version;
- browser, OS, camera resolution and frame rate;
- display resolution and physical size, zoom/device pixel ratio, viewing distance;
- lighting, glasses/contact lenses, head-motion or face-loss indicators;
- stimulus, task, transform, AOI, and QC rule versions;
- expected samples/trials, valid samples/trials, and structured invalidity reasons;
- known calibration target positions and pre/post validation drift where available.

Do not store direct identifiers in analysis filenames or tables. Raw gaze, video, and facial imagery may be identifiable and belong in approved encrypted storage, never this public repository.

## Analysis sequence

### Gate 1 — protocol and cohort comparability

- Produce a participant-flow table by cohort.
- Verify the common stimulus/task/transform/QC versions.
- Plot standardized mean differences for prespecified participant characteristics.
- Show acquisition-context variables separately; do not silently adjust away real workflow differences.

### Gate 2 — technical quality

Primary endpoints:

- calibration angular accuracy error;
- RMS precision at stable calibration targets;
- data-loss proportion;
- valid-trial share.

Use one summary per participant. Report cohort distributions, Webcam − Professional differences, Hedges' g, and Welch 90% CIs. Use independent-sample TOST-style equivalence only with margins chosen before unblinding.

Sampling rate is contextual and expected to differ by technology. Do not declare a lower frame rate a failure unless the scientific endpoint requires a prespecified minimum temporal resolution.

### Gate 3 — group attention distribution

- Transform all valid dwell events to the same normalized stimulus coordinates.
- Apply the same grid and smoothing rule to both cohorts.
- Weight by dwell duration when the event definition is device-neutral and versioned.
- For every stimulus, repeatedly split each cohort into two random halves.
- Compare within-Webcam, within-professional, and cross-domain half-cohort maps using histogram intersection (SIM).
- Evaluate cross-domain loss relative to the lower within-cohort repeatability benchmark using a prespecified noninferiority margin.

This bootstrap estimates stability under participant sampling. It does not recreate individual paired agreement.

### Gate 4 — domain distinguishability

Fit simple repeated cross-validated classifiers separately on:

1. technical-quality features; and
2. attention-pattern features such as stimulus-by-AOI dwell shares.

Report AUC and participant-bootstrap intervals. Never include a perfectly identifying administrative field such as site or recruitment source. AUC near 0.5 means this model did not separate the cohorts; it does not prove the distributions are identical.

## Multiplicity and endpoint priority

Preregister one primary intended use, such as:

- a population heatmap;
- AOI rank/share estimation; or
- individual gaze localization.

The current independent design is most defensible for the first two. Define primary and secondary endpoint families, margins, and any Holm adjustment before outcome inspection. Do not create one unvalidated composite “device quality score.”

## Sensitivity analyses

Plan a small, interpretable set:

- common-support restriction or weighting for materially imbalanced participant characteristics;
- exclusion of low-light or low-resolution Webcam sessions;
- alternate prespecified smoothing bandwidth;
- fixation-count versus duration-weighted maps;
- stimulus-stratified results rather than one pooled map;
- complete-case versus QC-threshold cohort flow.

Report whether conclusions change; do not search many variants and report only the favorable one.

## Recommended future causal comparison

If the research question becomes “does the Webcam device itself agree with the professional tracker?”, collect both streams on the same consented participant and same presentation, ideally simultaneously. If simultaneous acquisition is not possible, use randomized, counterbalanced repeated sessions and model retest/order variation. That future paired design answers a different question from this independent-cohort study.

## References

- Semmelmann and Weigelt, [Online Webcam-based eye tracking in cognitive science](https://pmc.ncbi.nlm.nih.gov/articles/PMC8787048/) — independent online/lab groups and data-quality dimensions.
- Yang and Krajbich, [Webcam-based online eye-tracking for behavioral research](https://pmc.ncbi.nlm.nih.gov/articles/PMC11289017/) — simultaneous device comparison and synchronization/quality methods.
- Lakens, [Equivalence tests: a practical primer](https://pmc.ncbi.nlm.nih.gov/articles/PMC5502906/) — TOST and smallest effect size of interest.
- Lopez-Paz and Oquab, [Revisiting classifier two-sample tests](https://arxiv.org/abs/1610.06545) — classifier-based distribution diagnostics.
