# NeuroFace External Validation v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply the already frozen PalsyNet Landmark 110D artifact once to all 36 Toronto NeuroFace participants, audit its participant-level transfer and exploratory clinical-score behavior, and quantify MediaPipe clinical-geometry agreement against the released manual 68-point landmarks without refitting or threshold selection.

**Architecture:** Raw Toronto NeuroFace ZIP archives remain read-only outside Git. A strict inventory layer authenticates the six archives, parses the 261 video/landmark pairs and two-rater SLP workbooks, and replaces publisher IDs with opaque IDs. A resumable extractor streams one authenticated AVI member at a time into an owner-private temporary file, reuses the frozen PalsyNet four-by-32-frame `clinical23_v2` contract, and writes only deidentified NPZ caches. A closed evaluator scores the fixed 110D JSON artifact, averages the three universally observed tasks (`NSM_KISS`, `NSM_OPEN`, `NSM_SPREAD`) once per participant for the primary endpoint, and emits aggregate-only metrics. A separate same-frame audit compares MediaPipe and manual-68 `semantic23_v1` geometry on the 3,306 annotated frames.

**Tech Stack:** Python 3.12/3.10, NumPy, SciPy, scikit-learn, OpenPyXL, OpenCV, MediaPipe Tasks, ZIP/JSON/SHA-256, the repository's custom `_testlib` runner.

---

## Locked scientific protocol

- Dataset identity: Toronto NeuroFace v1.0, 36 participants (11 ALS, 14 post-stroke, 11 healthy controls), 261 RGB task videos, 261 manual-landmark files, 3,306 annotated frames.
- External target: neurological oro-facial impairment (`ALS` or `post_stroke`) versus healthy control. This is not Bell's palsy, House-Brackmann grading, Mayo accuracy, or clinical validation.
- Model: the exact frozen `landmark_mi_110d` artifact, SHA-256 `cbc49d0aa54b504915bebd00fdbe005458378e5675b57461ce83d3385f9b60f9`; no fitting, calibration, feature selection, candidate selection, or threshold change on NeuroFace.
- Feature producer: MediaPipe Face Landmarker asset SHA-256 `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`, `clinical23_v2`, exactly four deterministic non-overlapping 32-frame windows, minimum per-video detection coverage 90 percent. The authorization must pin the extraction/evaluation source commit and exact isolated-environment lock before scoring.
- Primary participant score: arithmetic mean of frozen video probabilities for `NSM_KISS`, `NSM_OPEN`, and `NSM_SPREAD`, because every participant has exactly one recording of each task.
- Score direction: larger probability means more compatible with the affected PalsyNet class.
- Primary metrics: participant AUROC and balanced accuracy at threshold `0.5`.
- Secondary metrics: average precision, ordinary accuracy, sensitivity, specificity, Brier score, all-available-task mean aggregation, ALS-vs-control and stroke-vs-control endpoints, and task-wise descriptive endpoints.
- Participant eligibility and QC: the primary endpoint requires all three primary recordings to decode, meet 90 percent coverage, and produce finite frozen scores. QC uses no cohort or SLP values. There is no imputation; excluded participants remain in a flow table with group-hidden technical reason during extraction, and final denominators and exclusions are reported by cohort only after all QC is locked. All-task means use only individually eligible tasks and report each participant's task count; they are secondary because availability is unequal.
- Uncertainty: percentile 95 percent intervals from 5,000 cohort-stratified participant bootstrap draws, seed `20260813`; all repeated videos from a participant remain in one resample cluster. Draws lacking both binary classes or yielding an undefined metric are discarded and the valid/invalid draw counts are reported; fewer than 4,750 valid draws invalidates that interval.
- Clinical association: exploratory participant-level Spearman correlations between the three-task model score and two-rater mean symmetry, range-of-motion, speed, variability, fatigue, and total. Missing either rater excludes that recording from its participant domain mean, with no imputation. Report affected-only ALS and stroke strata separately when each has at least 8 eligible participants; the pooled affected estimate is diagnosis-adjusted by correlating within-diagnosis ranks. All-participant correlations are explicitly labelled spectrum effects. Holm correction is applied across the six domains within each analysis family.
- Landmark audit: for each annotated frame and each cross-topology semantic feature available in both representations, compute a signed normalized feature (`manual68 - MediaPipe478`) using each topology's own inter-ocular scale convention. Aggregate frame medians to video and video medians to participant. Report participant-level feature-wise Spearman agreement, median absolute signed residual, missing-detection denominator, and ALS-control/stroke-control median-residual contrasts with participant-stratified percentile bootstrap intervals. A predeclared median across the semantic features is the single summary; feature-wise p-values, if shown, use Holm correction. Raw 68-point NME is prohibited because MediaPipe-478 and Multi-PIE-68 anchors are not numerically interchangeable.
- Source-codec anomaly: report both inclusion and exclusion of `S002_02_BBP_NORMAL_color.avi`; it cannot affect the three-task primary endpoint.
- Exposure boundary: cohort identity is inherent in the release and the inventory program has already parsed SLP workbooks into a private manifest, although no prediction or association result has been generated or inspected. Therefore the frozen diagnostic endpoint is a held-fixed cross-disease external transfer test, while SLP associations are predeclared exploratory analyses rather than untouched confirmatory validation. After the first score report, NeuroFace is fully outcome-exposed and no later candidate can call performance on it untouched external validation.
- Privacy: publisher identifiers and archive members may exist only inside the authorized raw/private area or in process memory. Serialized IDs are deterministic one-way SHA-256 pseudonyms. Temporary sources are owner-only, created with exclusive access, always removed on success/error/signal, and logs/reports must contain neither publisher IDs, archive members, source paths, nor raw tracebacks. Public artifacts contain aggregate counts only.

The protocol, evaluator tests/code, artifact/model hashes, dependency lock, and result-free authorization are committed before the one-shot scoring command. The later cache authorization may add only authenticated cache hashes and QC counts; it cannot modify endpoints, missingness, estimands, or thresholds.

### Task 1: Strict archive inventory and private manifest

**Files:**
- Create: `src/datasets/neuroface_external_v1.py`
- Create: `scripts/build_neuroface_external_v1_manifest.py`
- Create: `tests/test_neuroface_external_manifest_v1.py`

- [x] **Step 1: Write failing tests for archive hashes, exact counts, duplicate/path rejection, filename/task parsing, video-landmark pairing, SLP averaging, opaque IDs, and deterministic ordering.**
- [x] **Step 2: Run `python tests/test_neuroface_external_manifest_v1.py`; expect failure because the module does not exist.**
- [x] **Step 3: Implement the smallest strict, ZIP-only inventory parser and private no-overwrite manifest writer.**
- [x] **Step 4: Re-run the test and the existing dataset/security tests; expect all pass.**

### Task 2: Resumable frozen feature extraction

**Files:**
- Create: `scripts/extract_neuroface_clinical23_v2_windows.py`
- Create: `tests/test_neuroface_dynamic_cache_v1.py`
- Reuse unchanged: `scripts/build_palsynet_v2_windows.py`
- Reuse unchanged: `src/preprocessing/action_bundle.py`

- [x] **Step 1: Write failing tests for authenticated ZIP-member copying, four-by-32 sampling, 90% coverage, exact schema/model hashes, resumability, no-overwrite, and absence of raw names in caches/manifests.**
- [x] **Step 2: Run the focused test and confirm the expected missing-feature failure.**
- [x] **Step 3: Implement per-member same-byte temporary decoding and validated deidentified NPZ output.**
- [x] **Step 4: Run focused and existing frozen-extractor tests.**
- [ ] **Step 5: Smoke one healthy-control, one ALS, and one stroke video before the full extraction.**

### Task 3: Frozen participant-level external evaluator

**Files:**
- Create: `src/evaluation/neuroface_external_v1.py`
- Create: `scripts/run_neuroface_external_v1.py`
- Create: `tests/test_neuroface_external_v1.py`
- Create after cache lock: `docs/registries/neuroface-external-v1-authorization.json`

- [x] **Step 1: Write failing tests proving no tuning CLI, exact artifact/cache authorization, common-task participant aggregation, group-cluster bootstrap, subgroup/task endpoints, identifier-free report, and no-overwrite.**
- [x] **Step 2: Run the focused test and confirm the expected failure.**
- [x] **Step 3: Implement authentication, frozen scoring, aggregate metrics, and independent report recomputation.**
- [x] **Step 4: Run focused tests plus the MEEI and 110D external evaluation regression suites.**
- [ ] **Step 5: Commit the result-free protocol/evaluator authorization before extraction; after extraction append only authenticated cache hashes and QC counts, then invoke the scorer once.**

### Task 4: SLP transfer and manual-landmark audit

**Files:**
- Create: `src/evaluation/neuroface_landmark_audit_v1.py`
- Create: `scripts/audit_neuroface_landmarks_v1.py`
- Create: `tests/test_neuroface_landmark_audit_v1.py`

- [ ] **Step 1: Write failing tests for 68-point parsing, exact frame joins, semantic23 transforms, participant clustering, missing-detection accounting, and cohort-safe aggregate output.**
- [ ] **Step 2: Run the focused test and confirm the expected failure.**
- [ ] **Step 3: Implement exact annotated-frame decoding, MediaPipe/manual semantic transforms, SLP aggregation, and bootstrap summaries.**
- [ ] **Step 4: Run focused and existing semantic-landmark adapter tests.**

### Task 5: Real-data execution and audit

**Files:**
- Create: `docs/results/neuroface_external_110d_v1.md`
- Create: `outputs/neuroface_external_v1/report.json`
- Keep private/ignored: participant manifest, per-video predictions, extracted NPZ caches, and landmark frame rows.

- [ ] **Step 1: Verify the six raw ZIP hashes and full CRC tests again.**
- [ ] **Step 2: Run extraction in a pinned isolated environment; resume safely if interrupted.**
- [ ] **Step 3: Freeze cache and implementation digests, then create the exact one-shot authorization.**
- [ ] **Step 4: Run the external scorer once and the manual-landmark audit once.**
- [ ] **Step 5: Independently recompute aggregate metrics from private rows and compare numerically.**
- [ ] **Step 6: Run all focused/regression tests, `py_compile`, `git diff --check`, identifier/secret scans, and confirm zero raw data is tracked.**
- [ ] **Step 7: Write the result and next-research decision with explicit clinical boundaries.**
