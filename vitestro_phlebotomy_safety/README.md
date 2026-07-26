# Vitestro Phlebotomy Safety — Real-Time Detector Evaluation

This project supports the Mayo–Vitestro collaboration on detecting presyncope,
syncope, and clinically meaningful discomfort during automated venipuncture.

## Current decision

Phase 1 is a measurement-system evaluation, not a production alerting system.
The current executive screen asks three questions:

1. Can a study application receive timestamped measurements during venipuncture?
2. Is the detector itself sufficiently accurate, continuous, and synchronized?
3. Is the signal valid in the intended workflow across motion, low
   perfusion, skin pigmentation, fit, and operating-system states?

Historical CSV export, sleep summaries, and values visible only inside a vendor
application are not procurement criteria. No reviewed device satisfies all three
questions without a controlled Vitestro pilot.

The main report retains ten systems with confirmed live paths. It now covers
watches, ECG chest straps and patches, finger pulse oximetry, fingertip/palmar
EDA for sweating, and continuous finger-cuff blood pressure. The broader
18-device matrices remain as background screening artifacts.

## Deliverables

- [`docs/wearable_device_evaluation.md`](docs/wearable_device_evaluation.md) —
  concise English real-time detector table, shortlist, and quality gates.
- [`data/device_feature_matrix.csv`](data/device_feature_matrix.csv) —
  broader legacy product screen using `1` (supported), `0.5` (conditional), and
  `0` (not confirmed); it is not the current procurement shortlist.
- [`data/modality_evidence_matrix.csv`](data/modality_evidence_matrix.csv) —
  evidence-ranked sensing modalities, intended roles, and failure modes.
- [`sources/evidence_registry.csv`](sources/evidence_registry.csv) —
  background source registry from the broader screen. Current shortlist sources
  are linked directly in the executive report.
- `outputs/019f8cc8-9802-7b01-8b3a-7fe5ef10eaa5/` — Excel scorecard.

## Interpretation boundary

- A device remains in the executive table only when a live sensor-to-host path
  is publicly documented.
- Real-time availability does not establish medical accuracy for acute
  presyncope detection.
- On-demand ECG/SpO2, hypertension notifications, and daily cuff-calibrated
  estimates are not treated as continuous signals.
- Continuous BP systems are development references until they pass the Vitestro
  workflow and fast-change tests.

## Data and privacy

This folder contains public-source research and planning artifacts only. Do not
commit participant identifiers, clinical recordings, API credentials, or raw
study exports. Future study data require an approved protocol and the repository's
private-data controls.

## Status

Research recommendation only. Not a medical device, diagnostic claim, or
authorization for patient-facing alerts or autonomous Vitestro intervention.
