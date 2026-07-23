# Vitestro Phlebotomy Safety — Wearable Device Evaluation

This project supports the Mayo–Vitestro collaboration on detecting presyncope,
syncope, and clinically meaningful discomfort during automated venipuncture.

## Current decision

Phase 1 is a measurement-system evaluation, not a production alerting system.
The evaluation separates three questions:

1. Does the wearable contain the relevant sensor?
2. Can a study application receive timestamped data with sufficiently low latency?
3. Is that signal valid in the intended venipuncture setting across motion, low
   perfusion, skin pigmentation, fit, and operating-system states?

No reviewed device currently satisfies all three questions without a controlled
Vitestro pilot.

## Deliverables

- [`docs/wearable_device_evaluation.md`](docs/wearable_device_evaluation.md) —
  English executive report, literature synthesis, device findings, recommendation,
  and evaluation protocol.
- [`data/device_feature_matrix.csv`](data/device_feature_matrix.csv) —
  machine-readable feature matrix using `1` (supported), `0.5` (conditional),
  and `0` (not confirmed).
- [`sources/evidence_registry.csv`](sources/evidence_registry.csv) —
  claim-level source registry with official product/developer sources, regulatory
  material, and peer-reviewed papers.
- `outputs/019f8cc8-9802-7b01-8b3a-7fe5ef10eaa5/` — Excel scorecard.

## Interpretation boundary

- `✓` means the capability is available in a relevant mode; it does not mean the
  device is medically accurate for acute presyncope detection.
- `△` means spot, sleep-only, intermittent, derived, region-limited, or gated by
  a vendor or partner program.
- `✗` means no supported capability was confirmed in public documentation as of
  2026-07-23.
- Hypertension notifications and daily cuff-calibrated estimates are not treated
  as continuous blood-pressure measurements.

## Data and privacy

This folder contains public-source research and planning artifacts only. Do not
commit participant identifiers, clinical recordings, API credentials, or raw
study exports. Future study data require an approved protocol and the repository's
private-data controls.

## Status

Research recommendation only. Not a medical device, diagnostic claim, or
authorization for patient-facing alerts or autonomous Vitestro intervention.
