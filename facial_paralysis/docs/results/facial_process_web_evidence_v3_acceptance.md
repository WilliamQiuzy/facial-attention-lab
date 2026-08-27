# Facial Process Web - Three-Layer Evidence Acceptance

**Date:** 2026-08-26

## Delivered evidence contract

Each active FACES action now exposes three separate layers in the browser and
downloaded PDF:

1. **Measured movement** - the existing MediaPipe geometry summarized over the
   registered three-second hold and compared with the recording's neutral
   repose;
2. **Model influence** - action-level Integrated Gradients on the shared action
   tokens, using neutral clinical geometry plus a zero dense-response baseline;
3. **Stability checks** - agreement across all three ensemble members, exact
   original/mirror view exchange, and deterministic minus/plus one-checkpoint
   timing shifts.

An action receives an upward or downward influence direction only when every
stability gate passes. Otherwise the response and report say that no stable
influence is available. The browser receives only the direction, relative
strength, and gate results; raw logit contributions and raw landmarks remain
inside the model service. The anatomy label is derived from the scripted action
(brow, eye, or mouth), not from a fabricated pointwise heatmap.

## Model invariance

The frozen Shared V9 scoring route is unchanged. The new explanation endpoint
first calls the existing predictor, then calculates attribution downstream of
the already-computed shared action tokens. A production-container comparison
confirmed that every prediction field from `/v1/predict/cue_aligned_action` is
byte-semantically identical to the corresponding fields returned by
`/v1/explain/cue_aligned_action`.

## Verification

- Related Python model, token, release, service, preprocessing, gateway, Docker,
  and public-container suites: **58/58 passed**.
- Frontend strict-contract, component, interaction, and lifecycle suites:
  **113/113 passed**; TypeScript and the production Vite build passed.
- Docker build reran the frontend **113/113** suite; model, gateway, and web
  containers all reached healthy readiness.
- Browser full loops passed for the eight-step desktop route and seven-step
  mobile route, including duplicate-click suppression, report navigation,
  direct PDF download, source-video download, and Blob URL cleanup.
- Poppler verified an **8-page A4 PDF with seven recorded context images**.
  Rendered inspection found no clipped action cards, overlapping text, or
  missing measurement, influence, or stability sections.

## Public synthetic evidence

- [Desktop full report](artifacts/facial_process_web_ui/evidence-v3/evidence-v3-desktop.png)
- [Mobile full report](artifacts/facial_process_web_ui/evidence-v3/evidence-v3-mobile.png)
- [Downloaded PDF](artifacts/facial_process_web_ui/evidence-v3/evidence-v3-report.pdf)

These artifacts use the checked synthetic camera fixture and contain no patient
or user recording.
