# Direction #6 — Home iPhone self-monitoring for facial palsy

A longitudinal self-monitoring tool that lets a patient track their own facial-palsy recovery at
home, using only their iPhone — no clinician visit, no manual grading, no training labels. It is a
direct payoff of the label-free measurement stack (#1 synkinesis, EAR dynamics) and its
now-established reliability (#5).

## Why it is feasible now (grounded, not aspirational)
The tool only claims what the measurement science supports:

1. **The measures run on-device, label-free.** MediaPipe FaceLandmarker + ARKit blendshapes give
   per-region left/right asymmetry, eye-closure/lagophthalmos dynamics, and synkinesis from a
   single selfie-style video. Nothing needs a clinician or a severity label.
2. **The measures are reliable (#5).** Split-half (pure measurement error) reliability: eye
   asymmetry ICC 0.97, brow 0.91, smile 0.74 — all good. So a number a patient gets today is
   reproducible tomorrow if their face is unchanged.
3. **We know the noise floor (MDC95).** Minimal detectable change: **eye 0.034, smile 0.082,
   brow 0.150** asymmetry-index units. A follow-up that moves by more than this is a REAL change,
   not measurement noise. This is what turns a one-off score into a trend you can trust.

## What the tool does
- **Protocol.** Patient records the 8-action FACES protocol (repose, brow raise, gentle + forced
  eye closure, smile, pucker, lower-teeth, reanimated smile) front-facing, ~60 s.
- **Per-session report card.** Per region (eye / smile / brow): the L/R asymmetry index with its
  ±MDC/2 "noise band" (see `home_monitor.png`, row 1). Color-coded by severity band.
- **Change detection.** Compares against the patient's own baseline; flags a region as
  improved/worsened only when |Δ| > MDC95 (row 2a). A 50% recovery in eye-closure asymmetry is
  detectable except when the baseline is already near-symmetric (Δ < 0.034).
- **Fixed provocation (important, from #5).** Cross-provocation agreement is only moderate
  (gentle vs forced eye closure ICC 0.58) because the probes are physiologically different — so the
  tool always compares like-to-like (gentle-to-gentle), never mixes probes across sessions.
- **Averaging sharpens it.** Sensitivity improves as ~1/√k with k repeated home sessions
  (row 2c): 4 sessions roughly halve the effective MDC.

## What it explicitly does NOT do (honest boundaries)
- **No severity grade.** It reports asymmetry/dynamics, not an House-Brackmann / eFACE score —
  that mapping requires clinician labels we do not have (the transfer/n-wall). It is a
  **change/trend** tool, not a diagnostic-grade classifier.
- **3D depth is not in the consumer path yet.** Per-frame 3D asymmetry (#4) is unreliable (ICC
  0.10) and only becomes usable after heavy frame-pooling (0.77); it stays a research channel.
- **Requires reasonable capture.** Frontal pose, even lighting, face filling the frame — the same
  conditions the reliability numbers were measured under. Out-of-distribution capture voids the MDC.

## Validation status
- Measurement reliability + MDC: **established** (#5, `reliability.json`).
- Between-patient resolvability: the cohort spread ≫ MDC (row 2b), so it distinguishes patients.
- Longitudinal change on real repeated recordings: **not yet** — no patient has a second session in
  this cohort. The detectable-change panel is a simulation bounded by the real MDC; confirming it on
  actual repeat visits is the single next step (needs follow-up recordings, not labels).

## Artifacts
- `scripts/home_monitor.py` — report-card generator + detectable-change analysis.
- `outputs/mayo_eface/home_monitor.png`, `home_scores.json`.
- Depends on `reliability.json` (#5) for the MDC thresholds.
