# Release verification — 2026-07-26

## Decision

The frontend is ready for continued **synthetic, session-only interface
rehearsal**. It is not ready for clinical use, patient interpretation, or a real
observer-attention model connection.

The available functional-assessment research outputs do not emit a spatial
observer-attention field. The web result is therefore labeled and governed as
a deterministic simulation. Connected mode is only a strict
transport-contract rehearsal and remains fail-closed.

## Model-output boundary

The separate functional-assessment research system produces non-spatial,
severity-oriented outputs. It does not supply the spatial display points
required by this workbench, and the browser does not convert those outputs into
a heatmap.

## Contract alignment

- Connected HTTP requests contain only the request profile, operational IDs,
  case/asset identity, source SHA-256, and exact full-image source-binding
  identity/version/geometry.
- Requests do not send the mock model profile, mock configuration, threshold,
  smoothing, configuration hash, mock metrics, or ROI approval.
- Connected responses must echo the exact request and provide a strict model
  identity: model ID/version, 64-hex artifact SHA-256, preprocessing version,
  calibration version, and display-scale ID.
- Connected spatial display points are capped at 4,096.
- Severity-only or ordinal payloads fail as `MALFORMED_RESPONSE`; no heatmap is
  synthesized from logits or scores.
- Connected AOI reporting is unavailable because version 1 explicitly declares
  `registration_geometry_unavailable_v1`.
- Result-payload gaze provenance and training-data provenance are represented
  separately.
- `HeatmapPoint[]` and patient-media reference are provisional integration
  choices, not a frozen scientific or production contract.

## Clinician workflow

The default clinician path is:

```text
Cases -> Run -> source image -> simulated density field -> density overlay
      -> Clinical AOI summary -> interpretation limits -> next step
```

The result is one vertical page. The default flow has no crop drawing, ROI
approval, model selector, threshold control, or smoothing control. Overlay
opacity is optional and collapsed under Display options.

The AOI presentation is post-inference and mock-only:

- one anatomical partition including outside-template, displayed total 100%;
- one patient-left/patient-right hemiface partition, displayed total 100%;
- central triangle explicitly overlapping and non-additive;
- patient laterality shown for frontal, non-mirrored synthetic images;
- point-center intensity assignment, with radius and boundary overlap not
  integrated;
- future surgical-site mask shown as absent and separately versioned.

The internal full-image source binding is neither an anatomical AOI nor a
surgical-site mask. Direct Run, Retry, and Batch launches revalidate it
immediately before gateway work. If it changes while queued, the attempt blocks
with `FULL_IMAGE_SOURCE_BINDING_REQUIRED` and makes no gateway call.

## Automated verification

Fresh commands:

```text
pnpm typecheck
CI=1 pnpm exec vitest run --maxWorkers=4
pnpm build
git diff --check -- . ../docs/superpowers
```

Results:

- TypeScript: passed, 0 errors.
- Vitest: **39 files, 662 tests passed**.
- Production build: passed; 1,845 modules transformed.
- Asset build boundary: exactly 10 approved synthetic PNG assets emitted.
- Scoped diff check: passed.
- Source-binding final regression: 3 files, 54 tests passed.
- AOI display rounding regression: 2 files, 45 tests passed.

## Browser and visual verification

Native Python Playwright with headless Chromium, `networkidle`, light color
scheme, and reduced motion:

- **44/44 checks passed**.
- Routes exercised: `/cases`, one exact `/analysis` run, exact `/runs/:runId`,
  `/jobs`, `/models?case=...`, `/cases/:caseId/roi`, `/reviews`, `/about`, and
  `/integration`.
- Viewports: 1440×1000, 390×844, and 360×800.
- Result heading order matched exactly on Analysis and Run Detail.
- Anatomical and hemiface displayed percentages each totaled exactly 100%.
- No horizontal overflow at any tested viewport.
- No console warnings/errors or page errors.
- No failed requests or HTTP error responses.
- No external network requests.
- No localStorage, sessionStorage, IndexedDB, or cookie writes.
- No duplicate rendered IDs.

Screenshots:

- `clinical-aoi-analysis-desktop-2026-07-26.png`
- `clinical-aoi-analysis-mobile-390-2026-07-26.png`
- `clinical-aoi-analysis-mobile-360-2026-07-26.png`
- `clinical-aoi-run-detail-desktop-2026-07-26.png`
- `clinical-aoi-jobs-desktop-2026-07-26.png`
- `clinical-aoi-models-desktop-2026-07-26.png`
- `clinical-aoi-source-binding-desktop-2026-07-26.png`

The screenshots were retained outside this public source snapshot as local
verification artifacts.

## Independent reviews

- Full-image source-binding totality: **APPROVED**, no Critical or Important
  findings after the launch-time and display-selector fixes.
- Connected contract and model-boundary review: **APPROVED**, no Critical or
  Important findings.
- Earlier image-stack and clinician-layout reviews found no remaining Critical
  or Important issue; the latest screenshots were rechecked after AOI and
  contract changes.

## Remaining hard gates

1. Train and validate the actual facial-defect observer-attention model.
2. Freeze the scientific spatial output, coordinate frame, normalization, and
   display-scale contract with the model team.
3. Define an authorized patient-media reference and server-side access control;
   this frontend currently has no patient upload.
4. Add validated registration geometry, orientation/mirror metadata, and
   registration quality control before connected anatomical AOI reporting.
5. Establish data provenance, clinical validation, governance, and a separate
   patient-use policy before any clinical or patient interpretation.
