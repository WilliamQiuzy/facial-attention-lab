# Facial Attention Lab

Public source snapshot of two clinician-facing, research-only React prototypes:
an eight-step FACES video-capture and inference-contract interface, and a
synthetic facial-attention workbench for standardized photo visits, local
simulation, and clinician-review exercises.

This repository contains no Mayo patient recordings, patient-derived features,
clinical labels, trained clinical or observer-attention model weights, or
real-image synthesis outputs. It includes a pinned, publicly sourced MediaPipe
Face Landmarker bundle used only for same-origin facial registration in the
browser prototype. Patient records and selected test-image bytes remain in
browser memory for the current session only. The ten bundled images are
separate AI-generated identities, hash-locked in the application allowlist, and
may only be presented as unpaired interface demonstrations. The interface is
not a diagnostic tool, clinical decision aid, human-gaze result, or validated
model.

## Projects

| Project | Current public scope |
| --- | --- |
| [`facial_defect_synthesis/`](facial_defect_synthesis/) | Synthetic-image generation code, methodology, and the exact 10 approved demonstration assets; the larger internal dataset and generation log are excluded |
| [`facial_paralysis/facial_paralysis_web/`](facial_paralysis/facial_paralysis_web/) | LifeLink Face upload or browser-camera capture with the FACES eight-step voice guide, explicit demonstration mode, and a fail-closed versioned research inference contract; no model runs in the browser |
| [`facial_paralysis/facial_defect_web/`](facial_paralysis/facial_defect_web/) | Session-only clinician workflow, local simulation, on-device face registration, research workbench, tests, and safety documentation |
| [`vitestro_phlebotomy_safety/`](vitestro_phlebotomy_safety/) | English, budget-focused evaluation of devices with programmatic live-data paths for presyncope measurement research |

## Run locally

Requirements: Node.js 22.12–24.x and pnpm 11.x.

FACES video capture and research inference interface:

```bash
cd facial_paralysis/facial_paralysis_web
pnpm install --frozen-lockfile
cp .env.example .env.local
pnpm dev
```

Configure `.env.local` only with an authorized research endpoint. Facial video
is identifiable, and the browser requires explicit confirmation before upload.
The v4 model checkpoint is not embedded in the browser. Demonstration mode is a
separate explicit preview and never substitutes for an unavailable or rejected
research response. See the [application README](facial_paralysis/facial_paralysis_web/README.md)
for the multipart manifest, strict response schema, and research-only boundary.

Synthetic facial-attention workbench:

```bash
cd facial_paralysis/facial_defect_web
pnpm install --frozen-lockfile
pnpm dev
```

## Verify

Run the same verification commands inside either web application directory:

```bash
pnpm typecheck
pnpm test:run
pnpm build
```

The synthetic workbench source is in `facial_paralysis/facial_defect_web/`.
Its ten approved synthetic assets remain in
`facial_defect_synthesis/output/synthetic/` so byte verification and static
imports use the same repository-relative paths as the research workspace.
Sanitized generation evidence is recorded in
`facial_paralysis/facial_defect_web/audits/approved-synthetic-provenance.json`.
The separate FACES capture source and its endpoint contract are in
`facial_paralysis/facial_paralysis_web/`.

No open-source license grant is included yet.
