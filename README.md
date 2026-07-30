# Facial Attention Lab

Public source snapshot of a clinician-facing, research-only React prototype for
rehearsing synthetic/test patient records, standardized photo visits, local
simulated attention results, and simple clinician review. The legacy research
workbench remains available for synthetic case, batch, model-comparison, and
structured-review exercises.

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
| [`facial_paralysis/facial_defect_web/`](facial_paralysis/facial_defect_web/) | Session-only clinician workflow, local simulation, on-device face registration, research workbench, tests, and safety documentation |
| [`vitestro_phlebotomy_safety/`](vitestro_phlebotomy_safety/) | English, budget-focused evaluation of devices with programmatic live-data paths for presyncope measurement research |

## Run locally

Requirements: Node.js 22.12–24.x and pnpm 11.x.

```bash
cd facial_paralysis/facial_defect_web
pnpm install --frozen-lockfile
pnpm dev
```

## Verify

```bash
cd facial_paralysis/facial_defect_web
pnpm typecheck
pnpm test:run
pnpm build
```

The application source is in `facial_paralysis/facial_defect_web/`. Its ten approved synthetic assets remain in `facial_defect_synthesis/output/synthetic/` so the byte-verification and static imports use the same repository-relative paths as the research workspace. Sanitized generation evidence is recorded in `facial_paralysis/facial_defect_web/audits/approved-synthetic-provenance.json`.

No open-source license grant is included yet.
