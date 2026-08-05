# Facial Attention Lab

Public source snapshot of a clinician-facing, research-only React prototype for
rehearsing synthetic/test patient records, standardized photo visits, local
simulated attention results, and simple clinician review. The legacy research
workbench remains available for synthetic case, batch, model-comparison, and
structured-review exercises.

This repository contains public research-model source and deidentified aggregate
PalsyNet development results, but no Mayo patient recordings, patient-derived
features, clinical labels, fitted model weights, or real-image synthesis
outputs. Patient records and selected test-image bytes remain in browser memory
for the current session only. The ten bundled images are separate AI-generated
identities, hash-locked in the application allowlist, and may only be presented
as unpaired interface demonstrations. The interface and research models are not
diagnostic tools, clinical decision aids, human-gaze results, or clinically
validated models.

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

## Facial-paralysis mirror-invariant Landmark 110D model

The public, collaboration-ready source package for the current development
champion is in
[`facial_paralysis/landmark_110d/`](facial_paralysis/landmark_110d/). It
contains the frozen MediaPipe-to-clinical23 transform, exact 110D trajectory
representation, horizontal-mirror transform, paired-view standardized
L2-logistic estimator, tests, model card, and deidentified aggregate result. It
intentionally contains no fitted weights or patient-level artifacts.

## Vitestro phlebotomy safety

The public-source wearable-device evaluation for the Mayo–Vitestro automated
phlebotomy collaboration is in
[`vitestro_phlebotomy_safety/`](vitestro_phlebotomy_safety/). It contains only
product research, literature review, procurement planning, and a measurement
validation protocol. It contains no participant or patient data.

No open-source license grant is included yet.
