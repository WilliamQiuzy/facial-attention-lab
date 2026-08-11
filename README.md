# Facial Attention Lab

Public source snapshot of two clinician-facing, research-only React prototypes:
an eight-step FACES video-capture and inference-contract interface, and a
synthetic facial-attention workbench for standardized photo visits, local
simulation, and clinician-review exercises.

This repository contains public research-model source, the frozen Shared V9
research weights, and deidentified aggregate development results, but no Mayo
patient recordings, patient-derived features, clinical labels, or real-image
synthesis outputs. Patient records and selected test-image bytes remain in browser memory
for the current session only. The ten bundled images are separate AI-generated
identities, hash-locked in the application allowlist, and may only be presented
as unpaired interface demonstrations. The interface and research models are not
diagnostic tools, clinical decision aids, human-gaze results, or clinically
validated models.

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

## Shared V9 public Docker service

The immutable public image includes the complete three-member V9 ensemble. On
an x86-64 Docker server, no GitHub or registry login is required:

```bash
git clone --depth 1 --branch codex/shared-v9-public-release \
  https://github.com/WilliamQiuzy/facial-attention-lab.git
cd facial-attention-lab/facial_paralysis/deploy/shared-v9
docker compose pull
docker compose up -d
curl --fail http://127.0.0.1:18090/readyz
```

This is a research API for validated preprocessed clinical-action tensors, not
a raw-video endpoint or a clinically validated diagnostic service.

The synthetic workbench source is in `facial_paralysis/facial_defect_web/`.
Its ten approved synthetic assets remain in
`facial_defect_synthesis/output/synthetic/` so byte verification and static
imports use the same repository-relative paths as the research workspace.
Sanitized generation evidence is recorded in
`facial_paralysis/facial_defect_web/audits/approved-synthetic-provenance.json`.
The separate FACES capture source and its endpoint contract are in
`facial_paralysis/facial_paralysis_web/`.

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
