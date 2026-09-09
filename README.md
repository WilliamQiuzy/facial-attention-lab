# Facial Attention Lab

Unified public snapshot of the Facial Attention Lab's research-only software,
synthetic methods, comparison analyses, and public model contracts. It includes
two clinician-facing React prototypes, reproducible synthetic facial-attention
assets, an independent-cohort gaze comparison, a mirror-invariant landmark
model package, and the Vitestro device-evaluation evidence base.

This repository contains public research-model source and deidentified
aggregate development results, but no Mayo patient recordings, patient-derived
features, clinical labels, fitted model weights, professional-camera or Webcam
participant data, or real-image synthesis outputs. It includes a pinned,
publicly sourced MediaPipe Face Landmarker bundle used only for same-origin
facial registration in a browser prototype. Patient records and selected
test-image bytes remain in browser memory for the current session only. The ten
bundled images are separate AI-generated identities, hash-locked in the
application allowlist, and may only be presented as unpaired interface
demonstrations. The interfaces and research models are not diagnostic tools,
clinical decision aids, human-gaze results, or clinically validated models.

## Projects

| Project | Current public scope |
| --- | --- |
| [`facial_defect_synthesis/`](facial_defect_synthesis/) | Synthetic-image generation code, methodology, and the exact 10 approved demonstration assets; the larger internal dataset and generation log are excluded |
| [`facial_paralysis/facial_paralysis_web/`](facial_paralysis/facial_paralysis_web/) | LifeLink Face upload or browser-camera capture with the FACES eight-step voice guide, explicit demonstration mode, and a fail-closed versioned research inference contract; no model runs in the browser |
| [`facial_paralysis/facial_defect_web/`](facial_paralysis/facial_defect_web/) | Session-only clinician workflow, local simulation, on-device face registration, research workbench, tests, and safety documentation |
| [`facial_paralysis/landmark_110d/`](facial_paralysis/landmark_110d/) | Mirror-invariant 110D facial-landmark feature and estimator contracts, tests, model card, and deidentified aggregate development result; no fitted weights or patient-level artifacts |
| [`facial_defect_gaze_comparison/`](facial_defect_gaze_comparison/) | Independent 500+500 Webcam-versus-professional cohort comparison, with synthetic inputs, equivalence analysis, split-half map benchmarking, source-domain diagnostics, and seven interpretable figures |
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

Synthetic independent-cohort Webcam/reference comparison:

```bash
cd facial_defect_gaze_comparison
uv sync --extra dev
uv run gaze-compare cohort-simulate
uv run gaze-compare cohort-analyze
```

Mirror-invariant Landmark 110D package:

```bash
cd facial_paralysis/landmark_110d
python -m pip install -e .
python -m unittest discover -s tests -v
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
