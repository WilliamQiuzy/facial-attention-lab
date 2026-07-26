# Facial Attention Lab

Public, history-free source snapshot of a research-only React workbench for exploring simulated facial-attention workflows on AI-generated synthetic faces.

This repository contains no Mayo patient recordings, patient-derived features, clinical labels, model weights, or real-image synthesis outputs. The ten bundled images are separate AI-generated identities, hash-locked in the application allowlist, and may only be presented as unpaired interface demonstrations. The interface is not a diagnostic tool, clinical decision aid, human-gaze result, or validated model.

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

## Vitestro phlebotomy safety

The public-source real-time detector evaluation for the Mayo–Vitestro automated
phlebotomy collaboration is in
[`vitestro_phlebotomy_safety/`](vitestro_phlebotomy_safety/). It contains only
product research, literature review, procurement planning, and a measurement
validation protocol. It contains no participant or patient data.

No open-source license grant is included yet.
