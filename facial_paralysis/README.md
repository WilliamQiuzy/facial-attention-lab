# Facial Paralysis Research

Universal Clinical Router v4 is the sole canonical scientific comparator in this project. It
routes authenticated recording evidence to a frozen free-recording, scripted
three-action, or externally timed seven-action expert without using dataset or
institution identity as a predictor.

This is a research model, not a clinically validated diagnostic device.

## Start here

- Current architecture, evidence, and limitations: `docs/CURRENT_MODEL.md`
- Current deployable shared model and limitations:
  `docs/CURRENT_DEPLOYMENT_MODEL.md`
- Authorized NVIDIA server quickstart: `deploy/shared-v8/README.md`
- Universal Clinical Router v6 development-candidate brief:
  [English](docs/results/universal_clinical_router_v6_mayo_brief_en.md) |
  [中文](docs/results/universal_clinical_router_v6_mayo_brief_zh.md)
- V8 shared-model structure (current deployment, not yet a clinical model):
  [Model Architecture](docs/architecture/model-architecture.md)
- Machine-readable model registry: `docs/model_registry.json`
- Active pipeline and verification commands: `docs/PIPELINE.md`
- Supported Python import: `src.models.current`
- Frozen model artifact:
  `docs/results/artifacts/universal_clinical_router_v4/model.json`
- Script support boundary: `scripts/README.md`

Historical modules and reports remain available only for reproducibility. They
are not alternative current models, and old generated checkpoints, patient
arrays, scorecards, and architecture-search scratch are intentionally excluded
from source control.

## Model-change rule

All new work starts from v4. A candidate must receive a new version, preserve
participant-disjoint evaluation and protected-data boundaries, report every
cohort separately, and pass an untouched validation gate before it can replace
v4 in `docs/model_registry.json`.
