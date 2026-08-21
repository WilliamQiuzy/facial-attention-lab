# Facial Paralysis Research

Shared V9 / BLV9-009 is the current research model. It combines 110D clinical
geometry and available 478-point action trajectories in one genuinely shared
patient encoder; a training-only masked clinical reconstruction objective helps
the shared representation retain distributed brow, eye, oral, temporal, and
bilateral evidence.

This is a research model, not a clinically validated diagnostic device.

## Start here

- Current V9 identity, evidence, and limitations: `docs/CURRENT_MODEL.md`
- Complete public V9 weights and loader: `releases/shared-v9-research-v1/`
- Current deployable shared model and limitations:
  `docs/CURRENT_DEPLOYMENT_MODEL.md`
- Public CPU/H200 Docker quickstart: `deploy/shared-v9/README.md`
- Universal Clinical Router v6 development-candidate brief:
  [English](docs/results/universal_clinical_router_v6_mayo_brief_en.md) |
  [中文](docs/results/universal_clinical_router_v6_mayo_brief_zh.md)
- V8 shared-model structure (current deployment, not yet a clinical model):
  [Model Architecture](docs/architecture/model-architecture.md)
- Machine-readable model registry: `docs/model_registry.json`
- Active pipeline and verification commands: `docs/PIPELINE.md`
- Supported V9 loader: `src.deployment.shared_v9_research_release.load_release`
- Frozen V9 selection artifact:
  `docs/results/artifacts/broad_literature_shared_v9/report.json`
- Script support boundary: `scripts/README.md`

Historical modules and reports remain available only for reproducibility. They
are not alternative current models, and old generated checkpoints, patient
arrays, scorecards, and architecture-search scratch are intentionally excluded
from source control.

## Model-change rule

All new work starts from V9. A candidate must receive a new version, preserve
participant-disjoint evaluation and protected-data boundaries, report every
cohort separately, and pass an untouched validation gate before it can replace
V9 in `docs/model_registry.json`.
