# Archived Model Research

Files under this directory preserve historical experiment context. They are not
current model specifications, supported inference entrypoints, or clinical
claims.

- `models/` contains pre-v4 model-selection narratives.
- `experiments/` contains pre-v4 run logs, autonomous-search findings, and Mayo
  exploratory analyses.
- `manuscripts/` contains drafts about superseded model families.

Some paths and generated artifacts named by archived documents now exist only
in Git history.

The current authority is:

- `../CURRENT_MODEL.md`
- `../CURRENT_RESEARCH_MODEL.md`
- `../CURRENT_DEPLOYMENT_MODEL.md`
- `../model_registry.json`
- `../../src/deployment/shared_v9_research_release.py`

The former Universal Clinical Router v4 authority is preserved as
`models/current_model_ucr4.md` with its v2 registry snapshot. The former
top-level project summary, Chinese progress report, and MARLIN/GRU architecture
note were moved into `experiments/` or `models/` because they are useful only
for historical reproduction.

Historical source modules may be imported directly only when reproducing an
archived experiment. They must never be re-exported from `src.models` or called
"current", "champion", or "deployment" without a new promotion review.
