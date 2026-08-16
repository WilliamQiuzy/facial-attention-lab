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

The sole current authority is:

- `../CURRENT_MODEL.md`
- `../model_registry.json`
- `../../src/models/current.py`

Historical source modules may be imported directly only when reproducing an
archived experiment. They must never be re-exported from `src.models` or called
"current", "champion", or "deployment" without a new promotion review.
