# Mayo Clinic — Facial Analysis Research

Two related research projects on automated facial analysis for clinical use.

> **Data note:** Clinical patient data (Mayo iPhone/LiveLinkFace recordings), large
> public datasets, model-extracted feature caches, and all API keys are **excluded**
> from this repo (see `.gitignore`). Only code, documentation, result summaries, key
> figures, lightweight warm-start checkpoints, and **synthetic** images are tracked.

---

## 1. `facial_paralysis/` — Facial-Palsy Severity Grading
Predicts clinical facial-palsy severity (House-Brackmann I–VI, ordinal) from video/image
of a patient performing facial actions.

- **Model:** two-stream (frozen MARLIN appearance encoder ⊕ trainable MediaPipe-geometry
  GRU) → shared latent severity `s` → multi-task ordered-threshold heads (HB / binary /
  3-level / region eyes+mouth). Handles heterogeneous label scales without fabricating labels.
- **Key docs** (`docs/`): `model_design.md` (architecture, source of truth),
  `training_runs.md` (Runs #1–#17), `mayo_data.md`, `data_acquisition.md`,
  `literature_datasets.md`, `leakage_policy.md`, `autoresearch_fp.md`,
  `模型与训练_中文说明.md` (中文说明).
- **Figures for slides** (`outputs/viz/`): `architecture.png`, `decision_basis_*.png`
  (per-patient left-right asymmetry — the model's explainable basis), `facogram_panel.png`.
- **Code:** `src/` (models, datasets, preprocessing, training, evaluation), `scripts/`,
  `tests/`. Warm-start checkpoints in `outputs/checkpoints/` (trainable head only;
  frozen MARLIN loaded separately).
- **Status:** pipeline complete + leak-safe; palsy detection AUC ≈ 0.86, region severity
  on public data measured. The clinical HB head awaits real HB labels + in-domain data.

## 2. `facial_defect_synthesis/` — Synthetic Clinical Face Photos
Large-scale generation of **synthetic** clinical photos of faces with facial defects
(facial_paralysis, mohs, hn_cancer, cleft, trauma) for facial-reconstruction research,
via OpenAI `gpt-image-2`.

- `generate.py samples|category|batch`, `config.py`, `prompts.py`, `openai_client.py`.
- Sample synthetic outputs in `output/images/` (no real patients).
- API key is read from the parent `.env` at runtime (not stored in code or this repo).

---
*Research code; not a medical device. Synthetic images are AI-generated.*
