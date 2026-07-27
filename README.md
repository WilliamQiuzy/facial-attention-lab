# Mayo Clinic — Facial Analysis Research

Two related research projects on automated facial analysis for clinical use.

> **Data note:** Clinical patient data (Mayo iPhone/LiveLinkFace recordings), large
> public datasets, model-extracted feature caches, and all API keys are **excluded**
> from this repo (see `.gitignore`). Only code, documentation, result summaries, key
> figures, lightweight warm-start checkpoints, and **synthetic** images are tracked.

---

## 1. `facial_paralysis/` — Facial-Palsy Severity Grading
Researches facial-palsy classification and eventual House-Brackmann I–VI
grading from facial video and images.

- **Current development champion:** a fixed, standardized L2 logistic regression
  over a 110-dimensional trajectory summary derived from 23 MediaPipe clinical
  landmark channels. On the PalsyNet grouped inner-OOF development screen it
  achieved AUROC **0.938**, balanced accuracy **0.905**, sensitivity **0.810**,
  and specificity **1.000**.
- **Canonical status:** `docs/CURRENT_MODEL.md` and
  `docs/results/current_development_model.json`. Earlier MARLIN, web-QWK,
  Blendshape/Fusion, and SSL results are preserved as historical baselines,
  not current champions.
- **Key docs** (`docs/`): `CURRENT_MODEL.md` (current result and claim boundary),
  `model_design.md` (architecture history),
  `training_runs.md` (Runs #1–#17), `mayo_data.md`, `data_acquisition.md`,
  `literature_datasets.md`, `leakage_policy.md`, `autoresearch_fp.md`,
  `模型与训练_中文说明.md` (中文说明).
- **Figures for slides** (`outputs/viz/`): `architecture.png`, `decision_basis_*.png`
  (per-patient left-right asymmetry — the model's explainable basis), `facogram_panel.png`.
- **Code:** `src/` (models, datasets, preprocessing, training, evaluation), `scripts/`,
  `tests/`. Warm-start checkpoints in `outputs/checkpoints/` (trainable head only;
  frozen MARLIN loaded separately).
- **Status:** strongest current evidence is a PalsyNet affected-vs-unaffected
  development result, not HB grading and not Mayo-65 accuracy. Identity/action
  review and the untouched outer test are still required; the clinical HB head
  awaits real HB labels and in-domain controls.

## 2. `facial_defect_synthesis/` — Synthetic Clinical Face Photos
Large-scale generation of **synthetic** clinical photos of faces with facial defects
(facial_paralysis, mohs, hn_cancer, cleft, trauma) for facial-reconstruction research,
via OpenAI `gpt-image-2`.

- `generate.py samples|category|batch`, `config.py`, `prompts.py`, `openai_client.py`.
- Sample synthetic outputs in `output/images/` (no real patients).
- API key is read from the parent `.env` at runtime (not stored in code or this repo).

---
*Research code; not a medical device. Synthetic images are AI-generated.*
