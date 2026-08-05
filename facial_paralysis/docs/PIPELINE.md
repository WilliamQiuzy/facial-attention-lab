# Pipeline reproducibility index (what each script does, how to re-run)

Everything runs locally on the cached bundles with the anaconda python3
(`KMP_DUPLICATE_LIB_OK=TRUE python3 ...`), except the 60fps EAR extraction (needs
mediapipe → a RunPod GPU box). Start here to reproduce any result in the docs.

## Current development model
- `docs/CURRENT_MODEL.md` — canonical human-readable result and claim boundary.
- `docs/results/current_development_model.json` — canonical machine-readable
  aggregate record.
- `scripts/run_action_clinical_geometry_v1.py` — fixed 110D Landmark /
  58D Clinical Dynamics development screen. It requires the private PalsyNet
  cache and never opens the protected outer fold.
- `scripts/run_mirror_invariant_110d.py` — fixed standard-110D versus
  mirror-invariant-110D successor screen, with training augmentation,
  symmetric inference, paired grouped bootstrap, and zero-use outer audit.
- `scripts/run_110d_generalization_v1.py` — current locked comparison of 110D,
  110D+Action proxy, and 110D+Action+Phase proxy. It authenticates the reviewed
  identity and deterministic person split before loading only development NPZs.
- Current development champion: mirror-invariant 110D Landmark, AUROC 0.980
  and balanced accuracy 0.952 on 38 reviewed patient groups. This is binary
  PalsyNet development evidence, not HB, Mayo, outer, or clinical validation.

## Historical web model (autoresearch)
- `autoresearch_fp/prepare_fp.py` — FIXED harness: data + leak-safe split + 3-seed QWK metric.
- `autoresearch_fp/runner.py <cfg.json>` — the model engine; run a config, print metric.
- `autoresearch_fp/best_config.json` — historical web champion (0.668).
  `deploy_config.json` — historical geometry-only candidate (0.577 web;
  drops MARLIN), not deployment-authorized.
- `autoresearch_fp/search.py <batch.json> <out.tsv>` — within-split QWK search (WITH MARLIN).
- `autoresearch_fp/xsearch.py` — cross-dataset generalization-objective search.
- `autoresearch_fp/results.tsv` — ranked logbook. `FINDINGS.md` — full write-up.
- Reproduce the historical web benchmark:
  `python3 autoresearch_fp/runner.py autoresearch_fp/best_config.json`

## Mayo per-action label-free analysis (needs FACES protocol semantics)
- `scripts/mayo_eface.py` — per-action asymmetry + synkinesis + forced recruitment → `outputs/mayo_eface/eface_scores.json`.
- `scripts/mayo_unsup_severity.py` — unsupervised severity + phenotype map.
- `scripts/ear_clips.py` (POD, mediapipe) + `scripts/ear_analyze.py` — 60fps EAR closure dynamics.
- `scripts/mouth_corner.py` — 60fps mouth-corner asymmetry (independent mouth target).
- `scripts/reliability.py` — cross-action side-consistency + cross-measure agreement.
- `scripts/mayo_scorecard.py` — **the deliverable**: per-patient clinical cards + cohort table.

## Generalization / transfer (the honest core)
- `scripts/mayo_generalization.py` — web→Mayo transfer (vs blendshape target; note circularity).
- `scripts/mayo_transfer_robust.py` — transfer vs INDEPENDENT geometric targets + bootstrap CI (the rigorous version).
- `scripts/mouth_transfer.py`, `scripts/cross_dataset.py`, `scripts/aug_generalization.py`, `scripts/mayo_coral.py` — the lever experiments.
- `scripts/power_analysis.py` — sample-size needs. `scripts/make_summary_figure.py` — the 4-panel arc figure.

## Data readiness (for new data)
- `scripts/au_pretrain.py` — AU-dynamics pretraining pipeline (fill `au_intensity_adapter.py` loaders when DISFA/BP4D land).
- `outputs/mayo_active_learning.json` — which takes to HB-label first.

## Read order for a new collaborator
`docs/CURRENT_MODEL.md` → `docs/SUMMARY.md` → `docs/loop_findings.md` → `docs/generalization.md` →
`docs/mayo_faces_analysis.md` → `docs/data_collection_plan.md` → `autoresearch_fp/FINDINGS.md`.
