# New-data ingestion runbook (turnkey)

> **Model-status note (2026-07-27):** This runbook predates the 110D Landmark
> development champion. Use `CURRENT_MODEL.md` as the model authority, and
> freeze a new evaluation protocol before fitting any newly labeled Mayo data.

Everything is built to consume new data. This is the exact "when X arrives, run Y" recipe.
Drop the files, run the commands, get the result. No re-architecting needed.

## A. HB labels arrive (≥40–60 takes) → supervised severity evaluation
The prerequisite for estimating a clinical accuracy number.
1. Put grades in `outputs/mayo_hb_labels.json` as `{"<take_id>": <HB 1-6 or region grades>}`
   (take_id like `20260313_FACES020`).
2. The model already has ordinal HB/region heads and per-sample task routing
   (`autoresearch_fp/runner.py`, `src/models/multitask.py`). Add the Mayo per-action bundles
   as a labeled task: they are in `outputs/mayo_action_bundles/<take>/<action>.npz` (MARLIN +
   mp_seq, same schema as web). Map each labeled take → its region/HB grade.
3. Treat the 110D Landmark model as the primary development reference and the
   older neural warm-starts as historical comparators. Under a newly frozen
   **leave-one-patient-out CV** protocol, train on k−1 patients, test the held-out;
   report QWK + bootstrap CI
   (reuse `scripts/mayo_transfer_robust.py`'s bootstrap; `power_analysis.py` says n≈40–60
   gives ±0.10–0.14). The historical geometry-only configuration may be compared,
   but must not be called the current or deployment model.
4. Re-run `scripts/mayo_scorecard.py`; keep `learned_severity` explicitly
   exploratory until the locked clinical-validation requirements pass.

## B. In-domain healthy controls arrive (record ~15–25, same FACES protocol)
The unlock for a trustworthy palsy DETECTOR (the one place MARLIN may transfer).
1. Drop the raw `.mov` into `data/livelinkface_data/<take>/` (label = healthy/0).
2. Segment + bundle: run the pod path (`ear_clips.py` for 60fps, or the existing
   `mayo_build_per_action.py` on the pod for MARLIN+MediaPipe bundles). Blendshape
   segmentation is automatic (`mayo_blendshape_segment.py`).
3. Re-run `scripts/mayo_binary.py` — now it reports real specificity (not just the saturated
   recall), so a threshold can be evaluated. Compare the current Landmark
   reference with historical MARLIN and geometry-only baselines on a locked,
   patient-held-out in-domain split.
4. Add the healthy asymmetry distribution as the "normal range" baseline in the scorecard.

## C. AU corpora arrive (DISFA / BP4D / FEAFA+)
The unlock for pretraining the geometric/temporal stream on real movement.
1. Drop the data; fill `load_disfa` / `load_bp4d` in `src/datasets/au_intensity_adapter.py`
   (return `{clip: [AUFrame,...]}`). The (T,72) feature construction + masked-reconstruction
   pretraining are done and synthetic-verified.
2. `python3 scripts/au_pretrain.py --disfa <dir>` → saves a pretrained BiGRU to
   `outputs/au_pretrain/geo_encoder.pt`.
3. Warm-start the model's temporal encoder from it and re-run the search — this is the main
   lever to help the EYE region (which needs movement dynamics stills lack).

## D. More palsy patients arrive (toward n≈35–50)
The unlock for POWER to establish transfer.
1. Same ingestion as (A) minus labels: `data/livelinkface_data/` → segment → bundles.
2. Re-run `scripts/mayo_transfer_robust.py` and `scripts/power_analysis.py` — as n grows the
   CIs shrink; the transfer question becomes answerable (currently 15–37% power at n=13).

## Sanity after any ingestion
- `python3 autoresearch_fp/runner.py autoresearch_fp/best_config.json`
  (historical web benchmark, not the current model)
- `python3 scripts/mayo_scorecard.py` (per-patient cards regenerate)
- check the new data isn't leaking across the train/val split (`docs/leakage_policy.md`).
