# Data-leakage policy (MUST follow when splitting / pooling)

_2026-06-18. Generated from `scripts/dedup_check.py` → `outputs/dedup_report.json`
(20,913 images, perceptual dHash). The palsy image sets are web-scraped and overlap;
Roboflow splits BY IMAGE not patient. Violating this inflates val/test metrics._

## Findings (measured)
- **anisa/paralysis-face: its train/valid/test split is LEAKED.** All 47 within-dataset
  cross-split duplicate groups are anisa — the same source face (+ augmentations) sits
  in train AND valid AND test. **Do NOT use anisa's provided split.**
- **Cross-dataset identical images (188 groups):**
  - `kaggle_facial_droop ↔ roboflow_fnp`: **181** identical images (kaggle is ~mostly FNP)
  - `kaggle_facial_droop ↔ sumin`: 15 · `sumin ↔ roboflow_fnp`: 12
- **FNP and YFP are clean** on the cross-split axis → **Runs #6/#12 (v2) and #15 (v3),
  trained on PalsyNet+FNP+YFP with per-source/subject holdout, are NOT contaminated.**
- 51,903 near-dup pairs (Hamming≤6) — mostly YFP within-subject video frames (expected;
  handled by subject-level YFP splitting) and anisa augmentation copies.

## Rules (enforce in every split / training run)
1. **Split videos by SUBJECT, never by clip/frame.** Already done for PalsyNet (subject
   CV), YFP (subject holdout), Mayo (note the duplicate `FACES018 ≡ MySlate_14`).
2. **Group-aware image splits.** Treat each exact/near-dup group as one unit; the whole
   group goes to exactly ONE of {train,val,test}. Use the dHash groups in dedup_report.
3. **Deduplicate across datasets before pooling.** Keep each image once. In particular,
   drop kaggle_facial_droop images that duplicate FNP (181) — kaggle adds only ~840 new.
4. **anisa & kaggle are TRAIN-ONLY** (or must be re-split by dedup-group). Never put them
   in val/test, and never report a metric on anisa's built-in test split.
5. **New web datasets (grboguz, future Roboflow/Kaggle) must pass `dedup_check.py`** vs
   the existing pool before entering any split.
6. Re-run `scripts/dedup_check.py` whenever a new image set is added.

## Status of current artifacts
- v2/v3 warm-starts: **clean** (FNP/YFP/PalsyNet, honest holdout).
- anisa, kaggle, stroke: usable as **extra TRAIN data only**, deduped, group-split.
