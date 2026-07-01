# autoresearch-FP — autonomous model search on our own data

Adapted from karpathy/autoresearch to the facial-palsy severity task. The idea is
unchanged: an agent (Claude Opus 4.8) edits ONE file, trains, checks if the metric
improved, keeps or reverts, and repeats — autonomously. What changes is the task,
the editable unit, and the metric.

## The three files

- **`prepare_fp.py`** — FIXED harness. Owns the data, the leak-safe train/val
  splits, and the ground-truth metric. **Never edit it.** (karpathy `prepare.py`.)
- **`train_fp.py`** — the ONE file you edit. Full model + feature engineering +
  loss + optimizer + training loop. Everything here is fair game. (karpathy
  `train.py`.)
- **`program_fp.md`** — this file: instructions + idea bank. Edited by the human.

## Metric (the only thing that matters)

`metric = mean(eyes_QWK, mouth_QWK)` — quadratic-weighted Cohen kappa of predicted
vs. true region severity grades on the leak-safe validation split, **averaged over
`prepare_fp.SEEDS` (3 seeds)**. **Higher is better.** Fixed budget
`prepare_fp.MAX_EPOCHS`. Multi-seed averaging is mandatory and lives in the harness
because Run #17 was fooled by single-seed noise (±0.015) — a real win must clear
that band. A change is only a KEEP if `metric` rises by **more than `metric_sd`**.

Reference baseline: **~0.635** (v2-attention / Run #17 plateau; eyes ~0.42,
mouth ~0.85). Beating it — especially lifting the weak **eyes** head — is the goal.

## What you CAN do (edit `train_fp.py`)

- Redesign the model: fusion, depth/width, normalization, attention, residuals.
- **Feature-engineer the geometric stream** (highest-value lever, see below).
- Change the ordinal loss (CORAL / CORN / weighted / focal-ordinal / label
  smoothing), the severity coupling, per-task loss weights, class weighting.
- Change optimizer, schedule, regularization, augmentation of the cached features.

## What you CANNOT do

- Edit `prepare_fp.py`, the metric, or the splits. They are ground truth.
- Add data sources beyond what the harness loads, or install new packages
  (numpy / torch / sklearn only).
- Peek at or fit the val set beyond the provided metric.

## Why the task is hard (read before theorizing)

Almost all metric-relevant records (FNP/YFP eyes+mouth) are **still images**:
`mp_seq` length 1 → the temporal GRU sees zero motion, so for the metric it is just
an MLP on a single 72-d geometric vector (52 blendshapes + 20 L/R asymmetry deltas)
plus the 768-d frozen MARLIN vector. Only 49 PalsyNet videos carry real dynamics
(binary task, val n=10, not in the metric). Consequences:

- **Temporal modeling cannot help the metric** on current data. Do not spend the
  budget there. The live levers are **geometric feature engineering, fusion, loss,
  and regularization**.
- **`eyes` is the bottleneck (~0.42).** Eye-closure severity ideally needs blink
  *dynamics*, absent in stills — so gains must come from squeezing the static
  geometric signal harder (asymmetry structure), not from motion.
- MARLIN (768) dominates the concat and is domain/appearance-heavy (Run #14). On
  web stills it *helps* the region labels (Run #15 B-1: naive down-weighting hurt),
  but smarter fusion (gating, per-region stream weighting, normalization) is open.

## Idea bank (starting points — invent your own too)

1. **Explicit asymmetry features.** The 72-d vector already holds 20 L/R deltas.
   Derive richer invariants from them: |L−R|/(L+R+eps) ratios, region aggregates
   (brow/eye/mouth), max-asymmetry, signed side. Feed as extra inputs. This is the
   clinically-grounded, domain-invariant signal (Run #14) the model under-uses.
2. **Per-region stream gating.** Let eyes vs. mouth heads weight MARLIN vs.
   geometric differently (learned gate), instead of one shared trunk.
3. **Loss redesign.** CORN (conditional) vs. CORAL; per-class / QWK-aligned
   weighting to fight the label imbalance (eyes is 71% grade-1); ordinal label
   smoothing; direct soft-QWK surrogate.
4. **Regularize the small trunk.** Input dropout / feature noise on MARLIN,
   weight decay sweeps, ensembling seeds, stochastic depth.
5. **Better severity readout.** Expected-grade (Σ j·P) vs. threshold-count decode;
   temperature on the cut-points.
6. **MARLIN dimensionality.** PCA/whiten or a learned low-rank projection with a
   LayerNorm, tuned jointly with a *stronger* geometric branch (v3 failed by
   shrinking MARLIN blindly — the point is to REBALANCE, not just cut).
7. **Capacity & schedule.** trunk width/depth, GRU→MLP swap for the static case,
   warmup + cosine, longer within budget.

## The loop (do this autonomously, do NOT stop to ask)

Run on branch `autoresearch/<tag>` (already created). LOOP:

1. Pick ONE idea; edit `train_fp.py`.
2. `KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=4 python3 train_fp.py > run.log 2>&1`
   (redirect; never flood context).
3. `grep -E "^(metric|metric_sd|eyes_qwk|mouth_qwk):" run.log`. Empty → crashed:
   `tail -40 run.log`, fix if trivial, else log `crash` and move on.
4. Append a row to `results.tsv` (TAB-separated): `iter  metric  metric_sd  eyes
   mouth  status  description`.
5. **Keep-if-better:** if `metric` improved by more than `metric_sd`, `git add -A
   && git commit` (advance). Else `git checkout -- train_fp.py` (revert to best).
6. Repeat. Target **≥20 iterations**. Combine near-misses, then try more radical
   redesigns. Simpler-and-equal is a KEEP (simplification win).

`results.tsv` is the logbook; keep it untracked. The first run is the baseline
(v0 as-is). Every later `train_fp.py` you keep must be a committed improvement.
