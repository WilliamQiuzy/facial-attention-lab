# autoresearch-FP — findings (Run: jul1 branch)

Autonomous model search on our own facial-palsy data, adapted from
karpathy/autoresearch. Driven by Claude Opus 4.8. **61 model versions** trained
and compared under one fixed, leak-safe, 3-seed metric. All numbers are on the
public-web-stills holdout (method validation, NOT clinical accuracy — same honest
framing as `docs/training_runs.md`).

## Metric
`mean(eyes_QWK, mouth_QWK)`, quadratic-weighted kappa on the leak-safe val split
(FNP-valid + held YFP subjects), averaged over 3 seeds. Fixed in `prepare_fp.py`.

## Headline
| model | metric | eyes QWK | mouth QWK | note |
|---|---|---|---|---|
| **v0 baseline** (BiGRU-attention ⊕ MARLIN, CORAL) | 0.530 | 0.293 | 0.767 | faithful v2 reimpl |
| GRU-baseline analog (raw feat, all data) | 0.549 | 0.311 | 0.787 | closest to Run #17 v2 recipe, in-harness |
| **champion** (`best_config.json`) | **0.649** | **0.472** | **0.826** | sd 0.003 |

**Within this fixed harness the search improved the model +0.119 over v0 (+22%),
and +0.100 over the GRU-baseline analog (+18%).** The gain is concentrated where
Run #17 was stuck: **eyes QWK 0.29 → 0.47** (+62%). Run #17's "config plateau
~0.635 / ceiling is data" held for its *config menu*; it did **not** hold once the
search could **redesign the architecture** — which is the whole point of using the
real autoresearch template instead of the config-JSON harness.

## The champion architecture
Fully **region-decoupled**, static (no temporal), per-action:
```
per region r ∈ {eyes, mouth}:
  geo_r   = MLP( [52 blendshapes | 20 L/R Δ | ENGINEERED asym invariants] )   # own encoder
  marlin_r= LayerNorm( Proj_r(MARLIN_768) )   # eyes→128 bottleneck, mouth→full 768
  h_r     = MLP( [marlin_r | geo_r] )         # own trunk
  s_r     = w_r · h_r  → CORN ordinal head (label-smoothed)
```
Config: `feat=asym, geo_encoder=mlp, loss=corn(ls=0.1), per_region_geo=true,
pr_marlin={eyes:128, mouth:768}, dropout=0.2, wd=5e-2`. ~0.3M trainable params,
**24 s / 3-seed run on CPU** (v0 GRU was ~164 s).

## What moved the needle (ranked)
1. **Engineered nonlinear asymmetry features** (|L−R|, energy, region aggregates):
   raw 0.55 → asym 0.58. The signed Δ's were already present, but a linear trunk
   can't form magnitudes/ratios; handing them in directly is the single biggest
   clean win. Clinically grounded + domain-invariant (Run #14).
2. **Per-region decoupling.** Eyes and mouth want *opposite* MARLIN treatment —
   eyes improves when MARLIN is bottlenecked (geometry forced to matter), mouth
   needs full MARLIN. One shared trunk can't serve both; separate geo+trunk+MARLIN-
   width per region unlocked eyes 0.47 **and** mouth 0.83 simultaneously.
3. **Drop the GRU → static MLP.** The metric data is single-frame (mp_seq len 1),
   so the temporal encoder was dead weight: equal metric, ~7× faster. (Confirms the
   "temporal can't help on stills" thesis — a *simplification win*.)
4. **CORN ordinal loss + light label smoothing** > plain CORAL: +0.01–0.02, mostly
   mouth stability.

## What did NOT help (honest negatives)
- Class weighting alone (hurt: reshapes shared trunk against mouth).
- `expected`-value decode (0.515, worst).
- Naive MARLIN down-weight / gate fusion without per-region structure.
- Bigger capacity, warmup, `expected` decode, eyes/mouth loss-weight nudges — noise.
- Beyond ~0.649 the search re-plateaus: R7's 8 refinements all land 0.637–0.649.
  So there IS a new, higher plateau — and, consistent with all prior runs, the
  next real lever is **data** (in-domain video dynamics, HB labels, AU corpora),
  not more config search.

## Caveats
- Absolute numbers are **not** directly comparable to Run #17's 0.635: this is a
  stricter, self-contained 3-seed harness (v0 reproduces 0.53, not 0.635). The
  honest claim is the **within-harness** lift (0.53→0.65) and the **eyes** jump.
- Still web-stills method validation. The champion's eyes gain comes from
  engineered asymmetry + MARLIN rebalancing — both are exactly the levers Run #14
  predicted would matter on the Mayo domain, so this model is a better *starting
  point* for HB fine-tuning, but clinical accuracy still needs HB labels.

## Reproduce
```bash
cd facial_paralysis
KMP_DUPLICATE_LIB_OK=TRUE python3 autoresearch_fp/runner.py autoresearch_fp/best_config.json
# full logbook: autoresearch_fp/results.tsv (61 rows, ranked)
# add an idea: drop a JSON config in experiments/, run search.py
```

## Update (2026-07-04 loop): v2 champion 0.668 (+0.02, verified)
A broad final sweep (12 novel configs beyond the original 61) + neighborhood search found
a genuine improvement over the 0.649 champion: **`pr_marlin eyes:256` (was 128) +
`trunk_layers:2`** → **0.668 ± 0.009** (5-seed), vs old 0.648 ± 0.002 — seed ranges barely
overlap, so real not noise. The eyes head benefits from more MARLIN capacity + a deeper
trunk on web. Neighborhood (eyes 320/384/512, trunk 3, ls/batch variants) confirmed 256+deep
is the local optimum. This is the new `best_config.json`. NB: it does not change any Mayo
conclusion — web models still don't transfer (a better web model is still a web model).
