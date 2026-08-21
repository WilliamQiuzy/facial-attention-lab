# autoresearch-FP — autonomous model/data search (Run #17)

> **Archived historical experiment:** The winners in this Run #17 log are not
> current. The sole current model is Universal Clinical Router v4; use
> `../../CURRENT_MODEL.md`. Deleted scratch outputs and the old generic search harness
> remain available in Git history for audit-only reproduction.

_2026-06-18. Adapted karpathy/autoresearch's loop to our task and ran it on the
RunPod A100. Harness: `scripts/fp_research.py` (config-driven trainer, cached
bundles), batches in `scripts/fp_batch{1,2,3}.json`, log `outputs/autoresearch_fp/fp_results.tsv`._

## Adaptation
- **Editable unit** = a CONFIG (model arch + data-processing + optimizer), not a GPT file.
- **Fixed metric** = mean(eyes_QWK, mouth_QWK) on the clean leak-safe holdouts
  (FNP-valid + YFP-held subjects). Higher = better. Eval splits FIXED (leakage_policy).
- **Loop** = keep-if-better, logged to `fp_results.tsv`. **40 experiments**, 3 adaptive
  batches. Search budget: 40 epochs / batch 256 (~2 min/exp); winners re-verified at
  the full budget (80 epochs / batch 64) over 3 seeds.

## What the search explored
data subsets (which sources/tasks), temporal pooling (mean/max/attention), stream
reweighting (marlin_proj 768→128 + LayerNorm), capacity (trunk/temporal dims),
dropout, lr, weight decay, action pooling, per-task loss weights.

## Findings
1. **Short-budget search winner:** `region-only (FNP+YFP, eyes+mouth) + reweight
   (proj128+LN) + lr1e-3` → 0.626/0.622 (eyes ↑ to ~0.45). reweight + region-loss-up
   helped the weak eyes head; adding binary/coarse3/anisa/kaggle DILUTED region QWK;
   dropout 0.3 collapsed eyes; pool_max worst.
2. **But it does NOT hold at the proper budget.** Verified at 80ep/batch64, mean over seeds:

   | config | metric (mean±sd) | eyes | mouth |
   |---|---|---|---|
   | **v2ref** (palsy+FNP+YFP, binary+eyes+mouth) | **0.635 ± 0.015** | 0.420 | 0.850 |
   | **region-only** (FNP+YFP, eyes+mouth) | **0.634 ± 0.016** | 0.404 | 0.863 |
   | region + reweight | 0.614 | 0.407 | 0.822 |
   | region + reweight + lr1e-3 | 0.605 | 0.440 | **0.770** |

   `lr1e-3`/reweight were **artifacts of the undertrained short budget** (lr1e-3 helps a
   40-epoch run converge but overshoots at 80 epochs → mouth crashes to 0.77).

## Conclusion (honest)
- At the proper budget, **nothing in the config space beats the existing v2 recipe
  beyond noise (±0.015).** We are at a **plateau ~0.635**; eyes stuck ~0.42, mouth ~0.85.
- **The ceiling is the DATA, not the model config** — exactly what Runs #7/#9/#16 also
  showed. Eyes severity is intrinsically capped on still web images (no blink dynamics);
  breaking it needs in-domain per-action *video* + HB labels, not more config search.
- **Simplification win (per autoresearch's own criterion):** `region-only` matches v2
  with fewer tasks/sources (drop binary+coarse3+PalsyNet) → a simpler model at equal
  metric. If only region severity is needed, prefer it.

## Reusable harness
`scripts/fp_research.py <batch.json>` runs a batch of configs against the fixed metric
and appends `fp_results.tsv`. Add new ideas as JSON configs and re-run. The search is
ready to re-point at any future labeled data (e.g., Mayo HB) by changing the metric/holdout.
