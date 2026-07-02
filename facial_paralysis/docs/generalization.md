# Generalization: what transfers and what doesn't (2026-07-01, no HB labels)

Using the Mayo per-action clips as a real OUT-OF-DOMAIN test set, plus cross-dataset
tests on the web data. Scripts: `scripts/mayo_generalization.py`, `scripts/cross_dataset.py`.

## 1. Within web-stills modality — the champion genuinely generalizes better
Cross-dataset region QWK (train one source, test the other):

| | champion | baseline |
|---|---|---|
| FNP→YFP eyes | **0.558** | 0.377 |
| FNP→YFP mouth | **0.746** | 0.658 |
| YFP→FNP eyes | 0.205 | 0.194 |
| YFP→FNP mouth | **0.530** | 0.476 |

The autoresearch redesign improved **generalization**, not just single-split fit —
better in every cell. (YFP→FNP eyes stays weak: FNP eyes are intrinsically hard.)

## 2. Across modality (web stills → Mayo video) — NOTHING transfers
Mayo clips scored by each model; Spearman(model severity, domain-invariant clinical
asymmetry) across 13 patients:

| | champion | baseline | **geometry-only** | old model (Run #14) |
|---|---|---|---|---|
| eyes | -0.06 | -0.14 | **+0.48** (p=.10) | -0.01 |
| mouth | -0.19 | +0.01 | 0.00 | -0.50 |

The appearance-driven models (champion, baseline) ≈ 0: even the improved champion does
NOT transfer to Mayo — its 0.649 web QWK says nothing about Mayo. **But dropping MARLIN
appearance entirely (geometry-only) flips the eyes head to +0.48**: it now tracks the
clinical asymmetry on Mayo. So the modality gap is specifically the **appearance
stream** (MARLIN, domain-confounded per Run #14); the geometric/asymmetry stream DOES
transfer. Actionable: the **Mayo/deployment model should be geometry-only**, while the
champion (with MARLIN) is best for the web benchmark. Caveats: n=13, p=0.10 marginal,
and partial circularity (geometry features overlap the asymmetry target) — directional,
needs HB labels to confirm. Mouth doesn't transfer even geometry-only (open problem).

## Conclusion
- The champion is a real improvement: higher web QWK **and** better cross-dataset
  generalization. It is the best starting point for fine-tuning.
- But **Mayo performance requires in-domain data.** Web gains don't cross the modality
  gap. This is why, with no HB labels, the deployable Mayo estimator is the
  **label-free dynamic clinical measure** (asymmetry / EAR closure / synkinesis /
  forced recruitment — `docs/mayo_faces_analysis.md`), which is domain-invariant by
  construction, not the web-trained learned severity.

## No-label interventions tried (2026-07-02)

**#3 Feature-noise augmentation** (`scripts/aug_generalization.py`) — regularizer to
fight appearance-overfit. Improves the WORST-case cross-dataset direction (YFP→FNP eyes
0.205→0.297, +45%; mouth 0.53→0.55) but costs the best-case (FNP→YFP eyes 0.56→0.44).
Net: a **robustness/peak tradeoff** — use noise if worst-case transfer matters, not a
free win.

**#1 CORAL domain adaptation** (`scripts/mayo_coral.py`) — align web↔Mayo feature
covariance (Mayo unlabeled) to make the FULL (MARLIN) model transfer without dropping
appearance. **NEGATIVE:** eyes stays ≈-0.05, mouth ≈-0.15 across λ∈{0,1,10}. Covariance
alignment can't fix the head's reliance on appearance directions (and n=13 target is
tiny). **Confirms: for Mayo you must drop MARLIN (geometry-only), not align it.**

**#2 AU-dynamics pretraining pipeline** (`scripts/au_pretrain.py`) — BUILT + verified on
synthetic AU (masked-recon MSE 0.27→0.09, 20 L/R pairs match the 72-d layout). Ready to
pretrain the temporal encoder the moment DISFA/BP4D land (loaders in
`au_intensity_adapter.py`). This is the main lever that trains the geometric/temporal
stream on real dynamics — but it needs the gated data.

## What can still improve things WITHOUT HB labels (ranked)
1. **Augmentation / regularization** (feature noise, mixup on cached features) → push
   cross-dataset web generalization further. Local, fast. *(I can run this.)*
2. **Self-supervised / domain-adaptation pretraining on Mayo clips** (action-ID,
   rest-vs-peak, gentle-vs-forced, CORAL feature alignment) → a domain-robust encoder
   ready for HB fine-tuning. Uses the clips for training. *(I can run this; n=13 so
   treat as proof-of-concept.)*
3. **AU-dynamics pretraining** (DISFA/BP4D) — trains the temporal stream on real
   movement; adapter built (`au_intensity_adapter.py`); needs the gated data.
4. **In-domain healthy controls** (record with the same FACES protocol) → unlocks a
   real Mayo palsy-detection metric.
5. **HB labels** on the highest-disagreement takes (`outputs/mayo_active_learning.json`)
   → the one true unlock for a supervised, validated clinical severity model.
