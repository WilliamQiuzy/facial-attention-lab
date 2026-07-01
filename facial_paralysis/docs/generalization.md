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

| | champion | baseline | old model (Run #14) |
|---|---|---|---|
| eyes | -0.06 | -0.14 | -0.01 |
| mouth | -0.19 | +0.01 | -0.50 |

**All ≈ 0.** A model trained on web stills — even the improved champion — does NOT
transfer to the Mayo domain. Its 0.649 web QWK says nothing about Mayo. The gap is
the **modality/domain** (web stills vs in-domain clinical video), which web-side model
work cannot close.

## Conclusion
- The champion is a real improvement: higher web QWK **and** better cross-dataset
  generalization. It is the best starting point for fine-tuning.
- But **Mayo performance requires in-domain data.** Web gains don't cross the modality
  gap. This is why, with no HB labels, the deployable Mayo estimator is the
  **label-free dynamic clinical measure** (asymmetry / EAR closure / synkinesis /
  forced recruitment — `docs/mayo_faces_analysis.md`), which is domain-invariant by
  construction, not the web-trained learned severity.

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
