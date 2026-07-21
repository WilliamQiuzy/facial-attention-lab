# Focused SSL Transfer Smoke Specification

## Objective

Determine quickly whether the authenticated Mayo Fusion masked-reconstruction
encoder provides a useful initialization for PalsyNet binary classification.

## Fixed development-only comparison

- Use PalsyNet `clinical23_v2` tensors and the existing grouped nested split.
- Fix outer fold 0 as a protected, untouched partition. Do not produce a
  prediction for any row in that partition.
- On the remaining outer-train rows, collect four-fold inner OOF predictions
  for exactly three candidates with seed 0 and a fixed 12-epoch budget:
  `landmark_random`, `fusion_random`, and `fusion_ssl_warmstart`.
- The warm-start candidate must authenticate the existing focused Fusion
  winner chain, copy only the four modality projections, BiGRU, attention, and
  pool projection, and leave a freshly initialized binary head unchanged.
- Use the same fold-local standardization, augmentation, optimizer, learning
  rate, and weight decay for all candidates.

## Report and decision boundary

Report group-level inner-OOF AUROC, average precision, Brier score, balanced
accuracy, sensitivity, and specificity. The report must state
`video_held_out`, `unreviewed`, and
`inner_oof_development_smoke_only`; it must contain no recording IDs, group
IDs, patient IDs, source paths, artifact paths, or outer predictions.

This experiment is a direction-selection smoke, not patient-held-out clinical
validation. Expand to three seeds only if warm-start improves random Fusion by
at least 0.02 AUROC without reducing sensitivity by more than 0.03. The locked
outer evaluator remains unchanged.
