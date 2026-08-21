# Shared V9 Specificity-First Search

## Decision

**No V9 candidate is promoted.** We evaluated 94 medically constrained,
genuinely shared candidates, but none simultaneously met the locked minimums
for accuracy (0.90), specificity (0.80), sensitivity (0.85), and AUROC (0.92)
on PalsyNet development, NeuroFace, and MEEI. **V8 remains the canonical deployment model**
and its registry and deployment files are unchanged.

## What was tested

| Search family | Candidates | Main question | Outcome |
|---|---:|---|---|
| Healthy-reference and operating point | 24 | Can a shared control anchor or training-fold threshold reduce false positives? | NeuroFace specificity reached 0.909 only by reducing sensitivity to 0.72. |
| Equal deep ensembles | 16 | Can seed/rank averaging reduce small-cohort variance? | No worst-source improvement; repeated runs exposed nondeterministic CUDA variation. |
| Nested clinical distillation | 13 | Can inner-OOF action-geometry teachers improve the shared encoder? | Distillation weakened AUROC and/or specificity; no candidate beat the comparator. |
| Low-dimensional mechanism encoder | 16 | Can 110D plus regional excursion/velocity reduce sample complexity? | PalsyNet improved, but NeuroFace AUROC fell sharply; full-mesh information was being discarded. |
| Full-mesh action phenotype heads | 25 | Can a shared 478D+110D encoder learn explicit script-specific action weights? | NeuroFace specificity recovered to 0.636, but AUROC and the other cohorts regressed. |

The last three searches used deterministic CUDA (`CUBLAS_WORKSPACE_CONFIG`,
deterministic PyTorch algorithms, no TF32). The exact comparator was reproduced
independently in the distillation and phenotype runners:

| Deterministic participant-disjoint comparator | Accuracy | Specificity | Sensitivity | AUROC |
|---|---:|---:|---:|---:|
| PalsyNet development | 92.11% | 88.24% | 95.24% | 0.905 |
| NeuroFace | 80.56% | 54.55% | 92.00% | 0.927 |
| MEEI | 89.29% | 80.00% | 91.30% | 0.930 |

These deterministic reruns are less favorable than the original adaptive V8
seed result. Two otherwise identical pre-determinism H200 runs also differed by
one MEEI participant and in NeuroFace AUROC. This is evidence of small-cohort
training instability, not a new clinical result, and is why no apparent
single-run gain was accepted.

## Model and clinical boundary

- Every neural candidate retained a trainable middle layer shared by all three
  datasets; dataset identity never entered the landmark encoder.
- Script-specific logic was restricted to small post-shared action queries,
  phenotype weighting, or binary heads; task-specific parameters stayed below
  10%.
- Mirroring was used only as a bilateral representation and never as evidence
  that one side was normal.
- PalsyNet protected reads: **0**.
- Mayo reads: **0**; Mayo predictions: **0**.
- No HB grading, clinical validation, or deployment-performance claim is made.

The evidence says the current bottleneck is not another threshold or small head
change. The next defensible upgrade needs new participant-disjoint clinical
labels/controls or a separately frozen external cohort. Until then, additional
selection on these same 130 exposed participants would mainly increase
development overfitting.

Machine-readable aggregate reports are in
[`artifacts/shared_v9_specificity_first/`](artifacts/shared_v9_specificity_first/).
