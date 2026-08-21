# Broad Literature-Grounded Shared V9 Screen

## Decision

We evaluated the exact frozen V8 comparator plus **20 mechanism-distinct shared
models** on one H200.  No V9 candidate passes the prespecified three-source
promotion gate, so **V8 remains the canonical deployment model**.  This is a
development experiment, not Mayo or clinical validation.

The full run completed 21 candidates x 3 seeds x 9 fits = **567 fits** in
555.9 seconds.  PalsyNet protected-test reads, Mayo reads, and Mayo predictions
were all **0**.

## What was tested

The 20 new candidates were frozen before results were reviewed.  Each changes
one medically motivated mechanism while keeping one shared facial-motor trunk:

- optimization: SAM, ASAM, SWA, and R-Drop;
- missing-evidence robustness: modality dropout and action-drop consistency;
- representation learning: VICReg, Barlow Twins, masked 110D reconstruction,
  masked action reconstruction, and clinical-to-dense reconstruction;
- clinical objectives: focal, LDAM, pairwise AUROC, high-specificity partial
  AUROC, and Brier-composite losses;
- shared architecture: progressive layered extraction, Cross-Stitch,
  action-conditioned FiLM, and an anatomy-action graph.

Learning-rate, width, seed, epoch, and threshold variants were not counted as
models.  Previously screened MMoE, deep ensembles, anatomical residuals,
kinematic supervision, PCGrad, CORAL, GroupDRO, TCN, BiGRU, and Transformer
models were not repeated.

## Main results

Metrics below use the prespecified mean-probability three-seed estimator and a
fixed 0.5 threshold.  `Worst` is the minimum over PalsyNet development,
NeuroFace, and MEEI.

| Rank | Candidate | Mechanism | Worst specificity | Worst AUROC | Worst accuracy |
|---:|---|---|---:|---:|---:|
| 1 | BLV9-001 | SAM | **0.800** | 0.862 | 0.821 |
| 2 | BLV9-002 | ASAM | 0.727 | 0.873 | 0.821 |
| 3 | BLV9-007 | cross-view VICReg | 0.727 | 0.811 | 0.778 |
| 4 | BLV9-009 | masked clinical reconstruction | 0.700 | **0.920** | **0.839** |
| 5 | BLV9-015 | high-specificity partial-AUROC loss | 0.636 | 0.924 | 0.833 |
| 6 | BLV9-014 | pairwise AUROC loss | 0.636 | 0.916 | 0.833 |
| 7 | BLV9-016 | Brier-composite loss | 0.636 | 0.909 | 0.833 |
| 8 | BLV9-010 | masked action reconstruction | 0.636 | 0.909 | 0.806 |
| 9 | BLV9-000 | exact V8 comparator | 0.636 | 0.905 | 0.833 |
| 10 | BLV9-018 | Cross-Stitch streams | 0.636 | 0.905 | 0.833 |
| 11 | BLV9-020 | anatomy-action graph | 0.636 | 0.905 | 0.833 |
| 12 | BLV9-004 | R-Drop | 0.636 | 0.905 | 0.833 |
| 13 | BLV9-017 | progressive layered extraction | 0.636 | 0.905 | 0.833 |
| 14 | BLV9-003 | SWA | 0.636 | 0.902 | 0.806 |
| 15 | BLV9-019 | action-conditioned FiLM | 0.636 | 0.898 | 0.833 |
| 16 | BLV9-011 | clinical-to-dense reconstruction | 0.636 | 0.895 | 0.806 |
| 17 | BLV9-006 | action-drop consistency | 0.636 | 0.891 | 0.806 |
| 18 | BLV9-012 | focal loss | 0.636 | 0.887 | 0.833 |
| 19 | BLV9-008 | cross-view Barlow Twins | 0.636 | 0.735 | 0.694 |
| 20 | BLV9-013 | LDAM | 0.600 | 0.907 | 0.833 |
| 21 | BLV9-005 | modality dropout | 0.545 | 0.898 | 0.778 |

The most useful signals are complementary rather than universally dominant:

| Candidate | PalsyNet acc/spec/AUROC | NeuroFace acc/spec/AUROC | MEEI acc/spec/AUROC |
|---|---:|---:|---:|
| Exact V8 | 0.921 / 0.882 / 0.950 | 0.833 / 0.636 / 0.905 | 0.875 / 0.700 / 0.928 |
| SAM | 0.921 / 0.882 / 0.952 | 0.861 / **0.818** / 0.862 | 0.821 / **0.800** / 0.911 |
| Masked clinical reconstruction | 0.921 / 0.882 / 0.952 | **0.889** / **0.818** / **0.920** | 0.839 / 0.700 / 0.926 |
| LDAM | **0.947** / **0.941** / **0.955** | 0.833 / 0.727 / 0.924 | 0.857 / 0.600 / 0.907 |

SAM is the only candidate that reaches the 0.80 specificity floor on every
source, but it lowers NeuroFace AUROC and MEEI accuracy/AUROC.  Masked clinical
reconstruction gives the strongest balanced representation result, especially
on NeuroFace, but MEEI specificity remains 0.70.  LDAM improves PalsyNet while
hurting MEEI.  These trade-offs are why none is promoted or combined after
outcome review.

## Frozen evaluation and gate

- Data: 38 PalsyNet development participants, 36 NeuroFace participants, and
  56 MEEI participants.
- Within-source evaluation: six participant-disjoint folds; three fixed seeds.
- Transfer stress test: train on two sources and leave one source out; this is
  descriptive only because the three disease endpoints are not identical.
- Promotion: every source must have accuracy >=0.90, specificity >=0.80,
  sensitivity >=0.85, and AUROC >=0.92; no accuracy or AUROC regression greater
  than 0.01 from V8; worst-source specificity must strictly improve.
- Candidate selection cannot use Mayo, the PalsyNet protected outer test, a
  best seed, or a post-hoc threshold.

## Interpretation and next action

This screen rules out the claim that one currently tested optimization,
regularizer, auxiliary loss, or small shared-architecture change is enough to
make the model uniformly strong.  It also identifies two preregisterable
ingredients for a future experiment: flat-minimum training for control
specificity and masked clinical reconstruction for distributed anatomical
evidence.  They must not be combined on these same outcomes and called a V9;
any combination needs a new frozen protocol or new independent participants.

Until then, V8 remains the formal model.  Mayo performance remains unknown
until participant-level labels and a protected participant-disjoint evaluation
are available.

## Reproducibility

- Machine report: `docs/results/artifacts/broad_literature_shared_v9/report.json`
- Report SHA-256: `27762fafb4923f043483bfc481d70948b3aaff12141f0a5fe2dbfeb1756c7ac4`
- Implementation SHA-256: `94e844478c7903ea03df0d554fa1ac4763f1a2100fb9b076c60653740bb09895`
- Runtime: Python 3.11.13, PyTorch 2.7.1+cu128, NumPy 2.4.6,
  NVIDIA H200, 20 epochs per fit.
