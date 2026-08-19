# Universal Orofacial Model v1/v2

## Outcome

The universal candidate is **not promoted**. We built and evaluated one
source-blind model across 38 identity-reviewed PalsyNet development
participants and all 36 NeuroFace participants, then applied the locked v1
candidate once to the already-exposed 60-participant MEEI diagnostic. The
experiment was useful, but the evidence rejects the claim that the present
MediaPipe representation is a universal affected-versus-unaffected signal.

All results are participant-level. AUROC measures how reliably affected people
rank above unaffected people across all thresholds; balanced accuracy is the
mean of sensitivity and specificity at the fixed 0.5 threshold, so the larger
affected class cannot dominate the score.

| Same model / representation | PalsyNet AUROC | PalsyNet balanced accuracy | NeuroFace AUROC | NeuroFace balanced accuracy |
|---|---:|---:|---:|---:|
| 110D source-balanced Logistic | 0.964 | 0.976 | 0.564 | 0.498 |
| 110D GroupDRO low-rank network | 0.894 | 0.728 | 0.513 | 0.522 |
| 110D multi-task low-rank network | 0.899 | 0.758 | 0.509 | 0.613 |
| Blendshape-288 Logistic | 0.922 | 0.828 | 0.655 | 0.578 |
| **Fusion-398 Logistic** | **0.958** | **0.947** | **0.680** | **0.598** |

Fusion-398 was the best multi-signal representation by the preregistered
worst-source rule. It improved NeuroFace AUROC by 0.116 over Landmark-110 while
preserving strong PalsyNet performance, but it remained far below the 0.90
gate. Its source-heldout AUROC was 0.422 from PalsyNet to NeuroFace and 0.714
from NeuroFace to PalsyNet, confirming that a dataset-specific boundary—not a
shared disease-invariant boundary—still drives the results.

The locked v1 110D model was also run once on MEEI without refitting,
recalibration, threshold selection, or candidate selection. It reached AUROC
0.820 and balanced accuracy 0.770. This repeated, already-exposed diagnostic is
not untouched external validation and did not pass the 0.90/0.90 gate.

## What changed and what did not

Universal v1 held Landmark-110 fixed and compared source-balanced Logistic,
GroupDRO, and a shared low-rank multi-task network. GroupDRO optimizes the
worst observed training group, but its original authors also emphasize that
strong regularization and model selection are necessary; here it did not
create a shared clinical signal from incompatible endpoints.

Universal v2 held the estimator fixed and changed only the upstream
representation: Landmark-110, Blendshape-288, and their exact Fusion-398. The
Blendshape block summarizes all 72 MediaPipe blendshape and left/right
difference channels with median, interquartile range, range, and maximum
per-second velocity. This was the correct direction—NeuroFace improved—but it
did not reproduce the independent Py-Feat AU signal.

The NeuroFace ALS development result near AUROC 0.95 came from a separate
Py-Feat action-unit model on original video frames, not from Landmark-110 or the
MediaPipe TCN. A label-free connector diagnostic could only bring the
MediaPipe-derived AU-like signal to about 0.80 while weakening PalsyNet. We
therefore do not relabel the ALS-specific Py-Feat result as performance of this
universal model.

## Scientific interpretation

PalsyNet chiefly rewards unilateral facial asymmetry. NeuroFace combines ALS
and post-stroke participants and chiefly rewards action capacity or AU
intensity, including bilateral weakness. MEEI is facial palsy but differs in
institution, acquisition, and class balance. A single small binary head cannot
make these endpoint definitions equivalent merely by changing its architecture.

The correct next architecture is a shared facial encoder with explicit
phenotype heads—unilateral asymmetry, bilateral action capacity, and later HB
severity—followed by a preregistered fusion rule. That requires either original
videos (so the same AU extractor can run everywhere) or a newly collected
multi-disease cohort with the same scripted actions and labels. Until then,
the current endpoint-specific PalsyNet 110D and NeuroFace ALS Py-Feat models
remain separate research models.

## Audit and claim boundary

- The H200 run used 74 development participants and completed in seconds after
  feature loading.
- PalsyNet protected cache reads and predictions were both zero; only 39
  development recordings from 38 reviewed participants were opened.
- Mayo, MEEI, and YFP reads during candidate selection were zero. MEEI was
  opened only after the v1 candidate was locked.
- Private model artifacts remain owner-only on H200; the public report contains
  no participant identifiers, row probabilities, paths, or model weights.
- This is **not clinical validation**, not Mayo accuracy, not HB grading, and
  not evidence for deployment.

Primary method context: [GroupDRO](https://arxiv.org/abs/1911.08731) and the
[MLST-Net multi-task facial-paralysis study](https://pubmed.ncbi.nlm.nih.gov/40031643/).

Machine-readable aggregate:
`docs/results/artifacts/universal_orofacial_v1/report.json`.
