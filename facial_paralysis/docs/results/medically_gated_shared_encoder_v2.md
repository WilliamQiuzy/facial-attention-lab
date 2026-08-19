# Medically-Gated Shared Clinical Encoder: 48-Version Result

## Outcome

We evaluated **48 genuinely shared models**: a 32-version full-mesh family and
a 16-version compact regional family. Every model used one trainable encoder
shared by PalsyNet, NeuroFace, and MEEI; dataset identity was allowed only at
the final small binary head. No shared candidate exceeded 90% three-seed mean
accuracy in all three cohorts, so this experiment is **not promoted** and makes
no Mayo or clinical-performance claim.

| Strongest stable shared candidate | PalsyNet | NeuroFace | MEEI |
|---|---:|---:|---:|
| MSC2-022, three-seed mean accuracy | 87.72% | 80.56% | 85.12% |
| Mean AUROC | 0.937 | 0.847 | 0.891 |

These are participant-disjoint results on exposed development cohorts, not
three independent external tests. Mayo was not read, trained on, or scored.

## Architecture that survived the search

`MSC2-022` takes the frozen 110D clinical geometry and the full
`32 x 478 x 3` MediaPipe action trajectory. The clinical branch, full-mesh
spatial/temporal branch, regional-excursion branch, fusion layer, cross-action
Transformer, and 64D patient embedding are all shared across the three
datasets. Only the final binary heads differ.

The mirror pair is combined only by commutative mean and absolute difference.
This is medically admissible for a binary weakness endpoint because the label
does not change when the affected side changes. It is explicitly forbidden for
affected-side prediction, signed regional scores, or laterality-specific HB
interpretation.

## What the 48 versions established

- The best seed-0 full-mesh direction used region-aware excursion and
  cross-action aggregation, but its NeuroFace result was initialization
  sensitive; three-seed confirmation prevented a false success claim.
- Extending training from 20 to 50, 100, and 200 updates reduced NeuroFace/MEEI
  performance, supporting an overfitting diagnosis rather than undertraining.
- Compressing all 478 points into brow/eye/mouth/global excursion and velocity
  summaries preserved PalsyNet (best seed-0 accuracy 94.74%) but reduced
  NeuroFace and MEEI (77.78% and 85.71%). Fine-grained mesh information is
  therefore still needed.
- Random crop, color jitter, arbitrary flip augmentation, dataset-specific
  trainable encoders, and score-driven landmark selection were not tried
  because they lacked an adequate clinical or measurement rationale for this
  protocol.

## Interpretation and next gate

This search shows that parameter sharing alone does not guarantee robust
cross-disease transfer. PalsyNet also remains an exposed development cohort,
and the shared candidate does not justify reopening its protected outer test.
The next scientifically useful input is labeled Mayo data with participant
identity and scripted-action timing. We should then keep this shared trunk as a
research comparator, add the Mayo endpoint head after the shared embedding,
and evaluate participant-disjoint Mayo binary/HB performance without selecting
on the same participants.

Primary rationale sources: [Sunnybrook](https://pubmed.ncbi.nlm.nih.gov/8649870/),
[eFACE](https://pubmed.ncbi.nlm.nih.gov/26218397/),
[dynamic 3D facial movement](https://pubmed.ncbi.nlm.nih.gov/30534499/),
[dynamic facial-function ML](https://pubmed.ncbi.nlm.nih.gov/40333095/), and
[MediaPipe contour definitions](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/python/vision/face_landmarker.py).

Machine-readable evidence:
[`artifacts/medically_gated_shared_encoder_v2/report.json`](artifacts/medically_gated_shared_encoder_v2/report.json).
