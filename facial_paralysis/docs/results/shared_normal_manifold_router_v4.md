# Shared Clinical Encoder V4–V9: Iteration Result

## Outcome

We completed six medically constrained shared-model iterations on the exposed
PalsyNet development, NeuroFace, and MEEI cohorts. None passed the requirement
for strong performance in all three cohorts, so **no candidate is promoted** and
the canonical UCR4 files remain byte-identical to `HEAD`.

The strongest single-seed genuinely shared candidate was `RSR8-001`:

| Participant-disjoint development result | PalsyNet | NeuroFace | MEEI |
|---|---:|---:|---:|
| Accuracy | **92.11%** | **86.11%** | **91.07%** |
| Balanced accuracy | 91.74% | 79.82% | 86.74% |
| Specificity | 88.24% | 63.64% | 80.00% |
| AUROC | 0.905 | 0.924 | 0.928 |

This is seed-0 adaptive development evidence, not a stable three-seed result or
external validation. It is below the preregistered worst-source gate and is not
the current model.

## What was tested

| Iteration | Medically motivated change | Result |
|---|---|---|
| V4 | Shared healthy-control anchor plus shared normality logit | Healthy specificity remained low; best seed-0 accuracies were 89.47%, 83.33%, and 87.50%. |
| V5 | Project measured conflicting source gradients without changing the representation | No improvement; the selected aggregate projection largely masked the local conflict. |
| V6 | Keep action physiology shared, then use tiny script-aware pooling and heads | One seed reached 92.11% PalsyNet and 91.07% MEEI, but NeuroFace remained 83.33% and the confirmed candidates were unstable. |
| V7 | Replace raw-mesh learning with fold-local, label-free PCA of median, quantile, range, and velocity statistics | PalsyNet improved, but NeuroFace fell to 75%; PCA discarded low-variance neurological evidence. |
| V8 | Add rank-limited disease residuals only after the shared patient representation | Best overall shared compromise: 92.11%, 86.11%, and 91.07%; still insufficient. |
| V9 | Freeze the shared core and train only script adapters/heads for 40 or 80 more updates | Both schedules degraded PalsyNet/MEEI and did not improve NeuroFace. |

The most important falsification is that the problem is not merely model size,
threshold, or insufficient epochs. PalsyNet gradients conflicted with the two
scripted cohorts, while NeuroFace and MEEI were more aligned. NeuroFace also has
bilateral ALS/post-stroke impairment rather than the predominantly unilateral
facial-palsy endpoint. Forcing one disease boundary into one shared embedding
therefore creates negative transfer.

## Why training stops here

The three datasets have already been repeatedly exposed to architecture
selection. Continuing to invent candidates until the same 130 participants
cross an arbitrary accuracy threshold would optimize to these cohorts, not
produce a more robust Mayo model. The separate UCR6 branches remain stronger
development predictors, but they do not provide the genuine trainable sharing
required by this experiment.

The next defensible improvement requires new information rather than another
architecture loop: participant-disjoint Mayo labels/controls, or a separately
preregistered teacher-distillation study in which the high-performing fixed
clinical experts supervise a shared motor encoder and an untouched cohort tests
transfer. Until then, `RSR8-001` is an archived research finding only.

## Boundaries and evidence

- PalsyNet protected reads: **0**.
- Mayo reads and predictions: **0**.
- No HB label or grading claim.
- No random image augmentation or presumed-normal contralateral side.
- H200: NVIDIA H200, 143,771 MiB reported by `nvidia-smi`; PyTorch
  `2.7.1+cu128` CUDA runtime verified.
- Raw machine reports are under
  [`artifacts/shared_normal_manifold_router_v4/`](artifacts/shared_normal_manifold_router_v4/).

Medical rationale: regional movement magnitude and velocity in facial palsy
([PMID 30534499](https://pubmed.ncbi.nlm.nih.gov/30534499/),
[PMID 27480299](https://pubmed.ncbi.nlm.nih.gov/27480299/)) and lip/jaw
kinematics in ALS ([PMID 29800359](https://pubmed.ncbi.nlm.nih.gov/29800359/)).
