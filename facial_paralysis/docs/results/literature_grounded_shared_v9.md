# Literature-Grounded Shared V9 Decision

## Decision

**No V9 candidate is promoted; V8 remains the canonical deployment model.**
Three new, paper-supported mechanisms were frozen before outcome review and
tested with the same participant-disjoint protocol across PalsyNet development
(38 people), NeuroFace (36), and MEEI (56). None improved the locked V8
comparator without a source regression, and none met the joint minimums of
0.90 accuracy, 0.80 specificity, 0.85 sensitivity, and 0.92 AUROC on every
source. The V8 model registry and deployment document remain byte-identical.

## New experiments that count

1. **Bilateral anatomical relation residual.** A single source-blind block
   related fixed eye, brow, oral, and global action tokens. This follows the
   landmark-defined local/global reasoning in
   [ALGRNet](https://arxiv.org/abs/2203.01800). It reduced NeuroFace and MEEI
   accuracy/specificity and was rejected.
2. **Clinical-kinematic auxiliary supervision.** A fold-local auxiliary head
   preserved label-free regional excursion, velocity, and bilateral synchrony,
   following FACS-region self-supervision from
   [CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Chang_Knowledge-Driven_Self-Supervised_Representation_Learning_for_Facial_Action_Unit_Recognition_CVPR_2022_paper.html)
   and cue-specific dynamic landmark evidence from
   [Rao et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC12584918/). It was
   discarded at inference and never saw held participants. It improved no
   worst-source endpoint and was rejected.
3. **Multi-gate shared experts.** Three rank-16 motor experts were shared by
   all datasets; only small task gates selected their mixture, following
   [MMoE](https://www.kdd.org/kdd2018/accepted-papers/view/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-).
   All three sources produced nonzero gradients in the same expert bank and
   task-specific parameters remained below 10%. Its ensemble decisions matched
   V8, while NeuroFace AUROC was slightly lower, so it was rejected.

The table uses the prespecified mean-probability estimator across seeds 0, 1,
and 2. This reduces initialization variance but is not a new experiment because
deep ensembles were already tested in the earlier 94-candidate V9 search.

| Model | Dataset | Accuracy | Specificity | Sensitivity | AUROC |
|---|---|---:|---:|---:|---:|
| V8 comparator | PalsyNet development | 92.11% | 88.24% | 95.24% | 0.950 |
|  | NeuroFace | 83.33% | 63.64% | 92.00% | 0.905 |
|  | MEEI | 87.50% | 70.00% | 91.30% | 0.928 |
| Bilateral anatomical relation residual | PalsyNet development | 92.11% | 88.24% | 95.24% | 0.950 |
|  | NeuroFace | 80.56% | 54.55% | 92.00% | 0.902 |
|  | MEEI | 85.71% | 60.00% | 91.30% | 0.917 |
| Clinical-kinematic auxiliary supervision | PalsyNet development | 92.11% | 88.24% | 95.24% | 0.952 |
|  | NeuroFace | 80.56% | 63.64% | 88.00% | 0.898 |
|  | MEEI | 87.50% | 70.00% | 91.30% | 0.928 |
| Multi-gate shared experts | PalsyNet development | 92.11% | 88.24% | 95.24% | 0.950 |
|  | NeuroFace | 83.33% | 63.64% | 92.00% | 0.902 |
|  | MEEI | 87.50% | 70.00% | 91.30% | 0.928 |

## Not counted as experiments

- **GradNorm and CAGrad:** the prespecified diagnostic did not authorize them.
  The gradient-norm ratio was 1.114, relative training-rate ratio was 1.088,
  and all three median source-gradient cosines were positive (0.185, 0.319,
  0.545). Their required imbalance/conflict was absent.
- **DANN/source-adversarial training:** source identity here combines disease,
  label semantics, and action script. Removing it could erase genuine clinical
  phenotype rather than only acquisition nuisance, so the assumptions of
  [DANN](https://www.jmlr.org/beta/papers/v17/15-239.html) are not met.
- **Supervised contrastive learning:** “affected” joins unilateral palsy, ALS,
  and post-stroke impairment; forcing these phenotypes into one positive cluster
  lacks medical justification.
- **Threshold, width, dropout, learning-rate, SAM, and SWA sweeps:** these were
  not run. Repeatedly selecting them on the same 130 exposed participants would
  add development overfitting without new clinical evidence.
- **Deep ensemble:** already evaluated in the earlier V9 search; this run only
  reconfirmed the locked three-seed reporting estimator and is not counted again.

## Transfer finding and clinical boundary

The universal-head leave-one-source-out stress test remained weak, especially
when NeuroFace was entirely withheld (V8 mean AUROC 0.516). This does not prove
that the representation is clinically invalid—the datasets use different
diseases and scripts—but it does show that a universal binary decision boundary
does not yet transfer reliably between them. More tuning on the same cohorts is
therefore not a defensible route to a robust Mayo classifier.

- PalsyNet protected reads: **0**.
- Mayo reads: **0**; Mayo predictions: **0**.
- No Mayo binary-performance, HB-grade, clinical-validation, or production
  claim is authorized.
- The next meaningful model experiment requires participant-disjoint Mayo
  labels/controls or a newly frozen external cohort, followed by separate V8,
  shared-representation, and preregistered fusion evaluation.

Machine-readable aggregate evidence is in
[`artifacts/literature_grounded_shared_v9/`](artifacts/literature_grounded_shared_v9/).
