# Literature-Grounded Shared V9 Research Note

## Objective

Improve the genuinely shared PalsyNet/NeuroFace/MEEI encoder without turning
the three cohorts into independent models and without repeatedly selecting on
the same participant outcomes.  Specificity is a primary operational metric,
but a threshold-only gain does not count as a representation improvement.

## Primary papers read

| Paper | Validated mechanism | Relevance to this project |
|---|---|---|
| [GradNorm, ICML 2018](https://proceedings.mlr.press/v80/chen18a.html) | Dynamically equalizes task gradient magnitudes and relative training rates in one multi-task network. | Eligible only if the frozen V8 diagnostic shows source gradient-norm or training-rate imbalance. |
| [PCGrad, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html) | Projects pairwise conflicting task gradients away from one another. | Already implemented as project V5 after negative source-gradient cosine was observed; it made little effective update and will not be repeated. |
| [CAGrad, NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/9d27fdf2477ffbff837d73ef7ae23db9-Abstract.html) | Optimizes average loss while regularizing toward the worst local task improvement. | Eligible because V5 showed that local pairwise projection can disappear after aggregation; CAGrad addresses the combined update directly. |
| [ALGRNet, TMI 2023](https://arxiv.org/abs/2203.01800) | Adaptive landmark-defined muscle regions, inter-region relational modeling, and local-global gated fusion improved AU recognition and transferred to facial-palsy grading. | Supports one shared, anatomy-bounded bilateral-region residual over the current 110D/full-mesh encoder. |
| [Knowledge-driven AU self-supervision, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Chang_Knowledge-Driven_Self-Supervised_Representation_Learning_for_Facial_Action_Unit_Recognition_CVPR_2022_paper.html) | FACS-defined facial partitions and region relationships provide label-free auxiliary supervision and improve data efficiency. | Supports fold-local kinematic auxiliary targets derived from region excursion, velocity, and bilateral synchrony; no extra disease labels are introduced. |
| [Dynamic facial-function landmarks, TBME 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12584918/) | Cue-specific bilateral landmark correlations, Mahalanobis distance, and Wasserstein prototypes separated palsy from controls and localized dysfunction. | Justifies preserving cue identity, bilateral synchrony, and eye/brow/oral region dynamics rather than learning arbitrary landmark mixtures. |
| [Supervised Contrastive Learning, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html) | Same-class embedding compactness can improve robustness and reduced-data classification. | Not selected: the affected class combines unilateral palsy, ALS, and post-stroke impairment, so forcing all positives into one cluster is not medically justified. |
| [DomainBed, ICLR 2021](https://openreview.net/forum?id=lQdXeXDoWtI) | Under controlled model selection, many domain-generalization algorithms failed to beat well-tuned ERM consistently. | Prevents another broad CORAL/DANN/Fishr sweep; project CORAL and GroupDRO have already failed, and the cohorts differ in pathology rather than acquisition domain alone. |
| [Neyman-Pearson classification, COLT 2011](https://proceedings.mlr.press/v19/rigollet11a.html) | Controls type-I error while minimizing type-II error under a strict constraint. | Used only for a prespecified inner-training operating point after a representation is locked; never counted as an AUROC or architecture gain. |
| [MMoE, KDD 2018](https://www.kdd.org/kdd2018/accepted-papers/view/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-) | Shares one expert bank while small task gates learn different expert mixtures when tasks are not equally related. | Supports one bounded shared-expert candidate for heterogeneous diseases and action scripts; all experts must receive gradients from every source. |
| [Deep Ensembles, NeurIPS 2017](https://papers.nips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html) | Averages independently trained probabilistic networks to improve uncertainty and distribution-shift behavior. | Already evaluated in the prior V9 search, so the current three-seed probability average is a reporting estimator, not a new attempt. |
| [DANN, JMLR 2016](https://www.jmlr.org/beta/papers/v17/15-239.html) | Uses gradient reversal to remove domain-discriminative information while retaining task discrimination. | Rejected before implementation because source identity here conflates pathology, endpoint, and script; removing it could erase the clinical signal itself. |

## Frozen, meaningful experiments

An item below is an experiment only when its paper mechanism, project
diagnostic, implementation, and participant-disjoint evaluation are all
complete.  Smoke tests, broken runs, thresholds, and arbitrary hyperparameter
changes are engineering checks, not scientific attempts.

1. **CAGrad shared update.** Run only because source-gradient conflict was
   measured previously.  It is successful only if it improves worst-source
   AUROC/specificity without a sensitivity or accuracy regression.
2. **GradNorm shared update.** Run only if a new frozen diagnostic demonstrates
   at least a two-fold gradient-norm ratio or materially different relative
   loss descent across sources.  Otherwise it is skipped and not counted.
3. **Bilateral anatomical relational residual.** Encode registered right/left
   eye, brow, and oral geometry, exchange information only through a small
   shared relation block, and gate it with the existing global action token.
   Region masks are fixed by clinical landmark names; source identity never
   enters the region encoder.
4. **Clinical-kinematic auxiliary supervision.** During each training fold,
   require the shared action representation to preserve label-free region
   excursion, velocity, and bilateral synchrony targets.  The auxiliary head
   is discarded at inference and cannot see held participants.
5. **Combined model.** Combine only individually non-degrading mechanisms.
   No factorial combination is run merely to increase the experiment count.

## Frozen H200 authorization result

The one permitted diagnostic used six participant-disjoint training folds and
the prespecified five-epoch observation point.  The shared gradient-norm ratio
was `1.114`, the relative remaining-loss ratio was `1.088`, and all three
median pairwise gradient cosines were positive (`0.185`, `0.319`, `0.545`).
Consequently, **neither GradNorm nor CAGrad is authorized**.  They are recorded
as literature-screened exclusions, not as model experiments.

## Evaluation boundary

- Freeze candidates before reading outcome metrics.
- Use deterministic CUDA and participant-disjoint folds.
- Report accuracy, balanced accuracy, sensitivity, specificity, AUROC, and
  Brier score for every source.
- Add a universal-head leave-one-source-out diagnostic; it is descriptive
  because the three endpoints are not clinically identical, but it is a more
  relevant transfer stress test than another within-source score.
- Do not read Mayo or the protected PalsyNet outer test.
- Promotion requires a stable multi-seed improvement, not the best seed.
