# Broad Literature-Grounded Shared V9 Implementation Plan

> **Execution rule:** implement with test-driven development; a candidate only
> counts after its mechanism is directly tested, trained on all three sources,
> and evaluated with the frozen participant-disjoint protocol.

## Goal

Evaluate **20 new mechanism-distinct shared models plus the exact frozen
RSR8-001/V8 comparator**.  Learning-rate, width, threshold, seed, or epoch
changes are not models.  Every candidate keeps one genuinely shared facial
motor trunk; task-specific capacity remains below 10%, and source identity may
only select the small endpoint head after the shared representation.

The 20 candidates are frozen before outcome review.  No Mayo video, Mayo label,
or protected PalsyNet outer-test artifact may be read.

## Frozen Candidate Registry

| ID | Mechanism changed from V8 | Paper basis | Clinical/project rationale | Frozen setting |
|---|---|---|---|---|
| BLV9-001 | SAM optimizer | [Sharpness-Aware Minimization, ICLR 2021](https://openreview.net/forum?id=6Tm1mposlrM) | The exposed cohorts are small; a locally flatter shared solution may be less dependent on cohort-specific landmark noise. | `rho=0.05`, otherwise the V8 AdamW schedule |
| BLV9-002 | ASAM optimizer | [ASAM, ICML 2021](https://proceedings.mlr.press/v139/kwon21b.html) | Landmark and dense branches have different parameter scales; scale-adaptive sharpness is a distinct, relevant flatness test. | `rho=0.5`, `eta=0.01` |
| BLV9-003 | stochastic weight averaging | [SWA, UAI 2018](https://auai.org/uai2018/proceedings/papers/313.pdf) | Averaging late shared-trunk solutions tests whether one broad basin transfers better than the final iterate. | equal average of epochs 11-20 |
| BLV9-004 | R-Drop consistency | [R-Drop, NeurIPS 2021](https://proceedings.neurips.cc/paper_files/paper/2021/hash/5a66b9200f29ac3fa0ae244cc2a51b39-Abstract.html) | Two dropout views of the same motor exam should give the same probability. | symmetric Bernoulli KL, weight `0.6` |
| BLV9-005 | modality dropout | [ModDrop, TPAMI 2016](https://arxiv.org/abs/1501.00102) | Mayo inference may lose dense landmarks while retaining 110D clinical geometry; training must not collapse when one representation is absent. | drop available dense evidence for 20% of training participants |
| BLV9-006 | action dropout consistency | [ModDrop, TPAMI 2016](https://arxiv.org/abs/1501.00102) | A patient may incompletely perform one prompted action; the decision should remain stable from the remaining valid script. | remove one valid nonsole action for 20% of training participants and enforce probability consistency |
| BLV9-007 | cross-view VICReg | [VICReg, ICLR 2022](https://openreview.net/forum?id=xm6YD62D1Ub) | 110D clinical geometry and dense trajectory evidence describe the same facial movement and should share noncollapsed motor factors. | 32D projectors; canonical invariance/variance/covariance weights `25/25/1` |
| BLV9-008 | cross-view Barlow Twins | [Barlow Twins, ICML 2021](https://proceedings.mlr.press/v139/zbontar21a.html) | Redundancy reduction can align clinical and dense motor views without treating heterogeneous diseases as one contrastive class. | 32D projectors; off-diagonal weight `0.005` |
| BLV9-009 | masked 110D reconstruction | [Masked Autoencoders, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.html) | Reconstructing held-out clinical feature groups encourages the trunk to retain distributed eye, brow, and oral geometry rather than a few shortcuts. | deterministic 25% feature-group mask; masked MSE only |
| BLV9-010 | masked action-token reconstruction | [Masked Autoencoders, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.html) | The scripted exam is a correlated set of muscle responses; one action should be inferable from the other observed actions. | mask one valid nonsole action; reconstruct its detached pre-context shared token |
| BLV9-011 | clinical-to-dense reconstruction | [MDL-CW, CVPR 2016](https://openaccess.thecvf.com/content_cvpr_2016/html/Rastegar_MDL-CW_A_Multimodal_CVPR_2016_paper.html) | When both views exist, 110D clinical features should preserve enough motion information to predict the dense regional token; the decoder is removed at inference. | 64D clinical token predicts detached dense token; supported actions only |
| BLV9-012 | focal classification loss | [Focal Loss, ICCV 2017](https://openaccess.thecvf.com/content_iccv_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html) | Source-balanced weighting does not distinguish easy controls from hard clinically ambiguous participants. | `gamma=2`, no alpha reweighting |
| BLV9-013 | LDAM margin loss | [LDAM, NeurIPS 2019](https://proceedings.neurips.cc/paper_files/paper/2019/hash/621461af90cadfdaf0e8d4cc25129f91-Abstract.html) | NeuroFace and MEEI have few controls; a fold-local class-count margin directly tests minority-control separation. | source-local class counts, `max_m=0.5`, scale `30` |
| BLV9-014 | pairwise AUROC loss | [Deep AUC Maximization, ICML 2020](https://proceedings.mlr.press/v119/guo20f.html) | AUROC is a locked endpoint; source-local positive-negative ranking may improve it without post-hoc threshold selection. | BCE plus `0.25` mean pairwise logistic ranking loss |
| BLV9-015 | high-specificity partial-AUC loss | [Two-way pAUC, ICML 2021](https://proceedings.mlr.press/v139/yang21k.html) | The requested operational failure is false-positive controls; training focuses ranking on the hardest control tail while retaining sensitivity. | BCE plus `0.25` pairwise loss on top 20% training-fold negatives |
| BLV9-016 | proper-score composite | [Brier score decomposition](https://doi.org/10.1175/1520-0493(1983)111%3C1089:TVOTBS%3E2.0.CO;2) | Mayo will initially be consumed as a confidence score; a proper probabilistic loss tests calibration without outcome-selected thresholds. | BCE plus `0.25` Brier loss |
| BLV9-017 | progressive layered extraction | [PLE, RecSys 2020](https://doi.org/10.1145/3383313.3412236) | Related but nonidentical disease endpoints may need progressively separated low-rank experts while retaining a universal shared expert path. | two shared rank-8 experts plus one rank-4 endpoint expert; task-specific fraction `<10%` |
| BLV9-018 | cross-stitch endpoint streams | [Cross-Stitch Networks, CVPR 2016](https://openaccess.thecvf.com/content_cvpr_2016/html/Misra_Cross-Stitch_Networks_for_CVPR_2016_paper.html) | Learned mixing tests how much post-shared motor evidence each endpoint can borrow without creating three independent encoders. | rank-8 endpoint streams; stitch initialized 0.9 self/0.1 shared; task-specific fraction `<10%` |
| BLV9-019 | action-conditioned FiLM | [FiLM, AAAI 2018](https://ojs.aaai.org/index.php/AAAI/article/view/11671) | Eye closure, brow elevation, smile, and lip pucker express different muscle groups; action identity may modulate a shared encoder without revealing dataset identity. | action-code affine modulation of shared tokens; source code forbidden |
| BLV9-020 | anatomy-action graph | [Dynamic Probabilistic Graph Convolution for AU intensity, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Song_Dynamic_Probabilistic_Graph_Convolution_for_Facial_Action_Unit_Intensity_Estimation_CVPR_2021_paper.html) | Fixed eye, brow, oral, and free-response relations allow local muscle evidence to interact through one source-blind graph rather than an arbitrary fully connected residual. | one 64D shared graph-attention layer over the frozen action ontology |

## Scientific Exclusions

- Do not count the already tested MMoE, deep ensemble, anatomical residual,
  kinematic auxiliary, PCGrad, CORAL, GroupDRO, TCN, BiGRU, or Transformer
  screens again.
- Do not use DANN: source identity confounds pathology, action script, and
  endpoint, so removing source information may remove real clinical signal.
- Do not use supervised contrastive positive-class collapse: the positive class
  joins unilateral palsy, ALS, and post-stroke impairment.
- Do not use arbitrary landmark jitter, temporal warping, or MixUp: each can
  synthesize medically implausible motor phenotypes.
- True flip/reread remains the existing bilateral canonicalization only.  It is
  not a new model and never substitutes left/right clinical evidence.

## Frozen Evaluation

1. Exact comparator: `RSR8-001`; its code and deployment artifacts are read-only.
2. Data: PalsyNet development 38, NeuroFace 36, MEEI 56; participants remain
   disjoint across six source/label-stratified folds.
3. Every new model: epochs 20, seeds `(0, 1, 2)`, all six within-source folds,
   and all three leave-one-source-out fits.  Threshold remains `0.5`.
4. Every fold fits its scaler, loss statistics, class counts, masks, and
   auxiliary targets from training participants only.
5. Report per source: accuracy, specificity, sensitivity, balanced accuracy,
   AUROC, and Brier score.  Report seed dispersion and the prespecified
   mean-probability three-seed estimator.
6. Leave-one-source-out uses only the universal head and is descriptive; the
   three clinical endpoints are not identical.
7. Promotion requires every source to reach accuracy `>=0.90`, specificity
   `>=0.80`, sensitivity `>=0.85`, and AUROC `>=0.92`; no source accuracy or
   AUROC may regress by more than `0.01` from V8, and worst-source specificity
   must strictly improve.  A best seed cannot promote a model.
8. If no candidate passes, V8 remains canonical.  No combination search follows
   this 20-model screen, preventing another outcome-driven combinatorial sweep.

## Implementation Tasks

### Task 1: Freeze and test the registry

**Files:**
- Create: `src/models/broad_literature_candidate_registry_v9.py`
- Create: `tests/test_broad_literature_candidate_registry_v9.py`

Write failing tests requiring exactly one comparator and 20 unique mechanisms,
closed IDs/settings, primary-paper URLs, medical rationales, no forbidden
repeats, and no mutable configuration.  Implement the minimum frozen registry.

### Task 2: Implement and unit-test the shared architecture mechanisms

**Files:**
- Create: `src/models/broad_literature_shared_router_v9.py`
- Create: `tests/test_broad_literature_shared_router_v9.py`

Start from the exact RSR8-001 path.  Add only the PLE, cross-stitch, FiLM, and
anatomy-action graph mechanisms.  Tests must prove shape/finite contracts,
source-blind action conditioning, nonzero gradients from every source into the
same trunk, task-specific fraction below 10%, deterministic inference, and an
observable output change for each mechanism.

### Task 3: Implement and unit-test the 16 training mechanisms

**Files:**
- Create: `src/training/broad_literature_objectives_v9.py`
- Create: `tests/test_broad_literature_objectives_v9.py`

Tests must prove the exact SAM/ASAM perturbation, SWA averaging, symmetric
R-Drop KL, bounded ModDrop/action masks, noncollapsed VICReg/Barlow terms,
masked-only reconstruction, supported-only cross-view reconstruction, and
closed formulas for focal, LDAM, AUC, pAUC, and Brier objectives.  Fold-local
statistics and deterministic masks are mandatory.

### Task 4: Add one closed evaluation loop

**Files:**
- Create: `src/evaluation/broad_literature_shared_search_v9.py`
- Create: `tests/test_broad_literature_shared_search_v9.py`

The evaluator must execute the comparator and all 20 candidates with identical
splits and expose aggregate OOF probabilities only.  Tests must prove exact
coverage, one-mechanism activation, scaler/statistic isolation, shared-gradient
coverage, LOSO source exclusion, and strict promotion logic.

### Task 5: Add the H200 runner and fail-closed report

**Files:**
- Create: `scripts/run_broad_literature_shared_search_v9.py`
- Create: `tests/test_run_broad_literature_shared_search_v9.py`

The runner must require CUDA H200, immutable cache/registry commitments, exact
21-candidate/three-seed coverage, deterministic settings, aggregate-only public
artifacts, implementation hashes, and audit values of zero protected/Mayo reads
and predictions.  It must write to a fresh release directory atomically.

### Task 6: Run H200, document the decision, and verify the freeze

**Files:**
- Create: `docs/results/broad_literature_shared_v9.md`
- Create: `docs/results/artifacts/broad_literature_shared_v9/report.json`
- Create: `tests/test_broad_literature_shared_v9_release.py`

Run all 20 models, not just smoke tests.  Document every mechanism and result,
including failures, without claiming Mayo or clinical validation.  Verify the
V8 registry/deployment SHA-256 values are unchanged.  Run all targeted tests,
`py_compile`, `git diff --check`, secret/private-path scans, and an independent
release consistency audit before any completion claim.
