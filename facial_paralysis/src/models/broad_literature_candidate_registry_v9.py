"""Frozen registry for the broad, paper-grounded shared V9 screen."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Union


SettingValue = Union[float, int, str]


@dataclass(frozen=True)
class BroadLiteratureCandidateV9:
    candidate_id: str
    mechanism: str
    family: str
    paper_title: str
    paper_url: str
    medical_rationale: str
    settings: tuple[tuple[str, SettingValue], ...]
    inference_change: str
    combinable: bool = False


def _candidate(
    index: int,
    mechanism: str,
    family: str,
    paper_title: str,
    paper_url: str,
    medical_rationale: str,
    settings: tuple[tuple[str, SettingValue], ...],
    inference_change: str = "training_only",
) -> BroadLiteratureCandidateV9:
    return BroadLiteratureCandidateV9(
        candidate_id=f"BLV9-{index:03d}",
        mechanism=mechanism,
        family=family,
        paper_title=paper_title,
        paper_url=paper_url,
        medical_rationale=medical_rationale,
        settings=settings,
        inference_change=inference_change,
    )


def candidate_registry_v9() -> tuple[BroadLiteratureCandidateV9, ...]:
    """Return one comparator and exactly twenty mechanism-distinct candidates."""
    return (
        _candidate(
            0, "exact_v8_comparator", "comparator",
            "Deterministic RSR8-001 comparator", "https://github.com/WilliamQiuzy/mayo-facial-analysis",
            "Exact frozen V8 shared clinical motor encoder and endpoint router.",
            (), "none",
        ),
        _candidate(
            1, "sam", "optimization", "Sharpness-Aware Minimization",
            "https://openreview.net/forum?id=6Tm1mposlrM",
            "A flatter shared solution may be less dependent on cohort-specific landmark noise.",
            (("rho", 0.05),),
        ),
        _candidate(
            2, "asam", "optimization", "ASAM: Adaptive Sharpness-Aware Minimization",
            "https://proceedings.mlr.press/v139/kwon21b.html",
            "Scale-adaptive sharpness addresses unequal parameter scales in clinical and dense branches.",
            (("rho", 0.5), ("eta", 0.01)),
        ),
        _candidate(
            3, "swa", "optimization", "Averaging Weights Leads to Wider Optima",
            "https://auai.org/uai2018/proceedings/papers/313.pdf",
            "Averaging late shared-trunk solutions tests whether one broad basin transfers better.",
            (("average_start_epoch", 11), ("average_end_epoch", 20)),
        ),
        _candidate(
            4, "r_drop", "optimization", "R-Drop: Regularized Dropout",
            "https://proceedings.neurips.cc/paper_files/paper/2021/hash/5a66b9200f29ac3fa0ae244cc2a51b39-Abstract.html",
            "Two dropout views of the same motor examination should yield one decision.",
            (("symmetric_kl_weight", 0.6),),
        ),
        _candidate(
            5, "modality_dropout", "missing_evidence_robustness",
            "ModDrop: Adaptive Multi-Modal Gesture Recognition",
            "https://arxiv.org/abs/1501.00102",
            "Mayo inference may lose dense landmarks while retaining 110D clinical geometry.",
            (("dense_drop_probability", 0.2),),
        ),
        _candidate(
            6, "action_dropout_consistency", "missing_evidence_robustness",
            "ModDrop: Adaptive Multi-Modal Gesture Recognition",
            "https://arxiv.org/abs/1501.00102",
            "An incompletely performed prompted action should not erase evidence from the remaining script.",
            (("action_drop_probability", 0.2), ("consistency_weight", 0.25)),
        ),
        _candidate(
            7, "cross_view_vicreg", "self_supervision",
            "VICReg", "https://openreview.net/forum?id=xm6YD62D1Ub",
            "Clinical geometry and dense trajectory views should share noncollapsed motor factors.",
            (("projector_dim", 32), ("invariance_weight", 25.0),
             ("variance_weight", 25.0), ("covariance_weight", 1.0)),
        ),
        _candidate(
            8, "cross_view_barlow_twins", "self_supervision",
            "Barlow Twins", "https://proceedings.mlr.press/v139/zbontar21a.html",
            "Redundancy reduction can align two motor views without collapsing heterogeneous diseases.",
            (("projector_dim", 32), ("off_diagonal_weight", 0.005)),
        ),
        _candidate(
            9, "masked_clinical_reconstruction", "self_supervision",
            "Masked Autoencoders Are Scalable Vision Learners",
            "https://openaccess.thecvf.com/content/CVPR2022/html/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.html",
            "Reconstructing hidden clinical groups encourages distributed eye, brow, and oral geometry.",
            (("mask_fraction", 0.25),),
        ),
        _candidate(
            10, "masked_action_reconstruction", "self_supervision",
            "Masked Autoencoders Are Scalable Vision Learners",
            "https://openaccess.thecvf.com/content/CVPR2022/html/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.html",
            "One scripted muscle response should be reconstructable from the remaining examination.",
            (("masked_actions_per_participant", 1),),
        ),
        _candidate(
            11, "clinical_to_dense_reconstruction", "self_supervision",
            "MDL-CW: A Multimodal Deep Learning Framework",
            "https://openaccess.thecvf.com/content_cvpr_2016/html/Rastegar_MDL-CW_A_Multimodal_CVPR_2016_paper.html",
            "Clinical geometry should retain enough motion information to predict supported dense evidence.",
            (("reconstruction_weight", 0.25),),
        ),
        _candidate(
            12, "focal_loss", "clinical_objective", "Focal Loss",
            "https://openaccess.thecvf.com/content_iccv_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html",
            "Focus learning on clinically ambiguous participants rather than already separated examples.",
            (("gamma", 2.0),),
        ),
        _candidate(
            13, "ldam_loss", "clinical_objective", "Learning Imbalanced Datasets with LDAM",
            "https://proceedings.neurips.cc/paper_files/paper/2019/hash/621461af90cadfdaf0e8d4cc25129f91-Abstract.html",
            "Fold-local margins directly test minority-control separation in NeuroFace and MEEI.",
            (("maximum_margin", 0.5), ("logit_scale", 30.0)),
        ),
        _candidate(
            14, "pairwise_auc_loss", "clinical_objective", "Deep AUC Maximization",
            "https://proceedings.mlr.press/v119/guo20f.html",
            "Source-local positive-control ranking directly targets the locked AUROC endpoint.",
            (("ranking_weight", 0.25),),
        ),
        _candidate(
            15, "high_specificity_partial_auc_loss", "clinical_objective",
            "Two-way Partial AUC Optimization", "https://proceedings.mlr.press/v139/yang21k.html",
            "The hardest control tail is the clinically relevant source of false-positive deployment calls.",
            (("negative_tail_fraction", 0.2), ("ranking_weight", 0.25)),
        ),
        _candidate(
            16, "brier_composite_loss", "clinical_objective",
            "The Verification of Probability Forecasts",
            "https://doi.org/10.1175/1520-0493(1983)111%3C1089:TVOTBS%3E2.0.CO;2",
            "Mayo will initially consume confidence scores, so proper probabilistic quality matters.",
            (("brier_weight", 0.25),),
        ),
        _candidate(
            17, "progressive_layered_extraction", "shared_architecture",
            "Progressive Layered Extraction", "https://doi.org/10.1145/3383313.3412236",
            "Related but nonidentical endpoints may need bounded experts while retaining a shared path.",
            (("shared_experts", 2), ("shared_rank", 8), ("endpoint_rank", 4)),
            "architecture",
        ),
        _candidate(
            18, "cross_stitch_endpoint_streams", "shared_architecture",
            "Cross-Stitch Networks", "https://openaccess.thecvf.com/content_cvpr_2016/html/Misra_Cross-Stitch_Networks_for_CVPR_2016_paper.html",
            "Learned post-trunk mixing can borrow evidence without creating three independent encoders.",
            (("stream_rank", 8), ("self_initialization", 0.9),
             ("shared_initialization", 0.1)), "architecture",
        ),
        _candidate(
            19, "action_conditioned_film", "shared_architecture", "FiLM",
            "https://ojs.aaai.org/index.php/AAAI/article/view/11671",
            "Eye, brow, smile, and oral actions may condition one source-blind motor encoder differently.",
            (("film_scale", 0.1),), "architecture",
        ),
        _candidate(
            20, "anatomy_action_graph", "shared_architecture",
            "Dynamic Probabilistic Graph Convolution for Facial Action Unit Intensity Estimation",
            "https://openaccess.thecvf.com/content/CVPR2021/html/Song_Dynamic_Probabilistic_Graph_Convolution_for_Facial_Action_Unit_Intensity_Estimation_CVPR_2021_paper.html",
            "Eye, brow, oral, and free-response evidence should interact through a fixed clinical ontology.",
            (("graph_layers", 1), ("graph_heads", 4)), "architecture",
        ),
    )


__all__ = ["BroadLiteratureCandidateV9", "candidate_registry_v9"]
