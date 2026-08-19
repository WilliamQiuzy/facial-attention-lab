"""Closed candidate registry for the shared normal-manifold router v4."""
from __future__ import annotations

from dataclasses import dataclass


DYNAMIC_3D = "https://pubmed.ncbi.nlm.nih.gov/30534499/"
DYNAMIC_VELOCITY = "https://pubmed.ncbi.nlm.nih.gov/27480299/"
ALS_KINEMATICS = "https://pubmed.ncbi.nlm.nih.gov/29800359/"
ALS_VIDEO = "https://pubmed.ncbi.nlm.nih.gov/36367528/"


NORMAL_MANIFOLD_RATIONALE = {
    "healthy_anchor": {
        "phenomenon": (
            "Healthy controls provide a common cross-source facial-motor reference, "
            "while affected facial-palsy and neurological phenotypes may differ."
        ),
        "evidence": (DYNAMIC_3D, DYNAMIC_VELOCITY, ALS_KINEMATICS),
        "valid_labels": ("affected_vs_control", "facial_motor_abnormality"),
        "contraindication": (
            "Compact controls only; never collapse affected diseases into one centroid "
            "or interpret distance as an HB grade."
        ),
    },
    "universal_normality": {
        "phenomenon": (
            "Each cohort contrasts a facial-motor disorder endpoint with healthy "
            "controls even though its affected disease semantics differ."
        ),
        "evidence": (DYNAMIC_3D, ALS_VIDEO),
        "valid_labels": ("affected_vs_control",),
        "contraindication": (
            "The shared normality logit is not a disease diagnosis and endpoint heads "
            "remain necessary."
        ),
    },
}


@dataclass(frozen=True)
class NormalManifoldCandidateV4:
    candidate_id: str
    normal_weight: float
    universal_blend: float


def candidate_registry_v4() -> tuple[NormalManifoldCandidateV4, ...]:
    candidates = []
    index = 0
    for normal_weight in (0.0, 0.05, 0.2):
        for universal_blend in (0.25, 0.5):
            candidates.append(NormalManifoldCandidateV4(
                candidate_id=f"NMR4-{index:03d}",
                normal_weight=normal_weight,
                universal_blend=universal_blend,
            ))
            index += 1
    if len(candidates) != 6:
        raise AssertionError("the normal-manifold registry drifted")
    return tuple(candidates)


__all__ = [
    "NORMAL_MANIFOLD_RATIONALE",
    "NormalManifoldCandidateV4",
    "candidate_registry_v4",
]
