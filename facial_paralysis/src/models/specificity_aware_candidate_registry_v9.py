"""Frozen medically bounded candidates for specificity-aware shared V9."""
from __future__ import annotations

from dataclasses import dataclass


SUNNYBROOK = "https://pubmed.ncbi.nlm.nih.gov/8649870/"
EFACE = "https://pubmed.ncbi.nlm.nih.gov/26218397/"
DYNAMIC_3D = "https://pubmed.ncbi.nlm.nih.gov/30534499/"
ALS_KINEMATICS = "https://pubmed.ncbi.nlm.nih.gov/29800359/"


COMPONENT_RATIONALES_V9 = {
    "healthy_mode": {
        "off": {
            "phenomenon": "Retain the exact V8 shared representation as a comparator.",
            "evidence": (SUNNYBROOK, EFACE),
            "contraindication": "Absence of a healthy reference is not evidence of normal function.",
        },
        "compact": {
            "phenomenon": (
                "Technically adequate healthy facial motor responses provide a shared "
                "reference manifold across acquisition protocols."
            ),
            "evidence": (SUNNYBROOK, EFACE, ALS_KINEMATICS),
            "contraindication": (
                "The learned reference is a binary control phenotype, not a normal "
                "House-Brackmann grade or population normative range."
            ),
        },
        "compact_margin": {
            "phenomenon": (
                "Affected facial motor responses should remain separated from a compact "
                "healthy reference without forcing different diseases to share a centroid."
            ),
            "evidence": (SUNNYBROOK, DYNAMIC_3D, ALS_KINEMATICS),
            "contraindication": (
                "Distance from the reference is not calibrated severity and cannot rank "
                "Bell palsy, stroke, and ALS on one clinical scale."
            ),
        },
    },
    "control_cost": {
        1.0: {
            "phenomenon": "Use source-and-class-balanced binary evidence without extra cost.",
            "evidence": (SUNNYBROOK,),
            "contraindication": "Equal class loss does not encode a clinical deployment utility.",
        },
        1.5: {
            "phenomenon": (
                "False-positive healthy calls create avoidable clinical review burden, so "
                "controls receive a bounded additional training cost."
            ),
            "evidence": (SUNNYBROOK, EFACE),
            "contraindication": (
                "The cost is invalid if sensitivity falls below the frozen safety floor."
            ),
        },
    },
    "universal_blend": {
        0.25: {
            "phenomenon": "Preserve the V8 balance between shared normality and protocol endpoint.",
            "evidence": (SUNNYBROOK, EFACE, ALS_KINEMATICS),
            "contraindication": "The blend is predictive and is not a clinical subscore weight.",
        },
        0.5: {
            "phenomenon": (
                "Increase the contribution of source-blind motor impairment while retaining "
                "a protocol-specific endpoint residual."
            ),
            "evidence": (DYNAMIC_3D, ALS_KINEMATICS),
            "contraindication": (
                "A universal impairment signal cannot replace disease-specific interpretation."
            ),
        },
    },
    "control_alignment_weight": {
        0.0: {
            "phenomenon": "Do not impose cross-protocol control alignment.",
            "evidence": (SUNNYBROOK,),
            "contraindication": "Unaligned controls may retain acquisition or script nuisance.",
        },
        0.02: {
            "phenomenon": (
                "Align only healthy-control centroids across protocols so the shared encoder "
                "focuses on motor phenotype rather than acquisition source."
            ),
            "evidence": (SUNNYBROOK, EFACE, ALS_KINEMATICS),
            "contraindication": (
                "Affected centroids must not be aligned because Bell palsy, stroke, and ALS "
                "have different pathophysiology."
            ),
        },
    },
}


@dataclass(frozen=True)
class SpecificityCandidateV9:
    candidate_id: str
    healthy_mode: str
    control_cost: float
    universal_blend: float
    control_alignment_weight: float


def candidate_registry_v9() -> tuple[SpecificityCandidateV9, ...]:
    rows = []
    index = 0
    for healthy_mode in ("off", "compact", "compact_margin"):
        for control_cost in (1.0, 1.5):
            for universal_blend in (0.25, 0.5):
                for control_alignment_weight in (0.0, 0.02):
                    rows.append(SpecificityCandidateV9(
                        candidate_id=f"SSR9-{index:03d}",
                        healthy_mode=healthy_mode,
                        control_cost=control_cost,
                        universal_blend=universal_blend,
                        control_alignment_weight=control_alignment_weight,
                    ))
                    index += 1
    if len(rows) != 24 or len(set(rows)) != 24:
        raise AssertionError("the specificity-aware V9 registry drifted")
    return tuple(rows)


__all__ = [
    "COMPONENT_RATIONALES_V9",
    "SpecificityCandidateV9",
    "candidate_registry_v9",
]
