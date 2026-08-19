"""Frozen, medically gated candidate registry for shared encoder v2."""
from __future__ import annotations

from dataclasses import dataclass


SUNNYBROOK = "https://pubmed.ncbi.nlm.nih.gov/8649870/"
EFACE = "https://pubmed.ncbi.nlm.nih.gov/26218397/"
DYNAMIC_3D = "https://pubmed.ncbi.nlm.nih.gov/30534499/"
DYNAMIC_ML = "https://pubmed.ncbi.nlm.nih.gov/40333095/"
ANGLE_MAP = "https://pubmed.ncbi.nlm.nih.gov/42072220/"
MEDIAPIPE_CONTOURS = (
    "https://github.com/google-ai-edge/mediapipe/blob/master/"
    "mediapipe/tasks/python/vision/face_landmarker.py"
)


COMPONENT_RATIONALES = {
    "view_mode": {
        "original_only": {
            "phenomenon": "Preserve observed capture laterality without augmentation.",
            "evidence": (SUNNYBROOK, EFACE),
            "valid_labels": ("binary_weakness", "regional_function"),
            "contraindication": (
                "Capture orientation is not an anatomical affected-side label unless "
                "orientation provenance is separately authenticated."
            ),
        },
        "bilateral_invariant": {
            "phenomenon": (
                "Binary facial weakness is independent of whether the affected side "
                "is left or right; retain only commutative bilateral magnitude."
            ),
            "evidence": (DYNAMIC_3D, ANGLE_MAP),
            "valid_labels": ("binary_weakness",),
            "contraindication": (
                "Do not use for affected-side prediction, signed regional scores, or "
                "laterality-specific HB interpretation."
            ),
        },
    },
    "regional_mode": {
        "none": {
            "phenomenon": "Distributed full-face motion may contain disease evidence.",
            "evidence": (ANGLE_MAP,),
            "valid_labels": ("binary_weakness",),
            "contraindication": (
                "A whole-face token is not a region-specific clinical explanation."
            ),
        },
        "all_excursion": {
            "phenomenon": (
                "Voluntary excursion is assessed separately in brow, eye, and oral "
                "regions during facial nerve examination."
            ),
            "evidence": (SUNNYBROOK, EFACE, DYNAMIC_ML, MEDIAPIPE_CONTOURS),
            "valid_labels": ("binary_weakness", "dynamic_function"),
            "contraindication": (
                "Regional excursion does not by itself measure synkinesis or establish "
                "an HB grade."
            ),
        },
        "matched_excursion": {
            "phenomenon": (
                "Prompted brow, eye, and oral actions have anatomically intended "
                "regions of movement."
            ),
            "evidence": (SUNNYBROOK, DYNAMIC_ML, MEDIAPIPE_CONTOURS),
            "valid_labels": ("binary_weakness", "dynamic_function"),
            "contraindication": (
                "Free-video windows without an authenticated prompt must use the "
                "global face rather than an inferred action region."
            ),
        },
        "matched_excursion_velocity": {
            "phenomenon": (
                "Facial paralysis alters both movement magnitude and movement velocity "
                "within the action-relevant region."
            ),
            "evidence": (DYNAMIC_3D, DYNAMIC_ML, MEDIAPIPE_CONTOURS),
            "valid_labels": ("binary_weakness", "dynamic_function"),
            "contraindication": (
                "Velocity is valid only on real timestamps with no cross-window step."
            ),
        },
    },
    "pooling_mode": {
        "meanmax_set": {
            "phenomenon": (
                "Facial grading combines regional/action findings without treating the "
                "recording order as disease severity."
            ),
            "evidence": (SUNNYBROOK, EFACE),
            "valid_labels": ("binary_weakness", "composite_function"),
            "contraindication": (
                "Set pooling cannot be interpreted as temporal progression."
            ),
        },
        "cross_action_transformer": {
            "phenomenon": (
                "A composite facial-function phenotype can depend on relationships "
                "among several standardized voluntary movements."
            ),
            "evidence": (SUNNYBROOK, EFACE, DYNAMIC_ML),
            "valid_labels": ("binary_weakness", "composite_function"),
            "contraindication": (
                "Cross-action attention is not a synkinesis score without explicit "
                "synkinesis labels."
            ),
        },
    },
    "fusion_mode": {
        "masked_concat": {
            "phenomenon": (
                "Static geometry and dynamic excursion are distinct complementary "
                "facial-function domains."
            ),
            "evidence": (SUNNYBROOK, EFACE),
            "valid_labels": ("binary_weakness", "composite_function"),
            "contraindication": (
                "Concatenation weights are predictive parameters, not clinical subscore "
                "weights."
            ),
        },
        "reliability_gate": {
            "phenomenon": (
                "Dense dynamic evidence is absent in PalsyNet but present in scripted "
                "cohorts, so fusion must explicitly respect modality availability."
            ),
            "evidence": (EFACE, DYNAMIC_ML),
            "valid_labels": ("binary_weakness", "missing_modality"),
            "contraindication": (
                "A learned gate measures predictive reliability, not medical importance."
            ),
        },
    },
}


@dataclass(frozen=True)
class SharedCandidateV2:
    candidate_id: str
    view_mode: str
    regional_mode: str
    pooling_mode: str
    fusion_mode: str


def candidate_registry() -> tuple[SharedCandidateV2, ...]:
    candidates = []
    index = 0
    for view_mode in COMPONENT_RATIONALES["view_mode"]:
        for regional_mode in COMPONENT_RATIONALES["regional_mode"]:
            for pooling_mode in COMPONENT_RATIONALES["pooling_mode"]:
                for fusion_mode in COMPONENT_RATIONALES["fusion_mode"]:
                    candidates.append(SharedCandidateV2(
                        candidate_id=f"MSC2-{index:03d}",
                        view_mode=view_mode,
                        regional_mode=regional_mode,
                        pooling_mode=pooling_mode,
                        fusion_mode=fusion_mode,
                    ))
                    index += 1
    if len(candidates) != 32:
        raise AssertionError("the medically gated candidate registry drifted")
    return tuple(candidates)


__all__ = [
    "COMPONENT_RATIONALES",
    "SharedCandidateV2",
    "candidate_registry",
]
