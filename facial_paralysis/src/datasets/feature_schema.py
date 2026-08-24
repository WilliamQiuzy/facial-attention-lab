"""Torch-free MediaPipe feature column and side-convention registry."""
from __future__ import annotations

from ..preprocessing.clinical_landmarks import (
    CLINICAL_LANDMARK_NAMES,
    CLINICAL_SIDE_CONVENTION,
)


_MEDIAPIPE_BLENDSHAPE_NAMES: tuple[str, ...] = (
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight", "cheekPuff", "cheekSquintLeft",
    "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft",
    "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft",
    "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft",
    "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft",
    "jawOpen", "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight",
    "mouthFrownLeft", "mouthFrownRight", "mouthFunnel", "mouthLeft",
    "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft",
    "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower",
    "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft",
    "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight",
)
_MEDIAPIPE_ASYMMETRY_NAMES = tuple(
    f"delta_left_minus_right_{name[:-4]}"
    for name in _MEDIAPIPE_BLENDSHAPE_NAMES
    if name.endswith("Left") and name[:-4] + "Right" in _MEDIAPIPE_BLENDSHAPE_NAMES
)
_MEDIAPIPE_BASE_NAMES = _MEDIAPIPE_BLENDSHAPE_NAMES + _MEDIAPIPE_ASYMMETRY_NAMES
_LEGACY_GEOMETRY_NAMES = (
    "ear_right", "ear_left", "ear_asym", "brow_asym", "mouthcorner_asym",
)

# A schema version is a complete column-order contract, not merely a dimension.
MP_FEATURE_NAMES_BY_SCHEMA: dict[str, tuple[str, ...]] = {
    "mediapipe_bs_lr_v1": _MEDIAPIPE_BASE_NAMES,
    "mediapipe_bs_lr_v1+legacy_geometry5_v1": (
        _MEDIAPIPE_BASE_NAMES + _LEGACY_GEOMETRY_NAMES
    ),
    "mediapipe_bs_lr_v1+clinical23_v2": (
        _MEDIAPIPE_BASE_NAMES + CLINICAL_LANDMARK_NAMES
    ),
}
MP_SIDE_CONVENTION_BY_SCHEMA: dict[str, str] = {
    "mediapipe_bs_lr_v1": "mediapipe_left_right_labels_capture_mirror_required",
    "mediapipe_bs_lr_v1+legacy_geometry5_v1": (
        "mediapipe_labels_plus_legacy_mesh_topology_capture_mirror_required"
    ),
    "mediapipe_bs_lr_v1+clinical23_v2": CLINICAL_SIDE_CONVENTION,
}


__all__ = ["MP_FEATURE_NAMES_BY_SCHEMA", "MP_SIDE_CONVENTION_BY_SCHEMA"]
