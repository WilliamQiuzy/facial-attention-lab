"""Source-neutral clinical geometry shared across facial landmark topologies.

``semantic23_v1`` deliberately describes measurements rather than detector
indices.  A 23-element vector is *not* assumed to be compatible merely because
its length matches: callers must select an explicit source-schema adapter.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .clinical_landmarks import CLINICAL_LANDMARK_NAMES


SEMANTIC23_SCHEMA = "semantic23_v1"

SEMANTIC23_FEATURE_NAMES: tuple[str, ...] = (
    "fissure_h_side_a", "fissure_h_side_b", "fissure_h_absdiff",
    "fissure_h_side_a_minus_side_b",
    "fissure_w_side_a", "fissure_w_side_b", "fissure_w_absdiff",
    "eye_measure_side_a", "eye_measure_side_b", "eye_measure_absdiff",
    "brow_h_side_a", "brow_h_side_b", "brow_h_absdiff",
    "brow_h_side_a_minus_side_b",
    "corner_y_side_a", "corner_y_side_b", "corner_y_absdiff",
    "corner_y_side_a_minus_side_b",
    "corner_x_side_a", "corner_x_side_b", "corner_x_absdiff",
    "mouth_width", "mouth_open",
)


@dataclass(frozen=True)
class SemanticFeatureDefinition:
    """Machine-readable definition of one position in ``semantic23_v1``."""

    name: str
    unit: str
    sign: str
    definition: str


def _linear(name: str, sign: str, definition: str) -> SemanticFeatureDefinition:
    return SemanticFeatureDefinition(name, "interocular_distance", sign, definition)


def _area(name: str, definition: str) -> SemanticFeatureDefinition:
    return SemanticFeatureDefinition(
        name, "interocular_distance_squared", "nonnegative", definition
    )


SEMANTIC23_DEFINITIONS: tuple[SemanticFeatureDefinition, ...] = (
    _linear("fissure_h_side_a", "nonnegative",
            "absolute mean lower-lid minus mean upper-lid vertical distance on side A"),
    _linear("fissure_h_side_b", "nonnegative",
            "absolute mean lower-lid minus mean upper-lid vertical distance on side B"),
    _linear("fissure_h_absdiff", "nonnegative",
            "absolute difference between side-A and side-B fissure heights"),
    _linear("fissure_h_side_a_minus_side_b", "side_a_minus_side_b",
            "side-A fissure height minus side-B fissure height"),
    _linear("fissure_w_side_a", "nonnegative",
            "absolute horizontal canthus-to-canthus distance on side A"),
    _linear("fissure_w_side_b", "nonnegative",
            "absolute horizontal canthus-to-canthus distance on side B"),
    _linear("fissure_w_absdiff", "nonnegative",
            "absolute difference between side-A and side-B fissure widths"),
    _area("eye_measure_side_a",
          "side-A fissure height multiplied by side-A fissure width; not polygon area"),
    _area("eye_measure_side_b",
          "side-B fissure height multiplied by side-B fissure width; not polygon area"),
    _area("eye_measure_absdiff",
          "absolute difference between side-A and side-B height-times-width eye measures"),
    _linear("brow_h_side_a", "nonnegative",
            "absolute vertical distance from side-A eye-ring mean to brow mean"),
    _linear("brow_h_side_b", "nonnegative",
            "absolute vertical distance from side-B eye-ring mean to brow mean"),
    _linear("brow_h_absdiff", "nonnegative",
            "absolute difference between side-A and side-B brow heights"),
    _linear("brow_h_side_a_minus_side_b", "side_a_minus_side_b",
            "side-A brow height minus side-B brow height"),
    _linear("corner_y_side_a", "eye_line_relative",
            "side-A oral commissure y coordinate minus the mean eye-line y coordinate"),
    _linear("corner_y_side_b", "eye_line_relative",
            "side-B oral commissure y coordinate minus the mean eye-line y coordinate"),
    _linear("corner_y_absdiff", "nonnegative",
            "absolute difference between side-A and side-B commissure y positions"),
    _linear("corner_y_side_a_minus_side_b", "side_a_minus_side_b",
            "side-A commissure y position minus side-B commissure y position"),
    _linear("corner_x_side_a", "nonnegative",
            "absolute horizontal distance from side-A commissure to the facial midline"),
    _linear("corner_x_side_b", "nonnegative",
            "absolute horizontal distance from side-B commissure to the facial midline"),
    _linear("corner_x_absdiff", "nonnegative",
            "absolute difference between side-A and side-B commissure-to-midline distances"),
    _linear("mouth_width", "nonnegative",
            "absolute horizontal distance between the two oral commissures"),
    _linear("mouth_open", "nonnegative",
            "absolute vertical distance between central inner upper- and lower-lip points"),
)


CLINICAL23_V2_SOURCE_NAMES: tuple[str, ...] = (
    "fissure_h_mesh33", "fissure_h_mesh263", "fissure_h_absdiff",
    "fissure_h_mesh33_minus_mesh263",
    "fissure_w_mesh33", "fissure_w_mesh263", "fissure_w_absdiff",
    "eye_area_mesh33", "eye_area_mesh263", "eye_area_absdiff",
    "brow_h_mesh33", "brow_h_mesh263", "brow_h_absdiff",
    "brow_h_mesh33_minus_mesh263",
    "corner_y_mesh61", "corner_y_mesh291", "corner_y_absdiff",
    "corner_y_mesh61_minus_mesh291",
    "corner_x_mesh61", "corner_x_mesh291", "commissure_x_absdiff",
    "mouth_width", "mouth_open",
)

# This tuple is intentionally explicit even though V2 currently has the same
# numeric order.  It prevents a future 23-dimensional source from being accepted
# by shape alone and makes schema drift visible in review.
CLINICAL23_V2_SOURCE_FOR_SEMANTIC_TARGET: tuple[str, ...] = (
    "fissure_h_mesh33", "fissure_h_mesh263", "fissure_h_absdiff",
    "fissure_h_mesh33_minus_mesh263",
    "fissure_w_mesh33", "fissure_w_mesh263", "fissure_w_absdiff",
    "eye_area_mesh33", "eye_area_mesh263", "eye_area_absdiff",
    "brow_h_mesh33", "brow_h_mesh263", "brow_h_absdiff",
    "brow_h_mesh33_minus_mesh263",
    "corner_y_mesh61", "corner_y_mesh291", "corner_y_absdiff",
    "corner_y_mesh61_minus_mesh291",
    "corner_x_mesh61", "corner_x_mesh291", "commissure_x_absdiff",
    "mouth_width", "mouth_open",
)
CLINICAL23_V2_TO_SEMANTIC23_INDEX: tuple[int, ...] = tuple(
    CLINICAL23_V2_SOURCE_NAMES.index(source_name)
    for source_name in CLINICAL23_V2_SOURCE_FOR_SEMANTIC_TARGET
)

CLINICAL23_V2_ADAPTER_METADATA: dict[str, object] = {
    "adapter_name": "clinical23_v2_to_semantic23_v1",
    "source_schema": "clinical23_v2",
    "target_schema": SEMANTIC23_SCHEMA,
    "source_topology": "mediapipe_facemesh_478",
    "source_side_a": "mesh33 capture side",
    "source_side_b": "mesh263 capture side",
    "patient_side_status": "unknown_until_capture_mirror_provenance_is_known",
    "scale_normalization": "interocular_distance",
    "roll_normalization": "eye_centres_horizontal",
    "eye_measure": "height_times_width",
    "numeric_mapping": "explicit_identity_reorder_for_clinical23_v2_only",
}


def clinical23_v2_to_semantic23(features: np.ndarray) -> np.ndarray:
    """Adapt one or more explicitly identified ``clinical23_v2`` vectors.

    The last dimension must be exactly 23 and every value must be finite.  This
    function does not attempt schema detection; choosing it is the caller's
    declaration that the source contract is ``clinical23_v2``.
    """
    array = np.asarray(features)
    if array.ndim < 1 or array.shape[-1] != len(CLINICAL23_V2_SOURCE_NAMES):
        raise ValueError(
            "clinical23_v2 features must have final dimension 23; "
            f"got shape {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError("clinical23_v2 features contain NaN or infinity")
    return np.take(array, CLINICAL23_V2_TO_SEMANTIC23_INDEX, axis=-1).astype(
        np.float32, copy=True
    )


if tuple(item.name for item in SEMANTIC23_DEFINITIONS) != SEMANTIC23_FEATURE_NAMES:
    raise RuntimeError("semantic23 feature definitions are out of order")
if tuple(CLINICAL_LANDMARK_NAMES) != CLINICAL23_V2_SOURCE_NAMES:
    raise RuntimeError("clinical23_v2 source schema drifted; explicit adapter review required")
if len(CLINICAL23_V2_SOURCE_FOR_SEMANTIC_TARGET) != len(SEMANTIC23_FEATURE_NAMES):
    raise RuntimeError("clinical23_v2 adapter does not define every semantic23 target")


__all__ = [
    "CLINICAL23_V2_ADAPTER_METADATA",
    "CLINICAL23_V2_SOURCE_NAMES",
    "CLINICAL23_V2_SOURCE_FOR_SEMANTIC_TARGET",
    "CLINICAL23_V2_TO_SEMANTIC23_INDEX",
    "SEMANTIC23_DEFINITIONS",
    "SEMANTIC23_FEATURE_NAMES",
    "SEMANTIC23_SCHEMA",
    "SemanticFeatureDefinition",
    "clinical23_v2_to_semantic23",
]
