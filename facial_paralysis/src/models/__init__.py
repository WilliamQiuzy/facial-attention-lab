"""Default model imports resolve exclusively to Universal Clinical Router v4.

Historical models remain available from their explicit modules for archived
experiment reproduction; they are intentionally absent from this package API.
"""
from src.models.current import (
    CURRENT_MODEL_ARTIFACT_SHA256,
    CURRENT_MODEL_NAME,
    CURRENT_MODEL_SCHEMA_VERSION,
    SCRIPTED_COMMON_TASKS,
    TIMING_AUTHORITIES,
    UPPER_PROMPT_TASKS,
    cue_aligned_upper_probability,
    evidence_profile,
    linear_head_probability,
    load_current_artifact,
    median_low_confidence_gate,
    scripted_multimechanism_probability,
    serialized_head_probability,
)


__all__ = (
    "CURRENT_MODEL_ARTIFACT_SHA256",
    "CURRENT_MODEL_NAME",
    "CURRENT_MODEL_SCHEMA_VERSION",
    "SCRIPTED_COMMON_TASKS",
    "TIMING_AUTHORITIES",
    "UPPER_PROMPT_TASKS",
    "cue_aligned_upper_probability",
    "evidence_profile",
    "linear_head_probability",
    "load_current_artifact",
    "median_low_confidence_gate",
    "scripted_multimechanism_probability",
    "serialized_head_probability",
)
