"""Canonical public model surface for facial-paralysis research.

All new development must import from this module. Historical model modules are
retained only for explicit reproducibility work and are not default exports.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.models.universal_clinical_router_v4 import (
    SCRIPTED_COMMON_TASKS,
    TIMING_AUTHORITIES,
    UPPER_PROMPT_TASKS,
    cue_aligned_upper_probability,
    evidence_profile,
    linear_head_probability,
    median_low_confidence_gate,
    scripted_multimechanism_probability,
    serialized_head_probability,
)


CURRENT_MODEL_NAME = "universal_clinical_router_v4"
CURRENT_MODEL_SCHEMA_VERSION = "universal_clinical_router_v4"
CURRENT_MODEL_ARTIFACT_SHA256 = (
    "c8f8c217d508b15bf0d8626b42cead857192ecd738b1fffab94f364c6ed80495"
)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ARTIFACT = (
    _PROJECT_ROOT
    / "docs/results/artifacts/universal_clinical_router_v4/model.json"
)


def load_current_artifact() -> dict[str, object]:
    """Load the exact v4 artifact after verifying its immutable commitment."""
    payload = _ARTIFACT.read_bytes()
    if hashlib.sha256(payload).hexdigest() != CURRENT_MODEL_ARTIFACT_SHA256:
        raise ValueError("current model artifact differs from its frozen SHA-256")
    document = json.loads(payload)
    if (
        type(document) is not dict
        or document.get("schema_version") != CURRENT_MODEL_SCHEMA_VERSION
    ):
        raise ValueError("current model artifact has an unexpected schema")
    return document


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
