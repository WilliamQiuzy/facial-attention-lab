#!/usr/bin/env python3
"""Evaluate the frozen UCR4 selective-decision candidates, aggregate only."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import sys
import tempfile
import zipfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.evaluation.selective_router_v5 import (  # noqa: E402
    CANDIDATE_ORDER,
    COVERAGES,
    EVIDENCE_PROFILES,
    PRIMARY_COVERAGE,
    evaluate_profile,
    select_candidate,
)


_NPZ_FIELDS = {
    "schema_version",
    "evidence_profile",
    "anonymous_groups",
    "labels",
    "final_probability",
    "component_probability",
    "decision_threshold",
}
_BOUNDARY_FIELDS = {
    "schema_version",
    "protected_palsynet_reads",
    "palsynet_sealed_outer_reads",
    "mayo_reads",
    "profile_generator_sha256",
    "helper_aggregate_sha256",
    "profile_payload_sha256",
}
_V4_EVALUATION_BY_PROFILE = {
    "free_asymmetry": "palsynet_development",
    "scripted_multimechanism": "neuroface_development",
    "cue_aligned_upper": "meei_development",
}
_BASELINE_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
)
_MAX_PRIVATE_PROFILE_BYTES = 64 * 1024 * 1024


def _immutable_exact(array: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(array)
    return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )


def _unique_json_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("JSON contains a duplicate key")
        result[key] = value
    return result


def _json_bytes(payload: bytes, name: str) -> Mapping[str, object]:
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_json_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{name} is not canonical JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def load_private_profile_bytes(payload: bytes) -> dict[str, object]:
    """Load one exact-byte private profile without exposing row evidence."""
    if type(payload) is not bytes or not 0 < len(payload) <= _MAX_PRIVATE_PROFILE_BYTES:
        raise ValueError("private profile payload size or type differs")
    expected_members = {f"{name}.npy" for name in _NPZ_FIELDS}
    try:
        with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
            names = archive.namelist()
            if (
                len(names) != len(expected_members)
                or len(names) != len(set(names))
                or set(names) != expected_members
            ):
                raise ValueError("private NPZ members differ from the closed schema")
        with np.load(io.BytesIO(payload), allow_pickle=False) as saved:
            if (
                len(saved.files) != len(_NPZ_FIELDS)
                or len(saved.files) != len(set(saved.files))
                or set(saved.files) != _NPZ_FIELDS
            ):
                raise ValueError("private NPZ fields differ from the closed schema")
            arrays = {name: np.asarray(saved[name]) for name in _NPZ_FIELDS}
    except (zipfile.BadZipFile, zipfile.LargeZipFile, EOFError, OSError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("private NPZ"):
            raise
        raise ValueError("private profile is not a safe exact NPZ") from exc
    schema = arrays["schema_version"]
    profile = arrays["evidence_profile"]
    groups = arrays["anonymous_groups"]
    threshold = arrays["decision_threshold"]
    if (
        schema.ndim != 0
        or schema.dtype.kind != "U"
        or profile.ndim != 0
        or profile.dtype.kind != "U"
        or groups.ndim != 1
        or groups.dtype.kind != "U"
        or threshold.ndim != 1
        or threshold.dtype != np.dtype(np.float64)
    ):
        raise ValueError("private profile scalar or string dtypes differ")
    labels = arrays["labels"]
    final = arrays["final_probability"]
    components = arrays["component_probability"]
    document = {
        "schema_version": str(schema.item()),
        "evidence_profile": str(profile.item()),
        "anonymous_groups": tuple(str(value) for value in groups.tolist()),
        "labels": _immutable_exact(labels),
        "final_probability": _immutable_exact(final),
        "component_probability": _immutable_exact(components),
        "decision_threshold": _immutable_exact(threshold),
    }
    evaluate_profile(document)
    return document


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _implementation_sha256() -> str:
    components = {
        "runner": _sha256(Path(__file__).read_bytes()),
        "evaluator": _sha256(
            (PROJECT_ROOT / "src/evaluation/selective_router_v5.py").read_bytes()
        ),
    }
    payload = (json.dumps(components, sort_keys=True, separators=(",", ":")) + "\n").encode()
    return _sha256(payload)


def _validate_boundary(
    value: object,
    profile_payloads: Mapping[str, bytes],
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != _BOUNDARY_FIELDS:
        raise ValueError("boundary attestation has an open or incomplete schema")
    if value["schema_version"] != "selective_router_v5_boundary_attestation":
        raise ValueError("boundary attestation schema version differs")
    for name in (
        "protected_palsynet_reads", "palsynet_sealed_outer_reads", "mayo_reads"
    ):
        if type(value[name]) is not int or value[name] != 0:
            raise ValueError(f"{name} must remain exactly zero")
    for name in ("profile_generator_sha256", "helper_aggregate_sha256"):
        digest = value[name]
        if (
            type(digest) is not str
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"{name} must be lowercase SHA-256")
    hashes = value["profile_payload_sha256"]
    if not isinstance(hashes, Mapping) or set(hashes) != set(EVIDENCE_PROFILES):
        raise ValueError("boundary profile commitments differ")
    for profile in EVIDENCE_PROFILES:
        if hashes[profile] != _sha256(profile_payloads[profile]):
            raise ValueError("private profile bytes differ from boundary commitment")
    return value


def _validate_v4_report(
    report: Mapping[str, object],
    *,
    model_sha256: str,
    evaluations: Mapping[str, Mapping[str, object]],
) -> None:
    if report.get("schema_version") != "universal_clinical_router_v4_aggregate_report":
        raise ValueError("v4 aggregate report schema differs")
    artifact = report.get("model_artifact")
    if not isinstance(artifact, Mapping) or artifact.get("sha256") != model_sha256:
        raise ValueError("v4 report does not bind the supplied model bytes")
    v4_evaluations = report.get("evaluations")
    if not isinstance(v4_evaluations, Mapping):
        raise ValueError("v4 aggregate evaluations are absent")
    for profile, name in _V4_EVALUATION_BY_PROFILE.items():
        expected = v4_evaluations.get(name)
        if not isinstance(expected, Mapping):
            raise ValueError(f"v4 evaluation {name!r} is absent")
        if expected.get("participants") != evaluations[profile]["participants"]:
            raise ValueError("private profile count differs from v4 report")
        metrics = expected.get("metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError("v4 metrics are absent")
        baseline = evaluations[profile]["baseline"]
        for metric in _BASELINE_METRICS:
            left, right = metrics.get(metric), baseline.get(metric)
            if (
                isinstance(left, bool)
                or not isinstance(left, (int, float))
                or right is None
                or not math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
            ):
                raise ValueError(
                    f"private {profile} {metric} does not reproduce UCR4"
                )


def _public_profile(evaluation: Mapping[str, object]) -> dict[str, object]:
    candidates = {}
    for candidate in CANDIDATE_ORDER:
        points = {}
        for coverage in COVERAGES:
            private = evaluation["candidates"][candidate][f"{coverage:.2f}"]
            points[f"{coverage:.2f}"] = {
                key: value
                for key, value in private.items()
                if key != "selection_sha256"
            }
        candidates[candidate] = points
    return {
        "participants": evaluation["participants"],
        "class_counts": evaluation["class_counts"],
        "decision_threshold_scope": evaluation["decision_threshold_scope"],
        "decision_threshold_range": evaluation["decision_threshold_range"],
        "baseline": evaluation["baseline"],
        "candidates": candidates,
    }


def build_aggregate_report(
    profile_payloads: Mapping[str, bytes],
    *,
    v4_model_bytes: bytes,
    v4_report_bytes: bytes,
    boundary_attestation: Mapping[str, object],
) -> dict[str, object]:
    """Recompute and return one deidentified aggregate-only experiment report."""
    if (
        not isinstance(profile_payloads, Mapping)
        or set(profile_payloads) != set(EVIDENCE_PROFILES)
        or type(v4_model_bytes) is not bytes
        or type(v4_report_bytes) is not bytes
    ):
        raise ValueError("aggregate evaluator inputs differ from the closed schema")
    _validate_boundary(boundary_attestation, profile_payloads)
    model_sha = _sha256(v4_model_bytes)
    v4_report = _json_bytes(v4_report_bytes, "v4 report")
    private_documents = {
        profile: load_private_profile_bytes(profile_payloads[profile])
        for profile in EVIDENCE_PROFILES
    }
    if any(
        document["evidence_profile"] != profile
        for profile, document in private_documents.items()
    ):
        raise ValueError("private profile payloads are misrouted")
    evaluations = {
        profile: evaluate_profile(document)
        for profile, document in private_documents.items()
    }
    _validate_v4_report(
        v4_report, model_sha256=model_sha, evaluations=evaluations
    )
    decision = select_candidate(evaluations)
    report = {
        "schema_version": "universal_clinical_router_v5_candidate_report",
        "status": "adaptive_development_candidate_not_current_model",
        "v4_model_sha256": model_sha,
        "v4_report_sha256": _sha256(v4_report_bytes),
        "implementation_sha256": _implementation_sha256(),
        "candidate_registry": list(CANDIDATE_ORDER),
        "primary_coverage": PRIMARY_COVERAGE,
        "evaluations": {
            profile: _public_profile(evaluations[profile])
            for profile in EVIDENCE_PROFILES
        },
        "decision": decision,
        "audit": {
            "protected_palsynet_reads": 0,
            "palsynet_sealed_outer_reads": 0,
            "mayo_reads": 0,
            "mayo_predictions": 0,
            "profile_generator_sha256": boundary_attestation[
                "profile_generator_sha256"
            ],
            "helper_aggregate_sha256": boundary_attestation[
                "helper_aggregate_sha256"
            ],
            "private_profile_payload_sha256": dict(
                boundary_attestation["profile_payload_sha256"]
            ),
        },
        "claim_boundary": {
            "development_only": True,
            "clinical_accuracy": False,
            "full_cohort_accuracy_changed": False,
            "current_model_changed": False,
        },
    }
    encoded = json.dumps(report, sort_keys=True, allow_nan=False).lower()
    for token in (
        "anonymous_groups", "labels", "final_probability",
        "component_probability", "selection_sha256", "/users/", "/home/",
        "recording_id", "participant_id", "source_sha256",
    ):
        if token in encoded:
            raise ValueError(f"public aggregate contains forbidden token {token!r}")
    return report


def _canonical_report_bytes(report: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            report, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False,
        ) + "\n"
    ).encode("utf-8")


def publish_report(path: Path, report: Mapping[str, object]) -> str:
    """Atomically publish a no-overwrite canonical aggregate report."""
    if not isinstance(path, Path) or path.exists() or path.is_symlink():
        raise FileExistsError("refusing to overwrite aggregate report")
    payload = _canonical_report_bytes(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            written = stream.write(payload)
            if written != len(payload):
                raise OSError("short aggregate report write")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return _sha256(payload)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--free-asymmetry", required=True, type=Path)
    parser.add_argument("--scripted-multimechanism", required=True, type=Path)
    parser.add_argument("--cue-aligned-upper", required=True, type=Path)
    parser.add_argument("--v4-model", required=True, type=Path)
    parser.add_argument("--v4-report", required=True, type=Path)
    parser.add_argument("--boundary-attestation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    profile_payloads = {
        "free_asymmetry": args.free_asymmetry.read_bytes(),
        "scripted_multimechanism": args.scripted_multimechanism.read_bytes(),
        "cue_aligned_upper": args.cue_aligned_upper.read_bytes(),
    }
    boundary = _json_bytes(
        args.boundary_attestation.read_bytes(), "boundary attestation"
    )
    report = build_aggregate_report(
        profile_payloads,
        v4_model_bytes=args.v4_model.read_bytes(),
        v4_report_bytes=args.v4_report.read_bytes(),
        boundary_attestation=boundary,
    )
    digest = publish_report(args.output, report)
    print(json.dumps({"report_sha256": digest, "decision": report["decision"]},
                     sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
