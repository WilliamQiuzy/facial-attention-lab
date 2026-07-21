#!/usr/bin/env python3
"""Run the fixed, development-only PalsyNet SSL-transfer smoke."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import secrets
import sys
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import pretrain_dynamic_landmarks as ssl_runner  # noqa: E402
from scripts.run_dynamic_landmark_classical import (  # noqa: E402
    binary_group_metrics,
    load_classical_dataset,
)
from src.evaluation.nested_group_cv import build_nested_group_splits  # noqa: E402
from src.models.dynamic_landmark import ARM_FUSION  # noqa: E402
from src.training.dynamic_landmark_benchmark import BenchmarkConfig  # noqa: E402
from src.training.dynamic_landmark_transfer_smoke import (  # noqa: E402
    DEVELOPMENT_CANDIDATES,
    FUSION_SSL_WARMSTART,
    run_development_inner_oof,
)


RUN_SEED = 0
RUN_EPOCHS = 12
RUN_OUTER_FOLD = 0
DEFAULT_REPORT_PATH = (
    PROJECT_ROOT / "outputs" / "dynamic_landmark" / "benchmarks"
    / "development" / "focused-ssl-transfer-smoke-v1" / "report.json"
)

COMMITMENT_FIELDS = (
    "trainer_sha256",
    "bridge_generation_sha256",
    "common_contract_sha256",
    "winner_report_sha256",
    "checkpoint_fingerprint",
    "checkpoint_receipt_sha256",
    "palsynet_manifest_sha256",
    "split_sha256",
    "runner_sha256",
)
_AUTH_COMMITMENT_FIELDS = COMMITMENT_FIELDS[:6]
_METRIC_FIELDS = (
    "auroc",
    "average_precision",
    "brier",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
)
_TOP_FIELDS = (
    "schema_version",
    "status",
    "claim_scope",
    "dataset",
    "claim_unit",
    "identity_status",
    "protocol",
    "accounting",
    "commitments",
    "candidates",
    "decision",
)
_ACCOUNTING_FIELDS = (
    "total_records",
    "total_groups",
    "development_records",
    "development_groups",
    "protected_records",
    "protected_groups",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_OPAQUE_IDENTIFIER = re.compile(r"(?:rec|grp)_[0-9a-f]{64}")
_PROTOCOL = {
    "seed": RUN_SEED,
    "epochs": RUN_EPOCHS,
    "outer_fold": RUN_OUTER_FOLD,
    "inner_folds": 4,
    "candidates": list(DEVELOPMENT_CANDIDATES),
    "optimizer": "AdamW",
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "mirror_probability": 0.5,
    "threshold": 0.5,
    "group_probability_aggregation": "mean",
    "outer_predictions": 0,
}


class _RedactingParser(argparse.ArgumentParser):
    def error(self, _message: str) -> None:
        self.exit(2, "focused transfer smoke arguments are invalid\n")


def _parser() -> argparse.ArgumentParser:
    parser = _RedactingParser(description=__doc__)
    parser.add_argument("--ssl-pretraining-root", required=True, type=Path)
    parser.add_argument("--palsynet-cache-root", required=True, type=Path)
    return parser


def _canonical_json_bytes(value: object) -> bytes:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("report is not canonical JSON data") from exc
    return encoded.encode("ascii")


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _finite_probability(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and lie within [0, 1]")
    return result


def _decision(candidate_metrics: Mapping[str, Mapping[str, float]]) -> dict[str, object]:
    best = DEVELOPMENT_CANDIDATES[0]
    for candidate in DEVELOPMENT_CANDIDATES[1:]:
        if candidate_metrics[candidate]["auroc"] > candidate_metrics[best]["auroc"]:
            best = candidate
    delta_auc = round(
        candidate_metrics[FUSION_SSL_WARMSTART]["auroc"]
        - candidate_metrics[DEVELOPMENT_CANDIDATES[1]]["auroc"],
        12,
    )
    delta_sensitivity = round(
        candidate_metrics[FUSION_SSL_WARMSTART]["sensitivity"]
        - candidate_metrics[DEVELOPMENT_CANDIDATES[1]]["sensitivity"],
        12,
    )
    expand = delta_auc >= 0.02 and delta_sensitivity >= -0.03
    return {
        "best_candidate": best,
        "warmstart_minus_random_fusion_auroc": delta_auc,
        "warmstart_minus_random_fusion_sensitivity": delta_sensitivity,
        "formal_expansion_gate": expand,
        "recommendation": (
            "expand_to_three_seed_development_evaluation_only"
            if expand
            else "stop_ssl_expansion_select_best_development_candidate"
        ),
    }


def _build_report(
    *,
    accounting: Mapping[str, int],
    commitments: Mapping[str, str],
    candidate_metrics: Mapping[str, Mapping[str, float]],
) -> dict[str, object]:
    checked_accounting = {
        name: accounting[name] for name in _ACCOUNTING_FIELDS
    }
    checked_commitments = {
        name: commitments[name] for name in COMMITMENT_FIELDS
    }
    checked_metrics: dict[str, dict[str, float]] = {}
    rows = []
    for candidate in DEVELOPMENT_CANDIDATES:
        metrics = {
            name: float(candidate_metrics[candidate][name])
            for name in _METRIC_FIELDS
        }
        checked_metrics[candidate] = metrics
        rows.append({
            "candidate": candidate,
            "initialization": (
                "authenticated_fusion_ssl_seed0"
                if candidate == FUSION_SSL_WARMSTART else "random"
            ),
            **metrics,
        })
    report = {
        "schema_version": "focused_ssl_transfer_smoke_v1",
        "status": "complete",
        "claim_scope": "inner_oof_development_smoke_only",
        "dataset": "PalsyNet",
        "claim_unit": "video_held_out",
        "identity_status": "unreviewed",
        "protocol": dict(_PROTOCOL),
        "accounting": checked_accounting,
        "commitments": checked_commitments,
        "candidates": rows,
        "decision": _decision(checked_metrics),
    }
    return _validate_report(report)


def _walk_report(value: object):
    yield value
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _walk_report(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_report(child)


def _validate_report(
    value: object,
    *,
    forbidden_paths: tuple[Path, ...] = (),
) -> dict[str, object]:
    if not isinstance(value, dict) or tuple(value) != _TOP_FIELDS:
        raise ValueError("report top-level schema is not exact")
    literals = {
        "schema_version": "focused_ssl_transfer_smoke_v1",
        "status": "complete",
        "claim_scope": "inner_oof_development_smoke_only",
        "dataset": "PalsyNet",
        "claim_unit": "video_held_out",
        "identity_status": "unreviewed",
    }
    if any(value.get(name) != expected for name, expected in literals.items()):
        raise ValueError("report literals are not exact")
    if value.get("protocol") != _PROTOCOL:
        raise ValueError("report protocol is not exact")

    accounting = value.get("accounting")
    if not isinstance(accounting, dict) or tuple(accounting) != _ACCOUNTING_FIELDS:
        raise ValueError("report accounting schema is not exact")
    for name in _ACCOUNTING_FIELDS:
        number = accounting[name]
        if isinstance(number, (bool, np.bool_)) or not isinstance(
            number, (int, np.integer)
        ) or int(number) < 1:
            raise ValueError(f"accounting {name} must be a positive integer")
    if accounting["development_records"] + accounting["protected_records"] != accounting["total_records"]:
        raise ValueError("report record accounting is inconsistent")
    if accounting["development_groups"] + accounting["protected_groups"] != accounting["total_groups"]:
        raise ValueError("report group accounting is inconsistent")

    commitments = value.get("commitments")
    if not isinstance(commitments, dict) or tuple(commitments) != COMMITMENT_FIELDS:
        raise ValueError("report commitment schema is not exact")
    for name in COMMITMENT_FIELDS:
        _require_sha256(commitments[name], name)

    rows = value.get("candidates")
    if not isinstance(rows, list) or len(rows) != len(DEVELOPMENT_CANDIDATES):
        raise ValueError("report candidates are incomplete")
    normalized_metrics: dict[str, dict[str, float]] = {}
    expected_row_fields = ("candidate", "initialization", *_METRIC_FIELDS)
    for expected_candidate, row in zip(DEVELOPMENT_CANDIDATES, rows):
        if not isinstance(row, dict) or tuple(row) != expected_row_fields:
            raise ValueError("candidate row schema is not exact")
        expected_initialization = (
            "authenticated_fusion_ssl_seed0"
            if expected_candidate == FUSION_SSL_WARMSTART else "random"
        )
        if row["candidate"] != expected_candidate or row["initialization"] != expected_initialization:
            raise ValueError("candidate identity or initialization is not exact")
        normalized_metrics[expected_candidate] = {
            name: _finite_probability(row[name], f"{expected_candidate}.{name}")
            for name in _METRIC_FIELDS
        }
    if value.get("decision") != _decision(normalized_metrics):
        raise ValueError("report decision contradicts candidate metrics")

    forbidden = tuple(str(Path(path)) for path in forbidden_paths)
    for item in _walk_report(value):
        if isinstance(item, (float, np.floating)) and not math.isfinite(float(item)):
            raise ValueError("report contains a nonfinite number")
        if isinstance(item, str):
            lowered = item.lower()
            if _OPAQUE_IDENTIFIER.search(lowered) is not None or any(
                token in lowered for token in (
                    "patient_id", "recording_id", "group_id",
                )
            ):
                raise ValueError("report contains a row-level identifier")
            if any(path and path in item for path in forbidden):
                raise ValueError("report contains a private path")

    try:
        return json.loads(_canonical_json_bytes(value).decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("report cannot be reconstructed") from exc


def _make_private_parents(parent: Path) -> None:
    missing = []
    current = parent
    while not current.exists():
        missing.append(current)
        current = current.parent
    if not current.is_dir():
        raise ValueError("report parent chain is invalid")
    for directory in reversed(missing):
        directory.mkdir(mode=0o700)
        os.chmod(directory, 0o700)


def _atomic_write_report(path: Path, report: Mapping[str, object]) -> None:
    destination = Path(path)
    payload = _canonical_json_bytes(_validate_report(dict(report)))
    _make_private_parents(destination.parent)
    if destination.exists():
        raise FileExistsError(destination.name)
    temporary = destination.parent / (
        f".{destination.name}.staging-{secrets.token_hex(8)}"
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("report write made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.link(temporary, destination)
        directory_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _extract_authenticated_winner(
    chain: Mapping[str, object],
    *,
    current_trainer_sha256: str,
) -> tuple[OrderedDict[str, torch.Tensor], dict[str, str]]:
    expected_top = {
        "trainer_sha256", "bridge_generation_sha256",
        "common_contract_sha256", "winner_report_sha256",
        "selected_arm", "checkpoints",
    }
    if not isinstance(chain, Mapping) or set(chain) != expected_top:
        raise ValueError("authenticated winner chain schema is not exact")
    trainer = _require_sha256(chain["trainer_sha256"], "trainer_sha256")
    if trainer != _require_sha256(current_trainer_sha256, "current trainer SHA-256"):
        raise ValueError("authenticated winner trainer has drifted")
    if chain["selected_arm"] != ARM_FUSION:
        raise ValueError("authenticated winner must be Fusion")
    checkpoints = chain["checkpoints"]
    if not isinstance(checkpoints, Mapping) or set(checkpoints) != {0, 1, 2}:
        raise ValueError("authenticated winner requires exact seeds 0, 1, and 2")
    expected_checkpoint_fields = {
        "metadata", "model_state", "checkpoint_fingerprint",
        "checkpoint_receipt_sha256",
    }
    for seed in (0, 1, 2):
        checkpoint = checkpoints[seed]
        if not isinstance(checkpoint, Mapping) or set(checkpoint) != expected_checkpoint_fields:
            raise ValueError("authenticated checkpoint schema is not exact")
        metadata = checkpoint["metadata"]
        if not isinstance(metadata, Mapping) or any(
            metadata.get(name) != expected
            for name, expected in {
                "phase": "winner", "arm": ARM_FUSION,
                "seed": seed, "epochs": 30,
            }.items()
        ):
            raise ValueError("authenticated checkpoint metadata is not exact")
        _require_sha256(checkpoint["checkpoint_fingerprint"], "checkpoint fingerprint")
        _require_sha256(checkpoint["checkpoint_receipt_sha256"], "checkpoint receipt")
        if not isinstance(checkpoint["model_state"], Mapping):
            raise ValueError("authenticated checkpoint lacks model_state")

    selected = checkpoints[RUN_SEED]
    state = OrderedDict()
    for name, tensor in selected["model_state"].items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError("authenticated model_state is malformed")
        state[name] = tensor.detach().clone()
    commitments = {
        "trainer_sha256": trainer,
        "bridge_generation_sha256": _require_sha256(
            chain["bridge_generation_sha256"], "bridge generation"
        ),
        "common_contract_sha256": _require_sha256(
            chain["common_contract_sha256"], "common contract"
        ),
        "winner_report_sha256": _require_sha256(
            chain["winner_report_sha256"], "winner report"
        ),
        "checkpoint_fingerprint": str(selected["checkpoint_fingerprint"]),
        "checkpoint_receipt_sha256": str(selected["checkpoint_receipt_sha256"]),
    }
    return state, commitments


def _authenticate_fusion_winner(
    pretraining_root: Path,
) -> tuple[OrderedDict[str, torch.Tensor], dict[str, str]]:
    root = Path(pretraining_root)
    trainer_sha256 = ssl_runner._focused_trainer_sha256()
    authorization = ssl_runner._authorize_focused_bridge(
        root / "bridge",
        root / ".mayo_ssl_hmac.key",
        producer_sha256=trainer_sha256,
    )
    common = ssl_runner._focused_common_contract(authorization)
    namespace = root / "development" / "focused-modality-v1"
    smoke = ssl_runner._validate_focused_smoke_phase(
        namespace / "smoke", authorization=authorization, common=common,
    )
    selection = ssl_runner._validate_focused_selection_phase(
        namespace / "selection",
        authorization=authorization,
        common=common,
        smoke=smoke,
    )
    if selection.get("selected_arm") != ARM_FUSION:
        raise ValueError("authenticated selection did not choose Fusion")
    winner = ssl_runner._validate_focused_winner_phase(
        namespace / "winner",
        authorization=authorization,
        common=common,
        selection=selection,
    )
    loaded_checkpoints = winner.get("checkpoints")
    if not isinstance(loaded_checkpoints, Mapping):
        raise ValueError("authenticated winner checkpoints are unavailable")
    checkpoints = {}
    for seed, loaded in loaded_checkpoints.items():
        if not isinstance(loaded, Mapping):
            raise ValueError("authenticated checkpoint is malformed")
        checkpoints[seed] = {
            "metadata": loaded.get("metadata"),
            "model_state": loaded.get("model_state"),
            "checkpoint_fingerprint": loaded.get("checkpoint_fingerprint"),
            "checkpoint_receipt_sha256": loaded.get("receipt_file_sha256"),
        }
    chain = {
        "trainer_sha256": trainer_sha256,
        "bridge_generation_sha256": getattr(
            authorization, "bridge_generation_sha256", None
        ),
        "common_contract_sha256": common.get("common_contract_sha256"),
        "winner_report_sha256": winner.get("report_sha256"),
        "selected_arm": winner.get("selected_arm"),
        "checkpoints": checkpoints,
    }
    return _extract_authenticated_winner(
        chain, current_trainer_sha256=trainer_sha256,
    )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _split_sha256(fold: object) -> str:
    payload = {
        "outer_train_indices": [int(value) for value in fold.train_indices],
        "protected_indices": [int(value) for value in fold.test_indices],
        "inner_folds": [
            {
                "train_indices": [int(value) for value in inner.train_indices],
                "validation_indices": [
                    int(value) for value in inner.validation_indices
                ],
            }
            for inner in fold.inner_folds
        ],
    }
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _same_state(
    first: Mapping[str, torch.Tensor], second: Mapping[str, torch.Tensor],
) -> bool:
    return tuple(first) == tuple(second) and all(
        torch.equal(first[name], second[name]) for name in first
    )


def main() -> None:
    args = _parser().parse_args()
    source_state, auth_commitments = _authenticate_fusion_winner(
        args.ssl_pretraining_root
    )
    dataset = load_classical_dataset(args.palsynet_cache_root)
    if (
        dataset.features.shape[0] != 49
        or len(set(dataset.group_ids.tolist())) != 48
        or int(dataset.labels.sum()) != 27
        or dataset.claim_unit != "video_held_out"
        or dataset.identity_status != "unreviewed"
    ):
        raise ValueError("PalsyNet development cohort is not the frozen 49-record set")

    folds = build_nested_group_splits(dataset.labels, dataset.group_ids)
    if len(folds) != 5:
        raise ValueError("PalsyNet grouped split must contain five outer folds")
    fold = folds[RUN_OUTER_FOLD]
    features = torch.from_numpy(np.asarray(dataset.features, dtype=np.float32).copy())
    valid_mask = torch.from_numpy(np.asarray(dataset.valid_masks, dtype=np.bool_).copy())
    timestamps = torch.from_numpy(np.asarray(dataset.timestamps, dtype=np.float32).copy())
    source_indices = torch.from_numpy(
        np.asarray(dataset.source_frame_indices, dtype=np.int64).copy()
    )
    labels = torch.from_numpy(np.asarray(dataset.labels, dtype=np.float32).copy())
    config = BenchmarkConfig(
        max_epochs=RUN_EPOCHS,
        learning_rate=1e-3,
        weight_decay=1e-4,
        mirror_probability=0.5,
    )

    results = {}
    metrics = {}
    for candidate in DEVELOPMENT_CANDIDATES:
        result = run_development_inner_oof(
            features,
            valid_mask,
            timestamps,
            source_indices,
            labels,
            fold=fold,
            candidate=candidate,
            seed=RUN_SEED,
            epochs=RUN_EPOCHS,
            config=config,
            source_state=(
                source_state if candidate == FUSION_SSL_WARMSTART else None
            ),
        )
        results[candidate] = result
        groups = dataset.group_ids[result.outer_train_indices]
        metrics[candidate] = binary_group_metrics(
            result.labels, groups, result.probabilities
        )

    reference = results[DEVELOPMENT_CANDIDATES[0]]
    for candidate in DEVELOPMENT_CANDIDATES[1:]:
        current = results[candidate]
        if not np.array_equal(current.outer_train_indices, reference.outer_train_indices):
            raise RuntimeError("candidate development rows are inconsistent")
        if not np.array_equal(current.labels, reference.labels):
            raise RuntimeError("candidate development labels are inconsistent")

    replay_state, replay_commitments = _authenticate_fusion_winner(
        args.ssl_pretraining_root
    )
    if replay_commitments != auth_commitments or not _same_state(
        source_state, replay_state
    ):
        raise ValueError("authenticated SSL winner changed during the smoke")

    train_indices = np.asarray(fold.train_indices, dtype=np.int64)
    protected_indices = np.asarray(fold.test_indices, dtype=np.int64)
    accounting = {
        "total_records": int(dataset.labels.shape[0]),
        "total_groups": len(set(dataset.group_ids.tolist())),
        "development_records": int(train_indices.size),
        "development_groups": len(set(dataset.group_ids[train_indices].tolist())),
        "protected_records": int(protected_indices.size),
        "protected_groups": len(set(dataset.group_ids[protected_indices].tolist())),
    }
    commitments = {
        **auth_commitments,
        "palsynet_manifest_sha256": _sha256_file(
            Path(args.palsynet_cache_root) / "collection_manifest.json"
        ),
        "split_sha256": _split_sha256(fold),
        "runner_sha256": _sha256_file(Path(__file__).resolve()),
    }
    report = _build_report(
        accounting=accounting,
        commitments=commitments,
        candidate_metrics=metrics,
    )
    report = _validate_report(
        report,
        forbidden_paths=(
            Path(args.ssl_pretraining_root),
            Path(args.palsynet_cache_root),
            DEFAULT_REPORT_PATH,
        ),
    )
    _atomic_write_report(DEFAULT_REPORT_PATH, report)
    sys.stdout.buffer.write(_canonical_json_bytes(report) + b"\n")


if __name__ == "__main__":
    main()
