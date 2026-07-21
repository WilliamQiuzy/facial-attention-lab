"""Publish the fixed authenticated Mayo Fusion development stress report."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import stat
import sys
from collections.abc import Mapping
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import pretrain_dynamic_landmarks as pretrain_cli  # noqa: E402
from src.evaluation import focused_fusion_robustness as fusion_core  # noqa: E402
from src.pretraining import dynamic_landmark_ssl as ssl_core  # noqa: E402


DEFAULT_REPORT_PATH = (
    PROJECT_ROOT / "outputs" / "dynamic_landmark" / "benchmarks"
    / "development" / "focused-fusion-robustness-v1" / "report.json"
)
_SEEDS = (0, 1, 2)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TOP_FIELDS = (
    "schema_version", "status", "claim_scope", "source", "selected_arm",
    "seeds", "metric_policy", "accounting", "protocol_registry",
    "commitments", "conditions",
)
_METRIC_POLICY = {
    "canonicalization": "decimal_round_half_even_v1",
    "decimal_places": 5,
    "input_metric_min": 0.0,
    "input_metric_max": 1e9,
    "input_metric_max_digits": 64,
    "input_metric_exponent_min": -100,
    "input_metric_exponent_max": 100,
    "primary_metric": "trained.raw_mae.equal_block_macro",
    "lower_is_better": True,
    "degradation_formula": "100*(condition_mean/clean_mean-1)",
    "degradation_range": [-100.0, 1e16],
}
_ACCOUNTING = {
    "heldout_packets": 160,
    "heldout_recording_groups": 10,
    "valid_positions": 20434,
    "scored_target_positions": 5120,
    "scored_target_scalars": 486400,
    "observed_context_positions": 15314,
    "feature_width": 95,
}
_COMMITMENT_FIELDS = (
    "benchmark_script_sha256", "evaluation_module_sha256", "trainer_sha256",
    "bridge_generation_sha256", "common_contract_sha256",
    "winner_report_sha256", "checkpoints",
)
_CHECKPOINT_FIELDS = (
    "seed", "checkpoint_fingerprint", "checkpoint_receipt_sha256",
)


class _RedactingParser(argparse.ArgumentParser):
    def error(self, _message: str) -> None:
        self.exit(2, "focused Fusion benchmark arguments are invalid\n")


def _parser() -> argparse.ArgumentParser:
    return _RedactingParser(description=__doc__)


def canonical_json_bytes(value: object) -> bytes:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("public report is not canonical JSON") from exc
    return encoded.encode("ascii")


def _exact(value: object, expected: object) -> bool:
    if type(value) is not type(expected):
        return False
    if type(expected) is dict:
        assert isinstance(value, dict)
        assert isinstance(expected, dict)
        return set(value) == set(expected) and all(
            _exact(value[name], expected[name]) for name in expected
        )
    if type(expected) is list:
        assert isinstance(value, list)
        assert isinstance(expected, list)
        return len(value) == len(expected) and all(
            _exact(observed, wanted)
            for observed, wanted in zip(value, expected)
        )
    return bool(value == expected)


def _protocol_registry() -> list[dict[str, object]]:
    return [
        {
            "name": condition.name,
            "input_arm": condition.input_arm,
            "context_dropout_probability": condition.context_dropout_probability,
            "landmark_noise_sd": condition.landmark_noise_sd,
            "rng_seed": condition.rng_seed,
        }
        for condition in fusion_core.BENCHMARK_CONDITIONS
    ]


def _require_sha256(value: object, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} is not an exact SHA-256")
    return value


def validate_public_report(value: object) -> dict[str, object]:
    """Return a fresh report reconstructed through the closed public schema."""
    if type(value) is not dict or set(value) != set(_TOP_FIELDS):
        raise ValueError("public report fields are not exact")
    report = value
    literals = {
        "schema_version": "focused_fusion_robustness_report_v1",
        "status": "complete",
        "claim_scope": (
            "recording_heldout_development_reconstruction_stress_only"
        ),
        "source": "mayo",
        "selected_arm": "fusion",
    }
    if any(type(report[name]) is not str or report[name] != expected
           for name, expected in literals.items()):
        raise ValueError("public report literals are not exact")
    if not _exact(report["seeds"], list(_SEEDS)):
        raise ValueError("public report requires exact seeds 0, 1, and 2")
    if not _exact(report["metric_policy"], _METRIC_POLICY):
        raise ValueError("public metric policy is not exact")
    if not _exact(report["accounting"], _ACCOUNTING):
        raise ValueError("public benchmark accounting is not exact")
    expected_protocol = _protocol_registry()
    if not _exact(report["protocol_registry"], expected_protocol):
        raise ValueError("public protocol registry is not exact")

    commitments = report["commitments"]
    if type(commitments) is not dict or set(commitments) != set(
        _COMMITMENT_FIELDS
    ):
        raise ValueError("public commitments are not exact")
    normalized_commitments: dict[str, object] = {}
    for name in _COMMITMENT_FIELDS[:-1]:
        normalized_commitments[name] = _require_sha256(
            commitments[name], f"public commitment {name}",
        )
    checkpoints = commitments["checkpoints"]
    if type(checkpoints) is not list or len(checkpoints) != 3:
        raise ValueError("public checkpoint commitments require three rows")
    normalized_checkpoints: list[dict[str, object]] = []
    for seed, row in zip(_SEEDS, checkpoints):
        if type(row) is not dict or set(row) != set(_CHECKPOINT_FIELDS):
            raise ValueError("public checkpoint commitment fields are not exact")
        if type(row["seed"]) is not int or row["seed"] != seed:
            raise ValueError("public checkpoint seed order is not exact")
        normalized_checkpoints.append({
            "seed": seed,
            "checkpoint_fingerprint": _require_sha256(
                row["checkpoint_fingerprint"], "checkpoint fingerprint",
            ),
            "checkpoint_receipt_sha256": _require_sha256(
                row["checkpoint_receipt_sha256"], "checkpoint receipt",
            ),
        })
    normalized_commitments["checkpoints"] = normalized_checkpoints

    aggregate = fusion_core.validate_deidentified_payload({
        "conditions": report["conditions"],
    })
    return {
        "schema_version": literals["schema_version"],
        "status": literals["status"],
        "claim_scope": literals["claim_scope"],
        "source": literals["source"],
        "selected_arm": literals["selected_arm"],
        "seeds": list(_SEEDS),
        "metric_policy": dict(_METRIC_POLICY),
        "accounting": dict(_ACCOUNTING),
        "protocol_registry": expected_protocol,
        "commitments": normalized_commitments,
        "conditions": aggregate["conditions"],
    }


def _authenticate_winner_chain() -> tuple[object, dict, dict, str]:
    trainer_sha256 = _require_sha256(
        pretrain_cli._focused_trainer_sha256(), "focused trainer",
    )
    authorization = pretrain_cli._authorize_focused_bridge(
        pretrain_cli._FOCUSED_BRIDGE_ROOT,
        pretrain_cli._FOCUSED_MAYO_KEY,
        producer_sha256=trainer_sha256,
    )
    common = pretrain_cli._focused_common_contract(authorization)
    namespace = pretrain_cli._FOCUSED_NAMESPACE
    smoke = pretrain_cli._validate_focused_smoke_phase(
        namespace / "smoke", authorization=authorization, common=common,
    )
    selection = pretrain_cli._validate_focused_selection_phase(
        namespace / "selection", authorization=authorization, common=common,
        smoke=smoke,
    )
    winner = pretrain_cli._validate_focused_winner_phase(
        namespace / "winner", authorization=authorization, common=common,
        selection=selection,
    )
    winner_report = winner.get("report")
    checkpoints = winner.get("checkpoints")
    if (
        winner.get("selected_arm") != "fusion"
        or type(winner_report) is not dict
        or winner_report.get("selected_arm") != "fusion"
        or winner_report.get("seeds") != [0, 1, 2]
        or type(checkpoints) is not dict
        or set(checkpoints) != set(_SEEDS)
    ):
        raise ValueError("authenticated winner is not the exact Fusion seed set")
    return authorization, common, winner, trainer_sha256


def _heldout_inputs(
    authorization: object, common: Mapping[str, object],
) -> dict[str, object]:
    features, valid, timestamps, source_indices = (
        pretrain_cli._load_focused_tensors(authorization)
    )
    split = common.get("split")
    scaler = common.get("scaler")
    target_mask = common.get("heldout_mask")
    group_ids = getattr(authorization, "group_ids", None)
    if (
        not isinstance(split, ssl_core.SSLGroupSplit)
        or not isinstance(scaler, ssl_core.SourceScaler)
        or type(group_ids) is not tuple
        or not isinstance(target_mask, torch.Tensor)
    ):
        raise ValueError("authenticated common contract is malformed")
    _, heldout_indices, _ = ssl_core._validate_split_partition(split, group_ids)
    heldout_rows = torch.as_tensor(heldout_indices, dtype=torch.int64)
    heldout_features = features.index_select(0, heldout_rows)
    heldout_valid = valid.index_select(0, heldout_rows)
    heldout_timestamps = timestamps.index_select(0, heldout_rows)
    heldout_source_indices = source_indices.index_select(0, heldout_rows)
    heldout_scaled = scaler.transform(
        heldout_features, heldout_valid, source=ssl_core.MAYO_SOURCE,
    )
    heldout_groups = {group_ids[int(index)] for index in heldout_indices.tolist()}
    accounting = {
        "heldout_packets": int(heldout_scaled.shape[0]),
        "heldout_recording_groups": len(heldout_groups),
        "valid_positions": int(heldout_valid.sum().item()),
        "scored_target_positions": int(target_mask.sum().item()),
        "scored_target_scalars": int(target_mask.sum().item())
        * int(heldout_scaled.shape[-1]),
        "observed_context_positions": int(
            (heldout_valid & ~target_mask).sum().item()
        ),
        "feature_width": int(heldout_scaled.shape[-1]),
    }
    if not _exact(accounting, _ACCOUNTING):
        raise ValueError("authenticated heldout accounting is not exact")
    return {
        "features": heldout_scaled,
        "valid_mask": heldout_valid,
        "timestamps": heldout_timestamps,
        "source_frame_indices": heldout_source_indices,
        "target_mask": target_mask,
        "scaler": scaler,
        "split": split,
        "evaluated_indices": heldout_indices,
        "group_ids": group_ids,
    }


def _models_and_clean_metrics(
    winner: Mapping[str, object],
) -> tuple[dict[int, object], dict[int, object], dict[int, dict]]:
    checkpoints = winner.get("checkpoints")
    if type(checkpoints) is not dict or set(checkpoints) != set(_SEEDS):
        raise ValueError("authenticated checkpoints are incomplete")
    trained_models: dict[int, object] = {}
    fresh_models: dict[int, object] = {}
    expected: dict[int, dict] = {}
    with torch.random.fork_rng(devices=[]):
        for seed in _SEEDS:
            loaded = checkpoints[seed]
            if type(loaded) is not dict:
                raise ValueError("authenticated checkpoint is malformed")
            model_state = loaded.get("model_state")
            pre_model_state = loaded.get("pre_model_state")
            metadata = loaded.get("metadata")
            if (
                not isinstance(model_state, Mapping)
                or not isinstance(pre_model_state, Mapping)
                or type(metadata) is not dict
            ):
                raise ValueError("authenticated checkpoint state is unavailable")
            trained = ssl_core.DynamicLandmarkSSLModel().to("cpu")
            trained.load_state_dict(model_state, strict=True)
            fresh = ssl_core.DynamicLandmarkSSLModel().to("cpu")
            fresh.load_state_dict(pre_model_state, strict=True)
            trained_models[seed] = trained
            fresh_models[seed] = fresh
            expected[seed] = fusion_core.validate_metric_bundle(
                metadata.get("metrics"),
            )
    roots = [*trained_models.values(), *fresh_models.values()]
    if len({id(model) for model in roots}) != 6:
        raise ValueError("benchmark models do not have distinct root identities")
    return trained_models, fresh_models, expected


def _safe_source_sha256(path: Path, label: str) -> str:
    _resolved, _payload, digest, identity = (
        ssl_core._regular_file_snapshot_with_identity(
            path, label, max_bytes=4 * 1024 * 1024,
        )
    )
    if identity.uid != os.geteuid() or identity.mode & 0o022:
        raise ValueError(f"{label} storage is unsafe")
    return _require_sha256(digest, label)


def run_benchmark() -> dict[str, object]:
    authorization, common, winner, trainer_sha256 = (
        _authenticate_winner_chain()
    )
    inputs = _heldout_inputs(authorization, common)
    trained, fresh, expected = _models_and_clean_metrics(winner)
    rows = fusion_core.evaluate_fusion_conditions(
        trained_models=trained,
        fresh_models=fresh,
        **inputs,
        expected_clean_metrics_by_seed=expected,
    )
    aggregate = fusion_core.aggregate_condition_metrics(rows)
    aggregate = fusion_core.validate_deidentified_payload(aggregate)
    checkpoints = winner["checkpoints"]
    assert isinstance(checkpoints, dict)
    commitments = {
        "benchmark_script_sha256": _safe_source_sha256(
            Path(__file__).resolve(), "focused benchmark script",
        ),
        "evaluation_module_sha256": _safe_source_sha256(
            Path(fusion_core.__file__).resolve(), "focused evaluation module",
        ),
        "trainer_sha256": trainer_sha256,
        "bridge_generation_sha256": _require_sha256(
            getattr(authorization, "bridge_generation_sha256", None),
            "bridge generation",
        ),
        "common_contract_sha256": _require_sha256(
            common.get("common_contract_sha256"), "common contract",
        ),
        "winner_report_sha256": _require_sha256(
            winner.get("report_sha256"), "winner report",
        ),
        "checkpoints": [
            {
                "seed": seed,
                "checkpoint_fingerprint": _require_sha256(
                    checkpoints[seed].get("checkpoint_fingerprint"),
                    "checkpoint fingerprint",
                ),
                "checkpoint_receipt_sha256": _require_sha256(
                    checkpoints[seed].get("receipt_file_sha256"),
                    "checkpoint receipt",
                ),
            }
            for seed in _SEEDS
        ],
    }
    report = {
        "schema_version": "focused_fusion_robustness_report_v1",
        "status": "complete",
        "claim_scope": (
            "recording_heldout_development_reconstruction_stress_only"
        ),
        "source": "mayo",
        "selected_arm": "fusion",
        "seeds": list(_SEEDS),
        "metric_policy": dict(_METRIC_POLICY),
        "accounting": dict(_ACCOUNTING),
        "protocol_registry": _protocol_registry(),
        "commitments": commitments,
        "conditions": aggregate["conditions"],
    }
    return validate_public_report(report)


def _require_directory(path: Path, label: str, *, private: bool) -> None:
    try:
        observed = path.lstat()
    except FileNotFoundError as exc:
        raise ValueError(f"{label} does not exist") from exc
    expected_mode = 0o700 if private else None
    if (
        not stat.S_ISDIR(observed.st_mode)
        or observed.st_uid != os.geteuid()
        or (expected_mode is not None
            and stat.S_IMODE(observed.st_mode) != expected_mode)
    ):
        raise ValueError(f"{label} storage is unsafe")


def _prepare_private_directories(private_root: Path, parent: Path) -> None:
    private_root = private_root.absolute()
    parent = parent.absolute()
    try:
        relative = parent.relative_to(private_root)
    except ValueError as exc:
        raise ValueError("report is outside its private benchmark root") from exc
    _require_directory(private_root.parent, "benchmark root parent", private=False)
    candidates = [private_root]
    current = private_root
    for component in relative.parts:
        current = current / component
        candidates.append(current)
    for candidate in candidates:
        if candidate.exists() or candidate.is_symlink():
            _require_directory(candidate, "private benchmark directory", private=True)
            continue
        try:
            os.mkdir(candidate, 0o700)
        except FileExistsError:
            pass
        else:
            os.chmod(candidate, 0o700)
        _require_directory(candidate, "private benchmark directory", private=True)


def _validate_existing_report(path: Path) -> None:
    try:
        observed = path.lstat()
    except FileNotFoundError:
        return
    if (
        not stat.S_ISREG(observed.st_mode)
        or observed.st_uid != os.geteuid()
        or stat.S_IMODE(observed.st_mode) != 0o600
        or observed.st_nlink != 1
    ):
        raise ValueError("existing public report storage is unsafe")


def _atomic_write_report(
    path: Path, payload: bytes, *, private_root: Path,
) -> None:
    if type(payload) is not bytes or not payload:
        raise ValueError("public report bytes are invalid")
    path = path.absolute()
    private_root = private_root.absolute()
    if path.name != "report.json":
        raise ValueError("only the fixed report filename may be published")
    _prepare_private_directories(private_root, path.parent)
    _validate_existing_report(path)
    temporary = path.parent / f".report.{secrets.token_hex(16)}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary, flags, 0o600)
    try:
        try:
            os.fchmod(descriptor, 0o600)
            offset = 0
            while offset < len(payload):
                written = os.write(descriptor, payload[offset:])
                if written < 1:
                    raise OSError("private report write made no progress")
                offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _validate_existing_report(path)
        os.replace(temporary, path)
        directory = os.open(
            path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    _validate_existing_report(path)
    _authorization, observed = ssl_core._private_file_snapshot(
        path, "focused public report", max_bytes=16 * 1024 * 1024,
    )
    if observed != payload:
        raise ValueError("public report changed during publication")


def main(argv: list[str] | None = None) -> None:
    _parser().parse_args(argv)
    report = validate_public_report(run_benchmark())
    payload = canonical_json_bytes(report)
    _atomic_write_report(
        DEFAULT_REPORT_PATH, payload, private_root=DEFAULT_REPORT_PATH.parents[2],
    )
    print(
        "status=complete report_sha256="
        + hashlib.sha256(payload).hexdigest()
    )


if __name__ == "__main__":
    main()
