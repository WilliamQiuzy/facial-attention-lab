"""Publish the fixed authenticated Mayo Fusion development stress report."""
from __future__ import annotations

import argparse
import fcntl
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


def _require_exact_dict(
    value: object, fields: tuple[str, ...], label: str,
) -> dict:
    if type(value) is not dict or len(value) != len(fields):
        raise ValueError(f"{label} fields are not exact")
    for key in value:
        if type(key) is not str:
            raise ValueError(f"{label} keys must be exact strings")
    if any(field not in value for field in fields):
        raise ValueError(f"{label} fields are not exact")
    return value


def _require_seed_mapping(value: object, label: str) -> dict:
    if type(value) is not dict or len(value) != len(_SEEDS):
        raise ValueError(f"{label} seeds are not exact")
    for key in value:
        if type(key) is not int:
            raise ValueError(f"{label} seed keys are not exact integers")
    if any(seed not in value for seed in _SEEDS):
        raise ValueError(f"{label} seeds are not exact")
    return value


def _exact(value: object, expected: object) -> bool:
    if type(value) is not type(expected):
        return False
    if type(expected) is dict:
        assert isinstance(expected, dict)
        try:
            mapping = _require_exact_dict(
                value, tuple(expected), "exact mapping",
            )
        except ValueError:
            return False
        return all(_exact(mapping[name], expected[name]) for name in expected)
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
    report = _require_exact_dict(value, _TOP_FIELDS, "public report")
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

    commitments = _require_exact_dict(
        report["commitments"], _COMMITMENT_FIELDS, "public commitments",
    )
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
        row = _require_exact_dict(
            row, _CHECKPOINT_FIELDS, "public checkpoint commitment",
        )
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

    provided_aggregate = fusion_core.validate_deidentified_payload({
        "conditions": report["conditions"],
    })
    seed_rows = [
        row
        for condition in provided_aggregate["conditions"]
        for row in condition["seed_rows"]
    ]
    if len(seed_rows) != 30:
        raise ValueError("public report does not contain the exact seed grid")
    recomputed_aggregate = fusion_core.validate_deidentified_payload(
        fusion_core.aggregate_condition_metrics(seed_rows)
    )
    if canonical_json_bytes(provided_aggregate) != canonical_json_bytes(
        recomputed_aggregate
    ):
        raise ValueError("public aggregates contradict their seed rows")
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
        "conditions": recomputed_aggregate["conditions"],
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
        type(winner.get("selected_arm")) is not str
        or winner.get("selected_arm") != "fusion"
        or type(winner_report) is not dict
        or type(winner_report.get("selected_arm")) is not str
        or winner_report.get("selected_arm") != "fusion"
        or not _exact(winner_report.get("seeds"), [0, 1, 2])
    ):
        raise ValueError("authenticated winner is not the exact Fusion seed set")
    _require_seed_mapping(checkpoints, "authenticated winner checkpoints")
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
    checkpoints = _require_seed_mapping(
        winner.get("checkpoints"), "authenticated checkpoints",
    )
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


def _source_commitments() -> dict[str, str]:
    return {
        "benchmark_script_sha256": _safe_source_sha256(
            Path(__file__).resolve(), "focused benchmark script",
        ),
        "evaluation_module_sha256": _safe_source_sha256(
            Path(fusion_core.__file__).resolve(), "focused evaluation module",
        ),
    }


def _checkpoint_commitments(winner: Mapping[str, object]) -> list[dict[str, object]]:
    checkpoints = _require_seed_mapping(
        winner.get("checkpoints"), "authenticated lineage checkpoints",
    )
    rows: list[dict[str, object]] = []
    for seed in _SEEDS:
        loaded = checkpoints[seed]
        if type(loaded) is not dict:
            raise ValueError("authenticated lineage checkpoint is malformed")
        for key in loaded:
            if type(key) is not str:
                raise ValueError(
                    "authenticated lineage checkpoint keys are not exact strings"
                )
        rows.append({
            "seed": seed,
            "checkpoint_fingerprint": _require_sha256(
                loaded.get("checkpoint_fingerprint"),
                "checkpoint fingerprint",
            ),
            "checkpoint_receipt_sha256": _require_sha256(
                loaded.get("receipt_file_sha256"),
                "checkpoint receipt",
            ),
        })
    return rows


def _authenticated_lineage_commitments(
    authorization: object,
    common: Mapping[str, object],
    winner: Mapping[str, object],
    trainer_sha256: object,
) -> dict[str, object]:
    return {
        "trainer_sha256": _require_sha256(trainer_sha256, "focused trainer"),
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
        "checkpoints": _checkpoint_commitments(winner),
    }


def run_benchmark() -> dict[str, object]:
    source_before = _source_commitments()
    authorization, common, winner, trainer_sha256 = (
        _authenticate_winner_chain()
    )
    lineage_before = _authenticated_lineage_commitments(
        authorization, common, winner, trainer_sha256,
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
    source_after = _source_commitments()
    if not _exact(source_after, source_before):
        raise ValueError("benchmark source changed during evaluation")
    pretrain_cli._require_focused_authority_unchanged(authorization)
    final_authorization, final_common, final_winner, final_trainer = (
        _authenticate_winner_chain()
    )
    lineage_after = _authenticated_lineage_commitments(
        final_authorization, final_common, final_winner, final_trainer,
    )
    if not _exact(lineage_after, lineage_before):
        raise ValueError("authenticated benchmark lineage changed during evaluation")
    commitments = {
        **source_before,
        **lineage_before,
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


def _require_publication_edge(report: Mapping[str, object]) -> None:
    commitments = _require_exact_dict(
        report.get("commitments"), _COMMITMENT_FIELDS,
        "publication commitments",
    )
    observed_sources = _source_commitments()
    expected_sources = {
        name: commitments[name]
        for name in ("benchmark_script_sha256", "evaluation_module_sha256")
    }
    if not _exact(observed_sources, expected_sources):
        raise ValueError("benchmark source changed before publication")
    authorization, common, winner, trainer_sha256 = (
        _authenticate_winner_chain()
    )
    pretrain_cli._require_focused_authority_unchanged(authorization)
    observed_lineage = _authenticated_lineage_commitments(
        authorization, common, winner, trainer_sha256,
    )
    expected_lineage = {
        name: commitments[name]
        for name in (
            "trainer_sha256", "bridge_generation_sha256",
            "common_contract_sha256", "winner_report_sha256", "checkpoints",
        )
    }
    if not _exact(observed_lineage, expected_lineage):
        raise ValueError("authenticated lineage changed before publication")


def _directory_open_flags() -> int:
    return (
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    )


def _regular_open_flags(access: int) -> int:
    return access | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)


def _require_anchor_directory(descriptor: int) -> None:
    observed = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(observed.st_mode)
        or observed.st_uid != os.geteuid()
        or stat.S_IMODE(observed.st_mode) & 0o022
    ):
        raise ValueError("benchmark root parent storage is unsafe")


def _require_private_directory(descriptor: int) -> None:
    observed = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(observed.st_mode)
        or observed.st_uid != os.geteuid()
        or stat.S_IMODE(observed.st_mode) != 0o700
    ):
        raise ValueError("private benchmark directory storage is unsafe")


def _require_bound_private_directory_name(
    parent: int, name: str, descriptor: int,
) -> None:
    held = os.fstat(descriptor)
    try:
        named = os.stat(name, dir_fd=parent, follow_symlinks=False)
    except OSError as exc:
        raise ValueError("private benchmark directory name is unavailable") from exc
    _require_private_directory(descriptor)
    if (
        not stat.S_ISDIR(named.st_mode)
        or named.st_uid != os.geteuid()
        or stat.S_IMODE(named.st_mode) != 0o700
    ):
        raise ValueError("private benchmark directory name is unsafe")
    if (held.st_dev, held.st_ino) != (named.st_dev, named.st_ino):
        raise ValueError("private benchmark directory name changed")


def _require_bound_private_directory_chain(
    directories: list[int], components: tuple[str, ...],
) -> None:
    if len(directories) != len(components) + 1:
        raise ValueError("private benchmark directory chain is incomplete")
    for parent, name, descriptor in zip(
        directories[:-1], components, directories[1:],
    ):
        _require_bound_private_directory_name(parent, name, descriptor)


def _open_private_directory_at(parent: int, name: str) -> int:
    created = False
    try:
        descriptor = os.open(name, _directory_open_flags(), dir_fd=parent)
    except FileNotFoundError:
        try:
            os.mkdir(name, 0o700, dir_fd=parent)
            created = True
            os.fsync(parent)
        except FileExistsError:
            pass
        try:
            descriptor = os.open(name, _directory_open_flags(), dir_fd=parent)
        except OSError as exc:
            raise ValueError("private benchmark directory cannot be held") from exc
    except OSError as exc:
        raise ValueError("private benchmark directory cannot be held") from exc
    try:
        if created:
            os.fchmod(descriptor, 0o700)
            os.fsync(descriptor)
        _require_private_directory(descriptor)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _require_private_regular_stat(observed: os.stat_result, label: str) -> None:
    if (
        not stat.S_ISREG(observed.st_mode)
        or observed.st_uid != os.geteuid()
        or stat.S_IMODE(observed.st_mode) != 0o600
        or observed.st_nlink != 1
    ):
        raise ValueError(f"{label} storage is unsafe")


def _require_bound_regular_name(
    directory: int, name: str, descriptor: int, label: str,
) -> None:
    held = os.fstat(descriptor)
    try:
        named = os.stat(name, dir_fd=directory, follow_symlinks=False)
    except OSError as exc:
        raise ValueError(f"{label} name is unavailable") from exc
    _require_private_regular_stat(held, label)
    _require_private_regular_stat(named, label)
    if (held.st_dev, held.st_ino) != (named.st_dev, named.st_ino):
        raise ValueError(f"{label} name changed")


def _open_publication_lock(directory: int) -> int:
    name = ".report.lock"
    created = False
    try:
        descriptor = os.open(
            name, _regular_open_flags(os.O_RDWR), dir_fd=directory,
        )
    except FileNotFoundError:
        try:
            descriptor = os.open(
                name,
                _regular_open_flags(os.O_RDWR) | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=directory,
            )
            created = True
        except FileExistsError:
            try:
                descriptor = os.open(
                    name, _regular_open_flags(os.O_RDWR), dir_fd=directory,
                )
            except OSError as exc:
                raise ValueError("publication lock cannot be held") from exc
        except OSError as exc:
            raise ValueError("publication lock cannot be created") from exc
    except OSError as exc:
        raise ValueError("publication lock cannot be held") from exc
    try:
        if created:
            os.fchmod(descriptor, 0o600)
            os.fsync(descriptor)
            os.fsync(directory)
        _require_bound_regular_name(
            directory, name, descriptor, "publication lock",
        )
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _validate_existing_report_at(directory: int) -> None:
    try:
        descriptor = os.open(
            "report.json", _regular_open_flags(os.O_RDONLY), dir_fd=directory,
        )
    except FileNotFoundError:
        return
    except OSError as exc:
        raise ValueError("existing public report cannot be held") from exc
    try:
        _require_bound_regular_name(
            directory, "report.json", descriptor, "existing public report",
        )
    finally:
        os.close(descriptor)


def _regular_identity(observed: os.stat_result) -> tuple[int, ...]:
    return (
        observed.st_dev, observed.st_ino, observed.st_mode, observed.st_uid,
        observed.st_nlink, observed.st_size, observed.st_mtime_ns,
        observed.st_ctime_ns,
    )


def _read_final_report_at(directory: int, expected: bytes) -> None:
    try:
        descriptor = os.open(
            "report.json", _regular_open_flags(os.O_RDONLY), dir_fd=directory,
        )
    except OSError as exc:
        raise ValueError("published report cannot be held") from exc
    try:
        _require_bound_regular_name(
            directory, "report.json", descriptor, "published report",
        )
        before = os.fstat(descriptor)
        if before.st_size != len(expected) or before.st_size > 16 * 1024 * 1024:
            raise ValueError("published report size is invalid")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise ValueError("published report read made no progress")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ValueError("published report grew during verification")
        after = os.fstat(descriptor)
        if _regular_identity(before) != _regular_identity(after):
            raise ValueError("published report identity changed during verification")
        if b"".join(chunks) != expected:
            raise ValueError("published report changed during publication")
    finally:
        os.close(descriptor)


def _atomic_write_report(
    path: Path, payload: bytes, *, private_root: Path,
) -> None:
    if (
        type(payload) is not bytes or not payload
        or len(payload) > 16 * 1024 * 1024
    ):
        raise ValueError("public report bytes are invalid")
    path = path.absolute()
    private_root = private_root.absolute()
    if path.name != "report.json":
        raise ValueError("only the fixed report filename may be published")
    try:
        relative = path.parent.relative_to(private_root)
    except ValueError as exc:
        raise ValueError("report is outside its private benchmark root") from exc
    components = (private_root.name, *relative.parts)
    if any(
        type(name) is not str or not name or name in {".", ".."}
        or "/" in name or "\\" in name
        for name in components
    ):
        raise ValueError("private benchmark components are invalid")
    try:
        anchor = os.open(private_root.parent, _directory_open_flags())
    except OSError as exc:
        raise ValueError("benchmark root parent cannot be held") from exc
    directories = [anchor]
    lock_descriptor: int | None = None
    temporary_name: str | None = None
    try:
        _require_anchor_directory(anchor)
        for component in components:
            directories.append(
                _open_private_directory_at(directories[-1], component)
            )
        final_directory = directories[-1]
        lock_descriptor = _open_publication_lock(final_directory)
        fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
        _require_bound_regular_name(
            final_directory, ".report.lock", lock_descriptor,
            "publication lock",
        )
        _validate_existing_report_at(final_directory)
        temporary_name = f".report.{secrets.token_hex(16)}.tmp"
        try:
            descriptor = os.open(
                temporary_name,
                _regular_open_flags(os.O_WRONLY) | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=final_directory,
            )
        except OSError as exc:
            raise ValueError("temporary report cannot be created") from exc
        try:
            os.fchmod(descriptor, 0o600)
            _require_private_regular_stat(
                os.fstat(descriptor), "temporary report",
            )
            offset = 0
            while offset < len(payload):
                written = os.write(descriptor, payload[offset:])
                if written < 1:
                    raise OSError("private report write made no progress")
                offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _validate_existing_report_at(final_directory)
        _require_bound_private_directory_chain(directories, components)
        os.replace(
            temporary_name,
            "report.json",
            src_dir_fd=final_directory,
            dst_dir_fd=final_directory,
        )
        temporary_name = None
        os.fsync(final_directory)
        _read_final_report_at(final_directory, payload)
        _require_bound_private_directory_chain(directories, components)
    finally:
        if temporary_name is not None and len(directories) == len(components) + 1:
            try:
                os.unlink(temporary_name, dir_fd=directories[-1])
            except FileNotFoundError:
                pass
        if lock_descriptor is not None:
            try:
                fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
            finally:
                os.close(lock_descriptor)
        for descriptor in reversed(directories):
            os.close(descriptor)


def main(argv: list[str] | None = None) -> None:
    _parser().parse_args(argv)
    report = validate_public_report(run_benchmark())
    _require_publication_edge(report)
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
