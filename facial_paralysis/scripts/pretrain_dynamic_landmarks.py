"""Receipt-bound two-stage dynamic-landmark pretraining runner."""
from __future__ import annotations

import argparse
import ctypes
import errno
import fcntl
import hashlib
import hmac
import io
import json
import math
import os
import re
import secrets
import stat
import statistics
import sys
import time
from collections.abc import Mapping
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRETRAINING_ROOT = (
    PROJECT_ROOT / "outputs" / "dynamic_landmark" / "pretraining"
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if __name__ == "__main__":
    from scripts import pretrain_dynamic_landmarks as canonical_runner

    canonical_runner._entrypoint()

from src.pretraining import dynamic_landmark_ssl as ssl_core  # noqa: E402
from src.models.dynamic_landmark import (  # noqa: E402
    ARM_BLENDSHAPE,
    ARM_FUSION,
    ARM_LANDMARK,
)


_RUN_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_MAX_EXACT_RESULT_TREE_ENTRIES = 64
_MAX_EXACT_RESULT_TREE_DEPTH = 4
_MAX_EXACT_RESULT_TREE_REGULAR_BYTES = 128 * 1024 * 1024
_ABLATION_ARMS = (ARM_BLENDSHAPE, ARM_LANDMARK, ARM_FUSION)
_ABLATION_BASELINES = ("trained", "fresh_untrained", "train_mean")
_FORMAL_SEEDS = (0, 1, 2)
_MAYO_ABLATION_NAMESPACE = (
    PRETRAINING_ROOT / "ablation" / "mayo-input-arm-v1"
)
_FOCUSED_NAMESPACE = (
    PRETRAINING_ROOT / "development" / "focused-modality-v1"
)
_FOCUSED_BRIDGE_ROOT = PRETRAINING_ROOT / "bridge"
_FOCUSED_MAYO_KEY = PRETRAINING_ROOT / ".mayo_ssl_hmac.key"
_FOCUSED_ARMS = (ARM_BLENDSHAPE, ARM_LANDMARK, ARM_FUSION)
_FOCUSED_EXPERIMENTS = {
    "smoke": "focused_modality_smoke_v1",
    "select": "focused_modality_selection_v1",
    "winner": "focused_modality_winner_v1",
}
_FOCUSED_PRIMARY_METRIC = (
    "common_target_metrics.trained.raw_mae.equal_block_macro"
)
_FOCUSED_METRIC_QUANTIZATION_POLICY = {
    "name": "decimal_round_half_even_v1",
    "decimal_places": 7,
}
_FOCUSED_CHECKPOINT_SCHEMA = "focused_mayo_checkpoint_v2"
_FOCUSED_CHECKPOINT_RECEIPT_SCHEMA = "focused_mayo_checkpoint_receipt_v2"
_FOCUSED_REPORT_SCHEMAS = {
    "smoke": "focused_mayo_smoke_report_v1",
    "select": "focused_mayo_selection_report_v1",
    "winner": "focused_mayo_winner_report_v1",
    "audit": "focused_mayo_audit_report_v1",
}
_FOCUSED_CONFIG_FIELDS = {
    "schema_version", "stage", "mode", "source", "objective",
    "sample_rate_hz", "seeds", "development_only", "optimizer",
    "learning_rate", "weight_decay", "epochs", "batch_policy",
    "span_length", "spans_per_window", "device",
    "bridge_receipt_sha256", "receipt_hmac", "experiment_kind",
    "input_arm", "input_active_indices", "target_schema",
    "initialization_policy", "producer_sha256",
    "mayo_generation_commitment_sha256", "heldout_mask_policy",
}
_FOCUSED_CHECKPOINT_RECEIPT_FIELDS = {
    "schema_version", "checkpoint_name", "checkpoint_file_sha256",
    "checkpoint_file_size_bytes", "checkpoint_file_identity_sha256",
    "checkpoint_fingerprint", "metadata_sha256", "phase", "arm", "seed",
    "epochs", "dependency_commitment_sha256",
    "bridge_generation_sha256", "bridge_producer_sha256", "trainer_sha256",
    "bundle_sha256", "common_contract_sha256", "target_schema",
    "receipt_sha256", "authority_hmac",
}
_FOCUSED_BRIDGE_STAGE_FIELDS = {
    "adapter_sha256", "bundle_file_count", "bundle_sha256",
    "bundle_size_bytes", "cache_integrity_ids", "closure_hmac",
    "covered_canonical_position_count", "exclusion_count",
    "feature_names_sha256", "group_ids", "original_mapping_sha256",
    "overlap_pair_count", "packet_policy", "producer_sha256",
    "sample_count", "sample_ids", "schema", "source_schema",
    "source_unit_count", "source_unit_ids", "stage",
    "unique_group_count", "upstream_cache_count",
    "upstream_generation_closure_hmac", "upstream_manifest_commitments",
    "window_starts",
}


class _FocusedBridgeAuthorization:
    """Opaque, locally authenticated bridge facts used by focused phases."""

    __slots__ = (
        "stage", "producer_sha256", "trainer_sha256",
        "bridge_generation_sha256",
        "bundle_sha256", "bundle_size_bytes", "feature_width",
        "exclusion_count", "sample_count", "group_ids", "source_unit_ids",
        "sample_ids", "cache_integrity_ids", "window_starts", "bundle_path",
        "key_path", "key_file_identity_sha256",
        "generation_commitment_sha256", "stage_record", "private_key",
    )

    def __init__(self, **values: object) -> None:
        if set(values) != set(self.__slots__):
            raise ValueError("focused bridge authorization fields are not exact")
        for name in self.__slots__:
            object.__setattr__(self, name, values[name])

    def __repr__(self) -> str:
        return "_FocusedBridgeAuthorization(<redacted>)"


def _authorization_factories(
    args: argparse.Namespace,
):
    from scripts import prepare_dynamic_landmark_ssl_inputs as inputs_cli

    return inputs_cli._authorization_factories(args)


def _transaction_authorizer(authorizer: Callable[[], object]):
    """Run one expensive live authorizer once inside a transaction boundary."""
    missing = object()
    authorization: object = missing

    def authorize():
        nonlocal authorization
        if authorization is missing:
            authorization = authorizer()
        return authorization

    return authorize


def _require_publication_edge_authorization(
    *,
    stage: str,
    evidence,
    authorization: object,
) -> None:
    """Compare one fresh live generation closure to its frozen stage receipt."""
    if stage == "ravdess":
        expected_schema = evidence.source_schema
        commitments = {
            "manifest_sha256": getattr(
                authorization, "manifest_sha256", None,
            ),
        }
    elif stage == "mayo":
        expected_schema = "mayo_mediapipe_clinical23_ssl_v2"
        generation_commitment = getattr(authorization, "commitment", None)
        if not isinstance(generation_commitment, Mapping):
            raise ValueError("live Mayo generation commitment is unavailable")
        commitments = {
            "collection_manifest_sha256": getattr(
                authorization, "collection_manifest_sha256", None,
            ),
            "exposure_manifest_sha256": getattr(
                authorization, "exposure_manifest_sha256", None,
            ),
            "generation_commitment_sha256": ssl_core._canonical_sha256(
                generation_commitment,
            ),
        }
    else:
        raise ValueError("publication-edge SSL stage is unsupported")
    if (
        getattr(authorization, "schema", None) != expected_schema
        or ssl_core._canonical_sha256(commitments)
        != evidence.upstream_manifest_commitments_sha256
        or getattr(authorization, "generation_closure_hmac", None)
        != evidence.upstream_generation_closure_hmac
        or getattr(authorization, "key_file_identity_sha256", None)
        != evidence.canonical_key_identity_sha256
    ):
        raise ValueError("live SSL generation changed before publication")


def _producer_sha256() -> str:
    from scripts import prepare_dynamic_landmark_ssl_inputs as inputs_cli

    return inputs_cli._producer_sha256()


def _focused_trainer_sha256() -> str:
    """Bind focused code without changing the canonical bridge producer set."""
    digest = hashlib.sha256()
    digest.update(b"focused-mayo-trainer-v1\0")
    sources = (
        Path(__file__).resolve(),
        PROJECT_ROOT / "src" / "pretraining" / "dynamic_landmark_ssl.py",
        PROJECT_ROOT / "src" / "models" / "dynamic_landmark.py",
    )
    for source in sources:
        _resolved, payload, source_sha256, identity = (
            ssl_core._regular_file_snapshot_with_identity(
                source, "focused trainer source", max_bytes=4 * 1024 * 1024,
            )
        )
        if identity.uid != os.geteuid() or identity.mode & 0o022:
            raise ValueError("focused trainer source storage is unsafe")
        label = source.relative_to(PROJECT_ROOT).as_posix().encode("ascii")
        digest.update(len(label).to_bytes(8, "big"))
        digest.update(label)
        digest.update(source_sha256.encode("ascii"))
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _privacy_forbidden(
    args: argparse.Namespace,
    ravdess_authorizer,
    mayo_authorizer,
):
    from scripts import prepare_dynamic_landmark_ssl_inputs as inputs_cli

    ravdess_inventory, mayo_inventory = inputs_cli._live_privacy_inventories(args)
    return inputs_cli._build_live_forbidden_tokens(
        mayo_roots=(args.mayo_data_root, args.mayo_existing_export_root),
        ravdess_authorization=ravdess_authorizer(),
        mayo_authorization=mayo_authorizer(),
        ravdess_inventory=ravdess_inventory,
        mayo_inventory=mayo_inventory,
    )


def _scan_private_results(roots: tuple[Path, ...], forbidden) -> None:
    from scripts import prepare_dynamic_landmark_ssl_inputs as inputs_cli

    modes_ok, privacy_ok, non_0600 = inputs_cli._scan_private_trees(
        roots, forbidden=forbidden,
    )
    if not modes_ok or not privacy_ok or non_0600 != 0:
        raise ValueError("private SSL result verification failed")


def _assert_summary_private(payload: bytes, forbidden) -> None:
    from scripts import prepare_dynamic_landmark_ssl_inputs as inputs_cli

    matcher = inputs_cli._ByteMatcher(forbidden.tokens)
    _, leaked = matcher.feed(payload)
    if leaked:
        raise ValueError("private fact entered the SSL summary")


def _quiet_call(function, /, *args, **kwargs):
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        value = function(*args, **kwargs)
    if stdout.getvalue() or stderr.getvalue():
        raise ValueError("pretraining dependency emitted unexpected output")
    return value


def _authorize_focused_bridge(
    bridge_root: Path,
    mayo_key: Path,
    *,
    producer_sha256: str,
) -> _FocusedBridgeAuthorization:
    """Authenticate the committed bridge locally, without source authorization."""
    from scripts import prepare_dynamic_landmark_ssl_inputs as inputs_cli
    from src.pretraining import dynamic_landmark_ssl_bridge as bridge_core

    trainer_sha256 = ssl_core._require_sha256(
        producer_sha256, "focused trainer",
    )
    bridge = _private_directory(bridge_root.absolute(), "focused bridge")
    if set(os.listdir(bridge)) != {"bundles", "bundle_generation.json"}:
        raise ValueError("focused bridge top-level schema is not exact")
    bundles = _private_directory(bridge / "bundles", "focused bridge bundles")
    if set(os.listdir(bundles)) != {"mayo_bundle.npz", "ravdess_bundle.npz"}:
        raise ValueError("focused bridge bundle set is not exact")
    _, generation_bytes, _, _ = ssl_core._private_regular_file_snapshot(
        bridge / "bundle_generation.json", "focused bridge generation",
    )
    generation = ssl_core._strict_json_mapping(
        generation_bytes, "focused bridge generation",
    )
    if set(generation) != {
        "schema", "producer_sha256", "stages",
        "dual_stage_closure_sha256", "dual_stage_closure_hmac",
    } or generation.get("schema") != "dynamic_landmark_bridge_generation_v1":
        raise ValueError("focused bridge generation schema is not exact")
    bridge_producer_sha256 = ssl_core._require_sha256(
        generation.get("producer_sha256"), "focused bridge producer",
    )
    stages = generation.get("stages")
    dual_hmac = generation.get("dual_stage_closure_hmac")
    if (
        type(stages) is not dict or set(stages) != {"ravdess", "mayo"}
        or type(dual_hmac) is not dict
        or set(dual_hmac) != {"ravdess", "mayo"}
    ):
        raise ValueError("focused bridge dual-stage closure is incomplete")
    mayo_record = stages["mayo"]
    if (
        type(mayo_record) is not dict
        or set(mayo_record) != _FOCUSED_BRIDGE_STAGE_FIELDS
        or mayo_record.get("schema") != "dynamic_landmark_bridge_stage_v1"
        or mayo_record.get("stage") != "mayo"
        or mayo_record.get("producer_sha256") != bridge_producer_sha256
    ):
        raise ValueError("focused Mayo bridge stage schema is not exact")
    key_path = mayo_key.absolute()
    key_authorization, key = ssl_core._private_file_snapshot(
        key_path, "focused Mayo canonical key", max_bytes=32,
    )
    if len(key) != 32:
        raise ValueError("focused Mayo canonical key is not exactly 32 bytes")
    key_identity = inputs_cli._key_identity_sha256(key_path, key)

    unsigned_stage = dict(mayo_record)
    observed_stage_hmac = unsigned_stage.pop("closure_hmac", None)
    expected_stage_hmac = hmac.new(
        key,
        b"dynamic-landmark-bridge-stage-closure-v1\0"
        + bridge_core._json_bytes(unsigned_stage),
        hashlib.sha256,
    ).hexdigest()
    if (
        type(observed_stage_hmac) is not str
        or not hmac.compare_digest(observed_stage_hmac, expected_stage_hmac)
    ):
        raise ValueError("focused Mayo bridge stage HMAC is invalid")
    dual_digest = hashlib.sha256()
    dual_digest.update(b"dynamic-landmark-bridge-dual-stage-v1\0")
    for name in ("ravdess", "mayo"):
        record = stages[name]
        if type(record) is not dict or type(record.get("closure_hmac")) is not str:
            raise ValueError("focused bridge stage closure is malformed")
        dual_digest.update(name.encode("ascii") + b"\0")
        dual_digest.update(record["closure_hmac"].encode("ascii") + b"\n")
    if generation.get("dual_stage_closure_sha256") != dual_digest.hexdigest():
        raise ValueError("focused bridge dual-stage digest is invalid")
    unsigned_generation = dict(generation)
    unsigned_generation.pop("dual_stage_closure_hmac")
    dual_material = (
        b"dynamic-landmark-bridge-dual-stage-keyed-v1\0"
        + bridge_core._json_bytes(unsigned_generation)
    )
    expected_dual_hmac = hmac.new(
        key, dual_material, hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(
        str(dual_hmac.get("mayo")), expected_dual_hmac,
    ):
        raise ValueError("focused bridge dual-stage Mayo HMAC is invalid")

    bundle_path = bundles / "mayo_bundle.npz"
    bundle_authorization, bundle_payload = (
        ssl_core._private_file_snapshot(
            bundle_path, "focused Mayo bridge bundle",
            max_bytes=100 * 1024 * 1024,
        )
    )
    bundle_sha256 = bundle_authorization.sha256
    bundle_identity = bundle_authorization.identity
    bridge_core._validate_bundle_payload(
        bundle_payload, stage="mayo", record=mayo_record,
    )
    if (
        bundle_sha256 != mayo_record.get("bundle_sha256")
        or bundle_identity.size != mayo_record.get("bundle_size_bytes")
    ):
        raise ValueError("focused Mayo bundle contradicts its keyed closure")

    def exact_strings(name: str) -> tuple[str, ...]:
        value = mayo_record.get(name)
        if (
            type(value) is not list or not value
            or any(type(item) is not str or not item for item in value)
        ):
            raise ValueError(f"focused Mayo {name} are malformed")
        return tuple(value)

    group_ids = exact_strings("group_ids")
    source_units = exact_strings("source_unit_ids")
    sample_ids = exact_strings("sample_ids")
    cache_ids = exact_strings("cache_integrity_ids")
    sample_count = mayo_record.get("sample_count")
    starts_value = mayo_record.get("window_starts")
    if (
        isinstance(sample_count, bool) or type(sample_count) is not int
        or sample_count < 2
        or len(group_ids) != sample_count
        or len(source_units) != sample_count
        or len(sample_ids) != sample_count
        or len(cache_ids) != sample_count
        or len(set(sample_ids)) != sample_count
        or type(starts_value) is not list
        or len(starts_value) != sample_count
    ):
        raise ValueError("focused Mayo bridge aggregate counts are inconsistent")
    starts: list[tuple[int, ...]] = []
    for packet in starts_value:
        if (
            type(packet) is not list or len(packet) != 4
            or any(isinstance(item, bool) or type(item) is not int or item < 0
                   for item in packet)
        ):
            raise ValueError("focused Mayo window starts are noncanonical")
        normalized = tuple(packet)
        starts.append(normalized)
    overlap_pair_count = mayo_record.get("overlap_pair_count")
    if (
        isinstance(overlap_pair_count, bool)
        or type(overlap_pair_count) is not int
        or overlap_pair_count < 0
        or mayo_record.get("bundle_file_count") != 1
        or mayo_record.get("exclusion_count") != 2
        or mayo_record.get("source_unit_count") != len(set(source_units))
        or mayo_record.get("unique_group_count") != len(set(group_ids))
        or mayo_record.get("upstream_cache_count") != len(set(cache_ids))
    ):
        raise ValueError("focused Mayo bridge grouping or exclusion is inconsistent")
    commitments = mayo_record.get("upstream_manifest_commitments")
    if type(commitments) is not dict:
        raise ValueError("focused Mayo upstream commitments are unavailable")
    generation_commitment = ssl_core._require_sha256(
        commitments.get("generation_commitment_sha256"),
        "focused Mayo generation commitment",
    )
    return _FocusedBridgeAuthorization(
        stage="mayo",
        producer_sha256=bridge_producer_sha256,
        trainer_sha256=trainer_sha256,
        bridge_generation_sha256=hashlib.sha256(generation_bytes).hexdigest(),
        bundle_sha256=bundle_sha256,
        bundle_size_bytes=bundle_identity.size,
        feature_width=95,
        exclusion_count=2,
        sample_count=sample_count,
        group_ids=group_ids,
        source_unit_ids=source_units,
        sample_ids=sample_ids,
        cache_integrity_ids=cache_ids,
        window_starts=tuple(starts),
        bundle_path=bundle_path,
        key_path=key_path,
        key_file_identity_sha256=key_identity,
        generation_commitment_sha256=generation_commitment,
        stage_record=dict(mayo_record),
        private_key=key,
    )


def _load_focused_tensors(
    authorization: _FocusedBridgeAuthorization,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not isinstance(authorization, _FocusedBridgeAuthorization):
        raise ValueError("focused bridge authorization is unavailable")
    bundle_authorization, payload = ssl_core._private_file_snapshot(
        authorization.bundle_path,
        "focused Mayo bridge bundle",
        max_bytes=100 * 1024 * 1024,
    )
    digest = bundle_authorization.sha256
    identity = bundle_authorization.identity
    if (
        digest != authorization.bundle_sha256
        or identity.size != authorization.bundle_size_bytes
    ):
        raise ValueError("focused Mayo bundle changed after authorization")
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as cached:
            features = np.asarray(cached["features"]).copy()
            valid = np.asarray(cached["valid_mask"]).copy()
            timestamps = np.asarray(cached["timestamps"]).copy()
            source_indices = np.asarray(cached["source_frame_indices"]).copy()
    except (OSError, EOFError, KeyError, TypeError, ValueError) as exc:
        raise ValueError("focused Mayo bundle cannot be loaded safely") from exc
    if (
        features.dtype != np.float32
        or features.shape != (authorization.sample_count, 4, 32, 95)
        or valid.dtype != np.bool_
        or valid.shape != features.shape[:-1]
        or not np.isfinite(features).all()
    ):
        raise ValueError("focused Mayo tensors violate the full-95 contract")
    return (
        torch.from_numpy(features), torch.from_numpy(valid),
        torch.from_numpy(timestamps), torch.from_numpy(source_indices),
    )


def _require_focused_authority_unchanged(
    authorization: _FocusedBridgeAuthorization,
) -> _FocusedBridgeAuthorization:
    if not isinstance(authorization, _FocusedBridgeAuthorization):
        raise ValueError("focused bridge authority is unavailable")
    fresh = _authorize_focused_bridge(
        authorization.bundle_path.parent.parent,
        authorization.key_path,
        producer_sha256=authorization.trainer_sha256,
    )
    fields = (
        "stage", "producer_sha256", "trainer_sha256",
        "bridge_generation_sha256", "bundle_sha256", "bundle_size_bytes",
        "feature_width", "exclusion_count", "sample_count", "group_ids",
        "source_unit_ids", "sample_ids", "cache_integrity_ids",
        "window_starts", "key_file_identity_sha256",
        "generation_commitment_sha256", "stage_record",
    )
    if any(getattr(fresh, name) != getattr(authorization, name) for name in fields):
        raise ValueError("focused bridge authority changed after authorization")
    if not hmac.compare_digest(fresh.private_key, authorization.private_key):
        raise ValueError("focused canonical authority changed after authorization")
    return fresh


def _focused_common_contract(
    authorization: _FocusedBridgeAuthorization,
) -> dict[str, object]:
    """Derive one recording split, unique-frame scaler, and heldout mask."""
    features_tensor, valid_tensor, timestamps_tensor, source_indices_tensor = (
        _load_focused_tensors(authorization)
    )
    features = features_tensor.numpy()
    valid = valid_tensor.numpy()

    ordered_groups = tuple(dict.fromkeys(authorization.group_ids))
    permutation = np.random.default_rng(0).permutation(len(ordered_groups))
    heldout_group_count = min(
        len(ordered_groups) - 1,
        max(1, math.ceil(len(ordered_groups) * 0.20)),
    )
    heldout_groups = {
        ordered_groups[int(index)]
        for index in permutation[:heldout_group_count]
    }
    train_indices = np.asarray([
        index for index, group in enumerate(authorization.group_ids)
        if group not in heldout_groups
    ], dtype=np.int64)
    heldout_indices = np.asarray([
        index for index, group in enumerate(authorization.group_ids)
        if group in heldout_groups
    ], dtype=np.int64)
    split = ssl_core.SSLGroupSplit(
        train_indices=train_indices,
        heldout_indices=heldout_indices,
        unit="recording",
        claim_unit="recording_held_out_not_patient_held_out",
        patient_held_out=False,
    )
    ssl_core._validate_split_partition(split, authorization.group_ids)

    seen: dict[tuple[str, int], np.ndarray] = {}
    rows: list[np.ndarray] = []
    for sample_index in train_indices.tolist():
        source_unit = authorization.source_unit_ids[sample_index]
        starts = authorization.window_starts[sample_index]
        for window_index, start in enumerate(starts):
            for frame_index in range(32):
                if not bool(valid[sample_index, window_index, frame_index]):
                    continue
                key = (source_unit, start + frame_index)
                row = features[
                    sample_index, window_index, frame_index,
                ].astype(np.float64, copy=True)
                prior = seen.get(key)
                if prior is not None:
                    if not np.array_equal(prior, row):
                        raise ValueError(
                            "focused Mayo repeated canonical frame is inconsistent"
                        )
                    continue
                seen[key] = row
                rows.append(row)
    if not rows:
        raise ValueError("focused Mayo train-only scaler has no observations")
    stacked = np.stack(rows)
    mean = stacked.mean(axis=0, dtype=np.float64)
    scale = stacked.std(axis=0, dtype=np.float64)
    scale[scale < np.finfo(np.float32).eps] = 1.0
    scaler = ssl_core.SourceScaler(
        source=ssl_core.MAYO_SOURCE,
        mean=torch.from_numpy(mean.copy()),
        scale=torch.from_numpy(scale.copy()),
        fit_indices=tuple(int(index) for index in train_indices.tolist()),
    )
    ssl_core._validate_scaler_artifact(
        scaler,
        source=ssl_core.MAYO_SOURCE,
        train_indices=train_indices,
        feature_width=95,
    )
    heldout_tensor = torch.as_tensor(heldout_indices, dtype=torch.int64)
    heldout_mask = ssl_core.make_contiguous_span_mask(
        valid_tensor.index_select(0, heldout_tensor),
        timestamps_tensor.index_select(0, heldout_tensor),
        source_indices_tensor.index_select(0, heldout_tensor),
        expected_source_step=1,
        span_length=4,
        spans_per_window=2,
        seed=10_000,
    )
    result = {
        "split": split,
        "scaler": scaler,
        "heldout_mask": heldout_mask,
        "heldout_mask_sha256": ssl_core._mask_sha256(heldout_mask),
        "split_sha256": ssl_core._canonical_sha256({
            "train": train_indices.tolist(),
            "heldout": heldout_indices.tolist(),
        }),
        "scaler_sha256": ssl_core._scaler_sha256(scaler),
        "scaler_policy": "focused_train_unique_canonical_frame_scaler_v1",
        "fit_unique_frame_count": len(seen),
        "target_schema": ssl_core.TARGET_FULL95,
    }
    public_contract = {
        "schema_version": "focused_mayo_common_contract_v1",
        "bridge_generation_sha256": authorization.bridge_generation_sha256,
        "bridge_producer_sha256": authorization.producer_sha256,
        "bundle_sha256": authorization.bundle_sha256,
        "bundle_size_bytes": authorization.bundle_size_bytes,
        "group_order_sha256": ssl_core._canonical_sha256(
            list(authorization.group_ids),
        ),
        "split_policy": "first_seen_groups_rng0_ceil20_recording_v1",
        "split_sha256": result["split_sha256"],
        "scaler_policy": result["scaler_policy"],
        "scaler_sha256": result["scaler_sha256"],
        "fit_unique_frame_count": result["fit_unique_frame_count"],
        "heldout_mask_policy": "focused_common_heldout_mask_seed_10000_v1",
        "heldout_mask_sha256": result["heldout_mask_sha256"],
        "target_schema": result["target_schema"],
        "metric_quantization_policy": dict(
            _FOCUSED_METRIC_QUANTIZATION_POLICY,
        ),
    }
    result["public_contract"] = public_contract
    result["common_contract_sha256"] = ssl_core._canonical_sha256(
        public_contract,
    )
    return result


def _focused_runtime_environment() -> dict[str, str]:
    """Return the only legal focused device policy and portable runtime facts."""
    device_type = "cuda" if bool(torch.cuda.is_available()) else "cpu"
    cuda_version = (
        "none" if torch.version.cuda is None else str(torch.version.cuda)
    )
    return {
        "device_policy": "cuda_if_available_else_cpu_v1",
        "device_type": device_type,
        "torch_version": str(torch.__version__),
        "cuda_runtime_version": cuda_version,
    }


def _validate_focused_runtime_environment(
    value: object,
) -> dict[str, str]:
    if (
        type(value) is not dict
        or set(value) != {
            "device_policy", "device_type", "torch_version",
            "cuda_runtime_version",
        }
        or value.get("device_policy") != "cuda_if_available_else_cpu_v1"
        or value.get("device_type") not in {"cpu", "cuda"}
        or any(type(value.get(name)) is not str or not value.get(name)
               for name in ("torch_version", "cuda_runtime_version"))
    ):
        raise ValueError("focused runtime environment is not exact")
    return dict(value)  # type: ignore[return-value]


@contextmanager
def _focused_deterministic_runtime(
    runtime_environment: Mapping[str, str],
) -> Iterator[None]:
    runtime = _validate_focused_runtime_environment(runtime_environment)
    previous_algorithms = torch.are_deterministic_algorithms_enabled()
    previous_deterministic = torch.backends.cudnn.deterministic
    previous_benchmark = torch.backends.cudnn.benchmark
    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if runtime["device_type"] == "cuda":
            configured = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
            if configured not in {None, ":4096:8"}:
                raise ValueError("CUDA deterministic workspace policy conflicts")
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        yield
    finally:
        torch.use_deterministic_algorithms(previous_algorithms)
        torch.backends.cudnn.deterministic = previous_deterministic
        torch.backends.cudnn.benchmark = previous_benchmark


def _focused_training_config(
    phase: str,
    *,
    arm: str,
    producer_sha256: str,
    mayo_generation_commitment_sha256: str,
    bridge_receipt_sha256: str,
    receipt_hmac: str,
    runtime_environment: Mapping[str, str] | None = None,
) -> dict[str, object]:
    contract = _focused_job_contract(
        phase,
        selected_arm=arm if phase == "winner" else None,
    )
    if arm not in contract["arms"]:
        raise ValueError("focused phase arm is outside its fixed schedule")
    active = ssl_core.validate_ssl_input_arm("mayo", arm)
    mode = "smoke" if phase == "smoke" else "formal"
    runtime = _validate_focused_runtime_environment(
        _focused_runtime_environment()
        if runtime_environment is None else runtime_environment,
    )
    value = {
        "schema_version": ssl_core.SSL_CONFIG_V3_SCHEMA,
        "stage": "mayo",
        "mode": mode,
        "source": ssl_core.MAYO_SOURCE,
        "objective": "masked_span_smooth_l1_only",
        "sample_rate_hz": 30.0,
        "seeds": list(contract["seeds"]),
        "development_only": True,
        "optimizer": "adamw",
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "epochs": int(contract["epochs"]),
        "batch_policy": "deterministic_microbatch_full_partition_64",
        "span_length": 4,
        "spans_per_window": 2,
        "device": runtime["device_type"],
        "bridge_receipt_sha256": ssl_core._require_sha256(
            bridge_receipt_sha256, "focused bridge receipt",
        ),
        "receipt_hmac": ssl_core._require_sha256(
            receipt_hmac, "focused bridge receipt HMAC",
        ),
        "experiment_kind": _FOCUSED_EXPERIMENTS[phase],
        "input_arm": arm,
        "input_active_indices": list(active),
        "target_schema": ssl_core.TARGET_FULL95,
        "initialization_policy": "same_seed_fresh",
        "producer_sha256": ssl_core._require_sha256(
            producer_sha256, "focused bridge producer",
        ),
        "mayo_generation_commitment_sha256": ssl_core._require_sha256(
            mayo_generation_commitment_sha256,
            "focused Mayo generation commitment",
        ),
        "heldout_mask_policy": (
            "not_evaluated_smoke_v1" if phase == "smoke"
            else "focused_common_heldout_mask_seed_10000_v1"
        ),
    }
    return _validate_focused_training_config(value)


def _validate_focused_training_config(
    value: Mapping[str, object],
) -> dict[str, object]:
    """Validate the separate fixed development protocol, never formal aliases."""
    if type(value) is not dict or set(value) != _FOCUSED_CONFIG_FIELDS:
        raise ValueError("focused Mayo training config schema is not exact")
    experiment = value.get("experiment_kind")
    inverse = {name: phase for phase, name in _FOCUSED_EXPERIMENTS.items()}
    if experiment not in inverse:
        raise ValueError("focused Mayo experiment is unsupported")
    phase = inverse[str(experiment)]
    arm = value.get("input_arm")
    if arm not in _FOCUSED_ARMS:
        raise ValueError("focused Mayo input arm is unsupported")
    contract = _focused_job_contract(
        phase, selected_arm=str(arm) if phase == "winner" else None,
    )
    if arm not in contract["arms"]:
        raise ValueError("focused Mayo phase and arm are incompatible")
    mode = "smoke" if phase == "smoke" else "formal"
    expected = {
        "schema_version": ssl_core.SSL_CONFIG_V3_SCHEMA,
        "stage": "mayo",
        "mode": mode,
        "source": ssl_core.MAYO_SOURCE,
        "objective": "masked_span_smooth_l1_only",
        "sample_rate_hz": 30.0,
        "seeds": contract["seeds"],
        "development_only": True,
        "optimizer": "adamw",
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "epochs": contract["epochs"],
        "batch_policy": "deterministic_microbatch_full_partition_64",
        "span_length": 4,
        "spans_per_window": 2,
        "device": value.get("device"),
        "experiment_kind": experiment,
        "input_arm": arm,
        "input_active_indices": list(ssl_core.validate_ssl_input_arm(
            "mayo", str(arm),
        )),
        "target_schema": ssl_core.TARGET_FULL95,
        "initialization_policy": "same_seed_fresh",
        "heldout_mask_policy": (
            "not_evaluated_smoke_v1" if phase == "smoke"
            else "focused_common_heldout_mask_seed_10000_v1"
        ),
    }
    for name, expected_value in expected.items():
        if not ssl_core._exact_json_value(value.get(name), expected_value):
            raise ValueError("focused Mayo training config is not fixed")
    if value.get("device") not in {"cpu", "cuda"}:
        raise ValueError("focused Mayo device must come from auto policy")
    for name in (
        "bridge_receipt_sha256", "receipt_hmac", "producer_sha256",
        "mayo_generation_commitment_sha256",
    ):
        ssl_core._require_sha256(value.get(name), f"focused config {name}")
    return dict(value)


def _canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    return ssl_core._canonical_json_bytes(dict(value))


def _focused_hmac(
    domain: str, value: Mapping[str, object], private_key: bytes,
) -> str:
    if type(domain) is not str or not domain or len(private_key) != 32:
        raise ValueError("focused authority material is invalid")
    return hmac.new(
        private_key,
        b"focused-mayo-" + domain.encode("ascii") + b"-v1\0"
        + _canonical_json_bytes(value),
        hashlib.sha256,
    ).hexdigest()


def _sign_focused_report(
    phase: str, core: dict[str, object], private_key: bytes,
) -> dict[str, object]:
    if phase not in _FOCUSED_REPORT_SCHEMAS or type(core) is not dict:
        raise ValueError("focused report phase is unsupported")
    if (
        core.get("schema_version") != _FOCUSED_REPORT_SCHEMAS[phase]
        or core.get("phase") != phase
        or "authority_hmac" in core
    ):
        raise ValueError("focused report core is not phase-exact")
    report = dict(core)
    report["authority_hmac"] = _focused_hmac(
        f"{phase}-report", core, private_key,
    )
    encoded = _canonical_json_bytes(report)
    if b"/" in encoded or b"\\" in encoded:
        raise ValueError("focused report contains path-like material")
    return report


def _validate_focused_report_bytes(
    payload: bytes,
    *,
    phase: str,
    private_key: bytes,
) -> dict[str, object]:
    report = ssl_core._strict_json_mapping(payload, f"focused {phase} report")
    if _canonical_json_bytes(report) != payload:
        raise ValueError("focused report JSON is not canonical")
    observed = report.pop("authority_hmac", None)
    expected = _focused_hmac(f"{phase}-report", report, private_key)
    if (
        type(observed) is not str
        or not hmac.compare_digest(observed, expected)
        or report.get("schema_version") != _FOCUSED_REPORT_SCHEMAS.get(phase)
        or report.get("phase") != phase
    ):
        raise ValueError("focused report authority is invalid")
    report["authority_hmac"] = observed
    encoded = _canonical_json_bytes(report)
    if b"/" in encoded or b"\\" in encoded:
        raise ValueError("focused report contains path-like material")
    return report


def _write_private_bytes(path: Path, payload: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        os.fchmod(descriptor, 0o600)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written < 1:
                raise OSError("focused private write made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _focused_metric_bundle(value: object) -> dict[str, object]:
    if type(value) is not dict or set(value) != {
        "trained", "fresh_untrained", "train_mean",
    }:
        raise ValueError("focused metrics do not contain the exact baselines")
    normalized = {
        name: _normalize_ablation_metric(value[name])
        for name in ("trained", "fresh_untrained", "train_mean")
    }
    return {
        name: {
            "raw_mae": {
                metric: _canonical_focused_metric(observed)
                for metric, observed in normalized[name]["raw_mae"].items()
            },
            "standardized_mae": _canonical_focused_metric(
                normalized[name]["standardized_mae"],
            ),
            "standardized_smooth_l1": _canonical_focused_metric(
                normalized[name]["standardized_smooth_l1"],
            ),
        }
        for name in ("trained", "fresh_untrained", "train_mean")
    }


def _canonical_focused_metric(value: object) -> float:
    """Canonicalize cross-platform diagnostics before any decision or audit."""
    observed = _finite_nonnegative(value)
    places = _FOCUSED_METRIC_QUANTIZATION_POLICY["decimal_places"]
    quantum = Decimal(1).scaleb(-int(places))
    try:
        canonical = Decimal(str(observed)).quantize(
            quantum, rounding=ROUND_HALF_EVEN,
        )
    except InvalidOperation as exc:
        raise ValueError("focused metric cannot be canonicalized") from exc
    result = float(canonical)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError("focused canonical metric is invalid")
    return 0.0 if result == 0.0 else result


def _train_focused_job(
    authorization: _FocusedBridgeAuthorization,
    common: Mapping[str, object],
    *,
    phase: str,
    arm: str,
    seed: int,
) -> dict[str, object]:
    """Run one fixed fresh Mayo job without entering formal live authority."""
    authorization = _require_focused_authority_unchanged(authorization)
    contract = _focused_job_contract(
        phase, selected_arm=arm if phase == "winner" else None,
    )
    if (
        arm not in contract["arms"]
        or isinstance(seed, bool) or seed not in contract["seeds"]
        or type(common) is not dict
        or common.get("target_schema") != ssl_core.TARGET_FULL95
    ):
        raise ValueError("focused job is outside its fixed schedule")
    split = common.get("split")
    scaler = common.get("scaler")
    heldout_mask = common.get("heldout_mask")
    if (
        not isinstance(split, ssl_core.SSLGroupSplit)
        or not isinstance(scaler, ssl_core.SourceScaler)
        or not isinstance(heldout_mask, torch.Tensor)
        or common.get("common_contract_sha256")
        != ssl_core._canonical_sha256(common.get("public_contract"))
    ):
        raise ValueError("focused common contract is unauthorized")
    features, valid, timestamps, source_indices = _load_focused_tensors(
        authorization,
    )
    runtime = _focused_runtime_environment()
    device = torch.device(runtime["device_type"])
    train_indices, heldout_indices, _groups = ssl_core._validate_split_partition(
        split, authorization.group_ids,
    )
    config = _focused_training_config(
        phase,
        arm=arm,
        producer_sha256=authorization.producer_sha256,
        mayo_generation_commitment_sha256=(
            authorization.generation_commitment_sha256
        ),
        bridge_receipt_sha256=authorization.bridge_generation_sha256,
        receipt_hmac=str(authorization.stage_record["closure_hmac"]),
        runtime_environment=runtime,
    )
    train_rows = torch.as_tensor(train_indices, dtype=torch.int64)
    train_features = features.index_select(0, train_rows).to(device)
    train_valid = valid.index_select(0, train_rows).to(device)
    train_times = timestamps.index_select(0, train_rows).to(device)
    train_source_indices = source_indices.index_select(0, train_rows).to(device)
    train_scaled = scaler.transform(
        train_features, train_valid, source=ssl_core.MAYO_SOURCE,
    )
    epochs = int(contract["epochs"])
    train_trace: list[dict[str, object]] = []
    mask_schedule: list[dict[str, object]] = []
    rng_devices = [torch.cuda.current_device()] if device.type == "cuda" else []
    with _focused_deterministic_runtime(runtime), torch.random.fork_rng(
        devices=rng_devices,
    ):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        model = ssl_core.DynamicLandmarkSSLModel().to(device)
        pre_state = ssl_core._clone_model_state(model)
        pre_state_sha256 = ssl_core._model_state_sha256(pre_state)
        fresh_state_sha256, optimizer_initial_sha256 = (
            _fresh_ablation_initialization(seed)
        )
        if fresh_state_sha256 != pre_state_sha256:
            raise RuntimeError("focused fresh initialization is not reproducible")
        parameter_by_name = dict(model.named_parameters())
        trainable_names = ssl_core._trainable_parameter_names(model, "mayo")
        optimizer = torch.optim.AdamW(
            [parameter_by_name[name] for name in trainable_names],
            lr=0.001,
            weight_decay=0.0001,
        )
        if optimizer.state_dict().get("state") != {}:
            raise RuntimeError("focused AdamW is not initially empty")
        for epoch in range(epochs):
            mask_seed = seed + epoch * 100_003
            train_mask = ssl_core.make_contiguous_span_mask(
                train_valid,
                train_times,
                train_source_indices,
                expected_source_step=1,
                span_length=4,
                spans_per_window=2,
                seed=mask_seed,
            )
            mask_sha256 = ssl_core._mask_sha256(train_mask)
            mask_schedule.append({
                "epoch": epoch,
                "seed": mask_seed,
                "mask_sha256": mask_sha256,
            })
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = ssl_core._backward_full_partition_microbatches(
                model,
                features=train_scaled,
                valid_mask=train_valid,
                timestamps=train_times,
                source_frame_indices=train_source_indices,
                reconstruction_mask=train_mask,
                source="mayo",
                input_arm=arm,
            )
            loss_value = float(loss.detach().cpu().item())
            if not math.isfinite(loss_value) or loss_value < 0.0:
                raise RuntimeError("focused training produced a nonfinite loss")
            optimizer.step()
            train_trace.append({
                "epoch": epoch,
                "mask_sha256": mask_sha256,
                "loss": loss_value,
            })
        model.eval()
        post_state = ssl_core._clone_model_state(model)
        post_state_sha256 = ssl_core._model_state_sha256(post_state)
        if post_state_sha256 == pre_state_sha256:
            raise RuntimeError("focused training produced no state update")
        metrics: dict[str, object] | None = None
        heldout_evaluation_count = 0
        if bool(contract["evaluate_heldout"]):
            heldout_rows = torch.as_tensor(heldout_indices, dtype=torch.int64)
            # Evaluation is intentionally CPU-fixed so a CUDA-trained state can
            # be independently re-audited after transfer back to the Mac.
            heldout_features = features.index_select(0, heldout_rows)
            heldout_valid = valid.index_select(0, heldout_rows)
            heldout_times = timestamps.index_select(0, heldout_rows)
            heldout_source_indices = source_indices.index_select(
                0, heldout_rows,
            )
            heldout_scaled = scaler.transform(
                heldout_features, heldout_valid, source=ssl_core.MAYO_SOURCE,
            )
            expected_mask = ssl_core.make_contiguous_span_mask(
                heldout_valid,
                heldout_times,
                heldout_source_indices,
                expected_source_step=1,
                span_length=4,
                spans_per_window=2,
                seed=10_000,
            )
            heldout_mask_device = heldout_mask.to("cpu")
            if (
                heldout_mask_device.dtype != torch.bool
                or heldout_mask_device.shape != expected_mask.shape
                or not torch.equal(heldout_mask_device, expected_mask)
                or ssl_core._mask_sha256(heldout_mask_device)
                != common.get("heldout_mask_sha256")
            ):
                raise ValueError("focused common heldout mask changed")
            evaluation_model = ssl_core.DynamicLandmarkSSLModel().to("cpu")
            evaluation_model.load_state_dict(post_state, strict=True)
            evaluation_model.eval()
            baseline = ssl_core.DynamicLandmarkSSLModel().to("cpu")
            baseline.load_state_dict(pre_state, strict=True)
            baseline.eval()
            with torch.no_grad():
                trained_prediction = evaluation_model(
                    heldout_scaled, heldout_valid, heldout_times,
                    heldout_source_indices,
                    reconstruction_mask=heldout_mask_device,
                    source="mayo", input_arm=arm,
                )
                fresh_prediction = baseline(
                    heldout_scaled, heldout_valid, heldout_times,
                    heldout_source_indices,
                    reconstruction_mask=heldout_mask_device,
                    source="mayo", input_arm=arm,
                )
            reconstruction = ssl_core.reconstruction_report(
                trained_prediction,
                fresh_prediction,
                heldout_scaled,
                heldout_mask_device,
                baseline=scaler,
                split=split,
                evaluated_indices=heldout_indices,
                group_ids=authorization.group_ids,
                source=ssl_core.MAYO_SOURCE,
            )
            common_metrics = dict(reconstruction["common_target_metrics"])
            common_metrics["fresh_untrained"] = common_metrics.pop("untrained")
            metrics = _focused_metric_bundle(common_metrics)
            heldout_evaluation_count = 1
    return {
        "phase": phase,
        "arm": arm,
        "seed": seed,
        "epochs": epochs,
        "config": config,
        "config_sha256": ssl_core._canonical_sha256(config),
        "model_state": post_state,
        "pre_model_state": pre_state,
        "pre_state_sha256": pre_state_sha256,
        "post_state_sha256": post_state_sha256,
        "fresh_untrained_state_sha256": fresh_state_sha256,
        "optimizer_initial_sha256": optimizer_initial_sha256,
        "optimizer_initial_empty": True,
        "train_mask_schedule_sha256": ssl_core._canonical_sha256(
            mask_schedule,
        ),
        "train_trace_sha256": ssl_core._canonical_sha256(train_trace),
        "train_loss": _finite_nonnegative(train_trace[-1]["loss"]),
        "metrics": metrics,
        "heldout_evaluation_count": heldout_evaluation_count,
        "heldout_evaluation_computed": heldout_evaluation_count == 1,
        "runtime_environment": runtime,
    }


def _focused_checkpoint_metadata(
    result: Mapping[str, object],
    *,
    authorization: _FocusedBridgeAuthorization,
    common: Mapping[str, object],
    dependency_commitment_sha256: str | None,
) -> dict[str, object]:
    if (
        type(result) is not dict
        or not isinstance(result.get("model_state"), Mapping)
        or not isinstance(result.get("pre_model_state"), Mapping)
    ):
        raise ValueError("focused checkpoint result is malformed")
    phase = result.get("phase")
    arm = result.get("arm")
    seed = result.get("seed")
    contract = _focused_job_contract(
        str(phase), selected_arm=str(arm) if phase == "winner" else None,
    )
    if (
        arm not in contract["arms"]
        or seed not in contract["seeds"]
        or result.get("epochs") != contract["epochs"]
        or result.get("post_state_sha256")
        != ssl_core._model_state_sha256(result["model_state"])
        or result.get("pre_state_sha256")
        != ssl_core._model_state_sha256(result["pre_model_state"])
        or result.get("fresh_untrained_state_sha256")
        != result.get("pre_state_sha256")
        or result.get("optimizer_initial_empty") is not True
    ):
        raise ValueError("focused checkpoint result violates its job contract")
    if dependency_commitment_sha256 is not None:
        ssl_core._require_sha256(
            dependency_commitment_sha256, "focused dependency commitment",
        )
    runtime = _validate_focused_runtime_environment(
        result.get("runtime_environment"),
    )
    metrics = result.get("metrics")
    if bool(contract["evaluate_heldout"]):
        metrics = _focused_metric_bundle(metrics)
    elif metrics is not None:
        raise ValueError("focused smoke checkpoint cannot contain heldout metrics")
    metadata: dict[str, object] = {
        "schema_version": "focused_mayo_checkpoint_metadata_v2",
        "phase": phase,
        "arm": arm,
        "seed": seed,
        "epochs": contract["epochs"],
        "optimizer": "adamw",
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "initialization_policy": "same_seed_fresh",
        "target_schema": ssl_core.TARGET_FULL95,
        "input_active_indices": list(ssl_core.validate_ssl_input_arm(
            "mayo", str(arm),
        )),
        "bridge_generation_sha256": authorization.bridge_generation_sha256,
        "bridge_producer_sha256": authorization.producer_sha256,
        "trainer_sha256": authorization.trainer_sha256,
        "bundle_sha256": authorization.bundle_sha256,
        "common_contract_sha256": common.get("common_contract_sha256"),
        "split_sha256": common.get("split_sha256"),
        "scaler_sha256": common.get("scaler_sha256"),
        "heldout_mask_sha256": common.get("heldout_mask_sha256"),
        "config_sha256": result.get("config_sha256"),
        "pre_state_sha256": result.get("pre_state_sha256"),
        "post_state_sha256": result.get("post_state_sha256"),
        "fresh_untrained_state_sha256": result.get(
            "fresh_untrained_state_sha256",
        ),
        "optimizer_initial_sha256": result.get("optimizer_initial_sha256"),
        "optimizer_initial_empty": result["optimizer_initial_empty"],
        "train_mask_schedule_sha256": result.get(
            "train_mask_schedule_sha256",
        ),
        "train_trace_sha256": result.get("train_trace_sha256"),
        "train_loss": _finite_nonnegative(result.get("train_loss")),
        "heldout_evaluation_computed": bool(
            result.get("heldout_evaluation_computed"),
        ),
        "heldout_evaluation_count": result.get("heldout_evaluation_count"),
        "metrics": metrics,
        "dependency_commitment_sha256": dependency_commitment_sha256,
        "runtime_environment": runtime,
        "metric_quantization_policy": dict(
            _FOCUSED_METRIC_QUANTIZATION_POLICY,
        ),
        "claim_scope": "recording_heldout_development_reconstruction_only",
    }
    digest_fields = (
        "common_contract_sha256", "split_sha256", "scaler_sha256",
        "heldout_mask_sha256", "config_sha256", "pre_state_sha256",
        "post_state_sha256", "fresh_untrained_state_sha256",
        "optimizer_initial_sha256", "train_mask_schedule_sha256",
        "train_trace_sha256", "trainer_sha256",
    )
    for name in digest_fields:
        ssl_core._require_sha256(metadata.get(name), f"focused metadata {name}")
    expected_eval = 1 if bool(contract["evaluate_heldout"]) else 0
    if (
        metadata["heldout_evaluation_count"] != expected_eval
        or metadata["heldout_evaluation_computed"] is not (expected_eval == 1)
    ):
        raise ValueError("focused heldout evaluation count is not exact")
    return metadata


def _focused_checkpoint_fingerprint(
    metadata: Mapping[str, object],
    model_state: Mapping[str, torch.Tensor],
    pre_model_state: Mapping[str, torch.Tensor],
) -> str:
    digest = hashlib.sha256()
    digest.update(b"focused-mayo-checkpoint-fingerprint-v1\0")
    digest.update(ssl_core._canonical_sha256(metadata).encode("ascii"))
    digest.update(b"\0")
    digest.update(ssl_core._model_state_sha256(model_state).encode("ascii"))
    digest.update(b"\0")
    digest.update(ssl_core._model_state_sha256(pre_model_state).encode("ascii"))
    return digest.hexdigest()


def _focused_portable_file_identity(
    file_sha256: str, size_bytes: int, mode: int,
) -> str:
    return ssl_core._canonical_sha256({
        "schema_version": "focused_portable_private_file_identity_v1",
        "sha256": ssl_core._require_sha256(file_sha256, "focused file"),
        "size_bytes": size_bytes,
        "mode": mode,
    })


def _write_focused_checkpoint(
    path: Path,
    result: Mapping[str, object],
    *,
    authorization: _FocusedBridgeAuthorization,
    common: Mapping[str, object],
    dependency_commitment_sha256: str | None,
) -> dict[str, object]:
    authorization = _require_focused_authority_unchanged(authorization)
    path = path.absolute()
    parent = _private_directory(path.parent, "focused checkpoint parent")
    if path.name != Path(path.name).name:
        raise ValueError("focused checkpoint path is not canonical")
    path = parent / path.name
    metadata = _focused_checkpoint_metadata(
        result,
        authorization=authorization,
        common=common,
        dependency_commitment_sha256=dependency_commitment_sha256,
    )
    model_state = result["model_state"]
    pre_model_state = result["pre_model_state"]
    assert isinstance(model_state, Mapping)
    assert isinstance(pre_model_state, Mapping)
    fingerprint = _focused_checkpoint_fingerprint(
        metadata, model_state, pre_model_state,
    )
    payload = {
        "schema_version": _FOCUSED_CHECKPOINT_SCHEMA,
        "metadata": metadata,
        "model_state": model_state,
        "pre_model_state": pre_model_state,
        "checkpoint_fingerprint": fingerprint,
    }
    output = io.BytesIO()
    torch.save(payload, output)
    checkpoint_bytes = output.getvalue()
    if not checkpoint_bytes or len(checkpoint_bytes) > 128 * 1024 * 1024:
        raise ValueError("focused checkpoint exceeds its byte budget")
    _write_private_bytes(path, checkpoint_bytes)
    file_authorization, observed = ssl_core._private_file_snapshot(
        path, "focused checkpoint", max_bytes=128 * 1024 * 1024,
    )
    if observed != checkpoint_bytes:
        raise ValueError("focused checkpoint changed after writing")
    receipt_core: dict[str, object] = {
        "schema_version": _FOCUSED_CHECKPOINT_RECEIPT_SCHEMA,
        "checkpoint_name": path.name,
        "checkpoint_file_sha256": file_authorization.sha256,
        "checkpoint_file_size_bytes": len(checkpoint_bytes),
        "checkpoint_file_identity_sha256": _focused_portable_file_identity(
            file_authorization.sha256,
            file_authorization.identity.size,
            file_authorization.identity.mode,
        ),
        "checkpoint_fingerprint": fingerprint,
        "metadata_sha256": ssl_core._canonical_sha256(metadata),
        "phase": metadata["phase"],
        "arm": metadata["arm"],
        "seed": metadata["seed"],
        "epochs": metadata["epochs"],
        "dependency_commitment_sha256": dependency_commitment_sha256,
        "bridge_generation_sha256": authorization.bridge_generation_sha256,
        "bridge_producer_sha256": authorization.producer_sha256,
        "trainer_sha256": authorization.trainer_sha256,
        "bundle_sha256": authorization.bundle_sha256,
        "common_contract_sha256": common["common_contract_sha256"],
        "target_schema": ssl_core.TARGET_FULL95,
    }
    receipt_core["receipt_sha256"] = ssl_core._canonical_sha256(receipt_core)
    receipt = dict(receipt_core)
    receipt["authority_hmac"] = _focused_hmac(
        "checkpoint-receipt", receipt_core, authorization.private_key,
    )
    _write_private_json(Path(f"{path}.receipt.json"), receipt)
    return receipt


def _load_focused_checkpoint(
    path: Path,
    *,
    authorization: _FocusedBridgeAuthorization,
    common: Mapping[str, object],
    expected_phase: str,
    expected_arm: str,
    expected_seed: int,
    dependency_commitment_sha256: str | None,
) -> dict[str, object]:
    authorization = _require_focused_authority_unchanged(authorization)
    file_authorization, checkpoint_bytes = ssl_core._private_file_snapshot(
        path.absolute(), "focused checkpoint", max_bytes=128 * 1024 * 1024,
    )
    _, receipt_bytes, _, _ = ssl_core._private_regular_file_snapshot(
        Path(f"{path.absolute()}.receipt.json"), "focused checkpoint receipt",
    )
    receipt = ssl_core._strict_json_mapping(
        receipt_bytes, "focused checkpoint receipt",
    )
    if (
        set(receipt) != _FOCUSED_CHECKPOINT_RECEIPT_FIELDS
        or _canonical_json_bytes(receipt) != receipt_bytes
    ):
        raise ValueError("focused checkpoint receipt schema is not exact")
    observed_hmac = receipt.pop("authority_hmac")
    if (
        type(observed_hmac) is not str
        or not hmac.compare_digest(
            observed_hmac,
            _focused_hmac(
                "checkpoint-receipt", receipt, authorization.private_key,
            ),
        )
        or receipt.get("receipt_sha256")
        != ssl_core._canonical_sha256({
            name: value for name, value in receipt.items()
            if name != "receipt_sha256"
        })
    ):
        raise ValueError("focused checkpoint receipt HMAC is invalid")
    expected_receipt = {
        "checkpoint_name": path.name,
        "checkpoint_file_sha256": file_authorization.sha256,
        "checkpoint_file_size_bytes": len(checkpoint_bytes),
        "checkpoint_file_identity_sha256": _focused_portable_file_identity(
            file_authorization.sha256,
            file_authorization.identity.size,
            file_authorization.identity.mode,
        ),
        "phase": expected_phase,
        "arm": expected_arm,
        "seed": expected_seed,
        "dependency_commitment_sha256": dependency_commitment_sha256,
        "bridge_generation_sha256": authorization.bridge_generation_sha256,
        "bridge_producer_sha256": authorization.producer_sha256,
        "trainer_sha256": authorization.trainer_sha256,
        "bundle_sha256": authorization.bundle_sha256,
        "common_contract_sha256": common.get("common_contract_sha256"),
        "target_schema": ssl_core.TARGET_FULL95,
    }
    if any(receipt.get(name) != value for name, value in expected_receipt.items()):
        raise ValueError("focused checkpoint receipt contradicts its lineage")
    try:
        loaded = torch.load(
            io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True,
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError("focused checkpoint payload is unsafe") from exc
    if type(loaded) is not dict or set(loaded) != {
        "schema_version", "metadata", "model_state", "pre_model_state",
        "checkpoint_fingerprint",
    } or loaded.get("schema_version") != _FOCUSED_CHECKPOINT_SCHEMA:
        raise ValueError("focused checkpoint payload schema is not exact")
    metadata = loaded.get("metadata")
    model_state = loaded.get("model_state")
    pre_model_state = loaded.get("pre_model_state")
    if (
        type(metadata) is not dict
        or not isinstance(model_state, Mapping)
        or not isinstance(pre_model_state, Mapping)
    ):
        raise ValueError("focused checkpoint metadata or state is malformed")
    fingerprint = _focused_checkpoint_fingerprint(
        metadata, model_state, pre_model_state,
    )
    if (
        loaded.get("checkpoint_fingerprint") != fingerprint
        or receipt.get("checkpoint_fingerprint") != fingerprint
        or receipt.get("metadata_sha256")
        != ssl_core._canonical_sha256(metadata)
        or metadata.get("phase") != expected_phase
        or metadata.get("arm") != expected_arm
        or metadata.get("seed") != expected_seed
        or metadata.get("dependency_commitment_sha256")
        != dependency_commitment_sha256
        or metadata.get("trainer_sha256") != authorization.trainer_sha256
        or metadata.get("pre_state_sha256")
        != ssl_core._model_state_sha256(pre_model_state)
        or metadata.get("fresh_untrained_state_sha256")
        != ssl_core._model_state_sha256(pre_model_state)
        or metadata.get("optimizer_initial_empty") is not True
        or metadata.get("metric_quantization_policy")
        != _FOCUSED_METRIC_QUANTIZATION_POLICY
        or metadata.get("runtime_environment", {}).get("device_type")
        not in {"cpu", "cuda"}
    ):
        raise ValueError("focused checkpoint payload contradicts its receipt")
    model = ssl_core.DynamicLandmarkSSLModel().to("cpu")
    model.load_state_dict(model_state, strict=True)
    pre_model = ssl_core.DynamicLandmarkSSLModel().to("cpu")
    pre_model.load_state_dict(pre_model_state, strict=True)
    if ssl_core._model_state_sha256(model.state_dict()) != metadata.get(
        "post_state_sha256",
    ):
        raise ValueError("focused checkpoint model state is inconsistent")
    receipt["authority_hmac"] = observed_hmac
    return {
        "metadata": metadata,
        "model_state": ssl_core._clone_model_state(model),
        "pre_model_state": ssl_core._clone_model_state(pre_model),
        "checkpoint_fingerprint": fingerprint,
        "receipt": receipt,
        "receipt_file_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
    }


class _PathRedactingArgumentParser(argparse.ArgumentParser):
    def error(self, _message: str) -> None:
        self.exit(2, "pretraining command arguments are invalid\n")


def _parser() -> argparse.ArgumentParser:
    parser = _PathRedactingArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    def add_execution_arguments(command: argparse.ArgumentParser) -> None:
        command.add_argument("--run-root", type=Path, required=True)
        command.add_argument("--bridge-root", type=Path, required=True)
        command.add_argument("--ravdess-data-root", type=Path, required=True)
        command.add_argument("--ravdess-key", type=Path, required=True)
        command.add_argument("--mayo-data-root", type=Path, required=True)
        command.add_argument(
            "--mayo-existing-export-root", type=Path, required=True,
        )
        command.add_argument("--mayo-cache-root", type=Path, required=True)
        command.add_argument(
            "--mayo-exposure-manifest", type=Path, required=True,
        )
        command.add_argument("--mayo-key", type=Path, required=True)

    command = commands.add_parser("two-stage")
    command.add_argument("--mode", choices=("smoke", "formal"), required=True)
    add_execution_arguments(command)
    add_execution_arguments(commands.add_parser("mayo-ablation"))
    focused = commands.add_parser("focused-mayo")
    focused.add_argument(
        "--phase", choices=("smoke", "select", "winner", "audit"),
        required=True,
    )
    commands.add_parser("dry-run")
    return parser


def _focused_job_contract(
    phase: str,
    *,
    selected_arm: str | None = None,
) -> dict[str, object]:
    """Return the immutable development-study schedule without tuning knobs."""
    common = {
        "target_schema": ssl_core.TARGET_FULL95,
        "initialization": "same_seed_fresh",
        "optimizer": "adamw",
        "early_stopping": False,
    }
    if phase == "smoke":
        if selected_arm is not None:
            raise ValueError("focused smoke cannot accept an arm")
        return {
            **common, "phase": phase, "arms": [ARM_FUSION],
            "seeds": [0], "epochs": 1, "evaluate_heldout": False,
        }
    if phase == "select":
        if selected_arm is not None:
            raise ValueError("focused selection cannot accept an arm")
        return {
            **common, "phase": phase, "arms": list(_FOCUSED_ARMS),
            "seeds": [0], "epochs": 5, "evaluate_heldout": True,
        }
    if phase == "winner":
        if selected_arm not in _FOCUSED_ARMS:
            raise ValueError("focused winner requires one authenticated arm")
        return {
            **common, "phase": phase, "arms": [selected_arm],
            "seeds": list(_FORMAL_SEEDS), "epochs": 30,
            "evaluate_heldout": True,
        }
    raise ValueError("focused phase is unsupported")


def _select_focused_arm(rows: list[dict[str, object]]) -> dict[str, object]:
    """Select the finite minimum with the preregistered arm-order tie break."""
    if type(rows) is not list or len(rows) != len(_FOCUSED_ARMS):
        raise ValueError("focused selection matrix is not exact")
    metrics: list[float] = []
    for expected_arm, row in zip(_FOCUSED_ARMS, rows):
        if (
            type(row) is not dict
            or set(row) != {"arm", "primary_metric"}
            or row.get("arm") != expected_arm
        ):
            raise ValueError("focused selection arm order is not exact")
        metrics.append(_canonical_focused_metric(row["primary_metric"]))
    selected_index = min(range(len(metrics)), key=lambda index: (
        metrics[index], index,
    ))
    return {
        "selected_arm": _FOCUSED_ARMS[selected_index],
        "selected_metric": metrics[selected_index],
        "metric_path": _FOCUSED_PRIMARY_METRIC,
        "direction": "lower_is_better",
        "tie_break_order": list(_FOCUSED_ARMS),
    }


def _expected_focused_result_files(
    phase: str,
    *,
    selected_arm: str | None = None,
) -> set[str]:
    if phase == "smoke":
        if selected_arm is not None:
            raise ValueError("focused smoke result contract cannot name an arm")
        return {"checkpoint.pt", "checkpoint.pt.receipt.json", "report.json"}
    if phase == "select":
        if selected_arm is not None:
            raise ValueError("focused selection result contract cannot name an arm")
        return {
            "report.json",
            *(name for arm in _FOCUSED_ARMS for name in (
                f"{arm}.pt", f"{arm}.pt.receipt.json",
            )),
        }
    if phase == "winner":
        if selected_arm not in _FOCUSED_ARMS:
            raise ValueError("focused winner result contract lacks its selected arm")
        return {
            "report.json",
            *(name for seed in _FORMAL_SEEDS for name in (
                f"seed_{seed}.pt", f"seed_{seed}.pt.receipt.json",
            )),
        }
    if phase == "audit":
        if selected_arm is not None:
            raise ValueError("focused audit result contract cannot name an arm")
        return {"report.json"}
    raise ValueError("focused result phase is unsupported")


def _focused_smoke_elapsed(start: float, end: float) -> float:
    if (
        isinstance(start, bool) or isinstance(end, bool)
        or not isinstance(start, (int, float))
        or not isinstance(end, (int, float))
        or not math.isfinite(float(start))
        or not math.isfinite(float(end))
    ):
        raise ValueError("focused smoke monotonic timestamps are invalid")
    elapsed = float(end) - float(start)
    if elapsed < 0.0 or elapsed > 900.0:
        raise ValueError("focused smoke exceeded its 900-second publication bound")
    return elapsed


def _write_focused_report(
    directory: Path,
    *,
    phase: str,
    core: dict[str, object],
    authorization: _FocusedBridgeAuthorization,
) -> tuple[dict[str, object], str]:
    authorization = _require_focused_authority_unchanged(authorization)
    report = _sign_focused_report(phase, core, authorization.private_key)
    payload = _canonical_json_bytes(report)
    _write_private_bytes(directory / "report.json", payload)
    return report, hashlib.sha256(payload).hexdigest()


def _read_focused_report(
    directory: Path,
    *,
    phase: str,
    authorization: _FocusedBridgeAuthorization,
) -> tuple[dict[str, object], str]:
    authorization = _require_focused_authority_unchanged(authorization)
    _, payload, digest, _ = ssl_core._private_regular_file_snapshot(
        directory / "report.json", f"focused {phase} report",
    )
    report = _validate_focused_report_bytes(
        payload, phase=phase, private_key=authorization.private_key,
    )
    return report, digest


def _focused_lineage_fields(
    authorization: _FocusedBridgeAuthorization,
    common: Mapping[str, object],
) -> dict[str, object]:
    return {
        "bridge_generation_sha256": authorization.bridge_generation_sha256,
        "bridge_producer_sha256": authorization.producer_sha256,
        "trainer_sha256": authorization.trainer_sha256,
        "bundle_sha256": authorization.bundle_sha256,
        "common_contract_sha256": common["common_contract_sha256"],
        "metric_quantization_policy": dict(
            _FOCUSED_METRIC_QUANTIZATION_POLICY,
        ),
    }


def _require_focused_report_lineage(
    report: Mapping[str, object],
    authorization: _FocusedBridgeAuthorization,
    common: Mapping[str, object],
) -> None:
    if any(
        report.get(name) != value
        for name, value in _focused_lineage_fields(
            authorization, common,
        ).items()
    ):
        raise ValueError("focused report lineage is not current")


def _focused_winner_aggregates(
    rows: list[dict[str, object]],
) -> dict[str, object]:
    if (
        type(rows) is not list or len(rows) != 3
        or [row.get("seed") for row in rows] != [0, 1, 2]
    ):
        raise ValueError("focused winner rows are not seed-exact")

    def summarize(values: list[float]) -> dict[str, float]:
        checked = [_canonical_focused_metric(value) for value in values]
        return {
            "mean": _canonical_focused_metric(statistics.fmean(checked)),
            "sd": _canonical_focused_metric(statistics.stdev(checked)),
        }

    result: dict[str, object] = {}
    for baseline in ("trained", "fresh_untrained", "train_mean"):
        result[baseline] = {
            "raw_mae": {
                metric: summarize([
                    row["metrics"][baseline]["raw_mae"][metric]  # type: ignore[index]
                    for row in rows
                ])
                for metric in (
                    "blendshape72", "clinical23", "equal_block_macro", "full95",
                )
            },
            "standardized_mae": summarize([
                row["metrics"][baseline]["standardized_mae"]  # type: ignore[index]
                for row in rows
            ]),
            "standardized_smooth_l1": summarize([
                row["metrics"][baseline]["standardized_smooth_l1"]  # type: ignore[index]
                for row in rows
            ]),
        }
    return result


def _validate_focused_exact_tree(
    directory: Path, expected_files: set[str],
) -> None:
    directory = _private_directory(directory, "focused phase directory")
    parent = _private_directory(directory.parent, "focused namespace")
    descriptor = os.open(
        parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        with _hold_exact_result_tree(
            descriptor, directory.name, expected_files,
        ) as validate:
            validate()
    finally:
        os.close(descriptor)


def _validate_focused_smoke_phase(
    directory: Path,
    *,
    authorization: _FocusedBridgeAuthorization,
    common: Mapping[str, object],
) -> dict[str, object]:
    expected = _expected_focused_result_files("smoke")
    _validate_focused_exact_tree(directory, expected)
    report, report_sha256 = _read_focused_report(
        directory, phase="smoke", authorization=authorization,
    )
    expected_fields = {
        "schema_version", "phase", "status", "bridge_generation_sha256",
        "bridge_producer_sha256", "trainer_sha256", "bundle_sha256",
        "common_contract_sha256", "metric_quantization_policy",
        "runtime_environment", "arm", "seed",
        "epochs", "elapsed_seconds", "train_loss",
        "heldout_evaluation_computed", "checkpoint_fingerprint",
        "checkpoint_receipt_sha256", "claim_scope", "authority_hmac",
    }
    if set(report) != expected_fields or report.get("status") != "complete":
        raise ValueError("focused smoke report schema is not exact")
    _require_focused_report_lineage(report, authorization, common)
    _validate_focused_runtime_environment(report.get("runtime_environment"))
    if (
        report.get("arm") != ARM_FUSION
        or report.get("seed") != 0
        or report.get("epochs") != 1
        or report.get("heldout_evaluation_computed") is not False
        or report.get("claim_scope")
        != "recording_heldout_development_reconstruction_only"
    ):
        raise ValueError("focused smoke report contradicts its fixed job")
    elapsed = _finite_nonnegative(report.get("elapsed_seconds"))
    _focused_smoke_elapsed(0.0, elapsed)
    _finite_nonnegative(report.get("train_loss"))
    loaded = _load_focused_checkpoint(
        directory / "checkpoint.pt",
        authorization=authorization,
        common=common,
        expected_phase="smoke",
        expected_arm=ARM_FUSION,
        expected_seed=0,
        dependency_commitment_sha256=None,
    )
    if (
        report.get("checkpoint_fingerprint")
        != loaded["checkpoint_fingerprint"]
        or report.get("checkpoint_receipt_sha256")
        != loaded["receipt_file_sha256"]
        or loaded["metadata"].get("metrics") is not None  # type: ignore[union-attr]
    ):
        raise ValueError("focused smoke report contradicts its checkpoint")
    return {
        "report": report,
        "report_sha256": report_sha256,
        "checkpoint": loaded,
    }


def _validate_focused_selection_phase(
    directory: Path,
    *,
    authorization: _FocusedBridgeAuthorization,
    common: Mapping[str, object],
    smoke: Mapping[str, object],
) -> dict[str, object]:
    _validate_focused_exact_tree(
        directory, _expected_focused_result_files("select"),
    )
    report, report_sha256 = _read_focused_report(
        directory, phase="select", authorization=authorization,
    )
    expected_fields = {
        "schema_version", "phase", "status", "bridge_generation_sha256",
        "bridge_producer_sha256", "trainer_sha256", "bundle_sha256",
        "common_contract_sha256", "metric_quantization_policy",
        "runtime_environment",
        "smoke_report_sha256", "arm_order", "seed", "epochs",
        "optimizer", "initialization_policy", "primary_metric",
        "runs", "selected_arm", "selected_metric",
        "selected_run_commitment_sha256", "claim_scope", "authority_hmac",
    }
    if set(report) != expected_fields or report.get("status") != "complete":
        raise ValueError("focused selection report schema is not exact")
    _require_focused_report_lineage(report, authorization, common)
    _validate_focused_runtime_environment(report.get("runtime_environment"))
    smoke_sha256 = smoke.get("report_sha256")
    if (
        report.get("smoke_report_sha256") != smoke_sha256
        or report.get("arm_order") != list(_FOCUSED_ARMS)
        or report.get("seed") != 0
        or report.get("epochs") != 5
        or report.get("optimizer") != "adamw"
        or report.get("initialization_policy") != "same_seed_fresh"
    ):
        raise ValueError("focused selection did not bind the exact smoke/fairness")
    primary = report.get("primary_metric")
    if primary != {
        "path": _FOCUSED_PRIMARY_METRIC,
        "direction": "lower_is_better",
        "tie_break_order": list(_FOCUSED_ARMS),
    }:
        raise ValueError("focused selection primary metric contract changed")
    runs = report.get("runs")
    if type(runs) is not list or len(runs) != 3:
        raise ValueError("focused selection report lacks three runs")
    selection_rows: list[dict[str, object]] = []
    loaded_by_arm: dict[str, object] = {}
    pre_states: set[str] = set()
    optimizer_states: set[str] = set()
    for arm, row in zip(_FOCUSED_ARMS, runs):
        if type(row) is not dict or row.get("arm") != arm:
            raise ValueError("focused selection run order changed")
        metrics = _focused_metric_bundle(row.get("metrics"))
        primary_metric = _finite_nonnegative(
            metrics["trained"]["raw_mae"]["equal_block_macro"],  # type: ignore[index]
        )
        if row.get("primary_metric") != primary_metric:
            raise ValueError("focused selection primary metric was mutated")
        loaded = _load_focused_checkpoint(
            directory / f"{arm}.pt",
            authorization=authorization,
            common=common,
            expected_phase="select",
            expected_arm=arm,
            expected_seed=0,
            dependency_commitment_sha256=str(smoke_sha256),
        )
        metadata = loaded["metadata"]
        if (
            row.get("checkpoint_fingerprint")
            != loaded["checkpoint_fingerprint"]
            or row.get("checkpoint_receipt_sha256")
            != loaded["receipt_file_sha256"]
            or row.get("pre_state_sha256")
            != metadata.get("pre_state_sha256")
            or row.get("optimizer_initial_sha256")
            != metadata.get("optimizer_initial_sha256")
            or row.get("optimizer_initial_empty") is not True
            or metadata.get("optimizer_initial_empty") is not True
            or not ssl_core._exact_json_value(metadata.get("metrics"), metrics)
        ):
            raise ValueError("focused selection row contradicts its checkpoint")
        pre_states.add(str(row.get("pre_state_sha256")))
        optimizer_states.add(str(row.get("optimizer_initial_sha256")))
        selection_rows.append({"arm": arm, "primary_metric": primary_metric})
        loaded_by_arm[arm] = loaded
    if len(pre_states) != 1 or len(optimizer_states) != 1:
        raise ValueError("focused selection arms did not share fresh initialization")
    decision = _select_focused_arm(selection_rows)
    selected_commitment = ssl_core._canonical_sha256({
        "selection": decision,
        "runs": selection_rows,
        "smoke_report_sha256": smoke_sha256,
        "common_contract_sha256": common["common_contract_sha256"],
    })
    if (
        report.get("selected_arm") != decision["selected_arm"]
        or report.get("selected_metric") != decision["selected_metric"]
        or report.get("selected_run_commitment_sha256")
        != selected_commitment
    ):
        raise ValueError("focused selection decision commitment is invalid")
    return {
        "report": report,
        "report_sha256": report_sha256,
        "selected_arm": decision["selected_arm"],
        "selected_run_commitment_sha256": selected_commitment,
        "checkpoints": loaded_by_arm,
    }


def _validate_focused_winner_phase(
    directory: Path,
    *,
    authorization: _FocusedBridgeAuthorization,
    common: Mapping[str, object],
    selection: Mapping[str, object],
) -> dict[str, object]:
    selected_arm = selection.get("selected_arm")
    if selected_arm not in _FOCUSED_ARMS:
        raise ValueError("focused winner lacks an authenticated selected arm")
    _validate_focused_exact_tree(
        directory,
        _expected_focused_result_files(
            "winner", selected_arm=str(selected_arm),
        ),
    )
    report, report_sha256 = _read_focused_report(
        directory, phase="winner", authorization=authorization,
    )
    expected_fields = {
        "schema_version", "phase", "status", "bridge_generation_sha256",
        "bridge_producer_sha256", "trainer_sha256", "bundle_sha256",
        "common_contract_sha256", "metric_quantization_policy",
        "runtime_environment",
        "selection_report_sha256", "selected_run_commitment_sha256",
        "selected_arm", "seeds", "epochs", "optimizer",
        "initialization_policy", "runs", "aggregates", "claim_scope",
        "authority_hmac",
    }
    if set(report) != expected_fields or report.get("status") != "complete":
        raise ValueError("focused winner report schema is not exact")
    _require_focused_report_lineage(report, authorization, common)
    _validate_focused_runtime_environment(report.get("runtime_environment"))
    selection_sha256 = selection.get("report_sha256")
    dependency = selection.get("selected_run_commitment_sha256")
    if (
        report.get("selection_report_sha256") != selection_sha256
        or report.get("selected_run_commitment_sha256") != dependency
        or report.get("selected_arm") != selected_arm
        or report.get("seeds") != [0, 1, 2]
        or report.get("epochs") != 30
        or report.get("optimizer") != "adamw"
        or report.get("initialization_policy") != "same_seed_fresh"
        or report.get("claim_scope")
        != "recording_heldout_development_reconstruction_only"
    ):
        raise ValueError("focused winner report contradicts selection or budget")
    runs = report.get("runs")
    if type(runs) is not list or len(runs) != 3:
        raise ValueError("focused winner report lacks three seed runs")
    normalized_rows: list[dict[str, object]] = []
    loaded_by_seed: dict[int, object] = {}
    for seed, row in zip(_FORMAL_SEEDS, runs):
        if type(row) is not dict or row.get("seed") != seed:
            raise ValueError("focused winner seed order changed")
        metrics = _focused_metric_bundle(row.get("metrics"))
        loaded = _load_focused_checkpoint(
            directory / f"seed_{seed}.pt",
            authorization=authorization,
            common=common,
            expected_phase="winner",
            expected_arm=str(selected_arm),
            expected_seed=seed,
            dependency_commitment_sha256=str(selection_sha256),
        )
        metadata = loaded["metadata"]
        if (
            row.get("checkpoint_fingerprint")
            != loaded["checkpoint_fingerprint"]
            or row.get("checkpoint_receipt_sha256")
            != loaded["receipt_file_sha256"]
            or row.get("pre_state_sha256")
            != metadata.get("pre_state_sha256")
            or row.get("optimizer_initial_sha256")
            != metadata.get("optimizer_initial_sha256")
            or row.get("optimizer_initial_empty") is not True
            or metadata.get("fresh_untrained_state_sha256")
            != metadata.get("pre_state_sha256")
            or metadata.get("optimizer_initial_empty") is not True
            or not ssl_core._exact_json_value(metadata.get("metrics"), metrics)
        ):
            raise ValueError("focused winner is not a fresh exact seed job")
        normalized_rows.append({"seed": seed, "metrics": metrics})
        loaded_by_seed[seed] = loaded
    aggregates = _focused_winner_aggregates(normalized_rows)
    if not ssl_core._exact_json_value(report.get("aggregates"), aggregates):
        raise ValueError("focused winner aggregates were mutated")
    return {
        "report": report,
        "report_sha256": report_sha256,
        "selected_arm": selected_arm,
        "checkpoints": loaded_by_seed,
        "aggregates": aggregates,
    }


def _recompute_focused_checkpoint_metrics(
    loaded: Mapping[str, object],
    *,
    authorization: _FocusedBridgeAuthorization,
    common: Mapping[str, object],
) -> dict[str, object]:
    metadata = loaded.get("metadata")
    model_state = loaded.get("model_state")
    pre_model_state = loaded.get("pre_model_state")
    if (
        type(metadata) is not dict
        or not isinstance(model_state, Mapping)
        or not isinstance(pre_model_state, Mapping)
    ):
        raise ValueError("focused audit checkpoint is malformed")
    arm = metadata.get("arm")
    seed = metadata.get("seed")
    if arm not in _FOCUSED_ARMS or seed not in _FORMAL_SEEDS:
        raise ValueError("focused audit checkpoint job is unsupported")
    features, valid, timestamps, source_indices = _load_focused_tensors(
        _require_focused_authority_unchanged(authorization),
    )
    split = common.get("split")
    scaler = common.get("scaler")
    heldout_mask = common.get("heldout_mask")
    if (
        not isinstance(split, ssl_core.SSLGroupSplit)
        or not isinstance(scaler, ssl_core.SourceScaler)
        or not isinstance(heldout_mask, torch.Tensor)
    ):
        raise ValueError("focused audit common contract is malformed")
    _, heldout_indices, _ = ssl_core._validate_split_partition(
        split, authorization.group_ids,
    )
    heldout_rows = torch.as_tensor(heldout_indices, dtype=torch.int64)
    heldout_features = features.index_select(0, heldout_rows)
    heldout_valid = valid.index_select(0, heldout_rows)
    heldout_times = timestamps.index_select(0, heldout_rows)
    heldout_source_indices = source_indices.index_select(0, heldout_rows)
    heldout_scaled = scaler.transform(
        heldout_features, heldout_valid, source=ssl_core.MAYO_SOURCE,
    )
    with torch.random.fork_rng(devices=[]):
        fresh = ssl_core.DynamicLandmarkSSLModel().to("cpu")
        fresh.load_state_dict(pre_model_state, strict=True)
        trained = ssl_core.DynamicLandmarkSSLModel().to("cpu")
        trained.load_state_dict(model_state, strict=True)
        trained.eval()
        fresh.eval()
        with torch.no_grad():
            trained_prediction = trained(
                heldout_scaled, heldout_valid, heldout_times,
                heldout_source_indices,
                reconstruction_mask=heldout_mask,
                source="mayo", input_arm=str(arm),
            )
            fresh_prediction = fresh(
                heldout_scaled, heldout_valid, heldout_times,
                heldout_source_indices,
                reconstruction_mask=heldout_mask,
                source="mayo", input_arm=str(arm),
            )
    reconstruction = ssl_core.reconstruction_report(
        trained_prediction,
        fresh_prediction,
        heldout_scaled,
        heldout_mask,
        baseline=scaler,
        split=split,
        evaluated_indices=heldout_indices,
        group_ids=authorization.group_ids,
        source=ssl_core.MAYO_SOURCE,
    )
    values = dict(reconstruction["common_target_metrics"])
    values["fresh_untrained"] = values.pop("untrained")
    return _focused_metric_bundle(values)


def _audit_focused_metrics(
    selection: Mapping[str, object],
    winner: Mapping[str, object],
    *,
    authorization: _FocusedBridgeAuthorization,
    common: Mapping[str, object],
) -> None:
    selection_report = selection.get("report")
    winner_report = winner.get("report")
    if type(selection_report) is not dict or type(winner_report) is not dict:
        raise ValueError("focused audit reports are unavailable")
    selection_rows = selection_report.get("runs")
    winner_rows = winner_report.get("runs")
    if type(selection_rows) is not list or type(winner_rows) is not list:
        raise ValueError("focused audit run matrices are unavailable")
    selection_checkpoints = selection.get("checkpoints")
    winner_checkpoints = winner.get("checkpoints")
    if type(selection_checkpoints) is not dict or type(winner_checkpoints) is not dict:
        raise ValueError("focused audit checkpoint matrices are unavailable")
    for arm, row in zip(_FOCUSED_ARMS, selection_rows):
        recomputed = _recompute_focused_checkpoint_metrics(
            selection_checkpoints[arm],
            authorization=authorization,
            common=common,
        )
        if not ssl_core._exact_json_value(row.get("metrics"), recomputed):
            raise ValueError("focused selection metrics failed strict recomputation")
    normalized_winner: list[dict[str, object]] = []
    for seed, row in zip(_FORMAL_SEEDS, winner_rows):
        recomputed = _recompute_focused_checkpoint_metrics(
            winner_checkpoints[seed],
            authorization=authorization,
            common=common,
        )
        if not ssl_core._exact_json_value(row.get("metrics"), recomputed):
            raise ValueError("focused winner metrics failed strict recomputation")
        normalized_winner.append({"seed": seed, "metrics": recomputed})
    expected_aggregate = _focused_winner_aggregates(normalized_winner)
    if not ssl_core._exact_json_value(
        winner_report.get("aggregates"), expected_aggregate,
    ):
        raise ValueError("focused winner aggregate failed strict recomputation")


def _build_focused_smoke(
    staging: Path,
    *,
    authorization: _FocusedBridgeAuthorization,
    common: Mapping[str, object],
) -> dict[str, object]:
    started = time.monotonic()
    result = _train_focused_job(
        authorization, common, phase="smoke", arm=ARM_FUSION, seed=0,
    )
    _write_focused_checkpoint(
        staging / "checkpoint.pt", result,
        authorization=authorization,
        common=common,
        dependency_commitment_sha256=None,
    )
    loaded = _load_focused_checkpoint(
        staging / "checkpoint.pt",
        authorization=authorization,
        common=common,
        expected_phase="smoke",
        expected_arm=ARM_FUSION,
        expected_seed=0,
        dependency_commitment_sha256=None,
    )
    elapsed = _focused_smoke_elapsed(started, time.monotonic())
    core = {
        "schema_version": _FOCUSED_REPORT_SCHEMAS["smoke"],
        "phase": "smoke",
        "status": "complete",
        **_focused_lineage_fields(authorization, common),
        "runtime_environment": result["runtime_environment"],
        "arm": ARM_FUSION,
        "seed": 0,
        "epochs": 1,
        "elapsed_seconds": elapsed,
        "train_loss": result["train_loss"],
        "heldout_evaluation_computed": False,
        "checkpoint_fingerprint": loaded["checkpoint_fingerprint"],
        "checkpoint_receipt_sha256": loaded["receipt_file_sha256"],
        "claim_scope": "recording_heldout_development_reconstruction_only",
    }
    report, report_sha256 = _write_focused_report(
        staging, phase="smoke", core=core, authorization=authorization,
    )
    return {"report": report, "report_sha256": report_sha256}


def _build_focused_selection(
    staging: Path,
    *,
    authorization: _FocusedBridgeAuthorization,
    common: Mapping[str, object],
    smoke: Mapping[str, object],
) -> dict[str, object]:
    smoke_sha256 = ssl_core._require_sha256(
        smoke.get("report_sha256"), "focused smoke report",
    )
    rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    runtimes: list[dict[str, str]] = []
    for arm in _FOCUSED_ARMS:
        result = _train_focused_job(
            authorization, common, phase="select", arm=arm, seed=0,
        )
        path = staging / f"{arm}.pt"
        _write_focused_checkpoint(
            path, result,
            authorization=authorization,
            common=common,
            dependency_commitment_sha256=smoke_sha256,
        )
        loaded = _load_focused_checkpoint(
            path,
            authorization=authorization,
            common=common,
            expected_phase="select",
            expected_arm=arm,
            expected_seed=0,
            dependency_commitment_sha256=smoke_sha256,
        )
        metrics = _focused_metric_bundle(result["metrics"])
        primary = _finite_nonnegative(
            metrics["trained"]["raw_mae"]["equal_block_macro"],  # type: ignore[index]
        )
        row = {
            "arm": arm,
            "primary_metric": primary,
            "checkpoint_fingerprint": loaded["checkpoint_fingerprint"],
            "checkpoint_receipt_sha256": loaded["receipt_file_sha256"],
            "pre_state_sha256": result["pre_state_sha256"],
            "optimizer_initial_sha256": result["optimizer_initial_sha256"],
            "optimizer_initial_empty": True,
            "metrics": metrics,
        }
        rows.append(row)
        selection_rows.append({"arm": arm, "primary_metric": primary})
        runtimes.append(_validate_focused_runtime_environment(
            result["runtime_environment"],
        ))
    if (
        len({str(row["pre_state_sha256"]) for row in rows}) != 1
        or len({str(row["optimizer_initial_sha256"]) for row in rows}) != 1
        or any(runtime != runtimes[0] for runtime in runtimes[1:])
    ):
        raise RuntimeError("focused selection fairness contract changed")
    decision = _select_focused_arm(selection_rows)
    selected_commitment = ssl_core._canonical_sha256({
        "selection": decision,
        "runs": selection_rows,
        "smoke_report_sha256": smoke_sha256,
        "common_contract_sha256": common["common_contract_sha256"],
    })
    core = {
        "schema_version": _FOCUSED_REPORT_SCHEMAS["select"],
        "phase": "select",
        "status": "complete",
        **_focused_lineage_fields(authorization, common),
        "runtime_environment": runtimes[0],
        "smoke_report_sha256": smoke_sha256,
        "arm_order": list(_FOCUSED_ARMS),
        "seed": 0,
        "epochs": 5,
        "optimizer": "adamw",
        "initialization_policy": "same_seed_fresh",
        "primary_metric": {
            "path": _FOCUSED_PRIMARY_METRIC,
            "direction": "lower_is_better",
            "tie_break_order": list(_FOCUSED_ARMS),
        },
        "runs": rows,
        "selected_arm": decision["selected_arm"],
        "selected_metric": decision["selected_metric"],
        "selected_run_commitment_sha256": selected_commitment,
        "claim_scope": "recording_heldout_development_reconstruction_only",
    }
    report, report_sha256 = _write_focused_report(
        staging, phase="select", core=core, authorization=authorization,
    )
    return {
        "report": report,
        "report_sha256": report_sha256,
        "selected_arm": decision["selected_arm"],
        "selected_run_commitment_sha256": selected_commitment,
    }


def _build_focused_winner(
    staging: Path,
    *,
    authorization: _FocusedBridgeAuthorization,
    common: Mapping[str, object],
    selection: Mapping[str, object],
) -> dict[str, object]:
    selected_arm = selection.get("selected_arm")
    if selected_arm not in _FOCUSED_ARMS:
        raise ValueError("focused winner cannot accept a caller-selected arm")
    selection_sha256 = ssl_core._require_sha256(
        selection.get("report_sha256"), "focused selection report",
    )
    selected_commitment = ssl_core._require_sha256(
        selection.get("selected_run_commitment_sha256"),
        "focused selected-run commitment",
    )
    rows: list[dict[str, object]] = []
    runtimes: list[dict[str, str]] = []
    for seed in _FORMAL_SEEDS:
        result = _train_focused_job(
            authorization, common, phase="winner",
            arm=str(selected_arm), seed=seed,
        )
        path = staging / f"seed_{seed}.pt"
        _write_focused_checkpoint(
            path, result,
            authorization=authorization,
            common=common,
            dependency_commitment_sha256=selection_sha256,
        )
        loaded = _load_focused_checkpoint(
            path,
            authorization=authorization,
            common=common,
            expected_phase="winner",
            expected_arm=str(selected_arm),
            expected_seed=seed,
            dependency_commitment_sha256=selection_sha256,
        )
        metrics = _focused_metric_bundle(result["metrics"])
        rows.append({
            "seed": seed,
            "checkpoint_fingerprint": loaded["checkpoint_fingerprint"],
            "checkpoint_receipt_sha256": loaded["receipt_file_sha256"],
            "pre_state_sha256": result["pre_state_sha256"],
            "optimizer_initial_sha256": result["optimizer_initial_sha256"],
            "optimizer_initial_empty": True,
            "metrics": metrics,
        })
        runtimes.append(_validate_focused_runtime_environment(
            result["runtime_environment"],
        ))
    if any(runtime != runtimes[0] for runtime in runtimes[1:]):
        raise RuntimeError("focused winner seeds changed runtime environment")
    aggregate_rows = [
        {"seed": row["seed"], "metrics": row["metrics"]} for row in rows
    ]
    aggregates = _focused_winner_aggregates(aggregate_rows)
    core = {
        "schema_version": _FOCUSED_REPORT_SCHEMAS["winner"],
        "phase": "winner",
        "status": "complete",
        **_focused_lineage_fields(authorization, common),
        "runtime_environment": runtimes[0],
        "selection_report_sha256": selection_sha256,
        "selected_run_commitment_sha256": selected_commitment,
        "selected_arm": selected_arm,
        "seeds": list(_FORMAL_SEEDS),
        "epochs": 30,
        "optimizer": "adamw",
        "initialization_policy": "same_seed_fresh",
        "runs": rows,
        "aggregates": aggregates,
        "claim_scope": "recording_heldout_development_reconstruction_only",
    }
    report, report_sha256 = _write_focused_report(
        staging, phase="winner", core=core, authorization=authorization,
    )
    return {"report": report, "report_sha256": report_sha256}


def _ensure_focused_namespace() -> Path:
    root = _private_directory(PRETRAINING_ROOT, "focused pretraining root")
    current = root
    for name in ("development", "focused-modality-v1"):
        child = current / name
        try:
            os.mkdir(child, mode=0o700)
        except FileExistsError:
            pass
        child = _private_directory(child, "focused namespace component")
        current = child
    if current != _FOCUSED_NAMESPACE.resolve(strict=True):
        raise ValueError("focused namespace is not canonical")
    return current


@contextmanager
def _focused_namespace_lock(namespace: Path) -> Iterator[int]:
    lock_path = namespace / ".focused.lock"
    descriptor = os.open(
        lock_path,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        os.fchmod(descriptor, 0o600)
        status = os.fstat(descriptor)
        if (
            not stat.S_ISREG(status.st_mode)
            or status.st_uid != os.geteuid()
            or stat.S_IMODE(status.st_mode) != 0o600
            or status.st_nlink != 1
        ):
            raise ValueError("focused namespace lock is unsafe")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield descriptor
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _publish_focused_phase(
    *,
    phase: str,
    authorization: _FocusedBridgeAuthorization,
    common: Mapping[str, object],
    expected_files: set[str],
    builder: Callable[[Path], dict[str, object]],
) -> dict[str, object]:
    destination_name = "selection" if phase == "select" else phase
    if destination_name not in {"smoke", "selection", "winner", "audit"}:
        raise ValueError("focused publication phase is unsupported")
    namespace = _ensure_focused_namespace()
    with _focused_namespace_lock(namespace):
        entries = set(os.listdir(namespace))
        if destination_name in entries:
            raise FileExistsError("focused phase is already published")
        prefix = f".{destination_name}.staging-"
        if any(name.startswith(prefix) for name in entries):
            raise ValueError("focused phase has indeterminate staging residue")
        staging_name = prefix + secrets.token_hex(8)
        staging = namespace / staging_name
        os.mkdir(staging, mode=0o700)
        staging = _private_directory(staging, "focused phase staging")
        result = builder(staging)
        _validate_focused_exact_tree(staging, expected_files)
        fresh = _require_focused_authority_unchanged(authorization)
        fresh_common = _focused_common_contract(fresh)
        if (
            fresh_common["common_contract_sha256"]
            != common["common_contract_sha256"]
            or fresh.trainer_sha256 != authorization.trainer_sha256
        ):
            raise ValueError("focused lineage changed before publication")
        _fsync_directory(staging)
        parent_descriptor = os.open(
            namespace,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            with _hold_exact_result_tree(
                parent_descriptor, staging.name, expected_files,
            ) as validate:
                validate()
                _rename_directory_no_replace(
                    staging.name, destination_name,
                    parent_descriptor=parent_descriptor,
                )
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    return result


def _focused_phase_tree_commitments(namespace: Path) -> dict[str, str]:
    commitments: dict[str, str] = {}
    for phase in ("smoke", "selection", "winner"):
        root = _private_directory(namespace / phase, f"focused {phase}")
        digest = hashlib.sha256()
        digest.update(b"focused-phase-tree-v1\0")
        for path in sorted(root.iterdir(), key=lambda item: item.name):
            _, payload, file_sha256, identity = (
                ssl_core._private_regular_file_snapshot(
                    path, f"focused {phase} artifact",
                )
            )
            digest.update(path.name.encode("ascii") + b"\0")
            digest.update(file_sha256.encode("ascii"))
            digest.update(str(identity.mode).encode("ascii") + b"\0")
            digest.update(len(payload).to_bytes(8, "big"))
        commitments[phase] = digest.hexdigest()
    return commitments


def _run_focused_live_audit() -> dict[str, object]:
    """The only focused operation allowed to reauthorize live sources."""
    from scripts import prepare_dynamic_landmark_ssl_inputs as inputs_cli

    args = argparse.Namespace(
        command="verify-determinism",
        ravdess_data_root=PROJECT_ROOT / "data" / "external"
        / "ravdess_facial_tracking",
        ravdess_key=PROJECT_ROOT / "data" / "external"
        / "ravdess_facial_tracking" / ".semantic23_private_id_key",
        mayo_data_root=PROJECT_ROOT / "data" / "livelinkface_data",
        mayo_existing_export_root=PROJECT_ROOT / "data" / "mediapipe_out",
        mayo_cache_root=PRETRAINING_ROOT / "mayo_ssl_cache",
        mayo_exposure_manifest=(
            PROJECT_ROOT / "outputs" / "dynamic_landmark"
            / "mayo_exposure_manifest.json"
        ),
        mayo_key=_FOCUSED_MAYO_KEY,
        bridge_root=_FOCUSED_BRIDGE_ROOT,
        run_root=None,
    )
    captured = inputs_cli._run_mayo_cli_captured(
        args, lambda: inputs_cli._run_mayo_cli_operation(args),
    )
    result = json.loads(captured.json_line)
    if (
        type(result) is not dict
        or set(result) != {
            "bundle_count", "bundle_total_bytes", "deterministic",
            "modes_ok", "non_0600_private_file_count", "privacy_ok", "size_ok",
        }
        or not all(bool(result[name]) for name in (
            "deterministic", "modes_ok", "privacy_ok", "size_ok",
        ))
        or result.get("non_0600_private_file_count") != 0
    ):
        raise ValueError("focused strict live audit did not certify the bridge")
    ravdess_authorizer, mayo_authorizer = _authorization_factories(args)
    forbidden = _privacy_forbidden(
        args, ravdess_authorizer, mayo_authorizer,
    )
    _scan_private_results((_FOCUSED_NAMESPACE,), forbidden)
    return {
        "deterministic": True,
        "modes_ok": True,
        "privacy_ok": True,
        "size_ok": True,
        "non_0600_private_file_count": 0,
    }


def _formal_job_matrix() -> dict[str, object]:
    """Return the preregistered path-free formal job schedule."""
    jobs: list[dict[str, object]] = []
    for seed in _FORMAL_SEEDS:
        jobs.extend((
            {
                "experiment": "two_stage_fusion",
                "stage": "ravdess",
                "input_arm": ssl_core.ARM_SEMANTIC23,
                "target_schema": ssl_core.TARGET_SEMANTIC23,
                "seed": seed,
                "epochs": 30,
                "optimizer": "adamw",
                "initialization": "same_seed_fresh",
            },
            {
                "experiment": "two_stage_fusion",
                "stage": "mayo",
                "input_arm": ARM_FUSION,
                "target_schema": ssl_core.TARGET_FULL95,
                "seed": seed,
                "epochs": 30,
                "optimizer": "adamw",
                "initialization": "seed_matched_ravdess_prior",
            },
        ))
    for arm in _ABLATION_ARMS:
        for seed in _FORMAL_SEEDS:
            jobs.append({
                "experiment": "mayo_input_arm_ablation",
                "stage": "mayo",
                "input_arm": arm,
                "target_schema": ssl_core.TARGET_FULL95,
                "seed": seed,
                "epochs": 30,
                "optimizer": "adamw",
                "initialization": "same_seed_fresh",
            })
    return {
        "schema_version": "dynamic_landmark_ssl_job_matrix_v1",
        "jobs": jobs,
    }


def _expected_ablation_result_files(
    arms: tuple[str, ...] = _ABLATION_ARMS,
    seeds: tuple[int, ...] = _FORMAL_SEEDS,
) -> set[str]:
    if arms != _ABLATION_ARMS or seeds != _FORMAL_SEEDS:
        raise ValueError("Mayo ablation result matrix is not exact")
    expected = {
        "reports/mayo_input_arm_ablation.json",
    }
    for arm in arms:
        for seed in seeds:
            checkpoint = f"checkpoints/{arm}/seed_{seed}.pt"
            expected.add(checkpoint)
            expected.add(f"{checkpoint}.receipt.json")
    return expected


def _finite_nonnegative(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("ablation metric must be finite and nonnegative")
    observed = float(value)
    if observed < 0.0 or observed == float("inf") or observed != observed:
        raise ValueError("ablation metric must be finite and nonnegative")
    return observed


def _normalize_ablation_metric(value: object) -> dict[str, object]:
    if (
        type(value) is not dict
        or set(value) != {
            "raw_mae", "standardized_mae", "standardized_smooth_l1",
        }
        or type(value["raw_mae"]) is not dict
        or set(value["raw_mae"]) != {
            "blendshape72", "clinical23", "equal_block_macro", "full95",
        }
    ):
        raise ValueError("ablation common-target metric schema is not exact")
    return {
        "raw_mae": {
            name: _finite_nonnegative(metric)
            for name, metric in value["raw_mae"].items()
        },
        "standardized_mae": _finite_nonnegative(value["standardized_mae"]),
        "standardized_smooth_l1": _finite_nonnegative(
            value["standardized_smooth_l1"]
        ),
    }


def _ablation_statistics(
    runs: list[dict[str, object]],
) -> tuple[dict[str, object], dict[str, object]]:
    """Aggregate exact common-target metrics and paired seed differences."""
    if type(runs) is not list or len(runs) != 9:
        raise ValueError("ablation statistics require exactly nine jobs")
    indexed: dict[tuple[str, int], dict[str, object]] = {}
    for run in runs:
        if type(run) is not dict or set(run) != {"arm", "seed", "metrics"}:
            raise ValueError("ablation metric row schema is not exact")
        arm, seed, metrics = run["arm"], run["seed"], run["metrics"]
        if (
            arm not in _ABLATION_ARMS
            or seed not in _FORMAL_SEEDS
            or isinstance(seed, bool)
            or type(metrics) is not dict
            or set(metrics) != set(_ABLATION_BASELINES)
            or (arm, seed) in indexed
        ):
            raise ValueError("ablation metric matrix is not exact")
        indexed[(str(arm), int(seed))] = {
            baseline: _normalize_ablation_metric(metrics[baseline])
            for baseline in _ABLATION_BASELINES
        }
    if set(indexed) != {
        (arm, seed) for arm in _ABLATION_ARMS for seed in _FORMAL_SEEDS
    }:
        raise ValueError("ablation metric matrix is incomplete")

    def summarize(values: list[float]) -> dict[str, float]:
        return {
            "mean": statistics.fmean(values),
            "sd": statistics.stdev(values),
        }

    aggregate: dict[str, object] = {}
    for arm in _ABLATION_ARMS:
        aggregate[arm] = {
            baseline: {
                "raw_mae": {
                    name: summarize([
                        indexed[(arm, seed)][baseline]["raw_mae"][name]  # type: ignore[index]
                        for seed in _FORMAL_SEEDS
                    ])
                    for name in (
                        "blendshape72", "clinical23", "equal_block_macro",
                        "full95",
                    )
                },
                "standardized_mae": summarize([
                    indexed[(arm, seed)][baseline]["standardized_mae"]  # type: ignore[index]
                    for seed in _FORMAL_SEEDS
                ]),
                "standardized_smooth_l1": summarize([
                    indexed[(arm, seed)][baseline][
                        "standardized_smooth_l1"
                    ]  # type: ignore[index]
                    for seed in _FORMAL_SEEDS
                ]),
            }
            for baseline in _ABLATION_BASELINES
        }

    paired: dict[str, object] = {}
    for left, right in (
        (ARM_LANDMARK, ARM_BLENDSHAPE),
        (ARM_FUSION, ARM_BLENDSHAPE),
        (ARM_FUSION, ARM_LANDMARK),
    ):
        comparison: dict[str, object] = {"raw_mae": {}}
        for name in (
            "blendshape72", "clinical23", "equal_block_macro", "full95",
        ):
            values = {
                str(seed): (
                    indexed[(left, seed)]["trained"]["raw_mae"][name]  # type: ignore[index]
                    - indexed[(right, seed)]["trained"]["raw_mae"][name]  # type: ignore[index]
                )
                for seed in _FORMAL_SEEDS
            }
            comparison["raw_mae"][name] = {  # type: ignore[index]
                "by_seed": values,
                **summarize(list(values.values())),
            }
        for name in ("standardized_mae", "standardized_smooth_l1"):
            values = {
                str(seed): (
                    indexed[(left, seed)]["trained"][name]  # type: ignore[index]
                    - indexed[(right, seed)]["trained"][name]  # type: ignore[index]
                )
                for seed in _FORMAL_SEEDS
            }
            comparison[name] = {
                "by_seed": values,
                **summarize(list(values.values())),
            }
        paired[f"{left}_minus_{right}"] = comparison
    return aggregate, paired


def _ablation_inputs_root(run_root: Path) -> Path:
    """Return the one common atomically frozen Mayo ablation input tree."""
    return _private_directory(run_root / "inputs", "ablation inputs")


def _common_ablation_evidence(
    evidence_by_arm: Mapping[str, object],
) -> tuple[str, str, str]:
    """Require the arm snapshots to share one split, scaler, and source."""
    if set(evidence_by_arm) != set(_ABLATION_ARMS):
        raise ValueError("ablation evidence arm set is not exact")
    common_fields = (
        "source", "cache_commitment_sha256", "cache_count", "split_unit",
        "claim_unit", "patient_held_out", "train_indices_sha256",
        "heldout_indices_sha256", "group_ids_sha256", "scaler_sha256",
        "train_count", "heldout_count", "development_only",
        "sample_ids_sha256", "source_unit_ids_sha256",
        "cache_integrity_ids_sha256", "original_mapping_sha256",
        "bundle_sha256", "bundle_size_bytes", "bundle_file_count",
        "sample_count", "source_unit_count", "unique_group_count",
        "upstream_cache_count", "exclusion_count", "feature_names_sha256",
        "adapter_sha256", "temporal_policy_sha256",
        "bridge_generation_sha256", "upstream_manifest_commitments_sha256",
        "upstream_generation_closure_hmac", "canonical_key_identity_sha256",
        "source_schema", "mode", "target_schema", "experiment_kind",
        "initialization_policy",
    )
    first = evidence_by_arm[_ABLATION_ARMS[0]]
    baseline = tuple(getattr(first, name) for name in common_fields)
    for arm in _ABLATION_ARMS:
        evidence = evidence_by_arm[arm]
        if (
            tuple(getattr(evidence, name) for name in common_fields) != baseline
            or getattr(evidence, "stage", None) != "mayo"
            or getattr(evidence, "mode", None) != "formal"
            or getattr(evidence, "experiment_kind", None)
            != "mayo_input_arm_ablation"
            or getattr(evidence, "input_arm", None) != arm
            or getattr(evidence, "target_schema", None) != ssl_core.TARGET_FULL95
            or getattr(evidence, "initialization_policy", None)
            != "same_seed_fresh"
            or getattr(evidence, "prior_checkpoint_sha256", None) is not None
        ):
            raise ValueError("ablation arms do not share one frozen input contract")
    return (
        str(getattr(first, "heldout_indices_sha256")),
        str(getattr(first, "scaler_sha256")),
        str(getattr(first, "cache_commitment_sha256")),
    )


def _materialized_heldout_mask(
    evidence: object,
) -> tuple[torch.Tensor, bytes, str]:
    """Load the one persisted common formal held-out mask before any job."""
    mask = ssl_core.load_frozen_mayo_ablation_mask(evidence)
    payload = ssl_core._tensor_fingerprint_bytes(mask)
    digest = hashlib.sha256(payload).hexdigest()
    if digest != ssl_core._mask_sha256(mask):
        raise RuntimeError("materialized held-out mask digest is inconsistent")
    return mask, payload, digest


def _fresh_ablation_initialization(seed: int) -> tuple[str, str]:
    """Fingerprint the same-seed model and empty AdamW state pre-job."""
    if isinstance(seed, bool) or seed not in _FORMAL_SEEDS:
        raise ValueError("ablation initialization seed is unsupported")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = ssl_core.DynamicLandmarkSSLModel().to("cpu")
        model_digest = ssl_core._model_state_sha256(model.state_dict())
        parameter_by_name = dict(model.named_parameters())
        trainable_names = ssl_core._trainable_parameter_names(model, "mayo")
        optimizer = torch.optim.AdamW(
            [parameter_by_name[name] for name in trainable_names],
            lr=0.001,
            weight_decay=0.0001,
        )
        initial_state = optimizer.state_dict()
        if initial_state.get("state") != {}:
            raise RuntimeError("fresh AdamW unexpectedly contains mutable state")
        optimizer_material = {
            "schema_version": "adamw_empty_named_state_v1",
            "trainable_parameters": [
                {
                    "name": name,
                    "shape": list(parameter_by_name[name].shape),
                    "dtype": str(parameter_by_name[name].dtype),
                }
                for name in trainable_names
            ],
            "state_dict": initial_state,
        }
        optimizer_digest = ssl_core._canonical_sha256(optimizer_material)
    return model_digest, optimizer_digest


def _ablation_common_target_metrics(result: object) -> dict[str, object]:
    report = getattr(result, "heldout_report", None)
    if type(report) is not dict:
        raise ValueError("ablation training report is unavailable")
    common = report.get("common_target_metrics")
    if type(common) is not dict or set(common) != set(_ABLATION_BASELINES):
        raise ValueError("ablation common-target metrics are unavailable")
    metrics = {
        baseline: _normalize_ablation_metric(common[baseline])
        for baseline in _ABLATION_BASELINES
    }
    return metrics


def _private_directory(path: Path, name: str) -> Path:
    lexical = path.absolute()
    try:
        resolved = lexical.resolve(strict=True)
        status = lexical.lstat()
    except OSError as exc:
        raise ValueError(f"{name} is unavailable") from exc
    canonical = resolved == lexical
    if not canonical and sys.platform == "darwin":
        for source, destination in (
            (Path("/var"), Path("/private/var")),
            (Path("/tmp"), Path("/private/tmp")),
        ):
            try:
                relative = lexical.relative_to(source)
            except ValueError:
                continue
            if resolved == destination / relative:
                canonical = True
                break
    if (
        not canonical
        or not stat.S_ISDIR(status.st_mode)
        or status.st_uid != os.geteuid()
        or stat.S_IMODE(status.st_mode) != 0o700
    ):
        raise ValueError(f"{name} must be an owner-only canonical directory")
    return resolved


def _require_existing_input(path: Path, *, directory: bool, name: str) -> None:
    """Reject an absent or symlinked live input before any run-side mutation."""
    lexical = path.absolute()
    try:
        status = lexical.lstat()
    except OSError as exc:
        raise ValueError(f"{name} is unavailable") from exc
    expected = stat.S_ISDIR(status.st_mode) if directory else stat.S_ISREG(
        status.st_mode
    )
    if not expected or lexical.is_symlink():
        raise ValueError(f"{name} has the wrong storage type")


def _preflight_live_inputs(args: argparse.Namespace) -> None:
    for path, name in (
        (args.ravdess_data_root, "RAVDESS data root"),
        (args.mayo_data_root, "Mayo data root"),
        (args.mayo_existing_export_root, "Mayo existing-export root"),
        (args.mayo_cache_root, "Mayo cache root"),
    ):
        _require_existing_input(path, directory=True, name=name)
    for path, name in (
        (args.ravdess_key, "RAVDESS key"),
        (args.mayo_exposure_manifest, "Mayo exposure manifest"),
        (args.mayo_key, "Mayo key"),
    ):
        _require_existing_input(path, directory=False, name=name)


def _anchor_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev), int(value.st_ino), int(value.st_mode),
        int(value.st_uid), int(value.st_gid),
    )


def _validate_run_root(
    descriptor: int,
    run_root: Path,
    expected_identity: tuple[int, ...],
) -> None:
    opened = os.fstat(descriptor)
    current = os.stat(run_root, follow_symlinks=False)
    if (
        _anchor_identity(opened) != expected_identity
        or _anchor_identity(current) != expected_identity
        or not stat.S_ISDIR(opened.st_mode)
        or opened.st_uid != os.geteuid()
        or stat.S_IMODE(opened.st_mode) != 0o700
    ):
        raise ValueError("run root changed during execution")


def _validate_results_lock(
    descriptor: int,
    lock_name: str,
    run_descriptor: int,
) -> None:
    opened = os.fstat(descriptor)
    current = os.stat(
        lock_name, dir_fd=run_descriptor, follow_symlinks=False,
    )
    identity = lambda value: (
        value.st_dev, value.st_ino, value.st_mode, value.st_uid,
        value.st_gid, value.st_nlink, value.st_size,
    )
    if (
        identity(opened) != identity(current)
        or not stat.S_ISREG(opened.st_mode)
        or opened.st_uid != os.geteuid()
        or stat.S_IMODE(opened.st_mode) != 0o600
        or opened.st_nlink != 1
        or opened.st_size != 0
    ):
        raise ValueError("results lock storage is unsafe")


def _close_descriptor_sequence(descriptors: tuple[int, ...]) -> None:
    """Attempt every close while preserving active and cleanup exceptions."""
    closer = ExitStack()
    for descriptor in descriptors:
        closer.callback(os.close, descriptor)
    closer.__exit__(*sys.exc_info())


def _generation_lease_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_uid),
        int(value.st_gid),
        int(value.st_nlink),
        int(value.st_size),
    )


def _validate_generation_lease(
    *,
    parent_descriptor: int,
    parent_path: Path,
    parent_identity: tuple[int, ...],
    descriptor: int,
    lock_name: str,
    lock_identity: tuple[int, ...],
    private_parent: bool,
) -> None:
    parent_opened = os.fstat(parent_descriptor)
    parent_current = os.stat(parent_path, follow_symlinks=False)
    if (
        _anchor_identity(parent_opened) != parent_identity
        or _anchor_identity(parent_current) != parent_identity
        or not stat.S_ISDIR(parent_opened.st_mode)
        or parent_opened.st_uid != os.geteuid()
        or (
            stat.S_IMODE(parent_opened.st_mode) != 0o700
            if private_parent
            else stat.S_IMODE(parent_opened.st_mode) & 0o022 != 0
        )
    ):
        raise ValueError("generation lease parent changed")
    opened = os.fstat(descriptor)
    current = os.stat(
        lock_name,
        dir_fd=parent_descriptor,
        follow_symlinks=False,
    )
    if (
        _generation_lease_identity(opened) != lock_identity
        or _generation_lease_identity(current) != lock_identity
        or not stat.S_ISREG(opened.st_mode)
        or opened.st_uid != os.geteuid()
        or stat.S_IMODE(opened.st_mode) != 0o600
        or opened.st_nlink != 1
        or opened.st_size != 0
    ):
        raise ValueError("generation lease storage is unsafe")


@contextmanager
def _shared_generation_leases(
    args: argparse.Namespace,
) -> Iterator[Callable[[], None]]:
    """Freeze both upstream generations through final result publication."""
    leases = sorted(
        (
            (
                args.ravdess_data_root.absolute(),
                ".derived_semantic23.lock",
                False,
            ),
            (
                PRETRAINING_ROOT.absolute(),
                ".mayo_ssl_cache.lock",
                True,
            ),
        ),
        key=lambda item: os.fspath(item[0] / item[1]),
    )
    closer = ExitStack()
    held: list[
        tuple[int, Path, tuple[int, ...], int, str, tuple[int, ...], bool]
    ] = []
    try:
        for parent_path, lock_name, private_parent in leases:
            parent_descriptor = os.open(
                parent_path,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
            )
            closer.callback(os.close, parent_descriptor)
            parent_info = os.fstat(parent_descriptor)
            parent_identity = _anchor_identity(parent_info)
            if (
                not stat.S_ISDIR(parent_info.st_mode)
                or parent_info.st_uid != os.geteuid()
                or (
                    stat.S_IMODE(parent_info.st_mode) != 0o700
                    if private_parent
                    else stat.S_IMODE(parent_info.st_mode) & 0o022 != 0
                )
            ):
                raise ValueError("generation lease parent is unsafe")
            try:
                descriptor = os.open(
                    lock_name,
                    os.O_RDONLY
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_CLOEXEC", 0),
                    dir_fd=parent_descriptor,
                )
            except OSError as exc:
                if exc.errno == errno.ELOOP:
                    raise ValueError(
                        "generation lease must not be a symlink"
                    ) from exc
                raise
            closer.callback(os.close, descriptor)
            lock_identity = _generation_lease_identity(os.fstat(descriptor))
            _validate_generation_lease(
                parent_descriptor=parent_descriptor,
                parent_path=parent_path,
                parent_identity=parent_identity,
                descriptor=descriptor,
                lock_name=lock_name,
                lock_identity=lock_identity,
                private_parent=private_parent,
            )
            fcntl.flock(descriptor, fcntl.LOCK_SH)
            _validate_generation_lease(
                parent_descriptor=parent_descriptor,
                parent_path=parent_path,
                parent_identity=parent_identity,
                descriptor=descriptor,
                lock_name=lock_name,
                lock_identity=lock_identity,
                private_parent=private_parent,
            )
            held.append((
                parent_descriptor,
                parent_path,
                parent_identity,
                descriptor,
                lock_name,
                lock_identity,
                private_parent,
            ))

        def validate() -> None:
            for (
                parent_descriptor,
                parent_path,
                parent_identity,
                descriptor,
                lock_name,
                lock_identity,
                private_parent,
            ) in held:
                _validate_generation_lease(
                    parent_descriptor=parent_descriptor,
                    parent_path=parent_path,
                    parent_identity=parent_identity,
                    descriptor=descriptor,
                    lock_name=lock_name,
                    lock_identity=lock_identity,
                    private_parent=private_parent,
                )

        yield validate
        validate()
    finally:
        closer.__exit__(*sys.exc_info())


@contextmanager
def _exclusive_results_lock(
    run_root: Path,
) -> Iterator[tuple[int, str, int, tuple[int, ...]]]:
    run_descriptor = os.open(
        run_root,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    descriptor: int | None = None
    try:
        run_identity = _anchor_identity(os.fstat(run_descriptor))
        _validate_run_root(run_descriptor, run_root, run_identity)
        lock_name = ".results.lock"
        created = False
        try:
            descriptor = os.open(
                lock_name,
                os.O_RDWR | os.O_CREAT | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=run_descriptor,
            )
            created = True
        except FileExistsError:
            descriptor = os.open(
                lock_name,
                os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=run_descriptor,
            )
        if created:
            os.fchmod(descriptor, 0o600)
            os.fsync(descriptor)
            os.fsync(run_descriptor)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        _validate_run_root(run_descriptor, run_root, run_identity)
        _validate_results_lock(descriptor, lock_name, run_descriptor)
        yield descriptor, lock_name, run_descriptor, run_identity
    finally:
        descriptors = (
            (run_descriptor,)
            + ((descriptor,) if descriptor is not None else ())
        )
        _close_descriptor_sequence(descriptors)


@contextmanager
def _exclusive_results_transaction(
    run_root: Path,
    args: argparse.Namespace,
) -> Iterator[
    tuple[int, str, int, tuple[int, ...], Callable[[], None]]
]:
    with _shared_generation_leases(args) as validate_generation_leases:
        with _exclusive_results_lock(run_root) as locked:
            yield (*locked, validate_generation_leases)


def _rename_directory_no_replace(
    source_name: str,
    destination_name: str,
    *,
    parent_descriptor: int,
) -> None:
    """Atomically publish one directory without replacing any destination."""
    library = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source_name)
    destination_bytes = os.fsencode(destination_name)
    if sys.platform == "darwin":
        rename = library.renameatx_np
        rename.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        rename.restype = ctypes.c_int
        result = rename(
            parent_descriptor,
            source_bytes,
            parent_descriptor,
            destination_bytes,
            0x00000004 | 0x00000010,
        )
    elif sys.platform.startswith("linux"):
        rename = library.renameat2
        rename.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        rename.restype = ctypes.c_int
        result = rename(
            parent_descriptor,
            source_bytes,
            parent_descriptor,
            destination_bytes,
            1,
        )
    else:
        raise OSError(
            errno.ENOTSUP,
            "atomic no-replace directory publication is unavailable",
        )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(
            error_number,
            os.strerror(error_number),
            destination_name,
        )


def _classify_results_publication(
    parent_descriptor: int,
    source_name: str,
    destination_name: str,
    expected_identity: tuple[int, int],
) -> str:
    """Classify an atomic rename outcome against the held staging inode."""
    observed: dict[str, tuple[int, int] | None] = {}
    for field, name in (
        ("source", source_name), ("destination", destination_name),
    ):
        try:
            status = os.stat(
                name, dir_fd=parent_descriptor, follow_symlinks=False,
            )
        except FileNotFoundError:
            observed[field] = None
        else:
            observed[field] = (int(status.st_dev), int(status.st_ino))
    if (
        observed["source"] == expected_identity
        and observed["destination"] is None
    ):
        return "staged"
    if (
        observed["destination"] == expected_identity
        and observed["source"] is None
    ):
        return "published"
    return "indeterminate"


def _write_private_json(path: Path, value: dict[str, object]) -> None:
    encoded = (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    ).encode("ascii")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        os.fchmod(descriptor, 0o600)
        offset = 0
        while offset < len(encoded):
            written = os.write(descriptor, encoded[offset:])
            if written < 1:
                raise OSError("private JSON write made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ledger_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev), int(value.st_ino), int(value.st_mode),
        int(value.st_uid), int(value.st_gid), int(value.st_nlink),
        int(value.st_size), int(value.st_mtime_ns), int(value.st_ctime_ns),
    )


def _directory_ledger_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev), int(value.st_ino), int(value.st_mode),
        int(value.st_uid), int(value.st_gid), int(value.st_nlink),
    )


def _hash_held_file(descriptor: int, expected_size: int) -> str:
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    total = 0
    while chunk := os.read(descriptor, 1024 * 1024):
        total += len(chunk)
        if total > expected_size:
            raise ValueError("result file grew during held verification")
        digest.update(chunk)
    if total != expected_size:
        raise ValueError("result file changed size during held verification")
    return digest.hexdigest()


@contextmanager
def _hold_exact_result_tree(
    run_descriptor: int,
    staging_name: str,
    expected_files: set[str],
) -> Iterator[Callable[..., None]]:
    directory_flags = (
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    root_descriptor = os.open(
        staging_name, directory_flags, dir_fd=run_descriptor,
    )
    directory_descriptors: dict[tuple[str, ...], int] = {
        (): root_descriptor,
    }
    file_descriptors: dict[tuple[str, ...], int] = {}
    identities: dict[tuple[str, ...], tuple[int, ...]] = {}
    digests: dict[tuple[str, ...], str] = {}
    expected_children: dict[tuple[str, ...], set[str]] = {(): set()}
    try:
        file_parts = {tuple(Path(value).parts) for value in expected_files}
        directory_parts: set[tuple[str, ...]] = set()
        for parts in file_parts:
            for size in range(1, len(parts)):
                directory_parts.add(parts[:size])
        if (
            not file_parts
            or 1 + len(directory_parts) + len(file_parts)
            > _MAX_EXACT_RESULT_TREE_ENTRIES
            or max(len(parts) for parts in file_parts)
            > _MAX_EXACT_RESULT_TREE_DEPTH
        ):
            raise ValueError("result tree exceeds its exact structural budget")
        for parts in sorted(directory_parts, key=lambda value: (len(value), value)):
            parent = parts[:-1]
            expected_children.setdefault(parent, set()).add(parts[-1])
            expected_children.setdefault(parts, set())
            descriptor = os.open(
                parts[-1], directory_flags,
                dir_fd=directory_descriptors[parent],
            )
            directory_descriptors[parts] = descriptor
            status = os.fstat(descriptor)
            if (
                not stat.S_ISDIR(status.st_mode)
                or status.st_uid != os.geteuid()
                or stat.S_IMODE(status.st_mode) != 0o700
            ):
                raise ValueError("result directory is not owner-only")
        for parts in sorted(file_parts):
            parent = parts[:-1]
            expected_children.setdefault(parent, set()).add(parts[-1])
            descriptor = os.open(
                parts[-1],
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_descriptors[parent],
            )
            file_descriptors[parts] = descriptor
            status = os.fstat(descriptor)
            if (
                not stat.S_ISREG(status.st_mode)
                or status.st_uid != os.geteuid()
                or stat.S_IMODE(status.st_mode) != 0o600
                or status.st_nlink != 1
            ):
                raise ValueError("result file is not private regular storage")
            identities[parts] = _ledger_identity(status)
        total_regular_bytes = sum(
            int(identity[6]) for identity in identities.values()
        )
        if total_regular_bytes > _MAX_EXACT_RESULT_TREE_REGULAR_BYTES:
            raise ValueError("result tree exceeds its regular-payload budget")
        for parts, descriptor in file_descriptors.items():
            digests[parts] = _hash_held_file(
                descriptor, int(identities[parts][6]),
            )
        for parts, descriptor in directory_descriptors.items():
            identities[parts] = _directory_ledger_identity(
                os.fstat(descriptor)
            )

        def validate(live_root_name: str = staging_name) -> None:
            for parts, descriptor in directory_descriptors.items():
                parent_descriptor = (
                    run_descriptor
                    if not parts
                    else directory_descriptors[parts[:-1]]
                )
                name = live_root_name if not parts else parts[-1]
                live = os.stat(
                    name, dir_fd=parent_descriptor, follow_symlinks=False,
                )
                if (
                    _directory_ledger_identity(os.fstat(descriptor))
                    != identities[parts]
                    or _directory_ledger_identity(live) != identities[parts]
                    or set(os.listdir(descriptor))
                    != expected_children.get(parts, set())
                ):
                    raise ValueError("result directory changed before publication")
            for parts, descriptor in file_descriptors.items():
                live = os.stat(
                    parts[-1],
                    dir_fd=directory_descriptors[parts[:-1]],
                    follow_symlinks=False,
                )
                expected = identities[parts]
                if (
                    _ledger_identity(os.fstat(descriptor)) != expected
                    or _ledger_identity(live) != expected
                    or _hash_held_file(descriptor, expected[6]) != digests[parts]
                ):
                    raise ValueError("result file changed before publication")

        validate()
        yield validate
    finally:
        directory_order = tuple(
            directory_descriptors[parts]
            for parts in sorted(
                directory_descriptors, key=lambda value: (len(value), value),
            )
        )
        _close_descriptor_sequence(
            directory_order + tuple(file_descriptors.values())
        )


def _stage_report(result, checkpoint_fingerprint: str) -> dict[str, object]:
    report = result.heldout_report
    value: dict[str, object] = {
        "checkpoint_fingerprint": checkpoint_fingerprint,
        "optimizer_steps": result.training_receipt.optimizer_steps,
        "seed": result.training_receipt.seed,
    }
    if result.training_receipt.prior_checkpoint_sha256 is not None:
        value["prior_checkpoint_fingerprint"] = (
            result.training_receipt.prior_checkpoint_sha256
        )
    if result.stage_evidence.mode == "smoke":
        value["train_loss"] = report["train_loss"]
    else:
        value["reconstruction"] = dict(report)
    return value


def _formal_aggregates(
    execution: list[dict[str, object]],
) -> dict[str, object]:
    metrics = {
        "ravdess_only": ("trained", "untrained", "train_mean"),
        "ravdess_then_mayo": (
            "trained", "prior_ravdess", "fresh_untrained", "train_mean",
        ),
    }
    aggregate: dict[str, object] = {}
    for stage, names in metrics.items():
        stage_aggregate: dict[str, object] = {}
        for name in names:
            values = [
                float(run[stage]["reconstruction"][name])  # type: ignore[index]
                for run in execution
            ]
            if len(values) != 3 or not all(
                value >= 0.0 and value < float("inf") for value in values
            ):
                raise ValueError("formal reconstruction evidence is incomplete")
            stage_aggregate[name] = {
                "mean": statistics.fmean(values),
                "sd": statistics.stdev(values),
            }
        aggregate[stage] = stage_aggregate
    return aggregate


def _expected_result_files(mode: str, seeds: tuple[int, ...]) -> set[str]:
    if mode not in {"smoke", "formal"}:
        raise ValueError("result file mode is unsupported")
    expected: set[str] = set()
    for seed in seeds:
        prefix = "checkpoints" if mode == "smoke" else (
            f"checkpoints/seed_{seed}"
        )
        for name in ("ravdess_only.pt", "ravdess_then_mayo.pt"):
            expected.add(f"{prefix}/{name}")
            expected.add(f"{prefix}/{name}.receipt.json")
    report_name = (
        "execution_only.json"
        if mode == "smoke"
        else "formal_pretraining_results.json"
    )
    expected.add(f"reports/{report_name}")
    return expected


def _publish_validated_results(
    *,
    run_descriptor: int,
    run_root: Path,
    run_identity: tuple[int, ...],
    results_lock: int,
    lock_name: str,
    staging_name: str,
    staged_identity: tuple[int, int],
    inputs_root: Path,
    privacy_forbidden,
    validate_result_tree: Callable[..., None],
    final_authorization: Callable[[], None],
) -> None:
    """Publish and classify one exact result tree through held descriptors."""
    staging = run_root / staging_name
    validate_result_tree()
    _quiet_call(
        _scan_private_results,
        (inputs_root, staging),
        privacy_forbidden,
    )
    validate_result_tree()
    _validate_run_root(run_descriptor, run_root, run_identity)
    _validate_results_lock(results_lock, lock_name, run_descriptor)
    os.fsync(run_descriptor)
    validate_result_tree()
    _quiet_call(final_authorization)
    rename_error: BaseException | None = None
    try:
        _rename_directory_no_replace(
            staging_name,
            "results",
            parent_descriptor=run_descriptor,
        )
    except BaseException as caught:
        rename_error = caught
    publication_state = _classify_results_publication(
        run_descriptor,
        staging_name,
        "results",
        staged_identity,
    )
    if publication_state != "published":
        if publication_state == "staged" and rename_error is not None:
            raise rename_error
        raise RuntimeError(
            "results publication outcome is indeterminate"
        ) from rename_error
    try:
        # A faulting wrapper may report failure after the syscall committed.
        # Success requires the canonical name, exact held tree, privacy closure,
        # lock identity, and parent durability all to revalidate.
        validate_result_tree("results")
        _quiet_call(
            _scan_private_results,
            (inputs_root, run_root / "results"),
            privacy_forbidden,
        )
        validate_result_tree("results")
        _validate_run_root(run_descriptor, run_root, run_identity)
        _validate_results_lock(results_lock, lock_name, run_descriptor)
        os.fsync(run_descriptor)
        _validate_run_root(run_descriptor, run_root, run_identity)
        _validate_results_lock(results_lock, lock_name, run_descriptor)
        validate_result_tree("results")
        _quiet_call(
            _scan_private_results,
            (inputs_root, run_root / "results"),
            privacy_forbidden,
        )
        validate_result_tree("results")
    except BaseException as caught:
        raise RuntimeError(
            "published results are retained as indeterminate evidence"
        ) from caught
    if rename_error is not None:
        raise RuntimeError(
            "published results are retained after a rename return fault"
        ) from rename_error


def _run_two_stage(
    args: argparse.Namespace,
    *,
    producer_sha256: str,
) -> dict[str, object]:
    run_root = _private_directory(args.run_root.absolute(), "run root")
    if args.mode == "smoke":
        namespace_root = run_root.parent.parent
        namespace_ok = (
            run_root.parent.name == "smoke"
            and _RUN_ID.fullmatch(run_root.name) is not None
        )
    else:
        namespace_root = run_root.parent
        namespace_ok = run_root.name == "formal"
    if (
        not namespace_ok
        or namespace_root != PRETRAINING_ROOT.absolute()
    ):
        raise ValueError("run root is outside the exact mode/run-id namespace")
    inputs_root = _private_directory(run_root / "inputs", "frozen inputs")
    bridge_root = _private_directory(
        args.bridge_root.absolute(), "bridge generation"
    )
    if bridge_root != (PRETRAINING_ROOT / "bridge").absolute():
        raise ValueError("bridge root is outside the canonical namespace")
    _preflight_live_inputs(args)
    raw_ravdess_authorizer, raw_mayo_authorizer = _quiet_call(
        _authorization_factories, args,
    )
    ravdess_authorizer = _transaction_authorizer(raw_ravdess_authorizer)
    mayo_authorizer = _transaction_authorizer(raw_mayo_authorizer)
    privacy_forbidden = _quiet_call(
        _privacy_forbidden,
        args, ravdess_authorizer, mayo_authorizer,
    )
    seeds = (0,) if args.mode == "smoke" else (0, 1, 2)
    summary = {
        "checkpoint_count": len(seeds) * 2,
        "mode": args.mode,
        "seed_count": len(seeds),
        "stage_count": 2,
    }
    _assert_summary_private(
        json.dumps(
            summary, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("ascii"),
        privacy_forbidden,
    )
    with _exclusive_results_transaction(run_root, args) as (
        results_lock,
        lock_name,
        run_descriptor,
        run_identity,
        validate_generation_leases,
    ):
        names = set(os.listdir(run_descriptor))
        if "results" in names:
            raise FileExistsError("results generation already exists")
        if any(name.startswith(".results.staging-") for name in names):
            raise ValueError("unresolved results staging state blocks execution")

        ravdess_evidence = _quiet_call(
            ssl_core.authorize_frozen_ssl_stage,
            stage="ravdess",
            mode=args.mode,
            inputs_root=inputs_root,
            bridge_root=bridge_root,
            ravdess_authorizer=ravdess_authorizer,
            mayo_authorizer=mayo_authorizer,
            producer_sha256=producer_sha256,
        )
        staging = run_root / f".results.staging-{secrets.token_hex(16)}"
        staging.mkdir(mode=0o700)
        os.chmod(staging, 0o700)
        staged_status = staging.stat()
        staged_identity = (
            staged_status.st_dev,
            staged_status.st_ino,
        )
        checkpoints = staging / "checkpoints"
        reports = staging / "reports"
        checkpoints.mkdir(mode=0o700)
        reports.mkdir(mode=0o700)
        os.chmod(checkpoints, 0o700)
        os.chmod(reports, 0o700)

        execution: list[dict[str, object]] = []
        reload_checks: list[tuple[Path, object, object, str]] = []
        publication_lineages: list[
            tuple[Path, object, str, Path, object, str]
        ] = []
        retained_runtime: list[object] = []
        checkpoint_directories = {checkpoints}
        for seed in seeds:
            seed_checkpoints = checkpoints
            if args.mode == "formal":
                seed_checkpoints = checkpoints / f"seed_{seed}"
                seed_checkpoints.mkdir(mode=0o700)
                os.chmod(seed_checkpoints, 0o700)
                checkpoint_directories.add(seed_checkpoints)
            ravdess_result = _quiet_call(
                ssl_core.train_ssl_stage,
                stage_evidence=ravdess_evidence,
                seed=seed,
            )
            ravdess_payload = _quiet_call(
                ssl_core.build_ssl_checkpoint_payload, ravdess_result,
            )
            ravdess_path = seed_checkpoints / "ravdess_only.pt"
            ravdess_receipt = _quiet_call(
                ssl_core.save_ssl_checkpoint,
                ravdess_path,
                ravdess_payload,
                stage_evidence=ravdess_evidence,
            )
            persisted_ravdess = _quiet_call(
                ssl_core.load_ssl_checkpoint,
                ravdess_path,
                receipt=ravdess_receipt,
                stage_evidence=ravdess_evidence,
            )
            retained_runtime.extend((ravdess_receipt, persisted_ravdess))
            ravdess_fingerprint = _quiet_call(
                ssl_core.ssl_checkpoint_fingerprint, persisted_ravdess,
            )
            reload_checks.append((
                ravdess_path, ravdess_receipt, ravdess_evidence,
                ravdess_fingerprint,
            ))

            mayo_evidence = _quiet_call(
                ssl_core.authorize_frozen_ssl_stage,
                stage="mayo",
                mode=args.mode,
                inputs_root=inputs_root,
                bridge_root=bridge_root,
                ravdess_authorizer=ravdess_authorizer,
                mayo_authorizer=mayo_authorizer,
                producer_sha256=producer_sha256,
                prior_ravdess_checkpoint=persisted_ravdess,
                prior_ravdess_evidence=ravdess_evidence,
            )
            retained_runtime.append(mayo_evidence)
            mayo_result = _quiet_call(
                ssl_core.train_ssl_stage,
                stage_evidence=mayo_evidence,
                seed=seed,
                prior_ravdess_checkpoint=persisted_ravdess,
                prior_stage_evidence=ravdess_evidence,
            )
            mayo_payload = _quiet_call(
                ssl_core.build_ssl_checkpoint_payload, mayo_result,
            )
            mayo_path = seed_checkpoints / "ravdess_then_mayo.pt"
            mayo_receipt = _quiet_call(
                ssl_core.save_ssl_checkpoint,
                mayo_path,
                mayo_payload,
                stage_evidence=mayo_evidence,
            )
            persisted_mayo = _quiet_call(
                ssl_core.load_ssl_checkpoint,
                mayo_path,
                receipt=mayo_receipt,
                stage_evidence=mayo_evidence,
            )
            retained_runtime.extend((mayo_receipt, persisted_mayo))
            mayo_fingerprint = _quiet_call(
                ssl_core.ssl_checkpoint_fingerprint, persisted_mayo,
            )
            reload_checks.append((
                mayo_path, mayo_receipt, mayo_evidence, mayo_fingerprint,
            ))
            publication_lineages.append((
                ravdess_path,
                ravdess_receipt,
                ravdess_fingerprint,
                mayo_path,
                mayo_receipt,
                mayo_fingerprint,
            ))
            execution.append({
                "seed": seed,
                "ravdess_only": _stage_report(
                    ravdess_result, ravdess_fingerprint,
                ),
                "ravdess_then_mayo": _stage_report(
                    mayo_result, mayo_fingerprint,
                ),
            })

        report = {
            "schema_version": "dynamic_landmark_ssl_execution_report_v1",
            "mode": args.mode,
            "seed_count": len(seeds),
            "stage_count": 2,
            "runs": execution,
            "medical_generalization": False,
            "outer_mayo_predictions_viewed": False,
        }
        if args.mode == "formal":
            report["aggregate"] = _formal_aggregates(execution)
        report_name = (
            "execution_only.json"
            if args.mode == "smoke"
            else "formal_pretraining_results.json"
        )
        report_path = reports / report_name
        _quiet_call(_write_private_json, report_path, report)
        (
            _report_resolved,
            report_bytes,
            _report_sha256,
            _report_identity,
        ) = _quiet_call(
            ssl_core._private_regular_file_snapshot,
            report_path,
            "SSL execution report",
        )
        report_value = _quiet_call(
            ssl_core._strict_json_mapping,
            report_bytes,
            "SSL execution report",
        )
        if not ssl_core._exact_json_value(report_value, report):
            raise ValueError("SSL execution report changed after writing")
        for path, receipt, evidence, expected in reload_checks:
            reloaded = _quiet_call(
                ssl_core.load_ssl_checkpoint,
                path, receipt=receipt, stage_evidence=evidence,
            )
            if _quiet_call(
                ssl_core.ssl_checkpoint_fingerprint, reloaded,
            ) != expected:
                raise ValueError("checkpoint changed during result finalization")
        expected_files = _expected_result_files(args.mode, seeds)

        def final_authorization() -> None:
            for (
                ravdess_path,
                ravdess_receipt,
                expected_ravdess,
                mayo_path,
                mayo_receipt,
                expected_mayo,
            ) in publication_lineages:
                final_ravdess = ssl_core.load_ssl_checkpoint(
                    ravdess_path,
                    receipt=ravdess_receipt,
                    stage_evidence=ravdess_evidence,
                )
                if (
                    ssl_core.ssl_checkpoint_fingerprint(final_ravdess)
                    != expected_ravdess
                ):
                    raise ValueError(
                        "RAVDESS checkpoint changed before publication"
                    )
                final_mayo_evidence = ssl_core.authorize_frozen_ssl_stage(
                    stage="mayo",
                    mode=args.mode,
                    inputs_root=inputs_root,
                    bridge_root=bridge_root,
                    ravdess_authorizer=ravdess_authorizer,
                    mayo_authorizer=mayo_authorizer,
                    producer_sha256=producer_sha256,
                    prior_ravdess_checkpoint=final_ravdess,
                    prior_ravdess_evidence=ravdess_evidence,
                )
                final_mayo = ssl_core.load_ssl_checkpoint(
                    mayo_path,
                    receipt=mayo_receipt,
                    stage_evidence=final_mayo_evidence,
                )
                if (
                    ssl_core.ssl_checkpoint_fingerprint(final_mayo)
                    != expected_mayo
                ):
                    raise ValueError(
                        "Mayo checkpoint changed before publication"
                    )
            validate_generation_leases()
            edge_mayo = raw_mayo_authorizer()
            validate_generation_leases()
            edge_ravdess = raw_ravdess_authorizer()
            validate_generation_leases()
            _require_publication_edge_authorization(
                stage="mayo",
                evidence=mayo_evidence,
                authorization=edge_mayo,
            )
            _require_publication_edge_authorization(
                stage="ravdess",
                evidence=ravdess_evidence,
                authorization=edge_ravdess,
            )
            validate_generation_leases()

        with _hold_exact_result_tree(
            run_descriptor, staging.name, expected_files,
        ) as validate_result_tree:
            _quiet_call(
                _scan_private_results,
                (inputs_root, staging), privacy_forbidden,
            )
            for checkpoint_directory in sorted(
                checkpoint_directories,
                key=lambda path: len(path.parts),
                reverse=True,
            ):
                _quiet_call(_fsync_directory, checkpoint_directory)
            _quiet_call(_fsync_directory, reports)
            _quiet_call(_fsync_directory, staging)
            if "results" in set(os.listdir(run_descriptor)):
                raise FileExistsError(
                    "results generation appeared during execution"
                )
            before_publish = os.stat(
                staging.name,
                dir_fd=run_descriptor,
                follow_symlinks=False,
            )
            if (before_publish.st_dev, before_publish.st_ino) != staged_identity:
                raise ValueError(
                    "results staging identity changed before publication"
                )
            _quiet_call(
                _scan_private_results,
                (inputs_root, staging), privacy_forbidden,
            )
            validate_result_tree()
            _validate_run_root(run_descriptor, run_root, run_identity)
            _validate_results_lock(
                results_lock, lock_name, run_descriptor,
            )
            os.fsync(run_descriptor)
            _publish_validated_results(
                run_descriptor=run_descriptor,
                run_root=run_root,
                run_identity=run_identity,
                results_lock=results_lock,
                lock_name=lock_name,
                staging_name=staging.name,
                staged_identity=staged_identity,
                inputs_root=inputs_root,
                privacy_forbidden=privacy_forbidden,
                validate_result_tree=validate_result_tree,
                final_authorization=final_authorization,
            )

    return summary


def _run_mayo_ablation(
    args: argparse.Namespace,
    *,
    producer_sha256: str,
) -> dict[str, object]:
    """Run and atomically publish the exact nine-job Mayo-only ablation."""
    run_root = _private_directory(args.run_root.absolute(), "ablation run root")
    if run_root != _MAYO_ABLATION_NAMESPACE.absolute():
        raise ValueError("ablation run root is outside its canonical namespace")
    inputs_root = _ablation_inputs_root(run_root)
    bridge_root = _private_directory(
        args.bridge_root.absolute(), "bridge generation",
    )
    if bridge_root != (PRETRAINING_ROOT / "bridge").absolute():
        raise ValueError("bridge root is outside the canonical namespace")
    _preflight_live_inputs(args)
    raw_ravdess_authorizer, raw_mayo_authorizer = _quiet_call(
        _authorization_factories, args,
    )
    ravdess_authorizer = _transaction_authorizer(raw_ravdess_authorizer)
    mayo_authorizer = _transaction_authorizer(raw_mayo_authorizer)
    privacy_forbidden = _quiet_call(
        _privacy_forbidden,
        args, ravdess_authorizer, mayo_authorizer,
    )
    summary = {
        "arm_count": 3,
        "checkpoint_count": 9,
        "job_count": 9,
        "mode": "formal",
        "seed_count": 3,
        "stage_count": 1,
    }
    _assert_summary_private(
        json.dumps(
            summary, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("ascii"),
        privacy_forbidden,
    )
    with _exclusive_results_transaction(run_root, args) as (
        results_lock,
        lock_name,
        run_descriptor,
        run_identity,
        validate_generation_leases,
    ):
        names = set(os.listdir(run_descriptor))
        if "results" in names:
            raise FileExistsError("results generation already exists")
        if any(name.startswith(".results.staging-") for name in names):
            raise ValueError("unresolved results staging state blocks execution")

        evidence_by_arm = {
            arm: _quiet_call(
                ssl_core.authorize_frozen_ssl_stage,
                stage="mayo",
                mode="formal",
                inputs_root=inputs_root,
                bridge_root=bridge_root,
                ravdess_authorizer=ravdess_authorizer,
                mayo_authorizer=mayo_authorizer,
                producer_sha256=producer_sha256,
                experiment_kind="mayo_input_arm_ablation",
                mayo_input_arm=arm,
            )
            for arm in _ABLATION_ARMS
        }
        (
            common_split_sha256,
            common_scaler_sha256,
            common_cache_sha256,
        ) = _quiet_call(_common_ablation_evidence, evidence_by_arm)
        common_mask, common_mask_bytes, common_mask_sha256 = _quiet_call(
            _materialized_heldout_mask, evidence_by_arm[ARM_FUSION],
        )

        initialization: dict[tuple[str, int], tuple[str, str]] = {}
        for seed in _FORMAL_SEEDS:
            seed_digests: list[tuple[str, str]] = []
            for arm in _ABLATION_ARMS:
                digests = _quiet_call(_fresh_ablation_initialization, seed)
                initialization[(arm, seed)] = digests
                seed_digests.append(digests)
            if len(set(seed_digests)) != 1:
                raise ValueError(
                    "same-seed ablation model or optimizer initialization differs"
                )

        staging = run_root / f".results.staging-{secrets.token_hex(16)}"
        staging.mkdir(mode=0o700)
        os.chmod(staging, 0o700)
        staged_status = staging.stat()
        staged_identity = (staged_status.st_dev, staged_status.st_ino)
        checkpoints = staging / "checkpoints"
        reports = staging / "reports"
        checkpoints.mkdir(mode=0o700)
        reports.mkdir(mode=0o700)
        os.chmod(checkpoints, 0o700)
        os.chmod(reports, 0o700)

        arm_directories: dict[str, Path] = {}
        checkpoint_directories = {checkpoints}
        for arm in _ABLATION_ARMS:
            directory = checkpoints / arm
            directory.mkdir(mode=0o700)
            os.chmod(directory, 0o700)
            arm_directories[arm] = directory
            checkpoint_directories.add(directory)

        metric_rows: list[dict[str, object]] = []
        report_runs: list[dict[str, object]] = []
        reload_checks: list[tuple[Path, object, object, str]] = []
        publication_lineages: list[
            tuple[str, int, Path, object, str]
        ] = []
        retained_runtime: list[object] = list(evidence_by_arm.values())
        for arm in _ABLATION_ARMS:
            evidence = evidence_by_arm[arm]
            for seed in _FORMAL_SEEDS:
                result = _quiet_call(
                    ssl_core.train_ssl_stage,
                    stage_evidence=evidence,
                    seed=seed,
                    heldout_mask=common_mask,
                )
                receipt = result.training_receipt
                model_init, optimizer_init = initialization[(arm, seed)]
                if (
                    receipt.input_arm != arm
                    or receipt.target_schema != ssl_core.TARGET_FULL95
                    or receipt.experiment_kind != "mayo_input_arm_ablation"
                    or receipt.initialization_policy != "same_seed_fresh"
                    or receipt.prior_checkpoint_sha256 is not None
                    or receipt.seed != seed
                    or receipt.epochs != 30
                    or receipt.optimizer != "adamw"
                    or receipt.optimizer_steps != 30
                    or receipt.pre_state_sha256 != model_init
                    or receipt.heldout_mask_schedule_sha256
                    != common_mask_sha256
                    or receipt.heldout_indices_sha256 != common_split_sha256
                    or receipt.scaler_sha256 != common_scaler_sha256
                    or receipt.cache_binding_sha256 != common_cache_sha256
                ):
                    raise ValueError("ablation job violated its frozen contract")
                payload = _quiet_call(
                    ssl_core.build_ssl_checkpoint_payload, result,
                )
                path = arm_directories[arm] / f"seed_{seed}.pt"
                checkpoint_receipt = _quiet_call(
                    ssl_core.save_ssl_checkpoint,
                    path,
                    payload,
                    stage_evidence=evidence,
                )
                persisted = _quiet_call(
                    ssl_core.load_ssl_checkpoint,
                    path,
                    receipt=checkpoint_receipt,
                    stage_evidence=evidence,
                )
                fingerprint = _quiet_call(
                    ssl_core.ssl_checkpoint_fingerprint, persisted,
                )
                retained_runtime.extend((
                    result, checkpoint_receipt, persisted,
                ))
                reload_checks.append((
                    path, checkpoint_receipt, evidence, fingerprint,
                ))
                publication_lineages.append((
                    arm, seed, path, checkpoint_receipt, fingerprint,
                ))
                metrics = _quiet_call(_ablation_common_target_metrics, result)
                metric_rows.append({
                    "arm": arm,
                    "seed": seed,
                    "metrics": metrics,
                })
                per_recording = result.heldout_report.get(
                    "per_recording_metrics",
                )
                if type(per_recording) is not list or not per_recording:
                    raise ValueError(
                        "ablation private per-recording metrics are unavailable"
                    )
                report_runs.append({
                    "arm": arm,
                    "seed": seed,
                    "checkpoint_fingerprint": fingerprint,
                    "optimizer_steps": receipt.optimizer_steps,
                    "pre_state_sha256": model_init,
                    "optimizer_initial_state_sha256": optimizer_init,
                    "heldout_mask_sha256": common_mask_sha256,
                    "common_target_metrics": metrics,
                    "per_recording_metrics": per_recording,
                })

        aggregate, paired = _quiet_call(_ablation_statistics, metric_rows)
        report = {
            "schema_version": "dynamic_landmark_ssl_ablation_report_v1",
            "mode": "formal",
            "experiment_kind": "mayo_input_arm_ablation",
            "arms": list(_ABLATION_ARMS),
            "seeds": list(_FORMAL_SEEDS),
            "job_count": 9,
            "epochs": 30,
            "optimizer": "adamw",
            "target_schema": ssl_core.TARGET_FULL95,
            "shared_split_sha256": common_split_sha256,
            "shared_scaler_sha256": common_scaler_sha256,
            "shared_cache_sha256": common_cache_sha256,
            "shared_heldout_mask_sha256": common_mask_sha256,
            "heldout_mask_materialized_once": True,
            "input_mask_after_common_scaler": True,
            "prior_ravdess": False,
            "early_stopping": False,
            "retry_count": 0,
            "aggregation": "per_recording_then_equal_recording_mean",
            "runs": report_runs,
            "aggregate": aggregate,
            "paired_differences": paired,
            "medical_generalization": False,
            "hb_evaluation": False,
        }
        report_path = reports / "mayo_input_arm_ablation.json"
        _quiet_call(_write_private_json, report_path, report)
        (
            _report_resolved,
            report_bytes,
            _report_sha256,
            _report_identity,
        ) = _quiet_call(
            ssl_core._private_regular_file_snapshot,
            report_path,
            "Mayo ablation report",
        )
        report_value = _quiet_call(
            ssl_core._strict_json_mapping,
            report_bytes,
            "Mayo ablation report",
        )
        if not ssl_core._exact_json_value(report_value, report):
            raise ValueError("Mayo ablation report changed after writing")
        for path, receipt, evidence, expected in reload_checks:
            reloaded = _quiet_call(
                ssl_core.load_ssl_checkpoint,
                path,
                receipt=receipt,
                stage_evidence=evidence,
            )
            if _quiet_call(
                ssl_core.ssl_checkpoint_fingerprint, reloaded,
            ) != expected:
                raise ValueError("ablation checkpoint changed during finalization")

        def final_authorization() -> None:
            edge_ravdess = _transaction_authorizer(raw_ravdess_authorizer)
            edge_mayo = _transaction_authorizer(raw_mayo_authorizer)
            final_evidence = {
                arm: ssl_core.authorize_frozen_ssl_stage(
                    stage="mayo",
                    mode="formal",
                    inputs_root=inputs_root,
                    bridge_root=bridge_root,
                    ravdess_authorizer=edge_ravdess,
                    mayo_authorizer=edge_mayo,
                    producer_sha256=producer_sha256,
                    experiment_kind="mayo_input_arm_ablation",
                    mayo_input_arm=arm,
                )
                for arm in _ABLATION_ARMS
            }
            _common_ablation_evidence(final_evidence)
            final_mask = ssl_core.load_frozen_mayo_ablation_mask(
                final_evidence[ARM_FUSION]
            )
            if (
                ssl_core._tensor_fingerprint_bytes(final_mask)
                != common_mask_bytes
                or ssl_core._mask_sha256(final_mask) != common_mask_sha256
            ):
                raise ValueError(
                    "common Mayo ablation mask changed before publication"
                )
            for arm, seed, path, receipt, expected in publication_lineages:
                loaded = ssl_core.load_ssl_checkpoint(
                    path,
                    receipt=receipt,
                    stage_evidence=final_evidence[arm],
                )
                if ssl_core.ssl_checkpoint_fingerprint(loaded) != expected:
                    raise ValueError(
                        "ablation checkpoint changed before publication"
                    )
                if loaded["metadata"]["seed"] != seed:
                    raise ValueError(
                        "ablation checkpoint seed changed before publication"
                    )
            validate_generation_leases()

        expected_files = _expected_ablation_result_files()
        with _hold_exact_result_tree(
            run_descriptor, staging.name, expected_files,
        ) as validate_result_tree:
            _quiet_call(
                _scan_private_results,
                (inputs_root, staging),
                privacy_forbidden,
            )
            for directory in sorted(
                checkpoint_directories,
                key=lambda path: len(path.parts),
                reverse=True,
            ):
                _quiet_call(_fsync_directory, directory)
            _quiet_call(_fsync_directory, reports)
            _quiet_call(_fsync_directory, staging)
            if "results" in set(os.listdir(run_descriptor)):
                raise FileExistsError(
                    "results generation appeared during execution"
                )
            before_publish = os.stat(
                staging.name,
                dir_fd=run_descriptor,
                follow_symlinks=False,
            )
            if (before_publish.st_dev, before_publish.st_ino) != staged_identity:
                raise ValueError(
                    "results staging identity changed before publication"
                )
            validate_result_tree()
            _validate_run_root(run_descriptor, run_root, run_identity)
            _validate_results_lock(results_lock, lock_name, run_descriptor)
            os.fsync(run_descriptor)
            # Scan the common parent as well as every arm-specific frozen tree.
            _quiet_call(
                _scan_private_results,
                (inputs_root, staging),
                privacy_forbidden,
            )
            _publish_validated_results(
                run_descriptor=run_descriptor,
                run_root=run_root,
                run_identity=run_identity,
                results_lock=results_lock,
                lock_name=lock_name,
                staging_name=staging.name,
                staged_identity=staged_identity,
                inputs_root=run_root / "inputs",
                privacy_forbidden=privacy_forbidden,
                validate_result_tree=validate_result_tree,
                final_authorization=final_authorization,
            )
    return summary


def _run_focused_phase(
    args: argparse.Namespace,
    *,
    producer_sha256: str,
) -> dict[str, object]:
    """Execute one fixed phase; local phases never touch live source roots."""
    trainer_sha256 = ssl_core._require_sha256(
        producer_sha256, "focused trainer",
    )
    if trainer_sha256 != _focused_trainer_sha256():
        raise ValueError("focused trainer changed before phase dispatch")
    authorization = _authorize_focused_bridge(
        _FOCUSED_BRIDGE_ROOT,
        _FOCUSED_MAYO_KEY,
        producer_sha256=trainer_sha256,
    )
    common = _focused_common_contract(authorization)
    phase = args.phase
    if phase == "smoke":
        def build(staging: Path) -> dict[str, object]:
            built = _build_focused_smoke(
                staging, authorization=authorization, common=common,
            )
            _validate_focused_smoke_phase(
                staging, authorization=authorization, common=common,
            )
            return built

        result = _publish_focused_phase(
            phase="smoke",
            authorization=authorization,
            common=common,
            expected_files=_expected_focused_result_files("smoke"),
            builder=build,
        )
        return {
            "phase": "smoke", "published": True,
            "report_sha256": result["report_sha256"],
        }
    namespace = _private_directory(_FOCUSED_NAMESPACE, "focused namespace")
    smoke = _validate_focused_smoke_phase(
        namespace / "smoke", authorization=authorization, common=common,
    )
    if phase == "select":
        def build(staging: Path) -> dict[str, object]:
            built = _build_focused_selection(
                staging, authorization=authorization, common=common,
                smoke=smoke,
            )
            _validate_focused_selection_phase(
                staging, authorization=authorization, common=common,
                smoke=smoke,
            )
            return built

        result = _publish_focused_phase(
            phase="select",
            authorization=authorization,
            common=common,
            expected_files=_expected_focused_result_files("select"),
            builder=build,
        )
        return {
            "phase": "select", "published": True,
            "report_sha256": result["report_sha256"],
            "selected_arm": result["selected_arm"],
        }
    selection = _validate_focused_selection_phase(
        namespace / "selection",
        authorization=authorization,
        common=common,
        smoke=smoke,
    )
    if phase == "winner":
        selected_arm = str(selection["selected_arm"])

        def build(staging: Path) -> dict[str, object]:
            built = _build_focused_winner(
                staging, authorization=authorization, common=common,
                selection=selection,
            )
            _validate_focused_winner_phase(
                staging, authorization=authorization, common=common,
                selection=selection,
            )
            return built

        result = _publish_focused_phase(
            phase="winner",
            authorization=authorization,
            common=common,
            expected_files=_expected_focused_result_files(
                "winner", selected_arm=selected_arm,
            ),
            builder=build,
        )
        return {
            "phase": "winner", "published": True,
            "report_sha256": result["report_sha256"],
            "selected_arm": selected_arm,
        }
    if phase != "audit":
        raise ValueError("focused phase is unsupported")
    winner = _validate_focused_winner_phase(
        namespace / "winner",
        authorization=authorization,
        common=common,
        selection=selection,
    )
    before = _focused_phase_tree_commitments(namespace)
    _audit_focused_metrics(
        selection, winner, authorization=authorization, common=common,
    )
    live = _run_focused_live_audit()
    after = _focused_phase_tree_commitments(namespace)
    if before != after:
        raise ValueError("focused evidence mutated during strict audit")
    audit_core = {
        "schema_version": _FOCUSED_REPORT_SCHEMAS["audit"],
        "phase": "audit",
        "status": "certified",
        **_focused_lineage_fields(authorization, common),
        "smoke_report_sha256": smoke["report_sha256"],
        "selection_report_sha256": selection["report_sha256"],
        "winner_report_sha256": winner["report_sha256"],
        "phase_tree_commitments": before,
        "metric_recomputation": "exact",
        "live_verification": live,
        "claim_scope": "recording_heldout_development_reconstruction_only",
    }

    def build_audit(staging: Path) -> dict[str, object]:
        _report, report_sha256 = _write_focused_report(
            staging, phase="audit", core=audit_core,
            authorization=authorization,
        )
        return {"report_sha256": report_sha256}

    result = _publish_focused_phase(
        phase="audit",
        authorization=authorization,
        common=common,
        expected_files=_expected_focused_result_files("audit"),
        builder=build_audit,
    )
    return {
        "phase": "audit", "published": True, "certified": True,
        "report_sha256": result["report_sha256"],
    }


def main(argv: list[str] | None = None) -> dict[str, object]:
    args = _parser().parse_args(argv)
    if args.command == "dry-run":
        result = _formal_job_matrix()
        print(json.dumps(
            result, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ))
        return result
    if args.command == "focused-mayo":
        result = _run_focused_phase(
            args, producer_sha256=_focused_trainer_sha256(),
        )
        encoded = json.dumps(
            result, sort_keys=True, separators=(",", ":"), allow_nan=False,
        )
        print(encoded)
        return result
    if args.command not in {"two-stage", "mayo-ablation"}:
        raise ValueError("unsupported pretraining command")
    producer_sha256 = _producer_sha256()
    from scripts import prepare_dynamic_landmark_ssl_inputs as inputs_cli

    captured = inputs_cli._run_mayo_cli_captured(
        args,
        lambda: (
            _run_two_stage(args, producer_sha256=producer_sha256)
            if args.command == "two-stage"
            else _run_mayo_ablation(args, producer_sha256=producer_sha256)
        ),
    )
    print(captured.json_line)
    result = json.loads(captured.json_line)
    if type(result) is not dict:
        raise ValueError("pretraining result is malformed")
    return result


def _entrypoint() -> None:
    try:
        main()
    except (FileExistsError, OSError, RuntimeError, ValueError):
        print("dynamic landmark pretraining failed closed", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(0)
