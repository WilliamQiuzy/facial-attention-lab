"""Synthetic-only contracts for dynamic landmark masked-span pretraining."""
from __future__ import annotations

import argparse
import base64
import fcntl
import hashlib
import importlib.util
import io
import json
import os
import runpy
import stat
import subprocess
import sys
import tempfile
import types
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _testlib import Check, run_all  # noqa: E402
from src.models.dynamic_landmark import DynamicLandmarkModel  # noqa: E402
from src.pretraining import dynamic_landmark_ssl as ssl_core  # noqa: E402
from src.pretraining import dynamic_landmark_ssl_bridge as bridge_core  # noqa: E402
from src.pretraining.dynamic_landmark_ssl import (  # noqa: E402
    CHECKPOINT_RAVDESS_MAYO,
    CHECKPOINT_RAVDESS_ONLY,
    DynamicLandmarkSSLModel,
    PretrainingLockedError,
    SourceScaler,
    authorize_ssl_checkpoint_receipt,
    build_ssl_checkpoint_payload,
    deterministic_group_split,
    fit_source_scaler,
    load_ssl_checkpoint,
    make_contiguous_span_mask,
    masked_smooth_l1,
    reconstruction_report,
    resample_trajectory_30hz,
    save_ssl_checkpoint,
    ssl_gap_safe_per_second_differences,
    transfer_ssl_weights,
    validate_ssl_checkpoint_payload,
)
from test_dynamic_landmark_ssl_bridge import (  # noqa: E402
    _PRODUCTION_BRIDGE_CONTRACT,
    _synthetic_authorizations_with_mayo_exclusion,
    _set_bridge_contract,
)

# Legacy v1 fixtures exercise only old synthetic invariants. Production callers
# must use ``authorize_frozen_ssl_stage`` and cannot mint v1 evidence.
build_ssl_stage_evidence = ssl_core._build_synthetic_ssl_stage_evidence_v1

_PUBLIC_TRAIN_SSL_STAGE = ssl_core.train_ssl_stage
_PUBLIC_INITIALIZE_MAYO_SSL_MODEL = ssl_core.initialize_mayo_ssl_model
_PUBLIC_BUILD_SSL_CHECKPOINT_PAYLOAD = build_ssl_checkpoint_payload
_PUBLIC_AUTHORIZE_SSL_CHECKPOINT_RECEIPT = authorize_ssl_checkpoint_receipt
_PUBLIC_SAVE_SSL_CHECKPOINT = save_ssl_checkpoint
_PUBLIC_LOAD_SSL_CHECKPOINT = load_ssl_checkpoint
_PUBLIC_TRANSFER_SSL_WEIGHTS = transfer_ssl_weights


def _test_train_ssl_stage(*, stage_evidence, **kwargs):
    function = (
        ssl_core._train_ssl_stage_impl
        if stage_evidence.mode is None
        else _PUBLIC_TRAIN_SSL_STAGE
    )
    return function(stage_evidence=stage_evidence, **kwargs)


def _test_initialize_mayo_ssl_model(
    prior_checkpoint, *, prior_stage_evidence,
):
    function = (
        ssl_core._initialize_mayo_ssl_model_impl
        if prior_stage_evidence.mode is None
        else _PUBLIC_INITIALIZE_MAYO_SSL_MODEL
    )
    return function(
        prior_checkpoint, prior_stage_evidence=prior_stage_evidence,
    )


def build_ssl_checkpoint_payload(training_result, *args, **kwargs):
    evidence = getattr(training_result, "stage_evidence", None)
    function = (
        ssl_core._build_ssl_checkpoint_payload_impl
        if evidence is not None and evidence.mode is None
        else _PUBLIC_BUILD_SSL_CHECKPOINT_PAYLOAD
    )
    return function(training_result, *args, **kwargs)


def authorize_ssl_checkpoint_receipt(*args, stage_evidence, **kwargs):
    function = (
        ssl_core._authorize_ssl_checkpoint_receipt_impl
        if stage_evidence.mode is None
        else _PUBLIC_AUTHORIZE_SSL_CHECKPOINT_RECEIPT
    )
    return function(*args, stage_evidence=stage_evidence, **kwargs)


def save_ssl_checkpoint(*args, stage_evidence, **kwargs):
    function = (
        ssl_core._save_ssl_checkpoint_impl
        if stage_evidence.mode is None
        else _PUBLIC_SAVE_SSL_CHECKPOINT
    )
    return function(*args, stage_evidence=stage_evidence, **kwargs)


def load_ssl_checkpoint(*args, stage_evidence, **kwargs):
    function = (
        ssl_core._load_ssl_checkpoint_impl
        if stage_evidence.mode is None
        else _PUBLIC_LOAD_SSL_CHECKPOINT
    )
    return function(*args, stage_evidence=stage_evidence, **kwargs)


def transfer_ssl_weights(*args, stage_evidence, **kwargs):
    if stage_evidence.mode is None:
        return ssl_core._transfer_ssl_weights_impl(
            *args,
            stage_evidence=stage_evidence,
            require_persisted=False,
            **kwargs,
        )
    return _PUBLIC_TRANSFER_SSL_WEIGHTS(
        *args, stage_evidence=stage_evidence, **kwargs,
    )


ssl_core.train_ssl_stage = _test_train_ssl_stage
ssl_core.initialize_mayo_ssl_model = _test_initialize_mayo_ssl_model


@contextmanager
def _frozen_bridge_inputs(
    root: Path,
    *,
    mode: str = "smoke",
):
    saved_contract = {
        name: getattr(bridge_core, name) for name in _PRODUCTION_BRIDGE_CONTRACT
    }
    try:
        ravdess, mayo = _synthetic_authorizations_with_mayo_exclusion()
        producer = "f" * 64
        bridge = root / "bridge"
        run_root = (
            root / "smoke" / "receipt-bound"
            if mode == "smoke"
            else root / "formal"
        )
        bridge_core.build_bridge_bundles(
            bridge,
            ravdess_authorizer=lambda: ravdess,
            mayo_authorizer=lambda: mayo,
            producer_sha256=producer,
        )
        bridge_core.freeze_bridge_stage(
            run_root,
            bridge,
            mode=mode,
            ravdess_authorizer=lambda: ravdess,
            mayo_authorizer=lambda: mayo,
            producer_sha256=producer,
        )
        arguments = {
            "inputs_root": run_root / "inputs",
            "bridge_root": bridge,
            "mode": mode,
            "ravdess_authorizer": lambda: ravdess,
            "mayo_authorizer": lambda: mayo,
            "producer_sha256": producer,
        }
        yield arguments, ravdess, mayo
    finally:
        _set_bridge_contract(saved_contract)


def _cache_commitment(groups: list[str], cache_paths: list[Path]) -> str:
    digest = hashlib.sha256()
    digest.update(b"dynamic-landmark-ssl-cache-commitment-v1\x00")
    encoded_groups = json.dumps(
        groups, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    digest.update(len(encoded_groups).to_bytes(8, "big"))
    digest.update(encoded_groups)
    for path in cache_paths:
        payload = path.read_bytes()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _same_byte_replace(path: Path) -> None:
    replacement = path.parent / f".{path.name}.same-byte-replacement"
    replacement.write_bytes(path.read_bytes())
    replacement.chmod(0o600)
    os.replace(replacement, path)


def _temporal(batch: int, windows: int = 4, frames: int = 32):
    source = torch.arange(frames, dtype=torch.int64).reshape(1, 1, frames)
    source = source.repeat(batch, windows, 1)
    timestamps = source.to(torch.float32) / 30.0
    mask = torch.ones(batch, windows, frames, dtype=torch.bool)
    return mask, timestamps, source


@contextmanager
def _mutated_microbatch_contract_globals():
    missing = object()
    saved_policy = getattr(ssl_core, "SSL_BATCH_POLICY", missing)
    saved_size = getattr(ssl_core, "SSL_MICRO_BATCH_SIZE", missing)
    ssl_core.SSL_BATCH_POLICY = "full_train_partition"
    ssl_core.SSL_MICRO_BATCH_SIZE = 65
    try:
        yield
    finally:
        if saved_policy is missing:
            delattr(ssl_core, "SSL_BATCH_POLICY")
        else:
            ssl_core.SSL_BATCH_POLICY = saved_policy
        if saved_size is missing:
            delattr(ssl_core, "SSL_MICRO_BATCH_SIZE")
        else:
            ssl_core.SSL_MICRO_BATCH_SIZE = saved_size


def _write_stage_artifacts(
    root: Path,
    *,
    stage: str,
    source: str,
    groups: list[str],
    cache_paths: list[Path],
    split,
    scaler: SourceScaler,
    config_overrides: dict[str, object] | None = None,
) -> tuple[Path, Path, Path, Path]:
    development_only = stage == "mayo"
    values = {
        "manifest": {
            "schema_version": "dynamic_landmark_ssl_manifest_v1",
            "stage": stage,
            "source": source,
            "sample_count": len(groups),
            "group_ids": groups,
            "cache_commitment_sha256": _cache_commitment(groups, cache_paths),
            "cache_count": len(cache_paths),
        },
        "config": {
            "schema_version": "dynamic_landmark_ssl_config_v1",
            "stage": stage,
            "source": source,
            "objective": "masked_span_smooth_l1_only",
            "sample_rate_hz": 30.0,
            "seeds": [0, 1, 2],
            "development_only": development_only,
            "optimizer": "adamw",
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "epochs": 1,
            "batch_policy": "deterministic_microbatch_full_partition_64",
            "span_length": 4,
            "spans_per_window": 1,
            "device": "cpu",
        },
        "split": {
            "schema_version": "dynamic_landmark_ssl_split_v1",
            "stage": stage,
            "source": source,
            "unit": split.unit,
            "claim_unit": split.claim_unit,
            "patient_held_out": split.patient_held_out,
            "train_indices": split.train_indices.tolist(),
            "heldout_indices": split.heldout_indices.tolist(),
        },
        "scaler": {
            "schema_version": "dynamic_landmark_ssl_scaler_v1",
            "stage": stage,
            "source": source,
            "fit_indices": list(scaler.fit_indices),
            "mean": scaler.mean.tolist(),
            "scale": scaler.scale.tolist(),
        },
    }
    if config_overrides:
        values["config"].update(config_overrides)
    paths = []
    for name, value in values.items():
        path = root / f"{stage}_{name}.json"
        path.write_text(
            json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        paths.append(path)
    return tuple(paths)


def _write_training_cache(
    path: Path,
    *,
    source: str,
    groups: list[str],
    seed: int,
    heldout_indices: np.ndarray | None = None,
    heldout_offset: float = 0.0,
    feature_multiplier: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    width = 23 if source == "ravdess_openface_semantic23" else 95
    generator = np.random.default_rng(seed)
    features = generator.normal(
        size=(len(groups), 4, 32, width)
    ).astype(np.float32)
    if heldout_indices is not None:
        features[np.asarray(heldout_indices, dtype=np.int64)] += np.float32(
            heldout_offset
        )
    features *= np.float32(feature_multiplier)
    valid_mask = np.ones((len(groups), 4, 32), dtype=np.bool_)
    timestamps = np.broadcast_to(
        np.arange(32, dtype=np.float32).reshape(1, 1, 32) / np.float32(30.0),
        valid_mask.shape,
    ).copy()
    source_frame_indices = np.broadcast_to(
        np.arange(32, dtype=np.int64).reshape(1, 1, 32),
        valid_mask.shape,
    ).copy()
    np.savez(
        path,
        features=features,
        valid_mask=valid_mask,
        timestamps=timestamps,
        source_frame_indices=source_frame_indices,
        group_ids=np.asarray(groups, dtype=np.str_),
    )
    return (
        torch.from_numpy(features),
        torch.from_numpy(valid_mask),
        torch.from_numpy(timestamps),
        torch.from_numpy(source_frame_indices),
    )


def _build_training_stage(
    root: Path,
    *,
    stage: str,
    groups: list[str],
    data_seed: int,
    heldout_offset: float = 0.0,
    feature_multiplier: float = 1.0,
    prior_checkpoint=None,
    prior_evidence=None,
    config_overrides: dict[str, object] | None = None,
):
    source = (
        "ravdess_openface_semantic23"
        if stage == "ravdess"
        else "mayo_mediapipe_clinical23_development_only"
    )
    unit = "actor" if stage == "ravdess" else "recording"
    split = deterministic_group_split(
        groups, heldout_fraction=0.5, seed=0, unit=unit,
    )
    cache_path = root / f"{stage}_cache.npz"
    features, valid_mask, _, _ = _write_training_cache(
        cache_path,
        source=source,
        groups=groups,
        seed=data_seed,
        heldout_indices=split.heldout_indices,
        heldout_offset=heldout_offset,
        feature_multiplier=feature_multiplier,
    )
    scaler = fit_source_scaler(
        features,
        valid_mask,
        source=source,
        fit_indices=split.train_indices,
        heldout_indices=split.heldout_indices,
    )
    manifest, config, split_artifact, scaler_artifact = _write_stage_artifacts(
        root,
        stage=stage,
        source=source,
        groups=groups,
        cache_paths=[cache_path],
        split=split,
        scaler=scaler,
        config_overrides=config_overrides,
    )
    evidence = build_ssl_stage_evidence(
        stage=stage,
        manifest_path=manifest,
        config_path=config,
        split_artifact_path=split_artifact,
        scaler_artifact_path=scaler_artifact,
        cache_paths=[cache_path],
        split=split,
        scaler=scaler,
        group_ids=groups,
        prior_ravdess_checkpoint=prior_checkpoint,
        prior_ravdess_evidence=prior_evidence,
    )
    return evidence, cache_path, split


def test_contiguous_span_mask_is_deterministic_valid_and_never_bridges_gaps(c: Check):
    valid = torch.ones(2, 2, 12, dtype=torch.bool)
    valid[:, :, 5] = False
    indices = torch.arange(12, dtype=torch.int64).reshape(1, 1, 12).repeat(2, 2, 1)
    timestamps = indices.to(torch.float32) / 30.0
    first = make_contiguous_span_mask(
        valid, timestamps, indices, expected_source_step=1,
        span_length=3, spans_per_window=1, seed=17,
    )
    second = make_contiguous_span_mask(
        valid, timestamps, indices, expected_source_step=1,
        span_length=3, spans_per_window=1, seed=17,
    )
    c.true(bool(torch.equal(first, second)), "same seed freezes span selection")
    c.true(bool((first & ~valid).sum() == 0), "detector gaps are never masked targets")
    for batch in range(2):
        for window in range(2):
            positions = torch.nonzero(
                first[batch, window], as_tuple=False
            ).reshape(-1).tolist()
            c.eq(len(positions), 3, "one exact span per eligible window")
            c.eq(positions, list(range(positions[0], positions[0] + 3)),
                 "each selected span is contiguous")
            c.true(5 not in positions, "a span never crosses a detector gap")
            c.true(bool((valid[batch, window] & ~first[batch, window]).any()),
                   "masking always leaves observed context")
    c.raises(lambda: make_contiguous_span_mask(
        torch.ones(1, 1, 3, dtype=torch.bool),
        timestamps[:1, :1, :3], indices[:1, :1, :3], expected_source_step=1,
        span_length=3,
        spans_per_window=1, seed=0,
    ), ValueError, "a span cannot mask every valid frame")
    c.raises(lambda: make_contiguous_span_mask(
        torch.zeros(1, 1, 8, dtype=torch.bool),
        timestamps[:1, :1, :8], indices[:1, :1, :8], expected_source_step=1,
        span_length=2,
        spans_per_window=1, seed=0,
    ), ValueError, "all-masked input is rejected")

    jumped = indices[:1, :1, :4].clone()
    jumped[..., 2:] += 1
    c.raises(lambda: make_contiguous_span_mask(
        torch.ones(1, 1, 4, dtype=torch.bool),
        timestamps[:1, :1, :4], jumped, expected_source_step=1,
        span_length=3, spans_per_window=1, seed=1,
    ), ValueError, "a contiguous target span cannot cross a source-index jump")

    seven_indices = torch.arange(7, dtype=torch.int64).reshape(1, 1, 7)
    feasible = make_contiguous_span_mask(
        torch.ones(1, 1, 7, dtype=torch.bool),
        seven_indices.to(torch.float32) / 30.0,
        seven_indices,
        expected_source_step=1, span_length=3, spans_per_window=2, seed=0,
    )
    c.eq(int(feasible.sum()), 6,
         "a feasible two-span request cannot fail because of a greedy first choice")


def test_masked_only_loss_and_conservative_reports_use_exact_baselines(c: Check):
    groups = ["actor_a", "actor_a", "actor_b", "actor_b"]
    split = deterministic_group_split(
        groups, heldout_fraction=0.5, seed=3, unit="actor",
    )
    heldout_count = int(split.heldout_indices.size)
    target = torch.zeros(heldout_count, 1, 4, 2)
    positions = torch.tensor([[[False, True, True, False]]]).repeat(
        heldout_count, 1, 1
    )
    trained = target.clone()
    untrained = torch.ones_like(target)
    changed_unmasked = trained.clone()
    changed_unmasked[..., 0, :] = 1000.0
    c.eq(float(masked_smooth_l1(trained, target, positions)), 0.0)
    c.eq(float(masked_smooth_l1(changed_unmasked, target, positions)), 0.0,
         "unmasked reconstruction values never enter the objective")
    baseline_features = torch.full((4, 1, 2, 2), 2.0)
    baseline_mask = torch.ones(4, 1, 2, dtype=torch.bool)
    baseline = fit_source_scaler(
        baseline_features, baseline_mask, source="ravdess_openface_semantic23",
        fit_indices=split.train_indices, heldout_indices=split.heldout_indices,
    )
    report = reconstruction_report(
        trained, untrained, target, positions,
        baseline=baseline, split=split,
        evaluated_indices=split.heldout_indices,
        group_ids=groups, source="ravdess_openface_semantic23",
    )
    c.eq(report["metric"], "masked_smooth_l1")
    c.eq(report["target_space"], "source_train_standardized")
    c.eq(report["trained"], 0.0)
    c.true(abs(report["untrained"] - 0.5) < 1e-7)
    c.eq(report["train_mean"], 0.0)
    c.eq(report["claim_unit"], "actor_held_out")
    c.eq(report["objective"], "masked_span_reconstruction_only")
    c.true(report["next_step_objective"] is False,
           "BiGRU next-step prediction is explicitly absent")
    c.true(len(report["baseline_scaler_sha256"]) == 64,
           "report binds the exact train-only baseline artifact")

    c.raises(lambda: reconstruction_report(
        trained[:1], untrained[:1], target[:1], positions[:1],
        baseline=baseline, split=split,
        evaluated_indices=split.heldout_indices[:1],
        group_ids=groups, source="ravdess_openface_semantic23",
    ), ValueError, "a one-row subset cannot inherit the complete actor-heldout claim")
    forged_baseline = SourceScaler(
        source="ravdess_openface_semantic23",
        mean=torch.full((2,), 2.0), scale=torch.ones(2),
        fit_indices=tuple(int(index) for index in split.heldout_indices),
    )
    c.raises(lambda: reconstruction_report(
        trained, untrained, target, positions,
        baseline=forged_baseline, split=split,
        evaluated_indices=split.heldout_indices,
        group_ids=groups, source="ravdess_openface_semantic23",
    ), ValueError, "the baseline must be fitted on the exact training partition")


def test_ssl_input_arms_are_stage_exact_and_keep_the_common_target(c: Check):
    expected = {
        ("ravdess", "semantic23_only"): tuple(range(23)),
        ("mayo", "blendshape_only"): tuple(range(72)),
        ("mayo", "landmark_only"): tuple(range(72, 95)),
        ("mayo", "fusion"): tuple(range(95)),
    }
    for (stage, arm), active in expected.items():
        c.eq(ssl_core.validate_ssl_input_arm(stage, arm), active)
    for stage, arm in (
        ("ravdess", "fusion"),
        ("ravdess", "landmark_only"),
        ("mayo", "semantic23_only"),
        ("mayo", "blendshape"),
        ("other", "fusion"),
    ):
        c.raises(
            lambda stage=stage, arm=arm: ssl_core.validate_ssl_input_arm(
                stage, arm,
            ),
            ValueError,
            "input arms cannot cross stages or accept aliases",
        )

    features = torch.arange(95, dtype=torch.float32).reshape(1, 1, 1, 95)
    blendshape = ssl_core.apply_ssl_input_arm(
        features, stage="mayo", input_arm="blendshape_only",
    )
    landmark = ssl_core.apply_ssl_input_arm(
        features, stage="mayo", input_arm="landmark_only",
    )
    fusion = ssl_core.apply_ssl_input_arm(
        features, stage="mayo", input_arm="fusion",
    )
    c.true(torch.equal(blendshape[..., :72], features[..., :72]))
    c.true(bool((blendshape[..., 72:] == 0).all()))
    c.true(bool((landmark[..., :72] == 0).all()))
    c.true(torch.equal(landmark[..., 72:], features[..., 72:]))
    c.true(torch.equal(fusion, features))
    c.true(torch.equal(features, torch.arange(95).reshape(1, 1, 1, 95)))


def test_ssl_mayo_arms_change_only_input_not_architecture_or_target(c: Check):
    valid, timestamps, indices = _temporal(batch=1)
    reconstruction = torch.zeros_like(valid)
    reconstruction[:, :, 4:8] = True
    generator = torch.Generator(device="cpu")
    generator.manual_seed(887)
    features = torch.randn(1, 4, 32, 95, generator=generator)
    observed = valid & ~reconstruction

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(99)
        first = DynamicLandmarkSSLModel()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(99)
        second = DynamicLandmarkSSLModel()
    c.eq(
        [(name, tuple(value.shape)) for name, value in first.state_dict().items()],
        [(name, tuple(value.shape)) for name, value in second.state_dict().items()],
    )
    for name, value in first.state_dict().items():
        c.true(torch.equal(value, second.state_dict()[name]))

    changed_landmarks = features.clone()
    changed_landmarks[..., 72:] += 1000.0
    bs_first = first.build_gru_input(
        features, observed, timestamps, indices,
        source="mayo", input_arm="blendshape_only",
    )
    bs_second = first.build_gru_input(
        changed_landmarks, observed, timestamps, indices,
        source="mayo", input_arm="blendshape_only",
    )
    c.true(torch.equal(bs_first, bs_second), "inactive landmark input is isolated")

    changed_blendshapes = features.clone()
    changed_blendshapes[..., :72] -= 1000.0
    lm_first = first.build_gru_input(
        features, observed, timestamps, indices,
        source="mayo", input_arm="landmark_only",
    )
    lm_second = first.build_gru_input(
        changed_blendshapes, observed, timestamps, indices,
        source="mayo", input_arm="landmark_only",
    )
    c.true(torch.equal(lm_first, lm_second), "inactive blendshape input is isolated")

    prediction = first(
        features, valid, timestamps, indices,
        reconstruction_mask=reconstruction, source="mayo",
        input_arm="landmark_only",
    )
    c.eq(tuple(prediction.shape), tuple(features.shape), "every arm predicts full95")


def test_mayo_reconstruction_report_uses_inverse_scaled_equal_recording_metrics(c: Check):
    split = ssl_core.SSLGroupSplit(
        train_indices=np.asarray([0], dtype=np.int64),
        heldout_indices=np.asarray([1, 2, 3], dtype=np.int64),
        unit="recording",
        claim_unit="recording_held_out_not_patient_held_out",
        patient_held_out=False,
    )
    groups = ["rec_train", "rec_a", "rec_b", "rec_b"]
    target = torch.zeros(3, 1, 2, 95)
    trained = torch.ones_like(target)
    untrained = torch.zeros_like(target)
    mask = torch.ones(3, 1, 2, dtype=torch.bool)
    scale = torch.cat((torch.full((72,), 2.0), torch.full((23,), 10.0)))
    scaler = SourceScaler(
        source="mayo_mediapipe_clinical23_development_only",
        mean=torch.zeros(95),
        scale=scale,
        fit_indices=(0,),
    )
    report = reconstruction_report(
        trained, untrained, target, mask,
        baseline=scaler, split=split,
        evaluated_indices=split.heldout_indices,
        group_ids=groups,
        source="mayo_mediapipe_clinical23_development_only",
    )
    raw = report["common_target_metrics"]["trained"]["raw_mae"]
    c.true(abs(raw["blendshape72"] - 2.0) < 1e-7)
    c.true(abs(raw["clinical23"] - 10.0) < 1e-7)
    c.true(abs(raw["equal_block_macro"] - 6.0) < 1e-7)
    c.true(abs(raw["full95"] - ((72 * 2 + 23 * 10) / 95)) < 1e-7)
    c.eq(len(report["per_recording_metrics"]), 2)
    c.eq(
        {item["recording_id"] for item in report["per_recording_metrics"]},
        {"rec_a", "rec_b"},
    )
    c.eq(report["aggregation"], "per_recording_then_equal_recording_mean")


def test_config_v3_binds_stage_arm_target_initialization_and_producer(c: Check):
    producer = "a" * 64
    commitment = "b" * 64
    common = {
        "schema_version": "dynamic_landmark_ssl_config_v3",
        "mode": "formal",
        "objective": "masked_span_smooth_l1_only",
        "sample_rate_hz": 30.0,
        "seeds": [0, 1, 2],
        "optimizer": "adamw",
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "epochs": 30,
        "batch_policy": "deterministic_microbatch_full_partition_64",
        "span_length": 4,
        "spans_per_window": 2,
        "device": "cpu",
        "producer_sha256": producer,
        "heldout_mask_policy": "frozen_common_heldout_mask_v1",
        "bridge_receipt_sha256": "c" * 64,
        "receipt_hmac": "d" * 64,
    }
    ravdess = {
        **common,
        "stage": "ravdess",
        "source": "ravdess_openface_semantic23",
        "development_only": False,
        "experiment_kind": "two_stage_fusion",
        "input_arm": "semantic23_only",
        "input_active_indices": list(range(23)),
        "target_schema": "semantic23_v1",
        "initialization_policy": "same_seed_fresh",
        "mayo_generation_commitment_sha256": None,
    }
    mayo = {
        **common,
        "stage": "mayo",
        "source": "mayo_mediapipe_clinical23_development_only",
        "development_only": True,
        "experiment_kind": "mayo_input_arm_ablation",
        "input_arm": "landmark_only",
        "input_active_indices": list(range(72, 95)),
        "target_schema": "mediapipe72_plus_clinical23_full95_v1",
        "initialization_policy": "same_seed_fresh",
        "mayo_generation_commitment_sha256": commitment,
    }
    c.eq(
        ssl_core._validate_v3_training_config(
            ravdess, stage="ravdess", mode="formal",
            source="ravdess_openface_semantic23", producer_sha256=producer,
            mayo_generation_commitment_sha256=None,
        )["input_arm"],
        "semantic23_only",
    )
    c.eq(
        ssl_core._validate_v3_training_config(
            mayo, stage="mayo", mode="formal",
            source="mayo_mediapipe_clinical23_development_only",
            producer_sha256=producer,
            mayo_generation_commitment_sha256=commitment,
        )["input_active_indices"],
        list(range(72, 95)),
    )
    for mutation in (
        {"input_arm": "semantic23_only"},
        {"input_active_indices": list(range(95))},
        {"target_schema": "clinical23_v2"},
        {"initialization_policy": "seed_matched_ravdess_prior"},
        {"schema_version": "dynamic_landmark_ssl_config_v2"},
        {"producer_sha256": "e" * 64},
    ):
        invalid = {**mayo, **mutation}
        c.raises(
            lambda invalid=invalid: ssl_core._validate_v3_training_config(
                invalid, stage="mayo", mode="formal",
                source="mayo_mediapipe_clinical23_development_only",
                producer_sha256=producer,
                mayo_generation_commitment_sha256=commitment,
            ),
            ValueError,
            "v3 config rejects cross-stage, mixed-schema, or lineage drift",
        )


def test_microbatch_backward_matches_full_partition_loss_gradients_and_update(c: Check):
    batch_size = 65
    valid, timestamps, source_indices = _temporal(batch=batch_size)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(4107)
    features = torch.randn(batch_size, 4, 32, 23, generator=generator)
    reconstruction_mask = torch.zeros_like(valid)
    for row in range(batch_size):
        masked_frames = row % 5 + 1
        reconstruction_mask[row, 0, :masked_frames] = True

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(91)
        full_model = DynamicLandmarkSSLModel()
    chunked_model = DynamicLandmarkSSLModel()
    chunked_model.load_state_dict(full_model.state_dict(), strict=True)
    full_optimizer = torch.optim.AdamW(
        full_model.parameters(), lr=0.001, weight_decay=0.0001,
    )
    chunked_optimizer = torch.optim.AdamW(
        chunked_model.parameters(), lr=0.001, weight_decay=0.0001,
    )

    full_optimizer.zero_grad(set_to_none=True)
    full_prediction = full_model(
        features, valid, timestamps, source_indices,
        reconstruction_mask=reconstruction_mask, source="ravdess",
    )
    full_loss = masked_smooth_l1(
        full_prediction, features, reconstruction_mask,
    )
    full_loss.backward()
    full_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in full_model.named_parameters()
        if parameter.grad is not None
    }

    chunked_optimizer.zero_grad(set_to_none=True)
    chunked_loss = ssl_core._backward_full_partition_microbatches(
        chunked_model,
        features=features,
        valid_mask=valid,
        timestamps=timestamps,
        source_frame_indices=source_indices,
        reconstruction_mask=reconstruction_mask,
        source="ravdess",
    )
    chunked_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in chunked_model.named_parameters()
        if parameter.grad is not None
    }
    torch.testing.assert_close(chunked_loss, full_loss.detach(), rtol=1e-6, atol=1e-7)
    c.eq(set(chunked_gradients), set(full_gradients))
    for name in full_gradients:
        torch.testing.assert_close(
            chunked_gradients[name], full_gradients[name], rtol=5e-5, atol=1e-6,
        )

    full_optimizer.step()
    chunked_optimizer.step()
    for name, full_value in full_model.state_dict().items():
        torch.testing.assert_close(
            chunked_model.state_dict()[name], full_value, rtol=3e-4, atol=5e-6,
        )


def test_microbatch_training_keeps_one_optimizer_step_per_frozen_epoch(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        evidence, _, _ = _build_training_stage(
            root,
            stage="ravdess",
            groups=["actor_a", "actor_a", "actor_b", "actor_b"],
            data_seed=771,
            config_overrides={"epochs": 2},
        )
        original_adamw = ssl_core.torch.optim.AdamW
        step_calls = 0

        class CountingAdamW(original_adamw):
            def step(self, *args, **kwargs):
                nonlocal step_calls
                step_calls += 1
                return super().step(*args, **kwargs)

        try:
            ssl_core.torch.optim.AdamW = CountingAdamW
            result = ssl_core.train_ssl_stage(stage_evidence=evidence, seed=0)
        finally:
            ssl_core.torch.optim.AdamW = original_adamw
        c.eq(step_calls, 2, "gradient accumulation cannot add optimizer steps")
        c.eq(result.training_receipt.optimizer_steps, 2)
        c.eq(
            result.training_receipt.batch_policy,
            "deterministic_microbatch_full_partition_64",
            "the frozen receipt names the exact accumulation algorithm and size",
        )
        payload = build_ssl_checkpoint_payload(result)
        c.eq(
            payload["metadata"]["training_receipt"]["batch_policy"],
            result.training_receipt.batch_policy,
            "checkpoint lineage retains the frozen microbatch policy",
        )


def test_microbatch_size_is_frozen_in_batch_policy_and_has_no_config_override(c: Check):
    groups = ["actor_a", "actor_a", "actor_b", "actor_b"]
    for overrides in (
        {"batch_policy": "full_train_partition"},
        {"micro_batch_size": 1},
    ):
        with tempfile.TemporaryDirectory() as td:
            c.raises(
                lambda value=overrides, root=Path(td): _build_training_stage(
                    root,
                    stage="ravdess",
                    groups=groups,
                    data_seed=909,
                    config_overrides=value,
                ),
                ValueError,
                "microbatch size cannot be changed outside the frozen policy",
            )


def test_mutated_module_globals_cannot_authorize_the_retired_batch_policy(c: Check):
    with _mutated_microbatch_contract_globals():
        with tempfile.TemporaryDirectory() as td:
            c.raises(
                lambda: _build_training_stage(
                    Path(td),
                    stage="ravdess",
                    groups=["actor_a", "actor_a", "actor_b", "actor_b"],
                    data_seed=419,
                    config_overrides={"batch_policy": "full_train_partition"},
                ),
                ValueError,
                "mutable module attributes cannot redefine the frozen policy",
            )


def test_mutated_module_globals_cannot_change_64_row_chunking(c: Check):
    batch_size = 65
    valid, timestamps, source_indices = _temporal(batch=batch_size)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(811)
    features = torch.randn(batch_size, 4, 32, 23, generator=generator)
    reconstruction_mask = torch.zeros_like(valid)
    reconstruction_mask[:, 0, :2] = True
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(117)
        model = DynamicLandmarkSSLModel()
    observed_batch_sizes: list[int] = []
    original_forward = model.forward

    def recording_forward(chunk, *args, **kwargs):
        observed_batch_sizes.append(int(chunk.shape[0]))
        return original_forward(chunk, *args, **kwargs)

    model.forward = recording_forward
    with _mutated_microbatch_contract_globals():
        ssl_core._backward_full_partition_microbatches(
            model,
            features=features,
            valid_mask=valid,
            timestamps=timestamps,
            source_frame_indices=source_indices,
            reconstruction_mask=reconstruction_mask,
            source="ravdess",
        )
    c.eq(
        observed_batch_sizes,
        [64, 1],
        "the exact 64-row execution contract cannot be changed at runtime",
    )


def test_mutated_module_globals_cannot_change_training_receipt_policy(c: Check):
    with tempfile.TemporaryDirectory() as td:
        evidence, _, _ = _build_training_stage(
            Path(td),
            stage="ravdess",
            groups=["actor_a", "actor_a", "actor_b", "actor_b"],
            data_seed=271,
        )
        with _mutated_microbatch_contract_globals():
            result = ssl_core.train_ssl_stage(stage_evidence=evidence, seed=0)
        c.eq(
            result.training_receipt.batch_policy,
            "deterministic_microbatch_full_partition_64",
            "receipt policy remains exact under hostile runtime mutation",
        )


def test_runtime_authorization_marker_remains_an_opaque_object_singleton(c: Check):
    marker = ssl_core._AUTHORIZATION_MARKER
    external = object()
    c.true(type(marker) is object, "runtime authority uses an opaque plain object")
    c.true(external is not marker)
    c.true(external != marker, "external callers cannot construct an equal marker")


def test_actor_and_recording_splits_are_disjoint_complete_and_conservative(c: Check):
    groups = ["a", "a", "b", "b", "c", "c", "d", "d"]
    first = deterministic_group_split(
        groups, heldout_fraction=0.25, seed=9, unit="actor"
    )
    second = deterministic_group_split(
        groups, heldout_fraction=0.25, seed=9, unit="actor"
    )
    c.true(bool(np.array_equal(first.train_indices, second.train_indices)),
           "split assignment is deterministic")
    c.eq(set(first.train_indices) | set(first.heldout_indices), set(range(8)))
    c.true(set(first.train_indices).isdisjoint(set(first.heldout_indices)))
    c.true(set(np.asarray(groups)[first.train_indices]).isdisjoint(
        set(np.asarray(groups)[first.heldout_indices])),
        "one actor cannot cross train and heldout")

    recording = deterministic_group_split(
        [f"rec_{index}" for index in range(8)],
        heldout_fraction=0.25, seed=9, unit="recording",
    )
    c.eq(recording.claim_unit, "recording_held_out_not_patient_held_out")
    c.true(recording.patient_held_out is False,
           "recording grouping never upgrades to a patient claim")


def test_30hz_resampling_selects_observed_frames_without_interpolation(c: Check):
    timestamps = np.arange(9, dtype=np.float64) / 60.0
    source_indices = np.arange(9, dtype=np.int64)
    features = np.stack((np.arange(9), np.arange(9) * 10), axis=1).astype(np.float32)
    valid = np.ones(9, dtype=bool)
    valid[4] = False
    result = resample_trajectory_30hz(features, valid, timestamps, source_indices)
    c.true(np.allclose(result.timestamps, np.arange(5) / 30.0))
    c.eq(result.valid_mask.tolist(), [True, True, False, True, True])
    c.true(bool((result.features[2] == 0).all()),
           "invalid source frame remains a detector gap, never interpolation")
    c.true(bool(np.array_equal(result.features[[0, 1, 3, 4]],
                               features[[0, 2, 6, 8]])),
           "60-Hz source is selected at exact 30-Hz positions")
    c.true(bool(np.array_equal(result.source_frame_indices, [0, 2, 4, 6, 8])),
           "resampled timeline preserves the original 60-Hz frame-index time base")
    c.eq(result.source_step, 2, "30-Hz targets advance two source frames at 60 Hz")

    missing_indices = np.asarray([0, 1, 3, 4], dtype=np.int64)
    missing_times = missing_indices.astype(np.float64) / 60.0
    missing_features = missing_indices[:, None].astype(np.float32)
    missing = resample_trajectory_30hz(
        missing_features, np.ones(4, dtype=bool), missing_times, missing_indices
    )
    c.true(bool(np.array_equal(missing.source_frame_indices, [0, 2, 4])),
           "expected original source positions remain explicit")
    c.eq(missing.valid_mask.tolist(), [True, False, True],
         "a missing exact source row stays a gap and is never nearest-filled")
    c.eq(missing.features[:, 0].tolist(), [0.0, 0.0, 4.0],
         "no neighbouring observation is copied into the missing 30-Hz target")
    c.raises(lambda: make_contiguous_span_mask(
        torch.as_tensor(missing.valid_mask).reshape(1, 1, 3),
        torch.as_tensor(missing.timestamps, dtype=torch.float32).reshape(1, 1, 3),
        torch.as_tensor(missing.source_frame_indices).reshape(1, 1, 3),
        expected_source_step=missing.source_step,
        span_length=2, spans_per_window=1, seed=0,
    ), ValueError, "masked spans cannot bridge the missing 60-Hz source row")

    rounded_count = 117
    rounded_indices = np.arange(1, rounded_count + 1, dtype=np.int64)
    rounded_times = np.round(
        np.arange(rounded_count, dtype=np.float64) / 30.0, 3
    )
    rounded = resample_trajectory_30hz(
        rounded_indices[:, None].astype(np.float32),
        np.ones(rounded_count, dtype=bool),
        rounded_times,
        rounded_indices,
    )
    c.eq(int(rounded.valid_mask.sum()), rounded_count,
         "millisecond-rounded real OpenFace timestamps retain every exact row")
    c.true(bool(np.array_equal(rounded.source_frame_indices, rounded_indices)),
           "rounded timestamps do not change original source provenance")

    official_count = 98
    official_indices = np.arange(1, official_count + 1, dtype=np.int64)
    official_times = np.round(
        np.arange(official_count, dtype=np.float64) / 29.97, 3
    )
    official_features = official_indices[:, None].astype(np.float32)
    official = resample_trajectory_30hz(
        official_features, np.ones(official_count, dtype=bool),
        official_times, official_indices,
    )
    c.eq(int(official.valid_mask.sum()), official_count,
         "official 29.97-Hz millisecond timestamps retain all exact source rows")
    c.true(bool(np.array_equal(official.source_frame_indices, official_indices)),
           "29.97-Hz canonicalization preserves official frame provenance")
    c.true(bool(np.allclose(
        official.timestamps,
        np.arange(official_count, dtype=np.float64) / 30.0,
        rtol=0.0, atol=1e-12,
    )), "official timestamps are mapped onto an explicit canonical 30-Hz timeline")

    gap_row = 49
    keep = np.arange(official_count) != gap_row
    official_gap = resample_trajectory_30hz(
        official_features[keep], np.ones(official_count - 1, dtype=bool),
        official_times[keep], official_indices[keep],
    )
    c.eq(official_gap.valid_mask[gap_row:gap_row + 2].tolist(), [False, True],
         "an absent official frame remains an explicit gap without nearest filling")
    c.eq(float(official_gap.features[gap_row, 0]), 0.0,
         "canonicalization never interpolates a missing official source row")
    canonical_values = torch.as_tensor(
        official_gap.features, dtype=torch.float32,
    ).reshape(1, 1, official_count, 1)
    canonical_mask = torch.as_tensor(official_gap.valid_mask).reshape(
        1, 1, official_count
    )
    canonical_times = torch.as_tensor(
        official_gap.timestamps, dtype=torch.float32,
    ).reshape(1, 1, official_count)
    canonical_indices = torch.as_tensor(
        official_gap.source_frame_indices, dtype=torch.int64,
    ).reshape(1, 1, official_count)
    official_dx, official_dx_mask = ssl_gap_safe_per_second_differences(
        canonical_values, canonical_mask, canonical_times, canonical_indices,
        expected_source_step=1,
    )
    c.true(bool(official_dx_mask[..., 1:gap_row].all()),
           "canonical official rows produce valid 30-Hz derivatives")
    c.eq(official_dx_mask[0, 0, gap_row:gap_row + 2].tolist(), [False, False],
         "official derivatives never bridge the preserved missing row")
    c.true(bool((official_dx[0, 0, gap_row:gap_row + 2] == 0).all()),
           "gap-adjacent official derivatives remain zero")


def test_source_scaler_is_train_only_and_cannot_cross_sources(c: Check):
    features = torch.zeros(4, 2, 4, 23)
    features[0] = 2.0
    features[1] = 4.0
    features[2:] = float("nan")
    valid = torch.ones(4, 2, 4, dtype=torch.bool)
    scaler = fit_source_scaler(
        features, valid, source="ravdess_openface_semantic23",
        fit_indices=np.asarray([0, 1]), heldout_indices=np.asarray([2, 3]),
    )
    c.true(bool(torch.allclose(scaler.mean, torch.full((23,), 3.0))))
    c.eq(scaler.fit_indices, (0, 1))
    transformed = scaler.transform(
        features[:2], valid[:2], source="ravdess_openface_semantic23"
    )
    c.true(bool(torch.isfinite(transformed).all()))
    c.raises(lambda: scaler.transform(
        features[:2], valid[:2], source="mayo_mediapipe_clinical23_development_only"
    ), ValueError, "a source scaler can never cross detector/source boundaries")
    c.raises(lambda: fit_source_scaler(
        features, valid, source="ravdess_openface_semantic23",
        fit_indices=np.asarray([0, 2]), heldout_indices=np.asarray([2, 3]),
    ), ValueError, "heldout samples cannot enter source scaler state")
    scaler.mean[0] = float("nan")
    c.raises(lambda: scaler.transform(
        features[:2], valid[:2], source="ravdess_openface_semantic23"
    ), ValueError, "corrupt nonfinite scaler state fails closed")


def test_training_cache_requires_exact_local_axes_and_zero_gaps(c: Check):
    groups = ("actor_a", "actor_b")
    leading = (len(groups), 4, 32)
    features = np.ones(leading + (23,), dtype=np.float32)
    valid_mask = np.ones(leading, dtype=np.bool_)
    valid_mask[0, 1, 7] = False
    features[0, 1, 7] = np.float32(0.0)
    expected_t = np.arange(32, dtype=np.float32) / np.float32(30.0)
    expected_i = np.arange(32, dtype=np.int64)
    timestamps = np.broadcast_to(expected_t, leading).copy()
    indices = np.broadcast_to(expected_i, leading).copy()

    def payload(**changes) -> bytes:
        fields = {
            "features": features,
            "valid_mask": valid_mask,
            "timestamps": timestamps,
            "source_frame_indices": indices,
            "group_ids": np.asarray(groups, dtype=np.str_),
        }
        fields.update(changes)
        buffer = io.BytesIO()
        np.savez(buffer, **fields)
        return buffer.getvalue()

    parsed = ssl_core._parse_training_cache_payloads(
        [payload()], stage="ravdess", group_ids=groups,
    )
    c.true(bool(torch.equal(
        parsed[2], torch.from_numpy(timestamps),
    )), "authorized bundle preserves the exact local float32 timeline")
    c.true(bool(torch.equal(
        parsed[3], torch.from_numpy(indices),
    )), "authorized bundle preserves exact local int64 indices")

    shifted_timestamps = timestamps.copy()
    shifted_timestamps[0, 2] += np.float32(1.0 / 30.0)
    c.raises(lambda: ssl_core._parse_training_cache_payloads(
        [payload(timestamps=shifted_timestamps)],
        stage="ravdess", group_ids=groups,
    ), ValueError, "recording offsets cannot masquerade as local bundle time")
    shifted_indices = indices.copy()
    shifted_indices[1, 3] += 100
    c.raises(lambda: ssl_core._parse_training_cache_payloads(
        [payload(source_frame_indices=shifted_indices)],
        stage="ravdess", group_ids=groups,
    ), ValueError, "original source offsets are private provenance, not bundle axes")
    nonzero_gap = features.copy()
    nonzero_gap[0, 1, 7] = np.float32(9.0)
    c.raises(lambda: ssl_core._parse_training_cache_payloads(
        [payload(features=nonzero_gap)],
        stage="ravdess", group_ids=groups,
    ), ValueError, "invalid bundle slots must be canonical zero")


def test_ssl_model_uses_full64_contract_and_source_specific_adapters(c: Check):
    torch.manual_seed(101)
    model = DynamicLandmarkSSLModel().eval()
    valid, timestamps, source_indices = _temporal(batch=2)
    ravdess = torch.randn(2, 4, 32, 23)
    masked = make_contiguous_span_mask(
        valid, timestamps, source_indices, expected_source_step=1,
        span_length=4, spans_per_window=1, seed=4,
    )
    gru_input = model.build_gru_input(
        ravdess, valid & ~masked, timestamps, source_indices, source="ravdess"
    )
    c.eq(tuple(gru_input.shape), (2, 4, 32, 64))
    c.true(bool((gru_input[..., :32] == 0).all()),
           "RAVDESS base latent is the exact zero first half")
    c.true(float(gru_input[..., 32:].abs().sum()) > 0,
           "RAVDESS semantic landmarks occupy the second 32 dimensions")
    reconstruction = model(
        ravdess, valid, timestamps, source_indices,
        reconstruction_mask=masked, source="ravdess",
    )
    c.eq(tuple(reconstruction.shape), tuple(ravdess.shape))
    repeated = model(
        ravdess, valid, timestamps, source_indices,
        reconstruction_mask=masked, source="ravdess",
    )
    c.true(bool(torch.equal(reconstruction, repeated)),
           "eval-mode SSL reconstruction is deterministic")

    mayo = torch.randn(2, 4, 32, 95)
    mayo_input = model.build_gru_input(
        mayo, valid & ~masked, timestamps, source_indices, source="mayo"
    )
    c.eq(tuple(mayo_input.shape), (2, 4, 32, 64))
    c.true(float(mayo_input[..., :32].abs().sum()) > 0,
           "Mayo compatible blendshape projections occupy base half")

    with torch.no_grad():
        model.proj_bs_x.weight.zero_()
        model.proj_lm_x.weight.zero_()
        model.proj_bs_dx.weight.zero_()
        model.proj_lm_dx.weight.zero_()
        model.proj_bs_dx.weight[0, 0] = 1.0
        model.proj_lm_dx.weight[0, 0] = 1.0
    mayo_ramp = torch.zeros(1, 4, 32, 95)
    ramp = torch.arange(32, dtype=torch.float32).reshape(1, 1, 32)
    mayo_ramp[..., 0] = ramp
    mayo_ramp[..., 72] = ramp
    mayo_valid, mayo_times, mayo_indices = _temporal(batch=1)
    derivative_input = model.build_gru_input(
        mayo_ramp, mayo_valid, mayo_times, mayo_indices, source="mayo"
    )
    c.true(bool(torch.allclose(
        derivative_input[..., 1:, 0], torch.full((1, 4, 31), 30.0)
    )), "Mayo bundle-local canonical step one retains per-second blendshape deltas")
    c.true(bool(torch.allclose(
        derivative_input[..., 1:, 32], torch.full((1, 4, 31), 30.0)
    )), "Mayo bundle-local canonical step one retains per-second landmark deltas")
    gap_mask = mayo_valid.clone()
    gap_mask[..., 2] = False
    gap_input = model.build_gru_input(
        mayo_ramp, gap_mask, mayo_times, mayo_indices, source="mayo"
    )
    c.eq(gap_input[0, 0, 2, [0, 32]].tolist(), [0.0, 0.0],
         "missing rows have zero derivative adapters")
    c.eq(gap_input[0, 0, 3, [0, 32]].tolist(), [0.0, 0.0],
         "derivatives never bridge a missing row")
    jumped_indices = mayo_indices.clone()
    jumped_indices[..., 16:] += 1
    jumped_input = model.build_gru_input(
        mayo_ramp, mayo_valid, mayo_times, jumped_indices, source="mayo"
    )
    c.eq(jumped_input[0, 0, 16, [0, 32]].tolist(), [0.0, 0.0],
         "derivatives never bridge an original-source index gap")
    downstream = DynamicLandmarkModel()
    c.eq({name: tuple(value.shape) for name, value in model.temporal.state_dict().items()},
         {name: tuple(value.shape) for name, value in downstream.temporal.state_dict().items()},
         "shared BiGRU checkpoint names and shapes match Task4 exactly")
    c.eq((model.temporal.input_size, model.temporal.hidden_size), (64, 32))

    c.raises(lambda: model(
        ravdess, torch.zeros_like(valid), timestamps, source_indices,
        reconstruction_mask=masked, source="ravdess",
    ), ValueError, "all-masked trajectory fails closed")
    bad = ravdess.clone()
    bad[0, 0, 0, 0] = float("nan")
    c.raises(lambda: model(
        bad, valid, timestamps, source_indices,
        reconstruction_mask=masked, source="ravdess",
    ), ValueError, "nonfinite SSL input fails closed")
    foreign_mask = torch.empty(valid.shape, dtype=torch.bool, device="meta")
    c.raises(lambda: model.build_gru_input(
        ravdess, foreign_mask, timestamps, source_indices, source="ravdess"
    ), ValueError, "SSL inputs must share one device")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        mps_model = DynamicLandmarkSSLModel().to(device).eval()
        with torch.no_grad():
            mps_output = mps_model(
                ravdess.to(device), valid.to(device), timestamps.to(device),
                source_indices.to(device), reconstruction_mask=masked.to(device),
                source="ravdess",
            )
        c.eq(mps_output.device.type, "mps", "SSL forward stays on caller device")
        c.true(bool(torch.isfinite(mps_output).all()), "MPS SSL output is finite")


def test_repository_training_is_the_only_checkpoint_minting_path(c: Check):
    c.raises(
        lambda: build_ssl_checkpoint_payload(DynamicLandmarkSSLModel()),
        ValueError,
        "a raw exact-shape model cannot mint a repository training claim",
    )
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        evidence, _, _ = _build_training_stage(
            root,
            stage="ravdess",
            groups=["actor_a", "actor_a", "actor_b", "actor_b"],
            data_seed=77,
        )
        c.eq(
            evidence.schema_version,
            "dynamic_landmark_ssl_stage_evidence_v1",
        )
        c.raises(
            lambda: _PUBLIC_TRAIN_SSL_STAGE(
                stage_evidence=evidence, seed=0,
            ),
            PretrainingLockedError,
            "public training rejects retired mode-null evidence before AdamW",
        )
        result = ssl_core.train_ssl_stage(stage_evidence=evidence, seed=0)
        receipt = result.training_receipt.to_dict()
        c.eq(receipt["optimizer"], "adamw")
        c.eq(receipt["learning_rate"], 0.001)
        c.eq(receipt["weight_decay"], 0.0)
        c.eq(receipt["epochs"], 1)
        c.eq(
            receipt["batch_policy"],
            "deterministic_microbatch_full_partition_64",
        )
        c.eq(receipt["span_length"], 4)
        c.eq(receipt["spans_per_window"], 1)
        c.eq(receipt["optimizer_steps"], 1)
        c.true(
            receipt["pre_state_sha256"] != receipt["post_state_sha256"],
            "one legal optimizer step changes the exact initialized state",
        )
        c.eq(
            receipt["baseline_state_sha256"], receipt["pre_state_sha256"],
            "RAVDESS heldout baseline is the same-seed fresh initialization",
        )
        payload = build_ssl_checkpoint_payload(result)
        c.raises(
            lambda: _PUBLIC_BUILD_SSL_CHECKPOINT_PAYLOAD(result),
            PretrainingLockedError,
            "mode-null training cannot mint a public checkpoint",
        )
        c.raises(
            lambda: _PUBLIC_SAVE_SSL_CHECKPOINT(
                root / "public-v1.pt",
                payload,
                stage_evidence=evidence,
            ),
            PretrainingLockedError,
            "mode-null payload cannot enter public checkpoint storage",
        )
        c.raises(
            lambda: _PUBLIC_INITIALIZE_MAYO_SSL_MODEL(
                payload, prior_stage_evidence=evidence,
            ),
            PretrainingLockedError,
            "mode-null payload cannot initialize public Mayo training",
        )
        c.raises(
            lambda: _PUBLIC_TRANSFER_SSL_WEIGHTS(
                payload,
                DynamicLandmarkModel(),
                stage_evidence=evidence,
            ),
            PretrainingLockedError,
            "mode-null payload cannot enter public downstream transfer",
        )
        c.raises(
            lambda: _PUBLIC_AUTHORIZE_SSL_CHECKPOINT_RECEIPT(
                root / "missing.receipt.json",
                root / "missing.pt",
                trusted_expected_receipt_sha256="0" * 64,
                stage_evidence=evidence,
            ),
            PretrainingLockedError,
            "mode-null external receipts cannot enter public authorization",
        )
        c.raises(
            lambda: _PUBLIC_LOAD_SSL_CHECKPOINT(
                root / "missing.pt",
                receipt=None,
                stage_evidence=evidence,
            ),
            PretrainingLockedError,
            "mode-null checkpoints cannot enter public loading",
        )
        c.true(not (root / "public-v1.pt").exists())
        c.eq(payload["metadata"]["seed"], 0)
        c.eq(
            payload["metadata"]["training_receipt"], receipt,
            "checkpoint serializes the exact authorized training receipt",
        )
        c.eq(
            payload["metadata"]["heldout_report"],
            dict(result.heldout_report),
            "checkpoint serializes the internally computed heldout report",
        )
        c.raises(
            lambda: build_ssl_checkpoint_payload(result, seed=0),
            TypeError,
            "checkpoint builder has no caller-controlled seed argument",
        )


def test_stage_evidence_recomputes_scaler_from_authorized_train_rows(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        groups = ["actor_a", "actor_a", "actor_b", "actor_b"]
        split = deterministic_group_split(
            groups, heldout_fraction=0.5, seed=0, unit="actor",
        )
        cache_path = root / "ravdess_cache.npz"
        features, valid_mask, _, _ = _write_training_cache(
            cache_path,
            source="ravdess_openface_semantic23",
            groups=groups,
            seed=71,
            heldout_indices=split.heldout_indices,
            heldout_offset=100.0,
        )
        leaked_rows = features[valid_mask]
        leaked_scaler = SourceScaler(
            source="ravdess_openface_semantic23",
            mean=leaked_rows.mean(dim=0),
            scale=leaked_rows.std(dim=0, unbiased=False),
            fit_indices=tuple(int(index) for index in split.train_indices),
        )
        manifest, config, split_artifact, scaler_artifact = (
            _write_stage_artifacts(
                root,
                stage="ravdess",
                source="ravdess_openface_semantic23",
                groups=groups,
                cache_paths=[cache_path],
                split=split,
                scaler=leaked_scaler,
            )
        )
        try:
            build_ssl_stage_evidence(
                stage="ravdess",
                manifest_path=manifest,
                config_path=config,
                split_artifact_path=split_artifact,
                scaler_artifact_path=scaler_artifact,
                cache_paths=[cache_path],
                split=split,
                scaler=leaked_scaler,
                group_ids=groups,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("MINTED_WITH_HELDOUT_SCALER")


def test_receipt_bound_v2_stage_authorizes_exact_single_bundle_and_claims(c: Check):
    c.raises(
        lambda: ssl_core.build_ssl_stage_evidence(),
        PretrainingLockedError,
        "the retired public v1 evidence constructor cannot authorize training",
    )
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        with _frozen_bridge_inputs(root) as (frozen, _ravdess, _mayo):
            evidence = ssl_core.authorize_frozen_ssl_stage(
                stage="ravdess",
                **frozen,
            )
            c.eq(evidence.schema_version, "dynamic_landmark_ssl_stage_evidence_v2")
            c.eq(evidence.stage, "ravdess")
            c.eq(evidence.mode, "smoke")
            c.eq(evidence.source, "ravdess_openface_semantic23")
            c.eq(evidence.bundle_file_count, 1)
            c.eq(evidence.sample_count, 2)


def test_frozen_mayo_manifest_binds_explicit_quality_exclusion(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        with _frozen_bridge_inputs(root) as (frozen, _ravdess, _mayo):
            receipt = json.loads(
                (frozen["inputs_root"] / "receipts" / "mayo.json").read_text(
                    encoding="utf-8"
                )
            )
            manifest = json.loads(
                (
                    frozen["inputs_root"] / "artifacts" / "mayo"
                    / "manifest.json"
                ).read_text(encoding="utf-8")
            )
            c.eq(receipt["exclusion_count"], 2)
            c.eq(manifest["exclusion_count"], 2)


def test_frozen_mayo_quality_exclusion_reauthorizes_before_training(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        with _frozen_bridge_inputs(root) as (frozen, _ravdess, _mayo):
            ravdess_evidence = ssl_core.authorize_frozen_ssl_stage(
                stage="ravdess", **frozen,
            )
            ravdess_result = ssl_core.train_ssl_stage(
                stage_evidence=ravdess_evidence, seed=0,
            )
            ravdess_payload = build_ssl_checkpoint_payload(ravdess_result)
            checkpoint_root = root / "quality-exclusion-prior"
            checkpoint_root.mkdir(mode=0o700)
            checkpoint = checkpoint_root / "ravdess_seed0.pt"
            checkpoint_receipt = save_ssl_checkpoint(
                checkpoint,
                ravdess_payload,
                stage_evidence=ravdess_evidence,
            )
            persisted = load_ssl_checkpoint(
                checkpoint,
                receipt=checkpoint_receipt,
                stage_evidence=ravdess_evidence,
            )
            mayo_evidence = ssl_core.authorize_frozen_ssl_stage(
                stage="mayo",
                prior_ravdess_checkpoint=persisted,
                prior_ravdess_evidence=ravdess_evidence,
                **frozen,
            )
            c.eq(mayo_evidence.exclusion_count, 2)
            c.eq(mayo_evidence.sample_count, 32)
            c.eq(mayo_evidence.source_unit_count, 2)


def test_frozen_exclusion_count_tamper_matrix_precedes_optimizer(c: Check):
    cases = (
        ("mayo-missing", "mayo", None),
        ("mayo-boolean", "mayo", True),
        ("mayo-negative", "mayo", -1),
        ("mayo-ravdess-value", "mayo", 0),
        ("mayo-other-value", "mayo", 3),
        ("ravdess-mayo-value", "ravdess", 2),
    )
    for name, stage, replacement in cases:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with _frozen_bridge_inputs(root) as (frozen, _ravdess, _mayo):
                receipt_path = (
                    frozen["inputs_root"] / "receipts" / f"{stage}.json"
                )
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                if replacement is None:
                    receipt.pop("exclusion_count")
                else:
                    receipt["exclusion_count"] = replacement
                receipt_path.write_bytes(ssl_core._canonical_json_bytes(receipt))
                receipt_path.chmod(0o600)
                optimizer_calls = 0
                original_adamw = torch.optim.AdamW

                def forbidden_adamw(*_args, **_kwargs):
                    nonlocal optimizer_calls
                    optimizer_calls += 1
                    raise AssertionError("OPTIMIZER_REACHED")

                torch.optim.AdamW = forbidden_adamw
                try:
                    c.raises(
                        lambda: ssl_core.authorize_frozen_ssl_stage(
                            stage=stage, **frozen,
                        ),
                        ValueError,
                        f"{name} exclusion claim fails frozen authorization",
                    )
                finally:
                    torch.optim.AdamW = original_adamw
                c.eq(
                    optimizer_calls,
                    0,
                    f"{name} fails before optimizer construction",
                )


def test_formal_receipt_freezes_three_seeds_and_thirty_epochs(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        with _frozen_bridge_inputs(
            root, mode="formal",
        ) as (frozen, _ravdess, _mayo):
            evidence = ssl_core.authorize_frozen_ssl_stage(
                stage="ravdess", **frozen,
            )
            authorization = evidence._runtime_authorization
            c.eq(evidence.mode, "formal")
            c.eq(authorization.training_config["seeds"], [0, 1, 2])
            c.eq(authorization.training_config["epochs"], 30)
            c.eq(authorization.training_config["optimizer"], "adamw")
            c.eq(authorization.training_config["learning_rate"], 0.001)
            c.eq(authorization.training_config["weight_decay"], 0.0001)
            c.eq(
                authorization.training_config["batch_policy"],
                "deterministic_microbatch_full_partition_64",
            )
            c.eq(evidence.source_unit_count, 2)
            c.eq(evidence.unique_group_count, 2)
            c.eq(evidence.upstream_cache_count, 2)
            c.true(bool(evidence.bridge_receipt_sha256))
            c.true(bool(evidence.receipt_hmac))
            c.true(bool(evidence.canonical_key_identity_sha256))
            c.true(bool(evidence.receipt_file_identity_sha256))
            c.true(bool(evidence.sample_ids_sha256))
            c.true(bool(evidence.source_unit_ids_sha256))
            c.true(bool(evidence.original_mapping_sha256))


def test_receipt_bound_v2_revalidates_every_authority_before_optimizer(c: Check):
    scenarios = (
        "receipt-byte",
        "manifest-v1",
        "config-mode",
        "bundle-replace",
        "canonical-key",
        "live-upstream",
    )
    for scenario in scenarios:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with _frozen_bridge_inputs(root) as (frozen, ravdess, _mayo):
                evidence = ssl_core.authorize_frozen_ssl_stage(
                    stage="ravdess",
                    **frozen,
                )
                inputs = frozen["inputs_root"]
                bridge = frozen["bridge_root"]
                if scenario == "receipt-byte":
                    path = inputs / "receipts" / "ravdess.json"
                    payload = bytearray(path.read_bytes())
                    payload[-2] ^= 1
                    path.write_bytes(bytes(payload))
                elif scenario == "manifest-v1":
                    path = inputs / "artifacts" / "ravdess" / "manifest.json"
                    value = json.loads(path.read_text(encoding="ascii"))
                    value["schema_version"] = "dynamic_landmark_ssl_manifest_v1"
                    path.write_text(
                        json.dumps(value, sort_keys=True, separators=(",", ":")),
                        encoding="ascii",
                    )
                elif scenario == "config-mode":
                    path = inputs / "artifacts" / "ravdess" / "config.json"
                    path.chmod(0o644)
                elif scenario == "bundle-replace":
                    path = bridge / "bundles" / "ravdess_bundle.npz"
                    replacement = path.with_name("replacement.npz")
                    replacement.write_bytes(path.read_bytes())
                    replacement.chmod(0o600)
                    os.replace(replacement, path)
                elif scenario == "canonical-key":
                    ravdess.private_key = b"x" * 32
                else:
                    ravdess.manifest_sha256 = "9" * 64

                optimizer_calls = 0
                original_adamw = torch.optim.AdamW

                def counted_adamw(*args, **kwargs):
                    nonlocal optimizer_calls
                    optimizer_calls += 1
                    return original_adamw(*args, **kwargs)

                torch.optim.AdamW = counted_adamw
                try:
                    c.raises(
                        lambda: ssl_core.train_ssl_stage(
                            stage_evidence=evidence,
                            seed=0,
                        ),
                        ValueError,
                        f"{scenario} drift fails closed",
                    )
                finally:
                    torch.optim.AdamW = original_adamw
                c.eq(
                    optimizer_calls,
                    0,
                    f"{scenario} fails before optimizer construction",
                )


def test_receipt_bound_frozen_file_storage_mutation_matrix(c: Check):
    targets = ("receipt", "manifest", "config", "split", "scaler", "bundle")
    operations = ("delete", "replace", "chmod", "one-byte")
    for target_name in targets:
        for operation in operations:
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                with _frozen_bridge_inputs(root) as (frozen, _ravdess, _mayo):
                    evidence = ssl_core.authorize_frozen_ssl_stage(
                        stage="ravdess", **frozen,
                    )
                    if target_name == "receipt":
                        target = (
                            Path(frozen["inputs_root"])
                            / "receipts" / "ravdess.json"
                        )
                    elif target_name == "bundle":
                        target = (
                            Path(frozen["bridge_root"])
                            / "bundles" / "ravdess_bundle.npz"
                        )
                    else:
                        target = (
                            Path(frozen["inputs_root"])
                            / "artifacts" / "ravdess"
                            / f"{target_name}.json"
                        )
                    if operation == "delete":
                        target.unlink()
                    elif operation == "replace":
                        _same_byte_replace(target)
                    elif operation == "chmod":
                        target.chmod(0o644)
                    else:
                        payload = bytearray(target.read_bytes())
                        payload[max(0, len(payload) // 2)] ^= 1
                        target.write_bytes(bytes(payload))
                    optimizer_calls = 0
                    original_adamw = torch.optim.AdamW

                    def counted_adamw(*args, **kwargs):
                        nonlocal optimizer_calls
                        optimizer_calls += 1
                        return original_adamw(*args, **kwargs)

                    torch.optim.AdamW = counted_adamw
                    try:
                        c.raises(
                            lambda: ssl_core.train_ssl_stage(
                                stage_evidence=evidence, seed=0,
                            ),
                            (OSError, ValueError),
                            f"{target_name} {operation} fails closed",
                        )
                    finally:
                        torch.optim.AdamW = original_adamw
                    c.eq(
                        optimizer_calls,
                        0,
                        f"{target_name} {operation} fails before AdamW",
                    )


def test_receipt_bound_key_and_live_generation_mutations_precede_optimizer(c: Check):
    scenarios = (
        ("key-delete", lambda ravdess: delattr(ravdess, "private_key")),
        (
            "key-same-byte-replace",
            lambda ravdess: setattr(
                ravdess, "key_file_identity_sha256", "3" * 64,
            ),
        ),
        (
            "key-mode-change",
            lambda ravdess: setattr(
                ravdess, "key_file_identity_sha256", "4" * 64,
            ),
        ),
        (
            "key-one-byte",
            lambda ravdess: setattr(ravdess, "private_key", b"x" * 32),
        ),
        (
            "generation-delete",
            lambda ravdess: delattr(ravdess, "manifest_sha256"),
        ),
        (
            "generation-replace",
            lambda ravdess: setattr(ravdess, "manifest_sha256", "9" * 64),
        ),
        (
            "generation-mode-change",
            lambda ravdess: setattr(
                ravdess, "generation_closure_hmac", "8" * 64,
            ),
        ),
        (
            "generation-one-byte",
            lambda ravdess: setattr(ravdess, "source_frames", 184),
        ),
    )
    for scenario, mutate in scenarios:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with _frozen_bridge_inputs(root) as (frozen, ravdess, _mayo):
                evidence = ssl_core.authorize_frozen_ssl_stage(
                    stage="ravdess", **frozen,
                )
                mutate(ravdess)
                optimizer_calls = 0
                original_adamw = torch.optim.AdamW

                def counted_adamw(*args, **kwargs):
                    nonlocal optimizer_calls
                    optimizer_calls += 1
                    return original_adamw(*args, **kwargs)

                torch.optim.AdamW = counted_adamw
                try:
                    c.raises(
                        lambda: ssl_core.train_ssl_stage(
                            stage_evidence=evidence, seed=0,
                        ),
                        (AttributeError, ValueError),
                        f"{scenario} fails closed",
                    )
                finally:
                    torch.optim.AdamW = original_adamw
                c.eq(optimizer_calls, 0)


def test_mayo_prior_checkpoint_storage_mutations_precede_optimizer(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        with _frozen_bridge_inputs(root) as (frozen, _ravdess, _mayo):
            ravdess_evidence = ssl_core.authorize_frozen_ssl_stage(
                stage="ravdess", **frozen,
            )
            ravdess_result = ssl_core.train_ssl_stage(
                stage_evidence=ravdess_evidence, seed=0,
            )
            ravdess_payload = build_ssl_checkpoint_payload(ravdess_result)
            for operation in ("delete", "replace", "chmod", "one-byte"):
                checkpoint_root = root / f"prior-{operation}"
                checkpoint_root.mkdir(mode=0o700)
                checkpoint = checkpoint_root / "ravdess_seed0.pt"
                receipt = save_ssl_checkpoint(
                    checkpoint,
                    ravdess_payload,
                    stage_evidence=ravdess_evidence,
                )
                persisted = load_ssl_checkpoint(
                    checkpoint,
                    receipt=receipt,
                    stage_evidence=ravdess_evidence,
                )
                mayo_evidence = ssl_core.authorize_frozen_ssl_stage(
                    stage="mayo",
                    prior_ravdess_checkpoint=persisted,
                    prior_ravdess_evidence=ravdess_evidence,
                    **frozen,
                )
                if operation == "delete":
                    checkpoint.unlink()
                elif operation == "replace":
                    _same_byte_replace(checkpoint)
                elif operation == "chmod":
                    checkpoint.chmod(0o644)
                else:
                    payload = bytearray(checkpoint.read_bytes())
                    payload[len(payload) // 2] ^= 1
                    checkpoint.write_bytes(bytes(payload))
                optimizer_calls = 0
                original_adamw = torch.optim.AdamW

                def counted_adamw(*args, **kwargs):
                    nonlocal optimizer_calls
                    optimizer_calls += 1
                    return original_adamw(*args, **kwargs)

                torch.optim.AdamW = counted_adamw
                try:
                    c.raises(
                        lambda: ssl_core.train_ssl_stage(
                            stage_evidence=mayo_evidence,
                            seed=0,
                            prior_ravdess_checkpoint=persisted,
                            prior_stage_evidence=ravdess_evidence,
                        ),
                        (OSError, ValueError),
                        f"prior checkpoint {operation} fails closed",
                    )
                finally:
                    torch.optim.AdamW = original_adamw
                c.eq(
                    optimizer_calls,
                    0,
                    f"prior checkpoint {operation} fails before Mayo AdamW",
                )


def test_mayo_rejects_persisted_ravdess_then_mayo_prior_at_every_gate(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        with _frozen_bridge_inputs(root) as (frozen, _ravdess, _mayo):
            ravdess_evidence = ssl_core.authorize_frozen_ssl_stage(
                stage="ravdess", **frozen,
            )
            ravdess_result = ssl_core.train_ssl_stage(
                stage_evidence=ravdess_evidence, seed=0,
            )
            ravdess_payload = build_ssl_checkpoint_payload(ravdess_result)
            checkpoint_root = root / "lineage-replay"
            checkpoint_root.mkdir(mode=0o700)
            ravdess_path = checkpoint_root / "ravdess_seed0.pt"
            ravdess_receipt = save_ssl_checkpoint(
                ravdess_path,
                ravdess_payload,
                stage_evidence=ravdess_evidence,
            )
            persisted_ravdess = load_ssl_checkpoint(
                ravdess_path,
                receipt=ravdess_receipt,
                stage_evidence=ravdess_evidence,
            )
            mayo_evidence = ssl_core.authorize_frozen_ssl_stage(
                stage="mayo",
                prior_ravdess_checkpoint=persisted_ravdess,
                prior_ravdess_evidence=ravdess_evidence,
                **frozen,
            )
            mayo_result = ssl_core.train_ssl_stage(
                stage_evidence=mayo_evidence,
                seed=0,
                prior_ravdess_checkpoint=persisted_ravdess,
                prior_stage_evidence=ravdess_evidence,
            )
            mayo_payload = build_ssl_checkpoint_payload(mayo_result)
            mayo_path = checkpoint_root / "mayo_seed0.pt"
            mayo_receipt = save_ssl_checkpoint(
                mayo_path,
                mayo_payload,
                stage_evidence=mayo_evidence,
            )
            persisted_mayo = load_ssl_checkpoint(
                mayo_path,
                receipt=mayo_receipt,
                stage_evidence=mayo_evidence,
            )
            c.eq(persisted_mayo["checkpoint_type"], CHECKPOINT_RAVDESS_MAYO)
            c.eq(persisted_mayo["metadata"]["seed"], 0)
            c.eq(mayo_evidence.mode, ravdess_evidence.mode)

            initial_authorization_rejected = False
            try:
                ssl_core.authorize_frozen_ssl_stage(
                    stage="mayo",
                    prior_ravdess_checkpoint=persisted_mayo,
                    prior_ravdess_evidence=mayo_evidence,
                    **frozen,
                )
            except ValueError:
                initial_authorization_rejected = True

            target_evidence = ssl_core.authorize_frozen_ssl_stage(
                stage="mayo",
                prior_ravdess_checkpoint=persisted_ravdess,
                prior_ravdess_evidence=ravdess_evidence,
                **frozen,
            )
            replay_values = target_evidence.to_dict()
            replay_values["prior_checkpoint_sha256"] = (
                ssl_core.ssl_checkpoint_fingerprint(persisted_mayo)
            )
            replay_unsigned = dict(replay_values)
            replay_unsigned.pop("evidence_sha256")
            replay_values["evidence_sha256"] = ssl_core._canonical_sha256(
                replay_unsigned
            )
            replay_evidence = ssl_core.SSLStageEvidence(**replay_values)
            replay_authorization = replace(
                target_evidence._runtime_authorization,
                evidence_sha256=replay_evidence.evidence_sha256,
                prior_ravdess_checkpoint=persisted_mayo,
                prior_ravdess_evidence=mayo_evidence,
            )
            object.__setattr__(
                replay_evidence,
                "_runtime_authorization",
                replay_authorization,
            )

            reauthorization_rejected = False
            try:
                ssl_core._require_authorized_stage_evidence(replay_evidence)
            except ValueError:
                reauthorization_rejected = True

            optimizer_calls = 0
            original_adamw = torch.optim.AdamW
            original_require_stage = ssl_core._require_authorized_stage_evidence

            def counted_adamw(*args, **kwargs):
                nonlocal optimizer_calls
                optimizer_calls += 1
                return original_adamw(*args, **kwargs)

            training_rejected = False
            torch.optim.AdamW = counted_adamw
            ssl_core._require_authorized_stage_evidence = lambda _evidence: None
            try:
                try:
                    ssl_core._train_ssl_stage_impl(
                        stage_evidence=replay_evidence,
                        seed=0,
                        prior_ravdess_checkpoint=persisted_mayo,
                        prior_stage_evidence=mayo_evidence,
                    )
                except ValueError:
                    training_rejected = True
            finally:
                ssl_core._require_authorized_stage_evidence = (
                    original_require_stage
                )
                torch.optim.AdamW = original_adamw

            c.true(
                initial_authorization_rejected,
                "initial Mayo authorization rejects a persisted ravdess_then_mayo prior",
            )
            c.true(
                reauthorization_rejected,
                "every Mayo reauthorization rejects a persisted ravdess_then_mayo prior",
            )
            c.true(
                training_rejected,
                "Mayo training independently rejects a persisted ravdess_then_mayo prior",
            )
            c.eq(
                optimizer_calls,
                0,
                "replayed Mayo lineage fails before AdamW construction",
            )


def test_receipt_bound_scaler_uses_unique_source_unit_canonical_frames(c: Check):
    features = torch.zeros(3, 4, 32, 1, dtype=torch.float32)
    valid = torch.zeros(3, 4, 32, dtype=torch.bool)
    canonical = np.arange(3 * 4 * 32, dtype=np.int64).reshape(3, 4, 32)
    valid[0, 0, :2] = True
    features[0, 0, 0, 0] = 1.0
    features[0, 0, 1, 0] = 3.0
    canonical[0, 0, 0] = 10_000
    canonical[0, 0, 1] = 10_001
    valid[1, 0, :2] = True
    features[1, 0, 0, 0] = 1.0
    features[1, 0, 1, 0] = 5.0
    canonical[1, 0, 0] = 10_000
    canonical[1, 0, 1] = 10_002
    valid[2, 0, 0] = True
    features[2, 0, 0, 0] = 999.0
    canonical[2, 0, 0] = 20_000

    scaler, unique_count, fit_sources = (
        ssl_core._fit_receipt_bound_source_scaler(
            features,
            valid,
            source="mayo_mediapipe_clinical23_development_only",
            train_indices=np.asarray([0, 1], dtype=np.int64),
            heldout_indices=np.asarray([2], dtype=np.int64),
            source_unit_ids=("recording_a", "recording_a", "recording_b"),
            original_canonical_frame_indices=canonical,
        )
    )
    c.eq(unique_count, 3)
    c.eq(fit_sources, ("recording_a",))
    c.true(torch.allclose(
        scaler.mean, torch.tensor([3.0], dtype=torch.float64)
    ), "overlapping packet slots contribute one unique source frame")
    c.true(torch.allclose(
        scaler.scale,
        torch.tensor([np.sqrt(8.0 / 3.0)], dtype=torch.float64),
    ), "unique-frame variance is not packet-frequency weighted")

    features[2, 0, 0, 0] = -999.0
    changed, changed_count, _ = ssl_core._fit_receipt_bound_source_scaler(
        features,
        valid,
        source="mayo_mediapipe_clinical23_development_only",
        train_indices=np.asarray([1, 0], dtype=np.int64),
        heldout_indices=np.asarray([2], dtype=np.int64),
        source_unit_ids=("recording_a", "recording_a", "recording_b"),
        original_canonical_frame_indices=canonical,
    )
    c.eq(changed_count, unique_count)
    c.true(torch.equal(changed.mean, scaler.mean))
    c.true(torch.equal(changed.scale, scaler.scale))


def test_receipt_bound_smoke_never_computes_or_emits_heldout_metrics(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        with _frozen_bridge_inputs(root, mode="smoke") as (
            frozen, _ravdess, _mayo,
        ):
            evidence = ssl_core.authorize_frozen_ssl_stage(
                stage="ravdess",
                **frozen,
            )
            original_report = ssl_core.reconstruction_report

            def forbidden_report(*_args, **_kwargs):
                raise AssertionError("SMOKE_COMPUTED_HELDOUT_REPORT")

            ssl_core.reconstruction_report = forbidden_report
            try:
                result = ssl_core.train_ssl_stage(
                    stage_evidence=evidence,
                    seed=0,
                )
            finally:
                ssl_core.reconstruction_report = original_report
            report = dict(result.heldout_report)
            c.eq(report["mode"], "smoke")
            c.eq(report["heldout_evaluation_computed"], False)
            c.true(np.isfinite(report["train_loss"]))
            c.eq(report["optimizer_steps"], 1)
            c.true(all(name not in report for name in (
                "trained", "untrained", "prior_ravdess", "fresh_untrained",
                "train_mean", "evaluated_indices_sha256",
            )))


def test_receipt_bound_stages_use_exact_source_parameter_allowlists(c: Check):
    model = DynamicLandmarkSSLModel()
    names = set(model.state_dict())
    ravdess_allowed = set(ssl_core._trainable_parameter_names(model, "ravdess"))
    mayo_allowed = set(ssl_core._trainable_parameter_names(model, "mayo"))
    c.eq(ravdess_allowed, {
        name for name in names if name.startswith((
            "ravdess_proj_x.", "ravdess_proj_dx.", "temporal.",
            "attention_score.", "pool_projection.", "ravdess_decoder.",
        ))
    })
    c.eq(mayo_allowed, {
        name for name in names if name.startswith((
            "proj_bs_x.", "proj_bs_dx.", "proj_lm_x.", "proj_lm_dx.",
            "temporal.", "attention_score.", "pool_projection.",
            "mayo_decoder.",
        ))
    })
    ssl_core._require_seed_matched_prior_checkpoint(
        {"metadata": {"seed": 0}}, 0,
    )
    c.raises(
        lambda: ssl_core._require_seed_matched_prior_checkpoint(
            {"metadata": {"seed": 0}}, 1,
        ),
        ValueError,
        "Mayo cannot consume a RAVDESS checkpoint from another seed",
    )

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        with _frozen_bridge_inputs(root) as (frozen, _ravdess, _mayo):
            ravdess_evidence = ssl_core.authorize_frozen_ssl_stage(
                stage="ravdess", **frozen,
            )
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(0)
                ravdess_before = {
                    name: value.detach().clone()
                    for name, value in DynamicLandmarkSSLModel().state_dict().items()
                }
            ravdess_result = ssl_core.train_ssl_stage(
                stage_evidence=ravdess_evidence, seed=0,
            )
            ravdess_after = ravdess_result.model.state_dict()
            c.true(all(torch.equal(ravdess_before[name], ravdess_after[name])
                       for name in names - ravdess_allowed))
            ravdess_payload = build_ssl_checkpoint_payload(ravdess_result)
            checkpoint_root = root / "allowlist-results"
            checkpoint_root.mkdir(mode=0o700)
            checkpoint_path = checkpoint_root / "ravdess_seed0.pt"
            checkpoint_receipt = save_ssl_checkpoint(
                checkpoint_path,
                ravdess_payload,
                stage_evidence=ravdess_evidence,
            )
            persisted_ravdess = load_ssl_checkpoint(
                checkpoint_path,
                receipt=checkpoint_receipt,
                stage_evidence=ravdess_evidence,
            )

            mayo_evidence = ssl_core.authorize_frozen_ssl_stage(
                stage="mayo",
                prior_ravdess_checkpoint=persisted_ravdess,
                prior_ravdess_evidence=ravdess_evidence,
                **frozen,
            )
            mayo_result = ssl_core.train_ssl_stage(
                stage_evidence=mayo_evidence,
                seed=0,
                prior_ravdess_checkpoint=persisted_ravdess,
                prior_stage_evidence=ravdess_evidence,
            )
            mayo_after = mayo_result.model.state_dict()
            c.true(all(torch.equal(
                persisted_ravdess["model_state"][name], mayo_after[name]
            ) for name in names - mayo_allowed))


def test_checkpoint_writer_is_fd_anchored_and_private_at_creation(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        with _frozen_bridge_inputs(root) as (frozen, _ravdess, _mayo):
            evidence = ssl_core.authorize_frozen_ssl_stage(
                stage="ravdess", **frozen,
            )
            result = ssl_core.train_ssl_stage(
                stage_evidence=evidence, seed=0,
            )
            payload = build_ssl_checkpoint_payload(result)
            original_open = ssl_core.os.open
            observed_initial_modes: list[int] = []

            def observed_open(path, flags, mode=0o777, *, dir_fd=None):
                descriptor = original_open(
                    path, flags, mode, dir_fd=dir_fd,
                )
                if dir_fd is not None and ".tmp-" in os.fsdecode(path):
                    observed_initial_modes.append(
                        stat.S_IMODE(os.fstat(descriptor).st_mode)
                    )
                return descriptor

            for index, hostile_umask in enumerate((0, 0o777)):
                target = root / f"anchored-checkpoint-{index}"
                target.mkdir(mode=0o700)
                previous_umask = os.umask(hostile_umask)
                ssl_core.os.open = observed_open
                try:
                    save_ssl_checkpoint(
                        target / "ravdess_seed0.pt",
                        payload,
                        stage_evidence=evidence,
                    )
                finally:
                    ssl_core.os.open = original_open
                    os.umask(previous_umask)
                c.eq(
                    stat.S_IMODE((target / "ravdess_seed0.pt").stat().st_mode),
                    0o600,
                )
                c.eq(
                    stat.S_IMODE((
                        target / "ravdess_seed0.pt.receipt.json"
                    ).stat().st_mode),
                    0o600,
                )
            c.eq(len(observed_initial_modes), 4)
            c.true(all(mode & 0o077 == 0 for mode in observed_initial_modes))

            parent = root / "checkpoint-parent-swap"
            parent.mkdir(mode=0o700)
            moved_parent = root / "held-checkpoint-parent"
            original_assert = ssl_core._assert_private_checkpoint_parent
            assertions = 0

            def swap_after_anchor(descriptor, lexical, expected_identity):
                nonlocal assertions
                original_assert(descriptor, lexical, expected_identity)
                assertions += 1
                if assertions == 2:
                    parent.rename(moved_parent)
                    parent.mkdir(mode=0o700)

            ssl_core._assert_private_checkpoint_parent = swap_after_anchor
            try:
                c.raises(
                    lambda: save_ssl_checkpoint(
                        parent / "ravdess_seed0.pt",
                        payload,
                        stage_evidence=evidence,
                    ),
                    ValueError,
                    "a replaced checkpoint parent fails without writing there",
                )
            finally:
                ssl_core._assert_private_checkpoint_parent = original_assert
            c.eq(list(parent.iterdir()), [])
            residues = list(moved_parent.iterdir())
            c.eq(len(residues), 1)
            c.true(residues[0].name.startswith(".ravdess_seed0.pt.tmp-"))
            c.eq(stat.S_IMODE(residues[0].stat().st_mode), 0o600)


def test_checkpoint_publication_faults_are_classified_without_cleanup(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        with _frozen_bridge_inputs(root) as (frozen, _ravdess, _mayo):
            evidence = ssl_core.authorize_frozen_ssl_stage(
                stage="ravdess", **frozen,
            )
            result = ssl_core.train_ssl_stage(
                stage_evidence=evidence, seed=0,
            )
            payload = build_ssl_checkpoint_payload(result)

            collision = root / "checkpoint-collision"
            collision.mkdir(mode=0o700)
            existing_receipt = collision / "ravdess_seed0.pt.receipt.json"
            existing_receipt.write_bytes(b"existing")
            existing_receipt.chmod(0o600)
            c.raises(
                lambda: save_ssl_checkpoint(
                    collision / "ravdess_seed0.pt",
                    payload,
                    stage_evidence=evidence,
                ),
                FileExistsError,
                "a preexisting final name blocks before temporary creation",
            )
            c.eq(list(collision.iterdir()), [existing_receipt])

            original_publish = (
                bridge_core._atomic_publish_directory_no_replace_at
            )
            rename_fault = root / "checkpoint-rename-return"
            rename_fault.mkdir(mode=0o700)

            def publish_then_raise(*args, **kwargs):
                original_publish(*args, **kwargs)
                raise OSError("synthetic checkpoint rename return fault")

            bridge_core._atomic_publish_directory_no_replace_at = (
                publish_then_raise
            )
            try:
                c.raises(
                    lambda: save_ssl_checkpoint(
                        rename_fault / "ravdess_seed0.pt",
                        payload,
                        stage_evidence=evidence,
                    ),
                    RuntimeError,
                    "checkpoint rename-return ambiguity cannot mint a receipt",
                )
            finally:
                bridge_core._atomic_publish_directory_no_replace_at = (
                    original_publish
                )
            c.true((rename_fault / "ravdess_seed0.pt").is_file())
            c.true(not (
                rename_fault / "ravdess_seed0.pt.receipt.json"
            ).exists())

            same_inode = root / "checkpoint-same-inode-mutation"
            same_inode.mkdir(mode=0o700)

            def publish_then_mutate(parent_fd, staging_name, output_name):
                original_publish(parent_fd, staging_name, output_name)
                mutated = os.open(
                    output_name,
                    os.O_WRONLY | os.O_TRUNC
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=parent_fd,
                )
                try:
                    os.write(mutated, b"same inode mutation")
                    os.fsync(mutated)
                finally:
                    os.close(mutated)

            bridge_core._atomic_publish_directory_no_replace_at = (
                publish_then_mutate
            )
            try:
                c.raises(
                    lambda: save_ssl_checkpoint(
                        same_inode / "ravdess_seed0.pt",
                        payload,
                        stage_evidence=evidence,
                    ),
                    ValueError,
                    "same-inode publication mutation cannot mint a receipt",
                )
            finally:
                bridge_core._atomic_publish_directory_no_replace_at = (
                    original_publish
                )
            c.eq(
                (same_inode / "ravdess_seed0.pt").read_bytes(),
                b"same inode mutation",
            )
            c.true(not (
                same_inode / "ravdess_seed0.pt.receipt.json"
            ).exists())

            foreign_swap = root / "checkpoint-foreign-swap"
            foreign_swap.mkdir(mode=0o700)

            def publish_foreign_inode(parent_fd, staging_name, output_name):
                held_name = f".held-{staging_name}"
                os.rename(
                    staging_name,
                    held_name,
                    src_dir_fd=parent_fd,
                    dst_dir_fd=parent_fd,
                )
                foreign = os.open(
                    staging_name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL
                    | getattr(os, "O_NOFOLLOW", 0),
                    0o600,
                    dir_fd=parent_fd,
                )
                try:
                    os.write(foreign, b"foreign checkpoint")
                    os.fsync(foreign)
                finally:
                    os.close(foreign)
                original_publish(parent_fd, staging_name, output_name)

            bridge_core._atomic_publish_directory_no_replace_at = (
                publish_foreign_inode
            )
            try:
                c.raises(
                    lambda: save_ssl_checkpoint(
                        foreign_swap / "ravdess_seed0.pt",
                        payload,
                        stage_evidence=evidence,
                    ),
                    RuntimeError,
                    "a foreign staging inode is indeterminate and never trusted",
                )
            finally:
                bridge_core._atomic_publish_directory_no_replace_at = (
                    original_publish
                )
            c.eq(
                (foreign_swap / "ravdess_seed0.pt").read_bytes(),
                b"foreign checkpoint",
            )
            c.true(any(
                path.name.startswith(".held-.ravdess_seed0.pt.tmp-")
                for path in foreign_swap.iterdir()
            ))
            c.true(not (
                foreign_swap / "ravdess_seed0.pt.receipt.json"
            ).exists())


def test_receipt_bound_smoke_persists_reloads_and_chains_two_stages(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        with _frozen_bridge_inputs(root) as (frozen, _ravdess, _mayo):
            ravdess_evidence = ssl_core.authorize_frozen_ssl_stage(
                stage="ravdess", **frozen,
            )
            ravdess_result = ssl_core.train_ssl_stage(
                stage_evidence=ravdess_evidence, seed=0,
            )
            ravdess_payload = build_ssl_checkpoint_payload(ravdess_result)
            results = root / "results-staging"
            results.mkdir(mode=0o700)
            checkpoint = results / "ravdess_seed0.pt"
            receipt = save_ssl_checkpoint(
                checkpoint,
                ravdess_payload,
                stage_evidence=ravdess_evidence,
            )
            c.eq(checkpoint.stat().st_mode & 0o777, 0o600)
            receipt_path = results / "ravdess_seed0.pt.receipt.json"
            c.eq(receipt_path.stat().st_mode & 0o777, 0o600)
            reloaded = load_ssl_checkpoint(
                checkpoint,
                receipt=receipt,
                stage_evidence=ravdess_evidence,
            )
            c.eq(
                ssl_core.ssl_checkpoint_fingerprint(reloaded),
                ssl_core.ssl_checkpoint_fingerprint(ravdess_payload),
            )

            for case, target_name, mutate in (
                (
                    "checkpoint chmod", "checkpoint",
                    lambda path: path.chmod(0o644),
                ),
                (
                    "receipt chmod", "receipt",
                    lambda path: path.chmod(0o644),
                ),
                (
                    "checkpoint same-byte replace", "checkpoint",
                    lambda path: _same_byte_replace(path),
                ),
                (
                    "receipt same-byte replace", "receipt",
                    lambda path: _same_byte_replace(path),
                ),
                (
                    "checkpoint hard link", "checkpoint",
                    lambda path: os.link(path, path.parent / "extra-link"),
                ),
            ):
                case_root = root / case.replace(" ", "-")
                case_root.mkdir(mode=0o700)
                case_checkpoint = case_root / "ravdess_seed0.pt"
                case_receipt = save_ssl_checkpoint(
                    case_checkpoint,
                    ravdess_payload,
                    stage_evidence=ravdess_evidence,
                )
                target = (
                    case_checkpoint
                    if target_name == "checkpoint"
                    else case_root / "ravdess_seed0.pt.receipt.json"
                )
                mutate(target)
                c.raises(
                    lambda checkpoint=case_checkpoint, receipt=case_receipt: (
                        load_ssl_checkpoint(
                            checkpoint,
                            receipt=receipt,
                            stage_evidence=ravdess_evidence,
                        )
                    ),
                    ValueError,
                    f"{case} invalidates persisted checkpoint authority",
                )
                c.raises(
                    lambda checkpoint=case_checkpoint,
                    receipt=case_receipt,
                    receipt_path=case_root / "ravdess_seed0.pt.receipt.json": (
                        authorize_ssl_checkpoint_receipt(
                            receipt_path,
                            checkpoint,
                            trusted_expected_receipt_sha256=(
                                receipt.receipt_sha256
                            ),
                            stage_evidence=ravdess_evidence,
                        )
                    ),
                    ValueError,
                    f"{case} cannot be trusted as a fresh external receipt",
                )

            duplicate_root = root / "duplicate-receipt-key"
            duplicate_root.mkdir(mode=0o700)
            duplicate_checkpoint = duplicate_root / "ravdess_seed0.pt"
            duplicate_receipt = save_ssl_checkpoint(
                duplicate_checkpoint,
                ravdess_payload,
                stage_evidence=ravdess_evidence,
            )
            duplicate_receipt_path = (
                duplicate_root / "ravdess_seed0.pt.receipt.json"
            )
            original_receipt_bytes = duplicate_receipt_path.read_bytes()
            duplicate_receipt_path.write_bytes(
                b'{"checkpoint_name":"duplicate.pt",'
                + original_receipt_bytes[1:]
            )
            c.raises(
                lambda: authorize_ssl_checkpoint_receipt(
                    duplicate_receipt_path,
                    duplicate_checkpoint,
                    trusted_expected_receipt_sha256=(
                        duplicate_receipt.receipt_sha256
                    ),
                    stage_evidence=ravdess_evidence,
                ),
                ValueError,
                "duplicate keyed receipt fields fail before first authorization",
            )

            mayo_evidence = ssl_core.authorize_frozen_ssl_stage(
                stage="mayo",
                prior_ravdess_checkpoint=reloaded,
                prior_ravdess_evidence=ravdess_evidence,
                **frozen,
            )
            mayo_result = ssl_core.train_ssl_stage(
                stage_evidence=mayo_evidence,
                seed=0,
                prior_ravdess_checkpoint=reloaded,
                prior_stage_evidence=ravdess_evidence,
            )
            c.eq(
                mayo_result.training_receipt.prior_checkpoint_sha256,
                ssl_core.ssl_checkpoint_fingerprint(reloaded),
            )
            forged = json.loads(receipt_path.read_text(encoding="utf-8"))
            forged["stage_authority_hmac"] = "0" * 64
            unsigned = dict(forged)
            unsigned.pop("receipt_sha256")
            forged["receipt_sha256"] = ssl_core._canonical_sha256(unsigned)
            receipt_path.write_text(
                json.dumps(forged, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            c.raises(
                lambda: authorize_ssl_checkpoint_receipt(
                    receipt_path,
                    checkpoint,
                    trusted_expected_receipt_sha256=forged["receipt_sha256"],
                    stage_evidence=ravdess_evidence,
                ),
                ValueError,
                "caller-supplied forged receipt SHA cannot replace keyed authority",
            )


def test_manifest_aggregate_commitment_binds_ordered_cache_bytes(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        groups = ["actor_a", "actor_a", "actor_b", "actor_b"]
        split = deterministic_group_split(
            groups, heldout_fraction=0.5, seed=0, unit="actor",
        )
        first_cache = root / "first_cache.npz"
        features, valid_mask, _, _ = _write_training_cache(
            first_cache,
            source="ravdess_openface_semantic23",
            groups=groups,
            seed=31,
            heldout_indices=split.heldout_indices,
        )
        scaler = fit_source_scaler(
            features,
            valid_mask,
            source="ravdess_openface_semantic23",
            fit_indices=split.train_indices,
            heldout_indices=split.heldout_indices,
        )
        manifest, config, split_artifact, scaler_artifact = (
            _write_stage_artifacts(
                root,
                stage="ravdess",
                source="ravdess_openface_semantic23",
                groups=groups,
                cache_paths=[first_cache],
                split=split,
                scaler=scaler,
            )
        )
        manifest_value = json.loads(manifest.read_text(encoding="utf-8"))
        manifest_value.update({
            "cache_commitment_sha256": _cache_commitment(groups, [first_cache]),
            "cache_count": 1,
        })
        manifest.write_text(
            json.dumps(manifest_value, sort_keys=True, separators=(",", ":"))
            + "\n",
            encoding="utf-8",
        )
        first_evidence = build_ssl_stage_evidence(
            stage="ravdess",
            manifest_path=manifest,
            config_path=config,
            split_artifact_path=split_artifact,
            scaler_artifact_path=scaler_artifact,
            cache_paths=[first_cache],
            split=split,
            scaler=scaler,
            group_ids=groups,
        )

        changed_cache = root / "changed_cache.npz"
        _write_training_cache(
            changed_cache,
            source="ravdess_openface_semantic23",
            groups=groups,
            seed=31,
            heldout_indices=split.heldout_indices,
            heldout_offset=50.0,
        )
        try:
            build_ssl_stage_evidence(
                stage="ravdess",
                manifest_path=manifest,
                config_path=config,
                split_artifact_path=split_artifact,
                scaler_artifact_path=scaler_artifact,
                cache_paths=[changed_cache],
                split=split,
                scaler=scaler,
                group_ids=groups,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("MINTED_WITH_UNCOMMITTED_CACHE")

        manifest_value["cache_commitment_sha256"] = _cache_commitment(
            groups, [changed_cache]
        )
        manifest.write_text(
            json.dumps(manifest_value, sort_keys=True, separators=(",", ":"))
            + "\n",
            encoding="utf-8",
        )
        changed_evidence = build_ssl_stage_evidence(
            stage="ravdess",
            manifest_path=manifest,
            config_path=config,
            split_artifact_path=split_artifact,
            scaler_artifact_path=scaler_artifact,
            cache_paths=[changed_cache],
            split=split,
            scaler=scaler,
            group_ids=groups,
        )
        c.true(
            first_evidence.evidence_sha256 != changed_evidence.evidence_sha256,
            "re-freezing changed cache bytes changes stage evidence",
        )
        c.eq(changed_evidence.cache_count, 1)
        c.eq(
            changed_evidence.cache_commitment_sha256,
            manifest_value["cache_commitment_sha256"],
        )
        c.true("cache_sha256" not in manifest_value)
        c.true("cache_files" not in manifest_value)


def test_authorized_training_is_deterministic_heldout_only_and_chained(c: Check):
    groups = ["actor_a", "actor_a", "actor_b", "actor_b"]
    with tempfile.TemporaryDirectory() as first_td:
        first_root = Path(first_td)
        evidence, _, _ = _build_training_stage(
            first_root, stage="ravdess", groups=groups, data_seed=91,
        )
        first = ssl_core.train_ssl_stage(stage_evidence=evidence, seed=0)
        repeated = ssl_core.train_ssl_stage(stage_evidence=evidence, seed=0)
        c.eq(first.training_receipt.to_dict(), repeated.training_receipt.to_dict(),
             "same seed, bytes, split, scaler, and config reproduce the receipt")
        c.eq(dict(first.heldout_report), dict(repeated.heldout_report),
             "same seed reproduces the exact heldout report")
        c.true(all(torch.equal(
            first.model.state_dict()[name], repeated.model.state_dict()[name]
        ) for name in first.model.state_dict()),
            "same seed reproduces the exact CPU post-state")

        different_seed = ssl_core.train_ssl_stage(
            stage_evidence=evidence, seed=1,
        )
        c.true(
            first.training_receipt.receipt_sha256
            != different_seed.training_receipt.receipt_sha256,
            "a different registered seed changes the training receipt",
        )
        c.raises(
            lambda: ssl_core.train_ssl_stage(stage_evidence=evidence, seed=3),
            ValueError,
            "only preregistered seeds 0, 1, 2 can train",
        )

        changed_root = first_root / "heldout_changed"
        changed_root.mkdir()
        changed_evidence, _, _ = _build_training_stage(
            changed_root,
            stage="ravdess",
            groups=groups,
            data_seed=91,
            heldout_offset=25.0,
        )
        heldout_changed = ssl_core.train_ssl_stage(
            stage_evidence=changed_evidence, seed=0,
        )
        c.true(
            evidence.evidence_sha256 != changed_evidence.evidence_sha256,
            "heldout cache-byte changes produce different frozen stage evidence",
        )
        c.eq(
            evidence.scaler_sha256,
            changed_evidence.scaler_sha256,
            "heldout-only changes cannot alter the recomputed train scaler",
        )
        c.eq(
            first.training_receipt.post_state_sha256,
            heldout_changed.training_receipt.post_state_sha256,
            "changing only heldout rows cannot affect the trained state",
        )
        c.true(
            first.training_receipt.heldout_report_sha256
            != heldout_changed.training_receipt.heldout_report_sha256,
            "heldout-only changes still change the exact evaluation report",
        )

        ravdess_payload = build_ssl_checkpoint_payload(first)
        mayo_root = first_root / "mayo"
        mayo_root.mkdir()
        mayo_evidence, _, _ = _build_training_stage(
            mayo_root,
            stage="mayo",
            groups=[f"recording_{index}" for index in range(4)],
            data_seed=27,
            prior_checkpoint=ravdess_payload,
            prior_evidence=evidence,
        )
        c.raises(
            lambda: ssl_core.train_ssl_stage(
                stage_evidence=mayo_evidence, seed=0,
            ),
            ValueError,
            "Mayo cannot train without the exact authorized RAVDESS checkpoint",
        )
        mayo = ssl_core.train_ssl_stage(
            stage_evidence=mayo_evidence,
            seed=0,
            prior_ravdess_checkpoint=ravdess_payload,
            prior_stage_evidence=evidence,
        )
        mayo_payload = build_ssl_checkpoint_payload(mayo)
        c.eq(mayo_payload["checkpoint_type"], CHECKPOINT_RAVDESS_MAYO)
        c.eq(
            mayo.training_receipt.pre_state_sha256,
            first.training_receipt.post_state_sha256,
            "Mayo starts from the exact authorized RAVDESS post-state",
        )
        c.eq(
            mayo.training_receipt.baseline_state_sha256,
            first.training_receipt.post_state_sha256,
            "Mayo heldout baseline is the exact prior RAVDESS state",
        )
        c.eq(
            mayo.training_receipt.prior_checkpoint_sha256,
            ssl_core.ssl_checkpoint_fingerprint(ravdess_payload),
        )
        c.true("prior_ravdess" in mayo.heldout_report,
               "Mayo names its exact initialization baseline explicitly")
        c.true("fresh_untrained" in mayo.heldout_report,
               "Mayo separately reports a same-seed fresh-model baseline")
        c.true("untrained" not in mayo.heldout_report,
               "a prior-trained RAVDESS model is never mislabeled untrained")
        c.eq(
            mayo.heldout_report["initialization_baseline_metric"],
            "prior_ravdess",
        )


def test_training_reconstructs_standardized_targets_not_raw_source_units(c: Check):
    groups = ["actor_a", "actor_a", "actor_b", "actor_b"]
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        unit_root = root / "unit"
        scaled_root = root / "scaled"
        unit_root.mkdir()
        scaled_root.mkdir()
        unit_evidence, _, _ = _build_training_stage(
            unit_root,
            stage="ravdess",
            groups=groups,
            data_seed=57,
            feature_multiplier=1.0,
        )
        scaled_evidence, _, _ = _build_training_stage(
            scaled_root,
            stage="ravdess",
            groups=groups,
            data_seed=57,
            feature_multiplier=8.0,
        )
        unit_result = ssl_core.train_ssl_stage(
            stage_evidence=unit_evidence, seed=0,
        )
        scaled_result = ssl_core.train_ssl_stage(
            stage_evidence=scaled_evidence, seed=0,
        )
        c.eq(
            unit_result.training_receipt.post_state_sha256,
            scaled_result.training_receipt.post_state_sha256,
            "raw detector units cannot change training after train-only standardization",
        )
        c.eq(
            unit_result.training_receipt.train_trace_sha256,
            scaled_result.training_receipt.train_trace_sha256,
            "training loss is computed in source-train-standardized target space",
        )
        for metric in ("trained", "untrained", "train_mean"):
            c.eq(
                unit_result.heldout_report[metric],
                scaled_result.heldout_report[metric],
                f"heldout {metric} is invariant to raw source units",
            )


def test_training_result_checkpoint_and_files_are_tamper_evident(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        evidence, cache_path, _ = _build_training_stage(
            root,
            stage="ravdess",
            groups=["actor_a", "actor_a", "actor_b", "actor_b"],
            data_seed=12,
        )
        result = ssl_core.train_ssl_stage(stage_evidence=evidence, seed=0)
        with torch.no_grad():
            next(result.model.parameters()).add_(1.0)
        c.raises(
            lambda: build_ssl_checkpoint_payload(result),
            ValueError,
            "in-place replacement after legal training invalidates authorization",
        )

        result = ssl_core.train_ssl_stage(stage_evidence=evidence, seed=0)
        forged_report = dict(result.heldout_report)
        forged_report["trained"] = float(forged_report["trained"]) + 1.0
        object.__setattr__(result, "heldout_report", forged_report)
        c.raises(
            lambda: build_ssl_checkpoint_payload(result),
            ValueError,
            "caller report replacement cannot mint an evaluation claim",
        )

        result = ssl_core.train_ssl_stage(stage_evidence=evidence, seed=0)
        forged_receipt = type(result.training_receipt)(
            **result.training_receipt.to_dict()
        )
        object.__setattr__(result, "training_receipt", forged_receipt)
        c.raises(
            lambda: build_ssl_checkpoint_payload(result),
            ValueError,
            "serialized receipt replacement lacks runtime training authorization",
        )

        result = ssl_core.train_ssl_stage(stage_evidence=evidence, seed=0)
        forged_result = type(result)(
            model=result.model,
            stage_evidence=result.stage_evidence,
            training_receipt=result.training_receipt,
            heldout_report=result.heldout_report,
        )
        c.raises(
            lambda: build_ssl_checkpoint_payload(forged_result),
            ValueError,
            "a reconstructed SSLTrainingResult cannot self-authorize",
        )

        result = ssl_core.train_ssl_stage(stage_evidence=evidence, seed=0)
        original_cache = cache_path.read_bytes()
        cache_path.write_bytes(original_cache + b"x")
        c.raises(
            lambda: build_ssl_checkpoint_payload(result),
            ValueError,
            "changing one authorized cache byte invalidates the result",
        )


def test_cache_authorization_never_mixes_read_bytes_with_a_postclose_inode(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cache_path = root / "cache.npz"
        replacement = root / "replacement.npz"
        cache_path.write_bytes(b"first-cache-bytes")
        replacement.write_bytes(b"other-cache-bytes")
        original_inode = cache_path.stat().st_ino
        snapshot_name = (
            "_regular_file_snapshot_with_identity"
            if hasattr(ssl_core, "_regular_file_snapshot_with_identity")
            else "_regular_file_snapshot"
        )
        original_snapshot = getattr(ssl_core, snapshot_name)

        def swap_after_close(path, name):
            snapshot = original_snapshot(path, name)
            os.replace(replacement, cache_path)
            return snapshot

        setattr(ssl_core, snapshot_name, swap_after_close)
        try:
            artifact = ssl_core._authorize_cache_artifact(cache_path)
            c.eq(
                artifact.inode,
                original_inode,
                "cache identity is the same descriptor that supplied its bytes",
            )
        finally:
            setattr(ssl_core, snapshot_name, original_snapshot)


def test_frozen_config_drives_training_and_every_artifact_revalidates(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        evidence, _, _ = _build_training_stage(
            root,
            stage="ravdess",
            groups=["actor_a", "actor_a", "actor_b", "actor_b"],
            data_seed=63,
            config_overrides={"epochs": 2},
        )
        result = ssl_core.train_ssl_stage(stage_evidence=evidence, seed=0)
        c.eq(result.training_receipt.epochs, 2)
        c.eq(result.training_receipt.optimizer_steps, 2)
        c.eq(result.training_receipt.optimizer, "adamw")
        c.eq(
            result.training_receipt.batch_policy,
            "deterministic_microbatch_full_partition_64",
        )
        payload = build_ssl_checkpoint_payload(result)
        public_manifest = json.loads(
            (root / "ravdess_manifest.json").read_text(encoding="utf-8")
        )
        c.true("cache_sha256" not in public_manifest)
        c.true("source_sha256" not in public_manifest)
        c.true("cache_files" not in public_manifest,
               "public manifest does not expose per-cache linkage")
        receipt = payload["metadata"]["training_receipt"]
        c.true("cache_binding_sha256" in receipt,
               "receipt exposes only one evidence-domain cache binding")
        c.true("cache_sha256" not in receipt)
        c.true("cache_names" not in receipt)

        for artifact_name in ("manifest", "config", "split", "scaler"):
            path = root / f"ravdess_{artifact_name}.json"
            original = path.read_bytes()
            path.write_bytes(original + b" ")
            c.raises(
                lambda: build_ssl_checkpoint_payload(result),
                ValueError,
                f"{artifact_name} bytes are revalidated before checkpoint minting",
            )
            path.write_bytes(original)
        validate_ssl_checkpoint_payload(payload)


def test_checkpoint_metadata_transfer_and_external_receipt_remain_exact(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        evidence, _, _ = _build_training_stage(
            root,
            stage="ravdess",
            groups=["actor_a", "actor_a", "actor_b", "actor_b"],
            data_seed=44,
        )
        result = ssl_core.train_ssl_stage(stage_evidence=evidence, seed=0)
        payload = build_ssl_checkpoint_payload(result)
        validate_ssl_checkpoint_payload(payload)
        c.eq(payload["checkpoint_type"], CHECKPOINT_RAVDESS_ONLY)
        c.true(all("scaler" not in key for key in payload["model_state"]),
               "source scalers are evidence, never transferable model weights")

        for metadata_key in ("training_receipt", "heldout_report"):
            tampered = dict(payload)
            tampered["metadata"] = dict(payload["metadata"])
            tampered["metadata"][metadata_key] = dict(
                payload["metadata"][metadata_key]
            )
            first_key = next(iter(tampered["metadata"][metadata_key]))
            tampered["metadata"][metadata_key][first_key] = "tampered"
            c.raises(
                lambda value=tampered: validate_ssl_checkpoint_payload(value),
                ValueError,
                f"{metadata_key} tampering is detected",
            )
        substituted = dict(payload)
        substituted["model_state"] = {
            name: value.detach().clone()
            for name, value in DynamicLandmarkSSLModel().state_dict().items()
        }
        c.raises(
            lambda: validate_ssl_checkpoint_payload(substituted),
            ValueError,
            "finite exact-shape state substitution contradicts the training receipt",
        )

        downstream = DynamicLandmarkModel()
        forward_base_before = downstream.temporal.weight_ih_l0[:, :32].clone()
        transferred = transfer_ssl_weights(
            payload, downstream, stage_evidence=evidence,
        )
        c.true(bool(transferred))
        c.true(torch.equal(
            downstream.temporal.weight_ih_l0[:, :32], forward_base_before
        ), "RAVDESS zero-base GRU columns retain downstream initialization")
        c.true(torch.equal(
            downstream.temporal.weight_ih_l0[:, 32:],
            payload["model_state"]["temporal.weight_ih_l0"][:, 32:],
        ), "trained RAVDESS landmark GRU columns transfer")

        path = root / "ssl.pt"
        file_receipt = save_ssl_checkpoint(
            path, payload, stage_evidence=evidence,
        )
        restored = authorize_ssl_checkpoint_receipt(
            root / "ssl.pt.receipt.json",
            path,
            trusted_expected_receipt_sha256=file_receipt.receipt_sha256,
            stage_evidence=evidence,
        )
        loaded = load_ssl_checkpoint(
            path, receipt=restored, stage_evidence=evidence,
        )
        c.eq(
            ssl_core.ssl_checkpoint_fingerprint(loaded),
            ssl_core.ssl_checkpoint_fingerprint(payload),
            "external file receipt restores the exact trained checkpoint",
        )


def test_checkpoint_validation_and_transfer_do_not_advance_global_torch_rng(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        evidence, _, _ = _build_training_stage(
            root,
            stage="ravdess",
            groups=["actor_a", "actor_a", "actor_b", "actor_b"],
            data_seed=82,
        )
        result = ssl_core.train_ssl_stage(stage_evidence=evidence, seed=0)
        model_state = {
            name: tensor.detach().clone()
            for name, tensor in result.model.state_dict().items()
        }
        payload = build_ssl_checkpoint_payload(result)
        downstream = DynamicLandmarkModel()
        actions = (
            ("model-state fingerprint", lambda: ssl_core._model_state_sha256(
                model_state
            )),
            ("training-receipt validation", lambda: ssl_core._validate_training_receipt(
                result.training_receipt
            )),
            ("checkpoint payload validation", lambda: validate_ssl_checkpoint_payload(
                payload
            )),
            ("checkpoint fingerprint", lambda: ssl_core.ssl_checkpoint_fingerprint(
                payload
            )),
            ("authorized checkpoint initialization", lambda: (
                ssl_core.initialize_mayo_ssl_model(
                    payload, prior_stage_evidence=evidence,
                )
            )),
            ("authorized transfer validation", lambda: transfer_ssl_weights(
                payload, downstream, stage_evidence=evidence,
            )),
        )
        for name, action in actions:
            before = torch.random.get_rng_state().clone()
            action()
            after = torch.random.get_rng_state()
            c.true(
                bool(torch.equal(before, after)),
                f"{name} is a pure validation action for caller RNG",
            )


def _load_runner():
    script = ROOT / "scripts" / "pretrain_dynamic_landmarks.py"
    spec = importlib.util.spec_from_file_location("receipt_bound_ssl_runner", script)
    if spec is None or spec.loader is None:
        raise AssertionError("SSL runner cannot be imported")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_import_order_preserves_canonical_producer_identity(c: Check):
    def producer_digest(import_source: str) -> bytes:
        read_descriptor, write_descriptor = os.pipe()
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(ROOT)
        environment["SSL_PRODUCER_DESCRIPTOR"] = str(write_descriptor)
        source = (
            "import os\n"
            f"{import_source}\n"
            "digest = producer._producer_sha256()\n"
            "os.write(int(os.environ['SSL_PRODUCER_DESCRIPTOR']), "
            "digest.encode('ascii'))\n"
        )
        try:
            completed = subprocess.run(
                [sys.executable, "-B", "-c", source],
                cwd=ROOT,
                env=environment,
                capture_output=True,
                check=False,
                pass_fds=(write_descriptor,),
            )
        finally:
            os.close(write_descriptor)
        try:
            digest = os.read(read_descriptor, 65)
        finally:
            os.close(read_descriptor)
        c.eq(completed.returncode, 0, "producer subprocess exits successfully")
        c.eq(completed.stdout, b"", "producer subprocess keeps stdout empty")
        c.eq(completed.stderr, b"", "producer subprocess keeps stderr empty")
        c.eq(len(digest), 64, "producer digest has the required length")
        c.true(
            all(byte in b"0123456789abcdef" for byte in digest),
            "producer digest is lowercase hexadecimal",
        )
        return digest

    canonical_digest = producer_digest(
        "from scripts import prepare_dynamic_landmark_ssl_inputs as producer"
    )
    runner_digest = producer_digest(
        "from scripts import pretrain_dynamic_landmarks as producer"
    )
    c.true(
        runner_digest == canonical_digest,
        "pretraining runner producer identity equals canonical bridge identity",
    )


def test_runner_freezes_producer_before_parser_and_authorization(c: Check):
    runner = _load_runner()
    from scripts import prepare_dynamic_landmark_ssl_inputs as inputs_cli

    events: list[object] = []
    producer = "a" * 64

    class Parser:
        def parse_args(self, _argv):
            events.append("parse")
            return argparse.Namespace(command="two-stage")

    def producer_sha256():
        events.append("producer")
        return producer

    def run_two_stage(_args, *, producer_sha256):
        events.append(("run", producer_sha256))
        return {"stage_count": 2}

    def captured(_args, action):
        events.append("capture")
        return types.SimpleNamespace(
            json_line=json.dumps(action(), sort_keys=True),
        )

    def quiet_call(function, /, *args, **kwargs):
        events.append(("quiet", function.__name__))
        return function(*args, **kwargs)

    original_capture = inputs_cli._run_mayo_cli_captured
    runner._producer_sha256 = producer_sha256
    runner._quiet_call = quiet_call
    runner._parser = Parser
    runner._run_two_stage = run_two_stage
    inputs_cli._run_mayo_cli_captured = captured
    stdout = io.StringIO()
    try:
        with redirect_stdout(stdout):
            result = runner.main([])
    finally:
        inputs_cli._run_mayo_cli_captured = original_capture

    c.eq(result, {"stage_count": 2})
    c.eq(stdout.getvalue(), '{"stage_count": 2}\n')
    c.eq(
        events,
        ["producer", "parse", "capture", ("run", producer)],
        "runner freezes the canonical producer identity before parser and "
        "authorization machinery can warm live producer semantics",
    )


def test_direct_runner_delegates_once_to_canonical_entrypoint(c: Check):
    import scripts

    canonical_name = "scripts.pretrain_dynamic_landmarks"
    attribute_name = "pretrain_dynamic_landmarks"
    sentinel_exit_code = 73
    calls: list[None] = []
    sentinel = types.ModuleType(canonical_name)

    def sentinel_entrypoint() -> None:
        calls.append(None)
        raise SystemExit(sentinel_exit_code)

    sentinel._entrypoint = sentinel_entrypoint
    missing = object()
    saved_module = sys.modules.get(canonical_name, missing)
    saved_attribute = getattr(scripts, attribute_name, missing)
    saved_argv = sys.argv
    script = ROOT / "scripts" / "pretrain_dynamic_landmarks.py"
    captured_stderr = io.StringIO()
    sys.modules[canonical_name] = sentinel
    setattr(scripts, attribute_name, sentinel)
    sys.argv = [str(script)]
    try:
        try:
            with redirect_stderr(captured_stderr):
                runpy.run_path(str(script), run_name="__main__")
        except SystemExit as error:
            exit_code = error.code
        else:
            exit_code = None
    finally:
        sys.argv = saved_argv
        if saved_module is missing:
            del sys.modules[canonical_name]
        else:
            sys.modules[canonical_name] = saved_module
        if saved_attribute is missing:
            delattr(scripts, attribute_name)
        else:
            setattr(scripts, attribute_name, saved_attribute)

    c.eq(
        (exit_code, len(calls)),
        (sentinel_exit_code, 1),
        "direct runner delegates exactly once to the canonical entrypoint",
    )
    c.eq(
        captured_stderr.getvalue(), "",
        "canonical entrypoint exit emits no local diagnostic",
    )


def _synthetic_runner_fixture(root: Path, frozen: dict[str, object]):
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    module = _load_runner()
    module.PRETRAINING_ROOT = Path(frozen["bridge_root"]).parent.resolve()
    module._authorization_factories = lambda _args: (
        frozen["ravdess_authorizer"], frozen["mayo_authorizer"],
    )
    module._producer_sha256 = lambda: frozen["producer_sha256"]

    def synthetic_privacy(args, ravdess_authorizer, mayo_authorizer):
        from scripts import prepare_dynamic_landmark_ssl_inputs as inputs_cli

        tokens: set[bytes] = set()
        for private_root in (
            args.mayo_data_root, args.mayo_existing_export_root,
        ):
            for representation in inputs_cli._root_text_representations(
                private_root,
            ):
                inputs_cli._add_text_variants(tokens, representation)
        for authorization in (ravdess_authorizer(), mayo_authorizer()):
            inputs_cli._add_binary_variants(
                tokens, authorization.private_key,
            )
        return inputs_cli._PrivacyForbidden(tokens=tuple(sorted(tokens)))

    module._privacy_forbidden = synthetic_privacy
    roots = {
        "mayo": root / "raw-mayo-root-secret",
        "legacy": root / "legacy-export-root-secret",
        "ravdess": root / "ravdess-root",
        "mayo_cache": root / "mayo-cache",
    }
    for directory in roots.values():
        directory.mkdir(mode=0o700)
    for lease in (
        roots["ravdess"] / ".derived_semantic23.lock",
        module.PRETRAINING_ROOT / ".mayo_ssl_cache.lock",
    ):
        lease.touch(exist_ok=True)
        lease.chmod(0o600)
    ravdess_key = root / "ravdess.key"
    mayo_key = root / "mayo.key"
    for path in (ravdess_key, mayo_key):
        path.write_bytes(b"k" * 32)
        path.chmod(0o600)
    exposure = root / "mayo-exposure.json"
    exposure.write_text("{}", encoding="ascii")
    exposure.chmod(0o600)
    run_root = Path(frozen["inputs_root"]).parent
    arguments = [
        "two-stage",
        "--mode", str(frozen["mode"]),
        "--run-root", str(run_root),
        "--bridge-root", str(frozen["bridge_root"]),
        "--ravdess-data-root", str(roots["ravdess"]),
        "--ravdess-key", str(ravdess_key),
        "--mayo-data-root", str(roots["mayo"]),
        "--mayo-existing-export-root", str(roots["legacy"]),
        "--mayo-cache-root", str(roots["mayo_cache"]),
        "--mayo-exposure-manifest", str(exposure),
        "--mayo-key", str(mayo_key),
    ]
    return module, arguments, run_root, roots


def test_real_pretraining_runner_requires_the_exact_two_stage_command(c: Check):
    module = _load_runner()
    c.true(module._RUN_ID.fullmatch("preflight-seed0") is not None)
    c.true(module._RUN_ID.fullmatch("unsafe.run") is None)
    aggregate = module._formal_aggregates([
        {
            "ravdess_only": {"reconstruction": {
                "trained": value,
                "untrained": value + 1.0,
                "train_mean": value + 2.0,
            }},
            "ravdess_then_mayo": {"reconstruction": {
                "trained": value,
                "prior_ravdess": value + 1.0,
                "fresh_untrained": value + 2.0,
                "train_mean": value + 3.0,
            }},
        }
        for value in (1.0, 2.0, 3.0)
    ])
    c.eq(aggregate["ravdess_only"]["trained"], {
        "mean": 2.0, "sd": 1.0,
    })
    formal_files = module._expected_result_files("formal", (0, 1, 2))
    c.eq(len(formal_files), 13)
    c.true(
        "checkpoints/seed_2/ravdess_then_mayo.pt.receipt.json"
        in formal_files
    )
    c.true("reports/formal_pretraining_results.json" in formal_files)
    with redirect_stderr(io.StringIO()):
        c.raises(
            lambda: module.main([]),
            SystemExit,
            "runner requires the exact two-stage command and all live roots",
        )
        c.raises(
            lambda: module.main(["two-stage", "--mode", "smoke"]),
            SystemExit,
            "mere mode selection cannot bypass frozen input authorization",
        )


def test_pretraining_runner_captures_native_mayo_root_output(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        with _frozen_bridge_inputs(root) as (frozen, _ravdess, _mayo):
            module, arguments, _run_root, roots = _synthetic_runner_fixture(
                root, frozen,
            )
            observed_producers: list[str] = []

            def leaking_run(_args, *, producer_sha256):
                observed_producers.append(producer_sha256)
                os.write(1, (str(roots["mayo"]) + "\n").encode("utf-8"))
                os.write(2, (str(roots["legacy"]) + "\n").encode("utf-8"))
                return {
                    "checkpoint_count": 2,
                    "mode": "smoke",
                    "seed_count": 1,
                    "stage_count": 2,
                }

            module._run_two_stage = leaking_run
            saved_stdout = os.dup(1)
            saved_stderr = os.dup(2)
            caught: BaseException | None = None
            with tempfile.TemporaryFile(mode="w+b") as native_stdout, \
                    tempfile.TemporaryFile(mode="w+b") as native_stderr:
                try:
                    os.dup2(native_stdout.fileno(), 1)
                    os.dup2(native_stderr.fileno(), 2)
                    try:
                        module.main(arguments)
                    except BaseException as exc:
                        caught = exc
                finally:
                    os.dup2(saved_stderr, 2)
                    os.dup2(saved_stdout, 1)
                    os.close(saved_stderr)
                    os.close(saved_stdout)
                native_stdout.seek(0)
                native_stderr.seek(0)
                emitted = native_stdout.read() + native_stderr.read()
            c.true(isinstance(caught, ValueError))
            c.eq(str(caught), "private Mayo command failed")
            c.eq(observed_producers, [frozen["producer_sha256"]])
            c.true(str(roots["mayo"]).encode("utf-8") not in emitted)
            c.true(str(roots["legacy"]).encode("utf-8") not in emitted)


def test_two_stage_missing_roots_and_mode_replay_fail_before_run_mutation(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        with _frozen_bridge_inputs(root) as (frozen, _ravdess, _mayo):
            module, arguments, run_root, roots = _synthetic_runner_fixture(
                root, frozen,
            )
            roots["mayo"].rmdir()
            factory_calls = 0

            def unexpected_factory(_args):
                nonlocal factory_calls
                factory_calls += 1
                raise AssertionError("authorization factory ran before root preflight")

            module._authorization_factories = unexpected_factory
            c.raises(
                lambda: module.main(arguments),
                ValueError,
                "a missing Mayo live root fails before authorization and run mutation",
            )
            c.eq(factory_calls, 0)
            c.true(not (run_root / ".results.lock").exists())
            c.true(not (run_root / "results").exists())
            c.true(not any(
                path.name.startswith(".results.staging-")
                for path in run_root.iterdir()
            ))

            roots["mayo"].mkdir(mode=0o700)
            arbitrary_parent = root / "arbitrary"
            arbitrary_parent.mkdir(mode=0o700)
            arbitrary_run = arbitrary_parent / "receipt-bound"
            arbitrary_run.mkdir(mode=0o700)
            wrong_run_arguments = list(arguments)
            run_value = wrong_run_arguments.index("--run-root") + 1
            wrong_run_arguments[run_value] = str(arbitrary_run)
            c.raises(
                lambda: module.main(wrong_run_arguments),
                ValueError,
                "copied inputs cannot run outside the canonical mode namespace",
            )
            c.true(not (arbitrary_run / ".results.lock").exists())

            arbitrary_bridge = root / "arbitrary-bridge"
            arbitrary_bridge.mkdir(mode=0o700)
            wrong_bridge_arguments = list(arguments)
            bridge_value = wrong_bridge_arguments.index("--bridge-root") + 1
            wrong_bridge_arguments[bridge_value] = str(arbitrary_bridge)
            c.raises(
                lambda: module.main(wrong_bridge_arguments),
                ValueError,
                "a copied bridge cannot replace the canonical generation",
            )
            c.true(not (run_root / ".results.lock").exists())

            module, arguments, run_root, _roots = _synthetic_runner_fixture(
                root / "second", frozen,
            )
            formal_arguments = list(arguments)
            formal_arguments[formal_arguments.index("smoke")] = "formal"
            train_calls = 0

            def unexpected_train(*_args, **_kwargs):
                nonlocal train_calls
                train_calls += 1
                raise AssertionError("training ran for a cross-mode receipt")

            original_train = module.ssl_core.train_ssl_stage
            try:
                module.ssl_core.train_ssl_stage = unexpected_train
                c.raises(
                    lambda: module.main(formal_arguments),
                    ValueError,
                    "a smoke receipt cannot authorize a formal run",
                )
            finally:
                module.ssl_core.train_ssl_stage = original_train
            c.eq(train_calls, 0)
            c.true(not (run_root / "results").exists())
            c.true(not any(
                path.name.startswith(".results.staging-")
                for path in run_root.iterdir()
            ))


def test_two_stage_all_live_inputs_fail_preflight_without_mutation(c: Check):
    required = (
        "--ravdess-data-root",
        "--ravdess-key",
        "--mayo-data-root",
        "--mayo-existing-export-root",
        "--mayo-cache-root",
        "--mayo-exposure-manifest",
        "--mayo-key",
    )
    for flag in required:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with _frozen_bridge_inputs(root) as (frozen, _ravdess, _mayo):
                module, arguments, run_root, roots = _synthetic_runner_fixture(
                    root / "live-preflight", frozen,
                )
                missing = Path(arguments[arguments.index(flag) + 1])
                if missing.is_dir():
                    for child in missing.iterdir():
                        child.unlink()
                    missing.rmdir()
                else:
                    missing.unlink()
                factory_calls = 0
                optimizer_calls = 0
                original_factory = module._authorization_factories
                original_adamw = torch.optim.AdamW

                def unexpected_factory(_args):
                    nonlocal factory_calls
                    factory_calls += 1
                    return original_factory(_args)

                def counted_adamw(*args, **kwargs):
                    nonlocal optimizer_calls
                    optimizer_calls += 1
                    return original_adamw(*args, **kwargs)

                module._authorization_factories = unexpected_factory
                torch.optim.AdamW = counted_adamw
                stdout = io.StringIO()
                stderr = io.StringIO()
                try:
                    with redirect_stdout(stdout), redirect_stderr(stderr):
                        c.raises(
                            lambda: module.main(arguments),
                            ValueError,
                            f"missing {flag} fails before any run mutation",
                        )
                finally:
                    torch.optim.AdamW = original_adamw
                    module._authorization_factories = original_factory
                c.eq(factory_calls, 0)
                c.eq(optimizer_calls, 0)
                c.eq(stdout.getvalue(), "")
                c.eq(stderr.getvalue(), "")
                c.true(not (run_root / ".results.lock").exists())
                c.true(not (run_root / "results").exists())
                c.true(not any(
                    path.name.startswith(".results.staging-")
                    for path in run_root.iterdir()
                ))
                persisted_output = (
                    stdout.getvalue() + stderr.getvalue()
                ).encode("utf-8")
                for private_root in (roots["mayo"], roots["legacy"]):
                    c.true(str(private_root).encode("utf-8") not in persisted_output)


def test_two_stage_rejects_every_frozen_stage_file_before_optimizer(c: Check):
    for stage in ("ravdess", "mayo"):
        for artifact in (
            "receipt", "manifest", "config", "split", "scaler", "bundle",
        ):
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                with _frozen_bridge_inputs(root) as (frozen, _ravdess, _mayo):
                    module, arguments, run_root, _roots = (
                        _synthetic_runner_fixture(
                            root / f"{stage}-{artifact}", frozen,
                        )
                    )
                    if artifact == "receipt":
                        target = (
                            Path(frozen["inputs_root"])
                            / "receipts" / f"{stage}.json"
                        )
                    elif artifact == "bundle":
                        target = (
                            Path(frozen["bridge_root"])
                            / "bundles" / f"{stage}_bundle.npz"
                        )
                    else:
                        target = (
                            Path(frozen["inputs_root"])
                            / "artifacts" / stage / f"{artifact}.json"
                        )
                    if target.suffix == ".json":
                        value = json.loads(target.read_text(encoding="ascii"))
                        value["synthetic_tamper"] = True
                        target.write_text(
                            json.dumps(
                                value,
                                sort_keys=True,
                                separators=(",", ":"),
                                ensure_ascii=True,
                            ),
                            encoding="ascii",
                        )
                    else:
                        payload = bytearray(target.read_bytes())
                        payload[len(payload) // 2] ^= 1
                        target.write_bytes(bytes(payload))
                    target.chmod(0o600)
                    optimizer_calls = 0
                    original_adamw = torch.optim.AdamW

                    def counted_adamw(*args, **kwargs):
                        nonlocal optimizer_calls
                        optimizer_calls += 1
                        return original_adamw(*args, **kwargs)

                    torch.optim.AdamW = counted_adamw
                    try:
                        c.raises(
                            lambda: module.main(arguments),
                            ValueError,
                            f"{stage} {artifact} tamper fails closed",
                        )
                    finally:
                        torch.optim.AdamW = original_adamw
                    c.eq(
                        optimizer_calls,
                        0,
                        f"{stage} {artifact} fails before optimizer",
                    )
                    c.true(not (run_root / "results").exists())
                    c.true(not any(
                        path.name.startswith(".results.staging-")
                        for path in run_root.iterdir()
                    ))


def test_two_stage_never_repairs_an_unsafe_existing_results_lock(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        with _frozen_bridge_inputs(root) as (frozen, _ravdess, _mayo):
            module, arguments, run_root, _roots = _synthetic_runner_fixture(
                root, frozen,
            )
            lock = run_root / ".results.lock"
            lock.write_bytes(b"")
            lock.chmod(0o644)
            train_calls = 0
            original_train = module.ssl_core.train_ssl_stage

            def unexpected_train(*_args, **_kwargs):
                nonlocal train_calls
                train_calls += 1
                raise AssertionError("training ran with an unsafe results lock")

            try:
                module.ssl_core.train_ssl_stage = unexpected_train
                c.raises(
                    lambda: module.main(arguments),
                    ValueError,
                    "an unsafe persistent lock fails closed without chmod repair",
                )
            finally:
                module.ssl_core.train_ssl_stage = original_train
            c.eq(train_calls, 0)
            c.eq(stat.S_IMODE(lock.stat().st_mode), 0o644)
            c.true(not (run_root / "results").exists())
            c.true(not any(
                path.name.startswith(".results.staging-")
                for path in run_root.iterdir()
            ))


def test_results_publication_ledger_rejects_file_root_and_destination_races(c: Check):
    module = _load_runner()
    with tempfile.TemporaryDirectory() as td:
        run_root = Path(td).resolve()
        run_descriptor = os.open(
            run_root,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            staging = run_root / ".results.staging-ledger"
            reports = staging / "reports"
            reports.mkdir(parents=True, mode=0o700)
            staging.chmod(0o700)
            reports.chmod(0o700)
            report = reports / "execution_only.json"
            report.write_bytes(b"{}")
            report.chmod(0o600)
            expected = {"reports/execution_only.json"}
            with module._hold_exact_result_tree(
                run_descriptor, staging.name, expected,
            ) as validate:
                _same_byte_replace(report)
                c.raises(
                    validate,
                    ValueError,
                    "same-byte result replacement contradicts the held ledger",
                )

            staging_two = run_root / ".results.staging-root-swap"
            reports_two = staging_two / "reports"
            reports_two.mkdir(parents=True, mode=0o700)
            staging_two.chmod(0o700)
            reports_two.chmod(0o700)
            report_two = reports_two / "execution_only.json"
            report_two.write_bytes(b"{}")
            report_two.chmod(0o600)
            with module._hold_exact_result_tree(
                run_descriptor, staging_two.name, expected,
            ) as validate:
                moved = run_root / ".held-original"
                os.rename(staging_two, moved)
                replacement_reports = staging_two / "reports"
                replacement_reports.mkdir(parents=True, mode=0o700)
                staging_two.chmod(0o700)
                replacement_reports.chmod(0o700)
                replacement = replacement_reports / "execution_only.json"
                replacement.write_bytes(b"{}")
                replacement.chmod(0o600)
                c.raises(
                    validate,
                    ValueError,
                    "a same-name staging tree cannot replace the held root",
                )

            source = run_root / ".results.staging-collision"
            destination = run_root / "results"
            source.mkdir(mode=0o700)
            destination.mkdir(mode=0o700)
            c.raises(
                lambda: module._rename_directory_no_replace(
                    source.name,
                    destination.name,
                    parent_descriptor=run_descriptor,
                ),
                OSError,
                "publication never replaces an existing results generation",
            )
            c.true(source.is_dir() and destination.is_dir())
        finally:
            os.close(run_descriptor)


def test_result_tree_aggregate_budget_precedes_file_hash(c: Check):
    module = _load_runner()
    with tempfile.TemporaryDirectory() as td:
        run_root = Path(td).resolve()
        run_root.chmod(0o700)
        staging = run_root / ".results.staging-aggregate-limit"
        reports = staging / "reports"
        reports.mkdir(parents=True, mode=0o700)
        staging.chmod(0o700)
        reports.chmod(0o700)
        report = reports / "execution_only.json"
        with report.open("wb") as handle:
            handle.truncate(128 * 1024 * 1024 + 1)
        report.chmod(0o600)
        run_descriptor = os.open(
            run_root,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        original_hash = module._hash_held_file
        hash_calls = 0

        def unexpected_hash(*_args, **_kwargs):
            nonlocal hash_calls
            hash_calls += 1
            raise AssertionError("aggregate-overflow result reached hashing")

        module._hash_held_file = unexpected_hash
        try:
            c.raises(
                lambda: module._hold_exact_result_tree(
                    run_descriptor,
                    staging.name,
                    {"reports/execution_only.json"},
                ).__enter__(),
                ValueError,
                "shared 128 MiB result budget fails before hashing",
            )
        finally:
            module._hash_held_file = original_hash
            os.close(run_descriptor)
        c.eq(hash_calls, 0, "aggregate overflow reaches no file hash")


def test_results_publication_classifies_commit_faults_and_rechecks_tree(c: Check):
    module = _load_runner()

    def run_case(case: str) -> tuple[Path, Path]:
        temporary = tempfile.TemporaryDirectory()
        cleanups.append(temporary)
        run_root = Path(temporary.name).resolve()
        run_root.chmod(0o700)
        inputs = run_root / "inputs"
        inputs.mkdir(mode=0o700)
        lock = run_root / ".results.lock"
        lock.write_bytes(b"")
        lock.chmod(0o600)
        staging = run_root / f".results.staging-{case}"
        reports = staging / "reports"
        reports.mkdir(parents=True, mode=0o700)
        staging.chmod(0o700)
        reports.chmod(0o700)
        report = reports / "execution_only.json"
        report.write_bytes(b"{}")
        report.chmod(0o600)
        run_descriptor = os.open(
            run_root,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        lock_descriptor = os.open(lock, os.O_RDWR)
        descriptors.append((lock_descriptor, run_descriptor))
        staged = staging.stat()
        identity = (int(staged.st_dev), int(staged.st_ino))
        original_scan = module._scan_private_results
        module._scan_private_results = lambda *_args, **_kwargs: None
        scans.append((module, original_scan))
        with module._hold_exact_result_tree(
            run_descriptor,
            staging.name,
            {"reports/execution_only.json"},
        ) as validate:
            module._publish_validated_results(
                run_descriptor=run_descriptor,
                run_root=run_root,
                run_identity=module._anchor_identity(os.fstat(run_descriptor)),
                results_lock=lock_descriptor,
                lock_name=lock.name,
                staging_name=staging.name,
                staged_identity=identity,
                inputs_root=inputs,
                privacy_forbidden=object(),
                validate_result_tree=validate,
                final_authorization=lambda: None,
            )
        return run_root, staging

    cleanups: list[tempfile.TemporaryDirectory] = []
    descriptors: list[tuple[int, int]] = []
    scans: list[tuple[object, object]] = []
    original_rename = module._rename_directory_no_replace
    original_fsync = module.os.fsync
    try:
        def rename_then_report_fault(*args, **kwargs):
            original_rename(*args, **kwargs)
            raise OSError("synthetic rename return fault")

        module._rename_directory_no_replace = rename_then_report_fault
        c.raises(
            lambda: run_case("rename-return"),
            RuntimeError,
            "a rename return fault retains but does not endorse the commit",
        )
        committed_root = Path(cleanups[-1].name)
        committed_staging = committed_root / ".results.staging-rename-return"
        c.true((committed_root / "results").is_dir())
        c.true(not committed_staging.exists())

        module._rename_directory_no_replace = original_rename

        def rename_then_inject(*args, **kwargs):
            original_rename(*args, **kwargs)
            destination = Path(cleanups[-1].name) / "results" / "reports"
            injected = destination / "late-private-root.txt"
            injected.write_bytes(b"late private material")
            injected.chmod(0o600)

        module._rename_directory_no_replace = rename_then_inject
        c.raises(
            lambda: run_case("late-injection"),
            RuntimeError,
            "a post-rename extra file cannot be reported as a valid result",
        )
        injected_root = Path(cleanups[-1].name)
        c.true((injected_root / "results").is_dir())
        c.true((
            injected_root / "results" / "reports" / "late-private-root.txt"
        ).is_file())

        module._rename_directory_no_replace = original_rename

        def postrename_fsync_fault(descriptor):
            if (Path(cleanups[-1].name) / "results").exists():
                raise OSError("synthetic post-rename fsync fault")
            return original_fsync(descriptor)

        module.os.fsync = postrename_fsync_fault
        c.raises(
            lambda: run_case("postrename-fsync"),
            RuntimeError,
            "a post-rename durability fault is retained as indeterminate",
        )
        fsync_root = Path(cleanups[-1].name)
        c.true((fsync_root / "results").is_dir())
    finally:
        module._rename_directory_no_replace = original_rename
        module.os.fsync = original_fsync
        for target, original_scan in scans:
            target._scan_private_results = original_scan
        for lock_descriptor, run_descriptor in descriptors:
            os.close(lock_descriptor)
            os.close(run_descriptor)
        for temporary in cleanups:
            temporary.cleanup()


def test_descriptor_cleanup_attempts_every_close_and_propagates_failure(c: Check):
    module = _load_runner()
    original_close = module.os.close
    for helper in (module._close_descriptor_sequence, ssl_core._close_ssl_descriptors):
        calls: list[int] = []

        def failing_close(descriptor):
            calls.append(descriptor)
            raise OSError(f"synthetic close failure {descriptor}")

        module.os.close = failing_close
        try:
            c.raises(
                lambda helper=helper: helper((101, 102, 103)),
                OSError,
                "descriptor cleanup errors cannot be swallowed",
            )
        finally:
            module.os.close = original_close
        c.eq(set(calls), {101, 102, 103})


def test_two_stage_parser_failure_never_echoes_mayo_roots(c: Check):
    module = _load_runner()
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        mayo_root = root / "mayo-private-root-sentinel"
        legacy_root = root / "mayo-legacy-root-sentinel"
        extra_root = root / "unexpected-private-root-sentinel"
        arguments = [
            "two-stage",
            "--mode", "smoke",
            "--run-root", str(root / "run"),
            "--bridge-root", str(root / "bridge"),
            "--ravdess-data-root", str(root / "ravdess"),
            "--ravdess-key", str(root / "ravdess-key"),
            "--mayo-data-root", str(mayo_root),
            "--mayo-existing-export-root", str(legacy_root),
            "--mayo-cache-root", str(root / "mayo-cache"),
            "--mayo-exposure-manifest", str(root / "mayo-exposure.json"),
            "--mayo-key", str(root / "mayo-key"),
            str(extra_root),
        ]
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            c.raises(
                lambda: module.main(arguments),
                SystemExit,
                "unknown two-stage arguments fail before any training action",
            )
        c.eq(stdout.getvalue(), "")
        emitted = stderr.getvalue()
        for private_root in (mayo_root, legacy_root, extra_root):
            c.true(
                str(private_root) not in emitted
                and private_root.name not in emitted,
                "two-stage parser failures never echo a Mayo root",
            )


def test_receipt_bound_smoke_runner_publishes_one_private_atomic_result(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        with _frozen_bridge_inputs(root) as (frozen, _ravdess, _mayo):
            module, arguments, run_root, roots = _synthetic_runner_fixture(
                root, frozen,
            )
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                result = module.main(arguments)
            c.eq(stderr.getvalue(), "")
            c.eq(result, {
                "checkpoint_count": 2,
                "mode": "smoke",
                "seed_count": 1,
                "stage_count": 2,
            })
            c.eq(json.loads(stdout.getvalue()), result)
            results = run_root / "results"
            c.true(results.is_dir())
            expected = {
                "checkpoints/ravdess_only.pt",
                "checkpoints/ravdess_only.pt.receipt.json",
                "checkpoints/ravdess_then_mayo.pt",
                "checkpoints/ravdess_then_mayo.pt.receipt.json",
                "reports/execution_only.json",
            }
            files = {
                str(path.relative_to(results))
                for path in results.rglob("*") if path.is_file()
            }
            c.eq(files, expected)
            c.true(all(
                path.stat().st_mode & 0o777 == 0o600
                for path in results.rglob("*") if path.is_file()
            ))
            report = json.loads(
                (results / "reports" / "execution_only.json")
                .read_text(encoding="ascii")
            )
            c.eq(report["mode"], "smoke")
            c.true("heldout" not in json.dumps(report).lower())
            persisted = b"\n".join(
                path.read_bytes() for path in sorted(results.rglob("*"))
                if path.is_file()
            ) + stdout.getvalue().encode("utf-8")
            for private_root in (roots["mayo"], roots["legacy"]):
                raw = str(private_root).encode("utf-8")
                tokens = {
                    raw,
                    private_root.name.encode("utf-8"),
                    raw.hex().encode("ascii"),
                    base64.b64encode(raw),
                    base64.urlsafe_b64encode(raw),
                }
                c.true(all(token not in persisted for token in tokens))
            c.true(not any(
                path.name.startswith(".results.staging-")
                for path in run_root.iterdir()
            ))


def test_smoke_runner_bounds_expensive_live_authorizers_to_transaction_edges(
    c: Check,
):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        with _frozen_bridge_inputs(root) as (frozen, _ravdess, _mayo):
            module, arguments, run_root, _roots = _synthetic_runner_fixture(
                root, frozen,
            )
            counts = {"ravdess": 0, "mayo": 0}
            raw_ravdess = frozen["ravdess_authorizer"]
            raw_mayo = frozen["mayo_authorizer"]

            def require_complete_staging() -> None:
                staging = [
                    path for path in run_root.iterdir()
                    if path.name.startswith(".results.staging-")
                ]
                c.eq(len(staging), 1, "publication-edge authorization is late")
                c.true((
                    staging[0] / "checkpoints" / "ravdess_only.pt"
                ).is_file())
                c.true((
                    staging[0] / "checkpoints" / "ravdess_then_mayo.pt"
                ).is_file())
                c.true((
                    staging[0] / "reports" / "execution_only.json"
                ).is_file())

            def counted_ravdess():
                counts["ravdess"] += 1
                if counts["ravdess"] == 2:
                    require_complete_staging()
                return raw_ravdess()

            def counted_mayo():
                counts["mayo"] += 1
                if counts["mayo"] == 2:
                    require_complete_staging()
                return raw_mayo()

            module._authorization_factories = lambda _args: (
                counted_ravdess, counted_mayo,
            )
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                module.main(arguments)
            c.eq(
                counts,
                {"ravdess": 2, "mayo": 2},
                "one live authorization at transaction entry and one before publish",
            )


def test_results_transaction_holds_both_generation_leases_through_edge_auth(
    c: Check,
):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        with _frozen_bridge_inputs(root) as (frozen, _ravdess, _mayo):
            module, arguments, _run_root, roots = _synthetic_runner_fixture(
                root, frozen,
            )
            locks = (
                roots["ravdess"] / ".derived_semantic23.lock",
                module.PRETRAINING_ROOT / ".mayo_ssl_cache.lock",
            )
            for path in locks:
                path.write_bytes(b"")
                path.chmod(0o600)
            counts = {"ravdess": 0, "mayo": 0}
            raw_ravdess = frozen["ravdess_authorizer"]
            raw_mayo = frozen["mayo_authorizer"]
            contender = (
                "import fcntl,os,sys; "
                "fd=os.open(sys.argv[1],os.O_RDWR); "
                "fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)"
            )

            def require_both_leases() -> None:
                for path in locks:
                    result = subprocess.run(
                        [sys.executable, "-c", contender, str(path)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False,
                    )
                    c.true(
                        result.returncode != 0,
                        f"exclusive contender must be blocked for {path.name}",
                    )

            def counted_ravdess():
                counts["ravdess"] += 1
                if counts["ravdess"] == 2:
                    require_both_leases()
                return raw_ravdess()

            def counted_mayo():
                counts["mayo"] += 1
                if counts["mayo"] == 2:
                    require_both_leases()
                return raw_mayo()

            module._authorization_factories = lambda _args: (
                counted_ravdess, counted_mayo,
            )
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                module.main(arguments)
            c.eq(counts, {"ravdess": 2, "mayo": 2})


def test_production_read_authorizers_support_reentrant_shared_leases(c: Check):
    from scripts import build_mayo_ssl_cache as mayo_cli
    from scripts import prepare_ravdess_semantic23 as ravdess_cli

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        root.chmod(0o700)

        ravdess_parent = root / "ravdess"
        ravdess_parent.mkdir(mode=0o700)
        ravdess_lock_name = ".derived_semantic23.lock"
        ravdess_lock = ravdess_parent / ravdess_lock_name
        ravdess_lock.touch()
        ravdess_lock.chmod(0o600)
        ravdess_parent_descriptor = os.open(
            ravdess_parent, os.O_RDONLY | os.O_DIRECTORY,
        )
        ravdess_outer = os.open(ravdess_lock, os.O_RDONLY)
        fcntl.flock(ravdess_outer, fcntl.LOCK_SH)
        try:
            descriptor, identity = ravdess_cli._acquire_output_lock(
                ravdess_parent_descriptor,
                ravdess_lock_name,
                create_if_missing=False,
                shared=True,
            )
            ravdess_cli._release_output_lock(
                ravdess_parent,
                ravdess_parent_descriptor,
                (
                    ravdess_parent.stat().st_dev,
                    ravdess_parent.stat().st_ino,
                ),
                ravdess_lock_name,
                descriptor,
                identity,
            )
        finally:
            os.close(ravdess_outer)
            os.close(ravdess_parent_descriptor)

        mayo_output = root / "mayo-cache"
        mayo_lock = root / ".mayo-cache.lock"
        mayo_lock.touch()
        mayo_lock.chmod(0o600)
        mayo_outer = os.open(mayo_lock, os.O_RDONLY)
        fcntl.flock(mayo_outer, fcntl.LOCK_SH)
        try:
            with mayo_cli.output_parent_lock(
                mayo_output,
                create_if_missing=False,
                shared=True,
            ):
                c.true(
                    mayo_output not in mayo_cli._HELD_OUTPUT_LOCKS,
                    "a read lease never grants output mutation authority",
                )
        finally:
            os.close(mayo_outer)


def test_generation_lease_path_swap_is_rejected_before_result_rename(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        with _frozen_bridge_inputs(root) as (frozen, _ravdess, _mayo):
            module, arguments, run_root, roots = _synthetic_runner_fixture(
                root, frozen,
            )
            raw_ravdess = frozen["ravdess_authorizer"]
            raw_mayo = frozen["mayo_authorizer"]
            mayo_calls = 0
            lock_path = roots["ravdess"] / ".derived_semantic23.lock"

            def swap_lock_path_at_edge():
                nonlocal mayo_calls
                mayo_calls += 1
                authorization = raw_mayo()
                if mayo_calls == 2:
                    replacement = lock_path.with_name(".replacement.lock")
                    replacement.touch()
                    replacement.chmod(0o600)
                    os.replace(replacement, lock_path)
                return authorization

            module._authorization_factories = lambda _args: (
                raw_ravdess, swap_lock_path_at_edge,
            )
            c.raises(
                lambda: module.main(arguments),
                ValueError,
                "a canonical generation lease swap blocks result rename",
            )
            c.eq(mayo_calls, 2)
            c.true(
                not (run_root / "results").exists(),
                "lease identity is revalidated before canonical publication",
            )
            residues = [
                path for path in run_root.iterdir()
                if path.name.startswith(".results.staging-")
            ]
            c.eq(len(residues), 1)
            c.eq(stat.S_IMODE(residues[0].stat().st_mode), 0o700)


def test_publication_edge_reauthorization_rejects_drift_after_final_scan(
    c: Check,
):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        with _frozen_bridge_inputs(root) as (frozen, ravdess, _mayo):
            module, arguments, run_root, _roots = _synthetic_runner_fixture(
                root, frozen,
            )
            original_scan = module._scan_private_results
            scan_calls = 0

            def drift_after_final_scan(*args, **kwargs):
                nonlocal scan_calls
                original_scan(*args, **kwargs)
                scan_calls += 1
                if scan_calls == 3:
                    ravdess.manifest_sha256 = "9" * 64

            module._scan_private_results = drift_after_final_scan
            try:
                c.raises(
                    lambda: module.main(arguments),
                    ValueError,
                    "a live generation change after the final scan blocks rename",
                )
            finally:
                module._scan_private_results = original_scan
            c.eq(scan_calls, 3, "failure occurs at the publication edge")
            c.true(not (run_root / "results").exists())
            residues = [
                path for path in run_root.iterdir()
                if path.name.startswith(".results.staging-")
            ]
            c.eq(len(residues), 1)
            c.eq(stat.S_IMODE(residues[0].stat().st_mode), 0o700)
            c.true((
                residues[0] / "reports" / "execution_only.json"
            ).is_file())


def test_publication_edge_compares_a_fresh_independent_authorization(
    c: Check,
):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        with _frozen_bridge_inputs(root) as (frozen, _ravdess, _mayo):
            module, arguments, run_root, _roots = _synthetic_runner_fixture(
                root, frozen,
            )
            raw_ravdess = frozen["ravdess_authorizer"]
            raw_mayo = frozen["mayo_authorizer"]
            ravdess_calls = 0

            def fresh_ravdess():
                nonlocal ravdess_calls
                ravdess_calls += 1
                authorization = raw_ravdess()
                if ravdess_calls == 1:
                    return authorization
                values = dict(vars(authorization))
                values["manifest_sha256"] = "9" * 64
                return types.SimpleNamespace(**values)

            module._authorization_factories = lambda _args: (
                fresh_ravdess, raw_mayo,
            )
            compared: list[str] = []
            original_compare = module._require_publication_edge_authorization

            def observed_compare(*, stage, evidence, authorization):
                compared.append(stage)
                return original_compare(
                    stage=stage,
                    evidence=evidence,
                    authorization=authorization,
                )

            module._require_publication_edge_authorization = observed_compare
            try:
                c.raises(
                    lambda: module.main(arguments),
                    ValueError,
                    "a fresh changed authorization blocks result publication",
                )
            finally:
                module._require_publication_edge_authorization = original_compare
            c.eq(ravdess_calls, 2)
            c.eq(
                compared,
                ["mayo", "ravdess"],
                "both independently refreshed generations are compared",
            )
            c.true(not (run_root / "results").exists())
            residues = [
                path for path in run_root.iterdir()
                if path.name.startswith(".results.staging-")
            ]
            c.eq(len(residues), 1)
            c.eq(stat.S_IMODE(residues[0].stat().st_mode), 0o700)


def test_receipt_bound_formal_runner_publishes_three_seed_contract(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        with _frozen_bridge_inputs(
            root, mode="formal",
        ) as (frozen, _ravdess, _mayo):
            module, arguments, run_root, roots = _synthetic_runner_fixture(
                root / "formal-runner", frozen,
            )
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                result = module.main(arguments)
            c.eq(result, {
                "checkpoint_count": 6,
                "mode": "formal",
                "seed_count": 3,
                "stage_count": 2,
            })
            c.eq(stderr.getvalue(), "")
            c.eq(json.loads(stdout.getvalue()), result)
            results = run_root / "results"
            files = {
                str(path.relative_to(results))
                for path in results.rglob("*") if path.is_file()
            }
            c.eq(
                files,
                module._expected_result_files("formal", (0, 1, 2)),
            )
            report = json.loads((
                results / "reports" / "formal_pretraining_results.json"
            ).read_text(encoding="ascii"))
            c.eq(report["seed_count"], 3)
            c.eq(len(report["runs"]), 3)
            c.eq({run["seed"] for run in report["runs"]}, {0, 1, 2})
            for run in report["runs"]:
                c.eq(run["ravdess_only"]["optimizer_steps"], 30)
                c.eq(run["ravdess_then_mayo"]["optimizer_steps"], 30)
                c.eq(
                    run["ravdess_then_mayo"][
                        "prior_checkpoint_fingerprint"
                    ],
                    run["ravdess_only"]["checkpoint_fingerprint"],
                )
            c.eq(set(report["aggregate"]), {
                "ravdess_only", "ravdess_then_mayo",
            })
            persisted = b"\n".join(
                path.read_bytes() for path in sorted(results.rglob("*"))
                if path.is_file()
            ) + stdout.getvalue().encode("utf-8")
            for private_root in (roots["mayo"], roots["legacy"]):
                raw = str(private_root).encode("utf-8")
                c.true(raw not in persisted)
                c.true(raw.hex().encode("ascii") not in persisted)
                c.true(base64.b64encode(raw) not in persisted)
                c.true(base64.urlsafe_b64encode(raw) not in persisted)
            c.true(not any(
                path.name.startswith(".results.staging-")
                for path in run_root.iterdir()
            ))


def test_formal_publication_reauthorizes_every_seed_lineage(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        with _frozen_bridge_inputs(
            root, mode="formal",
        ) as (frozen, _ravdess, _mayo):
            module, arguments, _run_root, _roots = _synthetic_runner_fixture(
                root / "formal-lineage", frozen,
            )
            counts = {"ravdess": 0, "mayo": 0}
            raw_ravdess = frozen["ravdess_authorizer"]
            raw_mayo = frozen["mayo_authorizer"]

            def counted_ravdess():
                counts["ravdess"] += 1
                return raw_ravdess()

            def counted_mayo():
                counts["mayo"] += 1
                return raw_mayo()

            module._authorization_factories = lambda _args: (
                counted_ravdess, counted_mayo,
            )
            original_authorize = module.ssl_core.authorize_frozen_ssl_stage
            final_mayo_priors: list[str] = []

            def observed_authorize(*args, **kwargs):
                evidence = original_authorize(*args, **kwargs)
                if kwargs.get("stage") == "mayo":
                    final_mayo_priors.append(
                        module.ssl_core.ssl_checkpoint_fingerprint(
                            kwargs["prior_ravdess_checkpoint"],
                        )
                    )
                return evidence

            module.ssl_core.authorize_frozen_ssl_stage = observed_authorize
            try:
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    module.main(arguments)
            finally:
                module.ssl_core.authorize_frozen_ssl_stage = original_authorize
            c.eq(counts, {"ravdess": 2, "mayo": 2})
            c.eq(
                len(final_mayo_priors),
                6,
                "each formal seed is authorized during training and publication",
            )
            c.eq(
                len(set(final_mayo_priors)),
                3,
                "formal authorization binds three distinct priors",
            )
            c.true(
                all(
                    final_mayo_priors.count(prior) == 2
                    for prior in set(final_mayo_priors)
                ),
                "every formal prior is reauthorized exactly once before publication",
            )


def test_two_stage_failure_retains_private_staging_and_blocks_retry(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        with _frozen_bridge_inputs(root) as (frozen, _ravdess, _mayo):
            module, arguments, run_root, _roots = _synthetic_runner_fixture(
                root, frozen,
            )
            original_train = module.ssl_core.train_ssl_stage
            stages: list[str] = []

            def fail_at_mayo(*args, **kwargs):
                evidence = kwargs["stage_evidence"]
                stages.append(evidence.stage)
                if evidence.stage == "mayo":
                    raise RuntimeError("synthetic fault after persisted RAVDESS")
                return original_train(*args, **kwargs)

            try:
                module.ssl_core.train_ssl_stage = fail_at_mayo
                c.raises(
                    lambda: module.main(arguments),
                    ValueError,
                    "a Mayo-stage fault cannot publish partial results",
                )
            finally:
                module.ssl_core.train_ssl_stage = original_train
            c.eq(stages, ["ravdess", "mayo"])
            c.true(not (run_root / "results").exists())
            residues = [
                path for path in run_root.iterdir()
                if path.name.startswith(".results.staging-")
            ]
            c.eq(len(residues), 1)
            c.eq(stat.S_IMODE(residues[0].stat().st_mode), 0o700)
            c.true((residues[0] / "checkpoints" / "ravdess_only.pt").is_file())

            retry_train_calls = 0

            def unexpected_retry(*_args, **_kwargs):
                nonlocal retry_train_calls
                retry_train_calls += 1
                raise AssertionError("retry trained despite retained staging")

            try:
                module.ssl_core.train_ssl_stage = unexpected_retry
                c.raises(
                    lambda: module.main(arguments),
                    ValueError,
                    "retained indeterminate staging blocks the next run",
                )
            finally:
                module.ssl_core.train_ssl_stage = original_train
            c.eq(retry_train_calls, 0)
            c.true(not (run_root / "results").exists())


if __name__ == "__main__":
    run_all("test_dynamic_landmark_ssl", dict(globals()))
