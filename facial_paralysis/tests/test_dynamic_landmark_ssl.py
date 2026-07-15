"""Synthetic-only contracts for dynamic landmark masked-span pretraining."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _testlib import Check, run_all  # noqa: E402
from src.models.dynamic_landmark import DynamicLandmarkModel  # noqa: E402
from src.pretraining import dynamic_landmark_ssl as ssl_core  # noqa: E402
from src.pretraining.dynamic_landmark_ssl import (  # noqa: E402
    CHECKPOINT_RAVDESS_MAYO,
    CHECKPOINT_RAVDESS_ONLY,
    DynamicLandmarkSSLModel,
    PretrainingLockedError,
    SourceScaler,
    authorize_ssl_checkpoint_receipt,
    build_ssl_checkpoint_payload,
    build_ssl_stage_evidence,
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


def _temporal(batch: int, windows: int = 4, frames: int = 32):
    source = torch.arange(frames, dtype=torch.int64).reshape(1, 1, frames)
    source = source.repeat(batch, windows, 1)
    timestamps = source.to(torch.float32) / 30.0
    mask = torch.ones(batch, windows, frames, dtype=torch.bool)
    return mask, timestamps, source


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
    development_only = stage == "mayo_development"
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
            "batch_policy": "full_train_partition",
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
    width = 23 if source == "ravdess_semantic23_v1" else 95
    source_step = 1 if source == "ravdess_semantic23_v1" else 2
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
        (
            np.arange(32, dtype=np.int64) * source_step
        ).reshape(1, 1, 32),
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
        "ravdess_semantic23_v1"
        if stage == "ravdess"
        else "mayo_mediapipe_v2"
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
        baseline_features, baseline_mask, source="ravdess_semantic23_v1",
        fit_indices=split.train_indices, heldout_indices=split.heldout_indices,
    )
    report = reconstruction_report(
        trained, untrained, target, positions,
        baseline=baseline, split=split,
        evaluated_indices=split.heldout_indices,
        group_ids=groups, source="ravdess_semantic23_v1",
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
        group_ids=groups, source="ravdess_semantic23_v1",
    ), ValueError, "a one-row subset cannot inherit the complete actor-heldout claim")
    forged_baseline = SourceScaler(
        source="ravdess_semantic23_v1",
        mean=torch.full((2,), 2.0), scale=torch.ones(2),
        fit_indices=tuple(int(index) for index in split.heldout_indices),
    )
    c.raises(lambda: reconstruction_report(
        trained, untrained, target, positions,
        baseline=forged_baseline, split=split,
        evaluated_indices=split.heldout_indices,
        group_ids=groups, source="ravdess_semantic23_v1",
    ), ValueError, "the baseline must be fitted on the exact training partition")


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
        features, valid, source="ravdess_semantic23_v1",
        fit_indices=np.asarray([0, 1]), heldout_indices=np.asarray([2, 3]),
    )
    c.true(bool(torch.allclose(scaler.mean, torch.full((23,), 3.0))))
    c.eq(scaler.fit_indices, (0, 1))
    transformed = scaler.transform(
        features[:2], valid[:2], source="ravdess_semantic23_v1"
    )
    c.true(bool(torch.isfinite(transformed).all()))
    c.raises(lambda: scaler.transform(
        features[:2], valid[:2], source="mayo_mediapipe_v2"
    ), ValueError, "a source scaler can never cross detector/source boundaries")
    c.raises(lambda: fit_source_scaler(
        features, valid, source="ravdess_semantic23_v1",
        fit_indices=np.asarray([0, 2]), heldout_indices=np.asarray([2, 3]),
    ), ValueError, "heldout samples cannot enter source scaler state")
    scaler.mean[0] = float("nan")
    c.raises(lambda: scaler.transform(
        features[:2], valid[:2], source="ravdess_semantic23_v1"
    ), ValueError, "corrupt nonfinite scaler state fails closed")


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
    mayo_indices = mayo_indices * 2
    derivative_input = model.build_gru_input(
        mayo_ramp, mayo_valid, mayo_times, mayo_indices, source="mayo"
    )
    c.true(bool(torch.allclose(
        derivative_input[..., 1:, 0], torch.full((1, 4, 31), 30.0)
    )), "Mayo 60-to-30 provenance step two retains per-second blendshape deltas")
    c.true(bool(torch.allclose(
        derivative_input[..., 1:, 32], torch.full((1, 4, 31), 30.0)
    )), "Mayo 60-to-30 provenance step two retains per-second landmark deltas")
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
    jumped_indices[..., 16:] += 2
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
        result = ssl_core.train_ssl_stage(stage_evidence=evidence, seed=0)
        receipt = result.training_receipt.to_dict()
        c.eq(receipt["optimizer"], "adamw")
        c.eq(receipt["learning_rate"], 0.001)
        c.eq(receipt["weight_decay"], 0.0)
        c.eq(receipt["epochs"], 1)
        c.eq(receipt["batch_policy"], "full_train_partition")
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
            source="ravdess_semantic23_v1",
            groups=groups,
            seed=71,
            heldout_indices=split.heldout_indices,
            heldout_offset=100.0,
        )
        leaked_rows = features[valid_mask]
        leaked_scaler = SourceScaler(
            source="ravdess_semantic23_v1",
            mean=leaked_rows.mean(dim=0),
            scale=leaked_rows.std(dim=0, unbiased=False),
            fit_indices=tuple(int(index) for index in split.train_indices),
        )
        manifest, config, split_artifact, scaler_artifact = (
            _write_stage_artifacts(
                root,
                stage="ravdess",
                source="ravdess_semantic23_v1",
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
            source="ravdess_semantic23_v1",
            groups=groups,
            seed=31,
            heldout_indices=split.heldout_indices,
        )
        scaler = fit_source_scaler(
            features,
            valid_mask,
            source="ravdess_semantic23_v1",
            fit_indices=split.train_indices,
            heldout_indices=split.heldout_indices,
        )
        manifest, config, split_artifact, scaler_artifact = (
            _write_stage_artifacts(
                root,
                stage="ravdess",
                source="ravdess_semantic23_v1",
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
            source="ravdess_semantic23_v1",
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
            stage="mayo_development",
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
        c.eq(result.training_receipt.batch_policy, "full_train_partition")
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


def test_real_pretraining_runner_is_locked_until_manifests_and_config_exist(c: Check):
    script = ROOT / "scripts" / "pretrain_dynamic_landmarks.py"
    spec = importlib.util.spec_from_file_location("locked_ssl_runner", script)
    if spec is None or spec.loader is None:
        raise AssertionError("SSL runner cannot be imported")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    c.raises(lambda: module.main([]), PretrainingLockedError,
             "real input cannot run without preregistered manifests/config")
    c.raises(lambda: module.main([
        "--ravdess-manifest", "fake.json", "--mayo-manifest", "fake2.json",
        "--config", "fake3.json",
    ]), PretrainingLockedError,
        "mere filenames cannot bypass frozen manifest validation")


if __name__ == "__main__":
    run_all("test_dynamic_landmark_ssl", dict(globals()))
