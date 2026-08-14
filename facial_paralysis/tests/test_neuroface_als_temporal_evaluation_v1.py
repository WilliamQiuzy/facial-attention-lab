"""Leakage and reproducibility contracts for NeuroFace temporal LOSO."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src.evaluation.neuroface_als_temporal_v1 import (  # noqa: E402
    FROZEN_EPOCHS,
    FROZEN_SEEDS,
    apply_feature_scaling,
    fit_masked_feature_scaler,
    participant_loso_splits,
    validate_temporal_dataset,
)
from _testlib import Check, run_all  # noqa: E402


def _dataset(count: int = 6):
    rng = np.random.default_rng(41)
    features = rng.normal(size=(count, 3, 4, 32, 95)).astype(np.float32)
    mask = np.ones((count, 3, 4, 32), dtype=bool)
    timestamps = np.broadcast_to(
        np.arange(32, dtype=np.float32) / 30.0,
        (count, 3, 4, 32),
    ).copy()
    labels = np.asarray([0, 1] * (count // 2), dtype=np.int64)
    groups = tuple(f"grp_{index:064x}" for index in range(count))
    return features, mask, timestamps, labels, groups


def test_protocol_is_fixed_and_participant_level(c: Check):
    c.eq(FROZEN_SEEDS, (17, 43, 79), "three seed ensemble is frozen")
    c.eq(FROZEN_EPOCHS, 200, "training duration is frozen before real scoring")
    dataset = validate_temporal_dataset(*_dataset())
    c.eq(dataset.features.shape[0], 6, "dataset contains one row per participant")
    splits = participant_loso_splits(dataset.labels, dataset.group_ids)
    c.eq(len(splits), 6, "one outer fold is created per participant")
    c.true(all(len(test) == 1 for _, test in splits), "outer tests contain one person")
    c.true(all(set(train).isdisjoint(test) for train, test in splits),
           "outer participants never enter their training fold")


def test_scaler_uses_train_rows_and_masks_only(c: Check):
    features, mask, timestamps, labels, groups = _dataset()
    mask[0, :, :, 2:5] = False
    features[0][~mask[0]] = 0.0
    train = np.arange(5, dtype=np.int64)
    scaler = fit_masked_feature_scaler(features[train], mask[train])
    expected = features[train][mask[train]].astype(np.float64).mean(axis=0)
    c.true(np.allclose(scaler.mean, expected.astype(np.float32), rtol=0, atol=1e-6),
           "only valid training rows contribute to the scaler")
    changed_held = features.copy()
    changed_held[5] += 10_000_000.0
    same = fit_masked_feature_scaler(changed_held[train], mask[train])
    c.true(np.array_equal(scaler.mean, same.mean),
           "held-out values cannot affect train statistics")
    scaled = apply_feature_scaling(features, mask, scaler)
    c.true(np.all(scaled[~mask] == 0), "invalid rows become canonical zero")
    c.true(np.isfinite(scaled).all(), "scaled features remain finite")


def test_dataset_rejects_record_expansion_and_bad_time(c: Check):
    features, mask, timestamps, labels, groups = _dataset()
    c.raises(lambda: validate_temporal_dataset(
        features.repeat(3, axis=0), mask.repeat(3, axis=0),
        timestamps.repeat(3, axis=0), labels, groups,
    ), ValueError, "three task recordings cannot become three labelled rows")
    bad_time = timestamps.copy()
    bad_time[..., 12] = bad_time[..., 11]
    c.raises(lambda: validate_temporal_dataset(
        features, mask, bad_time, labels, groups,
    ), ValueError, "timestamps must be ordered within each frozen window")


def test_scaler_output_is_immutable_and_torch_ready(c: Check):
    features, mask, _, _, _ = _dataset()
    scaler = fit_masked_feature_scaler(features[:5], mask[:5])
    c.raises(lambda: scaler.mean.__setitem__(0, 0.0), ValueError,
             "training mean is immutable evidence")
    scaled = apply_feature_scaling(features, mask, scaler)
    tensor = torch.from_numpy(scaled.copy())
    c.eq(tensor.dtype, torch.float32, "scaled representation preserves float32")


if __name__ == "__main__":
    run_all("test_neuroface_als_temporal_evaluation_v1", dict(globals()))
