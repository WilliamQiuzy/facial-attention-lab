"""Contracts for the compact task-aware NeuroFace temporal classifier."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src.models.dynamic_landmark import horizontal_mirror_features  # noqa: E402
from src.models.neuroface_als_temporal_v1 import (  # noqa: E402
    PARAMETER_CAP,
    PRIMARY_TASKS,
    TaskAwareTemporalALSClassifier,
    count_parameters,
    mirror_mean_probability,
    participant_balanced_bce,
    within_window_velocity,
)
from _testlib import Check, run_all  # noqa: E402


def _inputs(batch: int = 4):
    generator = torch.Generator().manual_seed(17)
    features = torch.randn(batch, 3, 4, 32, 95, generator=generator)
    mask = torch.ones(batch, 3, 4, 32, dtype=torch.bool)
    timestamps = torch.arange(32, dtype=torch.float32).view(1, 1, 1, 32)
    timestamps = timestamps.repeat(batch, 3, 4, 1) / 30.0
    return features, mask, timestamps


def test_architecture_is_small_fixed_and_participant_level(c: Check):
    c.eq(PRIMARY_TASKS, ("NSM_KISS", "NSM_OPEN", "NSM_SPREAD"),
         "task order is frozen")
    torch.manual_seed(3)
    model = TaskAwareTemporalALSClassifier()
    c.true(count_parameters(model) <= PARAMETER_CAP,
           "small-data temporal model stays below the parameter cap")
    features, mask, timestamps = _inputs()
    logits = model(features, mask, timestamps)
    c.eq(tuple(logits.shape), (4,), "one logit is emitted per participant")
    c.true(bool(torch.isfinite(logits).all()), "participant logits are finite")


def test_velocity_never_crosses_windows_or_missing_rows(c: Check):
    features, mask, timestamps = _inputs(batch=1)
    features.zero_()
    features[..., 0] = torch.arange(32, dtype=features.dtype)
    mask[..., 10] = False
    velocity = within_window_velocity(features, mask, timestamps)
    c.true(bool((velocity[..., 0, :] == 0).all()),
           "the first row of every window has no cross-window derivative")
    c.true(bool((velocity[..., 10, :] == 0).all()),
           "a missing destination row has no derivative")
    c.true(bool((velocity[..., 11, :] == 0).all()),
           "a missing source row cannot bridge a detector gap")


def test_masked_values_cannot_change_output(c: Check):
    features, mask, timestamps = _inputs(batch=2)
    mask[:, :, :, 5:9] = False
    changed = features.clone()
    changed[~mask] = 1_000_000.0
    torch.manual_seed(9)
    model = TaskAwareTemporalALSClassifier().eval()
    with torch.inference_mode():
        first = model(features, mask, timestamps)
        second = model(changed, mask, timestamps)
    c.true(torch.equal(first, second), "masked pixels have zero model influence")


def test_mirror_inference_is_probability_mean(c: Check):
    features, mask, timestamps = _inputs(batch=3)
    mirrored = horizontal_mirror_features(features)
    c.true(torch.equal(horizontal_mirror_features(mirrored), features),
           "raw 95D mirror remains an exact involution")
    torch.manual_seed(12)
    model = TaskAwareTemporalALSClassifier().eval()
    with torch.inference_mode():
        expected = 0.5 * (
            torch.sigmoid(model(features, mask, timestamps))
            + torch.sigmoid(model(mirrored, mask, timestamps))
        )
        observed = mirror_mean_probability(model, features, mask, timestamps)
    c.true(torch.equal(observed, expected), "mirror aggregation occurs in probability space")


def test_loss_weights_people_not_task_records(c: Check):
    logits = torch.asarray([0.0, 1.0, -1.0])
    labels = torch.asarray([0.0, 1.0, 0.0])
    loss = participant_balanced_bce(logits, labels)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    c.true(torch.equal(loss, expected), "each participant has exactly one equal loss weight")
    c.raises(lambda: participant_balanced_bce(
        logits.repeat_interleave(3), labels
    ), ValueError, "record-level label expansion is rejected")


if __name__ == "__main__":
    run_all("test_neuroface_als_temporal_v1", dict(globals()))
