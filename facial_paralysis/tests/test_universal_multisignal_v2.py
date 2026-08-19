"""Representation and evaluation contracts for Universal Multi-Signal v2."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src.evaluation.universal_multisignal_v2 import (  # noqa: E402
    REPRESENTATIONS,
    aggregate_multisignal_recordings,
    evaluate_multisignal_oof,
    fit_locked_multisignal,
    locked_multisignal_from_dict,
    locked_multisignal_to_dict,
    predict_locked_multisignal,
    select_multisignal_representation,
)
from src.preprocessing.universal_multisignal_v2 import (  # noqa: E402
    multisignal_feature_views,
)
from _testlib import Check, run_all  # noqa: E402


def _group(index: int) -> str:
    return f"grp_{index:064x}"


def _synthetic_dataset():
    labels = np.asarray([0] * 6 + [1] * 6 + [0] * 6 + [1] * 6, dtype=np.int64)
    sources = tuple(["palsynet"] * 12 + ["neuroface"] * 12)
    groups = tuple(_group(index) for index in range(24))
    rows = {}
    for name, dimension in REPRESENTATIONS.items():
        original = np.zeros((24, dimension), dtype=np.float64)
        mirrored = np.zeros_like(original)
        if name == "blendshape_288":
            original[:, 0] = labels * 4.0 - 2.0
            mirrored[:, 0] = original[:, 0]
        rows[name] = (original, mirrored)
    return aggregate_multisignal_recordings(rows, labels, groups, sources)


def test_representation_space_is_exact_and_fusion_is_concatenation(c: Check):
    c.eq(REPRESENTATIONS, {
        "landmark_110": 110, "blendshape_288": 288, "fusion_398": 398,
    }, "v2 changes the representation, not the estimator")
    rng = np.random.default_rng(4)
    features = rng.normal(size=(4, 32, 95)).astype(np.float32)
    mask = np.ones((4, 32), dtype=bool)
    timestamps = np.stack([w * 10 + np.arange(32) / 30 for w in range(4)])
    indices = np.stack([w * 100 + np.arange(32) for w in range(4)]).astype(np.int64)
    views = multisignal_feature_views(features, mask, timestamps, indices)
    c.eq({name: pair[0].shape[0] for name, pair in views.items()}, REPRESENTATIONS,
         "all three representations use the same authenticated recording")
    c.true(np.array_equal(
        views["fusion_398"][0],
        np.concatenate((views["blendshape_288"][0], views["landmark_110"][0])),
    ), "fusion contains exact Blendshape then Landmark features")


def test_participant_aggregation_is_shared_across_representations(c: Check):
    labels = np.asarray([0, 0, 1], dtype=np.int64)
    groups = (_group(0), _group(0), _group(1))
    sources = ("palsynet", "palsynet", "neuroface")
    rows = {
        name: (
            np.tile(np.arange(dimension, dtype=np.float64), (3, 1)),
            np.tile(np.arange(dimension, dtype=np.float64), (3, 1)),
        ) for name, dimension in REPRESENTATIONS.items()
    }
    dataset = aggregate_multisignal_recordings(rows, labels, groups, sources)
    c.eq(dataset.group_ids, (_group(0), _group(1)),
         "recording duplicates aggregate once per participant")
    c.eq(dataset.sources, ("palsynet", "neuroface"),
         "all representations share one participant/source order")


def test_oof_and_selection_use_worst_source_only(c: Check):
    result = evaluate_multisignal_oof(
        _synthetic_dataset(), "blendshape_288"
    )
    c.true(result.metrics["palsynet"]["auroc"] >= 0.99,
           "shared Blendshape signal is learned in PalsyNet")
    c.true(result.metrics["neuroface"]["auroc"] >= 0.99,
           "shared Blendshape signal is learned in NeuroFace")
    winner = select_multisignal_representation({
        "landmark_110": {
            "worst_source_auroc": 0.80,
            "worst_source_balanced_accuracy": 0.80,
            "overall_brier": 0.10,
        },
        "blendshape_288": {
            "worst_source_auroc": 0.91,
            "worst_source_balanced_accuracy": 0.85,
            "overall_brier": 0.14,
        },
        "fusion_398": {
            "worst_source_auroc": 0.90,
            "worst_source_balanced_accuracy": 0.99,
            "overall_brier": 0.01,
        },
    })
    c.eq(winner, "blendshape_288",
         "worst-source AUROC dominates pooled or secondary gains")


def test_locked_multisignal_roundtrip_is_prediction_exact(c: Check):
    dataset = _synthetic_dataset()
    locked = fit_locked_multisignal(dataset, "blendshape_288")
    before = predict_locked_multisignal(
        locked, dataset.original["blendshape_288"],
        dataset.mirrored["blendshape_288"],
    )
    payload = locked_multisignal_to_dict(locked)
    c.eq(set(payload), {
        "schema_version", "representation", "scaler", "model",
    }, "locked v2 artifact is closed")
    restored = locked_multisignal_from_dict(payload)
    after = predict_locked_multisignal(
        restored, dataset.original["blendshape_288"],
        dataset.mirrored["blendshape_288"],
    )
    c.true(np.array_equal(before, after),
           "locked representation model roundtrip changes no prediction")


if __name__ == "__main__":
    run_all("test_universal_multisignal_v2", dict(globals()))
