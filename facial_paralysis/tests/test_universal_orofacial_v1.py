"""Participant-level contracts for Universal Orofacial Model v1."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src.evaluation.universal_orofacial_v1 import (  # noqa: E402
    SOURCES,
    aggregate_participant_recordings,
    binary_metrics,
    source_class_balanced_weights,
    stratified_source_class_folds,
)
from _testlib import Check, run_all  # noqa: E402


def _group(index: int) -> str:
    return f"grp_{index:064x}"


def test_recordings_aggregate_once_per_participant_and_mirror_separately(c: Check):
    original = np.stack([
        np.full(110, 1.0), np.full(110, 3.0), np.full(110, 5.0),
    ]).astype(np.float64)
    mirrored = original + 10.0
    dataset = aggregate_participant_recordings(
        original, mirrored,
        labels=np.asarray([1, 1, 0], dtype=np.int64),
        group_ids=(_group(1), _group(1), _group(2)),
        sources=("palsynet", "palsynet", "neuroface"),
    )
    c.eq(dataset.original.shape, (2, 110),
         "one fixed 110D row is emitted per participant")
    c.true(np.allclose(dataset.original[0], 2.0),
           "repeated participant recordings are mean aggregated")
    c.true(np.allclose(dataset.mirrored[0], 12.0),
           "mirror values are aggregated independently")
    c.eq(dataset.labels.tolist(), [1, 0], "participant labels remain aligned")
    c.eq(dataset.group_ids, (_group(1), _group(2)),
         "participant order is deterministic")
    c.eq(dataset.sources, ("palsynet", "neuroface"),
         "source remains metadata and is not appended to features")
    c.true(not hasattr(dataset, "recording_counts"),
           "recording count cannot become a source shortcut feature")


def test_dataset_rejects_identity_source_label_and_numeric_drift(c: Check):
    values = np.ones((4, 110), dtype=np.float64)
    mirror = values.copy()
    c.raises(lambda: aggregate_participant_recordings(
        values, mirror, np.asarray([0, 1, 0, 1], dtype=np.int64),
        ("person-1", _group(2), _group(3), _group(4)),
        ("palsynet", "palsynet", "neuroface", "neuroface"),
    ), ValueError, "participant IDs must be opaque commitments")
    c.raises(lambda: aggregate_participant_recordings(
        values, mirror, np.asarray([0, 1, 0, 1], dtype=np.int64),
        (_group(1), _group(1), _group(3), _group(4)),
        ("palsynet", "palsynet", "neuroface", "neuroface"),
    ), ValueError, "one participant cannot carry inconsistent labels")
    bad = values.copy()
    bad[0, 0] = np.nan
    c.raises(lambda: aggregate_participant_recordings(
        bad, mirror, np.asarray([0, 1, 0, 1], dtype=np.int64),
        tuple(_group(index) for index in range(4)),
        ("palsynet", "palsynet", "neuroface", "neuroface"),
    ), ValueError, "nonfinite representation values fail closed")
    c.raises(lambda: aggregate_participant_recordings(
        values, mirror, np.asarray([0, 1, 0, 1], dtype=np.int64),
        tuple(_group(index) for index in range(4)),
        ("mayo", "palsynet", "neuroface", "neuroface"),
    ), ValueError, "only frozen development sources are accepted")


def test_arrays_are_immutable_even_via_setflags(c: Check):
    values = np.stack([np.zeros(110), np.ones(110)]).astype(np.float64)
    dataset = aggregate_participant_recordings(
        values, values, np.asarray([0, 1], dtype=np.int64),
        (_group(1), _group(2)), ("palsynet", "neuroface"),
    )
    for array in (dataset.original, dataset.mirrored, dataset.labels):
        c.raises(lambda array=array: array.setflags(write=True), ValueError,
                 "public dataset arrays cannot regain write access")


def test_source_class_weights_give_each_group_equal_total_mass(c: Check):
    labels = np.asarray([0, 0, 1, 0, 1, 1, 1], dtype=np.int64)
    sources = (
        "palsynet", "palsynet", "palsynet",
        "neuroface", "neuroface", "neuroface", "neuroface",
    )
    weights = source_class_balanced_weights(labels, sources)
    c.true(np.isclose(weights.sum(), 1.0), "participant weights total one")
    for source in SOURCES:
        for label in (0, 1):
            mask = np.asarray([
                observed_source == source and observed_label == label
                for observed_source, observed_label in zip(sources, labels)
            ])
            c.true(np.isclose(weights[mask].sum(), 0.25),
                   "every source-class group receives equal total mass")
    c.raises(lambda: weights.setflags(write=True), ValueError,
             "training weights are immutable")


def test_six_folds_are_deterministic_stratified_and_participant_disjoint(c: Check):
    labels = np.asarray(
        [0] * 12 + [1] * 12 + [0] * 12 + [1] * 12,
        dtype=np.int64,
    )
    sources = tuple(["palsynet"] * 24 + ["neuroface"] * 24)
    groups = tuple(_group(index) for index in range(48))
    first = stratified_source_class_folds(labels, groups, sources)
    second = stratified_source_class_folds(labels, groups, sources)
    c.eq(len(first), 6, "the development protocol has six fixed folds")
    c.true(all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
               for a, b in zip(first, second)), "fold creation is deterministic")
    held = np.concatenate([test for _, test in first])
    c.true(np.array_equal(np.sort(held), np.arange(48)),
           "every participant is held exactly once")
    for train, test in first:
        c.true(not set(train.tolist()) & set(test.tolist()),
               "train and held participants are disjoint")
        held_groups = {(sources[index], int(labels[index])) for index in test}
        c.eq(held_groups, {(source, label) for source in SOURCES for label in (0, 1)},
             "every held fold represents all source-class groups")


def test_binary_metrics_are_exact_and_reject_single_class(c: Check):
    metrics = binary_metrics(
        np.asarray([0, 0, 1, 1], dtype=np.int64),
        np.asarray([0.1, 0.6, 0.4, 0.9], dtype=np.float64),
    )
    c.eq(metrics["accuracy"], 0.5, "fixed 0.5-threshold accuracy is exact")
    c.eq(metrics["balanced_accuracy"], 0.5,
         "balanced accuracy is mean sensitivity and specificity")
    c.eq(metrics["sensitivity"], 0.5, "sensitivity is exact")
    c.eq(metrics["specificity"], 0.5, "specificity is exact")
    c.eq(metrics["auroc"], 0.75, "participant AUROC is exact")
    c.raises(lambda: binary_metrics(
        np.asarray([1, 1], dtype=np.int64), np.asarray([0.4, 0.6])
    ), ValueError, "metrics require both classes")


if __name__ == "__main__":
    run_all("test_universal_orofacial_v1", dict(globals()))
