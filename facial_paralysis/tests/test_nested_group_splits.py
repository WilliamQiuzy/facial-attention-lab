"""Leakage-proof contracts for frozen nested group cross-validation."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.nested_group_cv import (  # noqa: E402
    assert_outer_test_isolation,
    build_nested_group_splits,
)
from _testlib import Check, run_all  # noqa: E402


def _group_id(label: int, group_number: int) -> str:
    return f"grp_{label * 5 + group_number + 1:064x}"


def _cohort() -> tuple[np.ndarray, np.ndarray]:
    labels: list[int] = []
    groups: list[str] = []
    for label in (0, 1):
        for group_number in range(5):
            group = _group_id(label, group_number)
            for _ in range(group_number % 2 + 2):
                labels.append(label)
                groups.append(group)
    return np.asarray(labels, dtype=np.int64), np.asarray(groups)


def _signature(folds) -> tuple:
    return tuple(
        (
            tuple(outer.train_indices.tolist()),
            tuple(outer.test_indices.tolist()),
            tuple(
                (tuple(inner.train_indices.tolist()),
                 tuple(inner.validation_indices.tolist()))
                for inner in outer.inner_folds
            ),
        )
        for outer in folds
    )


def test_nested_splits_are_frozen_and_have_required_fold_counts(c: Check):
    labels, groups = _cohort()
    first = build_nested_group_splits(labels, groups)
    second = build_nested_group_splits(labels, groups)
    expected_outer_groups = (
        (_group_id(0, 3), _group_id(1, 1)),
        (_group_id(0, 1), _group_id(1, 3)),
        (_group_id(0, 4), _group_id(1, 2)),
        (_group_id(0, 2), _group_id(1, 4)),
        (_group_id(0, 0), _group_id(1, 0)),
    )
    outer_groups = tuple(
        tuple(sorted(set(groups[outer.test_indices].tolist())))
        for outer in first
    )
    expected_inner_groups = tuple(
        tuple(group_pair for pair_number, group_pair in enumerate(expected_outer_groups)
              if pair_number != outer_number)
        for outer_number in range(len(expected_outer_groups))
    )
    inner_groups = tuple(
        tuple(
            tuple(sorted(set(groups[inner.validation_indices].tolist())))
            for inner in outer.inner_folds
        )
        for outer in first
    )
    c.eq(len(first), 5, "five outer folds")
    c.true(all(len(outer.inner_folds) == 4 for outer in first),
           "four inner folds per outer train")
    c.eq(_signature(first), _signature(second), "split assignment is frozen")
    c.eq(outer_groups, expected_outer_groups,
         "group-to-fold assignment is pinned to the frozen protocol")
    c.eq(inner_groups, expected_inner_groups,
         "inner group-to-fold assignment is pinned inside each outer train")

    reversed_folds = build_nested_group_splits(labels[::-1], groups[::-1])
    reversed_outer_groups = tuple(
        tuple(sorted(set(groups[::-1][outer.test_indices].tolist())))
        for outer in reversed_folds
    )
    c.eq(reversed_outer_groups, expected_outer_groups,
         "group assignment does not depend on sample ordering")
    for label in (0, 1):
        class_counts = [int(np.sum(labels[outer.test_indices] == label))
                        for outer in first]
        c.true(max(class_counts) - min(class_counts) <= 1,
               f"class {label} sample counts are balanced across outer folds")
        for outer in first:
            inner_counts = [
                int(np.sum(labels[inner.validation_indices] == label))
                for inner in outer.inner_folds
            ]
            c.true(max(inner_counts) - min(inner_counts) <= 1,
                   f"class {label} sample counts are balanced across inner folds")


def test_nested_splits_accept_iterables_without_changing_assignment(c: Check):
    labels, groups = _cohort()
    expected = build_nested_group_splits(labels, groups)
    generated = build_nested_group_splits(
        (int(label) for label in labels),
        (str(group) for group in groups),
    )
    c.eq(_signature(generated), _signature(expected),
         "iterable inputs preserve the frozen split assignment")


def test_outer_folds_cover_each_sample_once_without_group_overlap(c: Check):
    labels, groups = _cohort()
    folds = build_nested_group_splits(labels, groups)
    test_counts = np.zeros(labels.shape[0], dtype=np.int64)
    for outer in folds:
        train = set(outer.train_indices.tolist())
        test = set(outer.test_indices.tolist())
        c.true(not train.intersection(test), "outer train and test are disjoint")
        c.eq(train.union(test), set(range(labels.shape[0])),
             "outer fold covers the cohort exactly")
        c.true(set(groups[outer.train_indices]).isdisjoint(
                   set(groups[outer.test_indices])),
               "a group never crosses outer train/test")
        c.eq(set(labels[outer.test_indices].tolist()), {0, 1},
             "outer test is stratified by binary label")
        test_counts[outer.test_indices] += 1
    c.true(np.all(test_counts == 1), "outer tests partition every sample once")


def test_inner_folds_cover_outer_train_once_and_never_touch_outer_test(c: Check):
    labels, groups = _cohort()
    for outer in build_nested_group_splits(labels, groups):
        outer_train = set(outer.train_indices.tolist())
        outer_test = set(outer.test_indices.tolist())
        validation_counts = {index: 0 for index in outer_train}
        for inner in outer.inner_folds:
            train = set(inner.train_indices.tolist())
            validation = set(inner.validation_indices.tolist())
            c.true(not train.intersection(validation),
                   "inner train and validation are disjoint")
            c.eq(train.union(validation), outer_train,
                 "inner fold covers outer train exactly")
            c.true(not outer_test.intersection(train.union(validation)),
                   "outer test is absent from all inner work")
            c.true(set(groups[inner.train_indices]).isdisjoint(
                       set(groups[inner.validation_indices])),
                   "a group never crosses inner train/validation")
            c.eq(set(labels[inner.validation_indices].tolist()), {0, 1},
                 "inner validation is stratified by binary label")
            for index in validation:
                validation_counts[index] += 1
        c.true(all(count == 1 for count in validation_counts.values()),
               "inner validations partition the outer train once")


def test_splits_reject_nonbinary_inconsistent_or_insufficient_groups(c: Check):
    labels, groups = _cohort()
    inconsistent = labels.copy()
    inconsistent[1] = 1 - inconsistent[1]
    c.raises(lambda: build_nested_group_splits(inconsistent, groups), ValueError,
             "labels must be consistent within a group")

    nonbinary = labels.copy()
    nonbinary[-1] = 2
    c.raises(lambda: build_nested_group_splits(nonbinary, groups), ValueError,
             "labels must be binary")

    keep = groups != _group_id(1, 4)
    c.raises(lambda: build_nested_group_splits(labels[keep], groups[keep]),
             ValueError, "each class needs five groups for frozen outer folds")

    for invalid_group in (None, 17, f"{_group_id(0, 0)} ", "group_alpha"):
        invalid_groups = groups.tolist()
        invalid_groups[0] = invalid_group
        c.raises(lambda invalid_groups=invalid_groups: build_nested_group_splits(
            labels, invalid_groups), ValueError,
            "group ids must be canonical opaque strings")


def test_outer_test_isolation_guard_accepts_disjoint_training_state(c: Check):
    assert_outer_test_isolation(
        np.asarray([8, 9]),
        train_indices=np.asarray([0, 1, 2]),
        validation_indices=np.asarray([3, 4]),
        scaler_fit_indices=np.asarray([0, 1]),
        prototype_fit_indices=np.asarray([2, 3]),
        selection_indices=np.asarray([4, 5]),
    )
    assert_outer_test_isolation(
        np.asarray([8, 9]),
        validation_indices=[],
    )
    c.true(True, "disjoint non-test sets are accepted")


def test_outer_test_cannot_be_used_for_validation_or_fitting(c: Check):
    outer_test = np.asarray([8, 9])
    c.raises(lambda: assert_outer_test_isolation(
        outer_test, validation_indices=np.asarray([3, 8])), ValueError,
        "outer test cannot be validation data")
    c.raises(lambda: assert_outer_test_isolation(
        outer_test, scaler_fit_indices=np.asarray([0, 9])), ValueError,
        "outer test cannot fit a scaler")
    c.raises(lambda: assert_outer_test_isolation(
        outer_test, prototype_fit_indices=np.asarray([8])), ValueError,
        "outer test cannot fit prototypes")
    c.raises(lambda: assert_outer_test_isolation(
        outer_test, selection_indices=np.asarray([9])), ValueError,
        "outer test cannot influence selection")
    c.raises(lambda: assert_outer_test_isolation(
        outer_test, train_indices=np.asarray([0, 8])), ValueError,
        "outer test cannot enter model fitting")
    c.raises(lambda: assert_outer_test_isolation(
        outer_test,
        {"validation_indices": np.asarray([8])},
        validation_indices=np.asarray([3])), ValueError,
        "a duplicate guard name cannot overwrite a leaking index set")
    c.raises(lambda: assert_outer_test_isolation(
        outer_test,
        {"auxiliary_fit_indices": np.asarray([8])},
        auxiliary_fit_indices=np.asarray([3])), ValueError,
        "an additional guard name cannot overwrite a leaking index set")


def test_outer_test_guard_rejects_indices_outside_int64(c: Check):
    too_large = np.asarray([np.iinfo(np.uint64).max], dtype=np.uint64)
    c.raises(lambda: assert_outer_test_isolation(too_large), ValueError,
             "outer test indices must fit int64 without wrapping")
    c.raises(lambda: assert_outer_test_isolation(
        np.asarray([8]), train_indices=too_large), ValueError,
        "training indices must fit int64 without wrapping")


if __name__ == "__main__":
    run_all("test_nested_group_splits", dict(globals()))
