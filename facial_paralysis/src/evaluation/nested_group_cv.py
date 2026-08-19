"""Frozen, leakage-guarded nested cross-validation for binary groups."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np


OUTER_FOLDS = 5
INNER_FOLDS = 4
_FROZEN_SPLIT_NAMESPACE = "dynamic-landmark-nested-group-cv-v1"
_GROUP_ID = re.compile(r"^grp_[0-9a-f]{64}$")


@dataclass(frozen=True)
class InnerGroupFold:
    """Global sample indices for one inner train/validation split."""

    train_indices: np.ndarray
    validation_indices: np.ndarray

    @property
    def val_indices(self) -> np.ndarray:
        return self.validation_indices


@dataclass(frozen=True)
class NestedGroupFold:
    """Global sample indices for one outer split and its inner splits."""

    train_indices: np.ndarray
    test_indices: np.ndarray
    inner_folds: tuple[InnerGroupFold, ...]

    @property
    def outer_train_indices(self) -> np.ndarray:
        return self.train_indices

    @property
    def outer_test_indices(self) -> np.ndarray:
        return self.test_indices


def _validated_inputs(
    labels: Iterable[int],
    group_ids: Iterable[object],
) -> tuple[np.ndarray, np.ndarray]:
    try:
        labels_array = (
            np.asarray(labels) if isinstance(labels, np.ndarray)
            else np.asarray(tuple(labels))
        )
        groups_array = (
            np.asarray(group_ids) if isinstance(group_ids, np.ndarray)
            else np.asarray(tuple(group_ids))
        )
    except TypeError as exc:
        raise ValueError("labels and group_ids must be iterable") from exc
    if labels_array.ndim != 1 or groups_array.ndim != 1:
        raise ValueError("labels and group_ids must be one-dimensional")
    if labels_array.shape[0] == 0 or labels_array.shape != groups_array.shape:
        raise ValueError("labels and group_ids must have the same nonzero length")
    if labels_array.dtype.kind not in {"b", "i", "u"}:
        raise ValueError("labels must be binary integers")
    labels_array = labels_array.astype(np.int64, copy=False)
    if not np.isin(labels_array, (0, 1)).all():
        raise ValueError("labels must contain only binary values 0 and 1")

    group_values = groups_array.tolist()
    if any(not isinstance(value, str) for value in group_values):
        raise ValueError("group_ids must be canonical opaque strings")
    if any(_GROUP_ID.fullmatch(value) is None for value in group_values):
        raise ValueError(
            "group_ids must use canonical opaque format grp_ followed by "
            "64 lowercase hexadecimal characters"
        )
    groups = np.asarray(group_values, dtype=str)

    group_labels: dict[str, int] = {}
    for group, label in zip(groups.tolist(), labels_array.tolist()):
        previous = group_labels.setdefault(group, label)
        if previous != label:
            raise ValueError(
                f"binary labels must be consistent within group {group!r}"
            )
    return labels_array, groups


def _stable_group_key(group: str) -> bytes:
    return hashlib.sha256(
        f"{_FROZEN_SPLIT_NAMESPACE}:{group}".encode("utf-8")
    ).digest()


def _stratified_group_holdouts(
    labels: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
) -> tuple[np.ndarray, ...]:
    group_indices = {
        group: np.flatnonzero(groups == group)
        for group in sorted(set(groups.tolist()))
    }
    groups_by_label: dict[int, list[str]] = {0: [], 1: []}
    for group, indices in group_indices.items():
        groups_by_label[int(labels[indices[0]])].append(group)
    for label, class_groups in groups_by_label.items():
        if len(class_groups) < n_splits:
            raise ValueError(
                f"binary class {label} has {len(class_groups)} groups; "
                f"at least {n_splits} are required"
            )

    fold_groups: list[set[str]] = [set() for _ in range(n_splits)]
    for label in (0, 1):
        class_counts = [0] * n_splits
        ordered = sorted(
            groups_by_label[label],
            key=lambda group: (-len(group_indices[group]), _stable_group_key(group)),
        )
        for group in ordered:
            target = min(
                range(n_splits),
                key=lambda fold: (class_counts[fold], len(fold_groups[fold]), fold),
            )
            fold_groups[target].add(group)
            class_counts[target] += len(group_indices[group])

    return tuple(
        np.flatnonzero(np.isin(groups, sorted(held_out))).astype(np.int64)
        for held_out in fold_groups
    )


def _assert_partition(
    universe: np.ndarray,
    train: np.ndarray,
    held_out: np.ndarray,
    name: str,
) -> None:
    universe_set = set(universe.tolist())
    train_set = set(train.tolist())
    held_out_set = set(held_out.tolist())
    if train_set.intersection(held_out_set):
        raise AssertionError(f"{name} train and held-out indices overlap")
    if train_set.union(held_out_set) != universe_set:
        raise AssertionError(f"{name} does not cover its sample universe exactly")
    if len(train_set) != train.size or len(held_out_set) != held_out.size:
        raise AssertionError(f"{name} contains duplicate indices")


def build_nested_group_splits(
    labels: Iterable[int],
    group_ids: Iterable[object],
) -> tuple[NestedGroupFold, ...]:
    """Build frozen 5-fold outer and 4-fold inner stratified group splits."""
    labels_array, groups = _validated_inputs(labels, group_ids)
    all_indices = np.arange(labels_array.shape[0], dtype=np.int64)
    outer_tests = _stratified_group_holdouts(labels_array, groups, OUTER_FOLDS)
    outer_test_counts = np.zeros(labels_array.shape[0], dtype=np.int64)
    nested: list[NestedGroupFold] = []

    for outer_number, outer_test in enumerate(outer_tests):
        outer_test = np.sort(outer_test)
        outer_train = np.setdiff1d(all_indices, outer_test, assume_unique=True)
        _assert_partition(
            all_indices, outer_train, outer_test, f"outer fold {outer_number}"
        )
        if not set(groups[outer_train]).isdisjoint(set(groups[outer_test])):
            raise AssertionError(f"outer fold {outer_number} splits a group")
        outer_test_counts[outer_test] += 1

        relative_validations = _stratified_group_holdouts(
            labels_array[outer_train], groups[outer_train], INNER_FOLDS
        )
        inner_validation_counts = np.zeros(outer_train.shape[0], dtype=np.int64)
        inner_folds: list[InnerGroupFold] = []
        for inner_number, relative_validation in enumerate(relative_validations):
            relative_validation = np.sort(relative_validation)
            inner_validation_counts[relative_validation] += 1
            validation = np.sort(outer_train[relative_validation])
            train = np.setdiff1d(outer_train, validation, assume_unique=True)
            _assert_partition(
                outer_train,
                train,
                validation,
                f"outer fold {outer_number} inner fold {inner_number}",
            )
            if not set(groups[train]).isdisjoint(set(groups[validation])):
                raise AssertionError(
                    f"outer fold {outer_number} inner fold {inner_number} splits a group"
                )
            assert_outer_test_isolation(
                outer_test,
                train_indices=train,
                validation_indices=validation,
            )
            inner_folds.append(InnerGroupFold(train, validation))

        if not np.all(inner_validation_counts == 1):
            raise AssertionError(
                f"outer fold {outer_number} inner validation folds do not cover "
                "the outer train exactly once"
            )
        nested.append(NestedGroupFold(outer_train, outer_test, tuple(inner_folds)))

    if not np.all(outer_test_counts == 1):
        raise AssertionError("outer test folds do not cover every sample exactly once")
    return tuple(nested)


def _index_array(name: str, values: Iterable[int] | np.ndarray) -> np.ndarray:
    if isinstance(values, np.ndarray):
        array = np.asarray(values)
    else:
        try:
            array = np.asarray(tuple(values))
        except TypeError as exc:
            raise ValueError(f"{name} must be an iterable of indices") from exc
    if array.ndim == 1 and array.size == 0:
        return array.astype(np.int64)
    if array.ndim != 1 or array.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{name} must be a one-dimensional integer index set")
    int64_info = np.iinfo(np.int64)
    integer_values = [int(value) for value in array.tolist()]
    if any(value < int64_info.min or value > int64_info.max
           for value in integer_values):
        raise ValueError(f"{name} values must fit in signed int64")
    if np.any(array < 0):
        raise ValueError(f"{name} cannot contain negative indices")
    if np.unique(array).size != array.size:
        raise ValueError(f"{name} cannot contain duplicate indices")
    return np.asarray(integer_values, dtype=np.int64)


def assert_outer_test_isolation(
    outer_test_indices: Iterable[int] | np.ndarray,
    non_test_index_sets: Mapping[str, Iterable[int] | np.ndarray] | None = None,
    *,
    train_indices: Iterable[int] | np.ndarray | None = None,
    validation_indices: Iterable[int] | np.ndarray | None = None,
    scaler_fit_indices: Iterable[int] | np.ndarray | None = None,
    prototype_fit_indices: Iterable[int] | np.ndarray | None = None,
    selection_indices: Iterable[int] | np.ndarray | None = None,
    **additional_non_test_indices: Iterable[int] | np.ndarray,
) -> None:
    """Raise if any outer-test sample enters training or model selection state."""
    outer_test = _index_array("outer_test_indices", outer_test_indices)
    named_sets: dict[str, Iterable[int] | np.ndarray] = {}
    if non_test_index_sets is not None:
        if not isinstance(non_test_index_sets, Mapping):
            raise ValueError("non_test_index_sets must be a name-to-indices mapping")
        named_sets.update(non_test_index_sets)
    explicit = {
        "train_indices": train_indices,
        "validation_indices": validation_indices,
        "scaler_fit_indices": scaler_fit_indices,
        "prototype_fit_indices": prototype_fit_indices,
        "selection_indices": selection_indices,
    }
    explicit = {name: values for name, values in explicit.items()
                if values is not None}
    duplicate_names = sorted(named_sets.keys() & explicit.keys())
    if duplicate_names:
        raise ValueError(
            f"non-test index sets were supplied more than once: {duplicate_names}"
        )
    named_sets.update(explicit)
    duplicate_names = sorted(
        named_sets.keys() & additional_non_test_indices.keys()
    )
    if duplicate_names:
        raise ValueError(
            f"non-test index sets were supplied more than once: {duplicate_names}"
        )
    named_sets.update(additional_non_test_indices)

    for name, values in named_sets.items():
        indices = _index_array(name, values)
        overlap = np.intersect1d(outer_test, indices, assume_unique=True)
        if overlap.size:
            raise ValueError(
                f"outer test leakage into {name}: sample indices {overlap.tolist()}"
            )


# Stable terminology aliases for training and selection callers.
frozen_nested_group_splits = build_nested_group_splits
make_nested_group_splits = build_nested_group_splits
assert_no_outer_test_leakage = assert_outer_test_isolation
guard_outer_test_isolation = assert_outer_test_isolation
InnerGroupSplit = InnerGroupFold
NestedGroupSplit = NestedGroupFold
