from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _testlib import run_all

from src.evaluation.shared_clinical_encoder_v1 import (
    SOURCES,
    SharedClinicalDataset,
    evaluate_shared_model,
    fit_clinical_scaler,
    participant_disjoint_folds,
    source_class_balanced_weights,
)


def _dataset(per_cell: int = 2) -> SharedClinicalDataset:
    rng = np.random.default_rng(19)
    labels = []
    sources = []
    groups = []
    for source_index, source in enumerate(SOURCES):
        for label in (0, 1):
            for member in range(per_cell):
                sources.append(source)
                labels.append(label)
                groups.append(f"grp_{len(groups):064x}")
    count = len(labels)
    actions = 3
    clinical_original = rng.normal(0.0, 0.2, (count, actions, 110)).astype(np.float32)
    clinical_original += np.asarray(labels, dtype=np.float32)[:, None, None] * 0.35
    clinical_mirrored = clinical_original.copy()
    clinical_mirrored[:, :, 0] *= -1.0
    dense_original = rng.normal(
        0.0, 0.03, (count, actions, 32, 478, 3)
    ).astype(np.float32)
    dense_original += np.asarray(labels, dtype=np.float32)[:, None, None, None, None] * 0.02
    dense_mirrored = dense_original.copy()
    dense_mirrored[..., 0] *= -1.0
    dense_valid = np.ones((count, actions, 32), dtype=bool)
    dense_available = np.ones((count, actions), dtype=bool)
    action_mask = np.ones((count, actions), dtype=bool)
    action_codes = np.tile(np.arange(actions, dtype=np.int64), (count, 1))
    return SharedClinicalDataset(
        clinical_original=clinical_original,
        clinical_mirrored=clinical_mirrored,
        dense_original=dense_original,
        dense_mirrored=dense_mirrored,
        dense_valid_mask=dense_valid,
        dense_available=dense_available,
        action_mask=action_mask,
        action_codes=action_codes,
        labels=np.asarray(labels, dtype=np.int64),
        group_ids=tuple(groups),
        sources=tuple(sources),
    )


def test_participant_folds_are_group_disjoint_and_source_class_stratified(c):
    dataset = _dataset(per_cell=6)
    folds = participant_disjoint_folds(dataset, n_splits=6)
    c.eq(len(folds), 6)
    all_held = []
    for train, held in folds:
        c.true(set(train).isdisjoint(set(held)))
        c.eq(len(set(dataset.group_ids[index] for index in held)), len(held))
        held_cells = {(dataset.sources[index], int(dataset.labels[index])) for index in held}
        c.eq(held_cells, {(source, label) for source in SOURCES for label in (0, 1)})
        all_held.extend(int(index) for index in held)
    c.eq(sorted(all_held), list(range(len(dataset.labels))))


def test_source_class_weights_give_every_cell_equal_mass(c):
    dataset = _dataset(per_cell=3)
    weights = source_class_balanced_weights(dataset.labels, dataset.sources)
    c.true(np.isclose(weights.sum(), 1.0))
    for source in SOURCES:
        for label in (0, 1):
            selected = np.asarray([
                observed_source == source and int(observed_label) == label
                for observed_source, observed_label in zip(dataset.sources, dataset.labels)
            ])
            c.true(np.isclose(weights[selected].sum(), 1.0 / 6.0))


def test_fold_scaler_ignores_held_participants_and_padding(c):
    dataset = _dataset(per_cell=2)
    train = np.arange(0, len(dataset.labels) - 1, dtype=np.int64)
    first = fit_clinical_scaler(dataset, train)
    changed_original = dataset.clinical_original.copy()
    changed_mirrored = dataset.clinical_mirrored.copy()
    changed_original[-1] = 1e6
    changed_mirrored[-1] = -1e6
    changed = SharedClinicalDataset(**{
        **dataset.__dict__,
        "clinical_original": changed_original,
        "clinical_mirrored": changed_mirrored,
    })
    second = fit_clinical_scaler(changed, train)
    c.true(np.array_equal(first.mean, second.mean))
    c.true(np.array_equal(first.scale, second.scale))
    c.true(not dataset.clinical_original.flags.writeable)


def test_cross_fitted_smoke_returns_one_probability_and_metrics_per_source(c):
    dataset = _dataset(per_cell=2)
    result = evaluate_shared_model(
        dataset,
        use_dense=False,
        epochs=2,
        n_splits=2,
        seed=0,
        device="cpu",
    )
    c.eq(result.probabilities.shape, dataset.labels.shape)
    c.true(np.isfinite(result.probabilities).all())
    c.true(np.all((result.probabilities >= 0.0) & (result.probabilities <= 1.0)))
    c.eq(set(result.metrics), set(SOURCES))
    c.eq(result.model_fits, 2)
    c.eq(result.threshold, 0.5)
    c.eq(result.shared_gradient_sources, SOURCES)


def test_duplicate_groups_and_unknown_sources_fail_closed(c):
    dataset = _dataset(per_cell=2)
    duplicate = list(dataset.group_ids)
    duplicate[-1] = duplicate[0]
    c.raises(
        lambda: SharedClinicalDataset(**{
            **dataset.__dict__, "group_ids": tuple(duplicate)
        }),
        ValueError,
    )
    sources = list(dataset.sources)
    sources[-1] = "mayo"
    c.raises(
        lambda: SharedClinicalDataset(**{
            **dataset.__dict__, "sources": tuple(sources)
        }),
        ValueError,
    )


if __name__ == "__main__":
    run_all("test_shared_clinical_encoder_evaluation_v1", dict(globals()))
