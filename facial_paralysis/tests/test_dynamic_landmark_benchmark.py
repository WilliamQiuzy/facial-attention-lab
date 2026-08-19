"""Leakage and budget tests for the dynamic neural benchmark infrastructure."""
from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _testlib import Check, run_all  # noqa: E402
from src.evaluation.nested_group_cv import InnerGroupFold, NestedGroupFold  # noqa: E402
from src.models.dynamic_landmark import (  # noqa: E402
    ARM_BLENDSHAPE,
    ARM_FUSION,
    ARM_LANDMARK,
)
from src.training.dynamic_landmark_benchmark import (  # noqa: E402
    HISTORICAL_POOLED_AUC_REFERENCE,
    RANDOM_INIT_SEEDS,
    BenchmarkConfig,
    OuterEvaluationLockedError,
    build_candidate_registry,
    ensemble_seed_probabilities,
    fit_fold_standardizer,
    refit_outer_train,
    require_frozen_outer_registry,
    select_inner_epoch,
)


def _data(n: int = 10):
    generator = torch.Generator().manual_seed(904)
    features = torch.randn(n, 4, 32, 95, generator=generator)
    mask = torch.ones(n, 4, 32, dtype=torch.bool)
    frame_indices = torch.arange(128, dtype=torch.int64).reshape(4, 32)
    frame_indices = frame_indices.unsqueeze(0).repeat(n, 1, 1)
    timestamps = frame_indices.to(torch.float32) / 30.0
    labels = torch.tensor([index % 2 for index in range(n)], dtype=torch.float32)
    return features, mask, timestamps, frame_indices, labels


def _fold() -> NestedGroupFold:
    outer_train = np.arange(8, dtype=np.int64)
    outer_test = np.arange(8, 10, dtype=np.int64)
    inner = []
    for start in range(0, 8, 2):
        validation = np.arange(start, start + 2, dtype=np.int64)
        train = np.setdiff1d(outer_train, validation)
        inner.append(InnerGroupFold(train, validation))
    return NestedGroupFold(outer_train, outer_test, tuple(inner))


def test_registry_freezes_equal_budgets_seeds_and_historical_caveat(c: Check):
    c.raises(lambda: BenchmarkConfig(max_epochs=1.5), ValueError,
             "fixed epoch budget must be an integer")
    config = BenchmarkConfig(max_epochs=7, learning_rate=1e-3, mirror_probability=0.5)
    registry = build_candidate_registry(config)
    c.eq(tuple(registry), (ARM_BLENDSHAPE, ARM_LANDMARK, ARM_FUSION))
    c.eq(RANDOM_INIT_SEEDS, (0, 1, 2), "exact registered seed ensemble")
    c.true(all(spec.seeds == RANDOM_INIT_SEEDS and spec.max_epochs == 7
               for spec in registry.values()),
           "all candidates share seeds and fixed inner budget")
    c.eq({spec.arm for spec in registry.values()}, set(registry),
         "registry contains only the three random-init arms")
    c.eq(HISTORICAL_POOLED_AUC_REFERENCE, 0.860)
    c.true(all("reference" in spec.historical_context.lower()
               and "baseline" not in spec.historical_context.lower()
               for spec in registry.values()),
           "historical pooled AUC is never described as comparison baseline")


def test_fold_standardizer_fits_only_explicit_nonouter_valid_rows(c: Check):
    features, mask, _timestamps, _indices, _labels = _data(n=4)
    features.zero_()
    features[0] = 2.0
    features[1] = 4.0
    features[2] = float("nan")  # outer test must never be inspected
    mask[1, 0, 0] = False
    features[1, 0, 0] = 999.0
    scaler = fit_fold_standardizer(
        features, mask, fit_indices=np.array([0, 1]),
        outer_test_indices=np.array([2, 3]),
    )
    expected = (2.0 * 128 + 4.0 * 127) / 255.0
    c.true(bool(torch.allclose(scaler.mean, torch.full((95,), expected))),
           "mean uses only valid rows from fit indices")
    c.eq(scaler.fit_indices, (0, 1), "fit provenance is explicit")
    c.raises(lambda: fit_fold_standardizer(
        features, mask, fit_indices=np.array([0, 2]),
        outer_test_indices=np.array([2, 3])), ValueError,
        "outer test cannot enter scaler state")


def test_inner_fixed_budget_median_selection_and_outer_refit_have_no_outer_validation(c: Check):
    features, mask, timestamps, source_indices, labels = _data()
    # Outer-test corruption must be irrelevant before the one-shot Task 7 run.
    features[8:] = float("nan")
    config = BenchmarkConfig(
        max_epochs=2, learning_rate=2e-3, weight_decay=0.0,
        mirror_probability=0.0,
    )
    selection = select_inner_epoch(
        features, mask, timestamps, source_indices, labels,
        fold=_fold(), arm=ARM_FUSION, seed=0, config=config,
    )
    c.eq(len(selection.traces), 4, "all four inner folds are used")
    c.true(all(trace.epochs_ran == 2 and len(trace.validation_losses) == 2
               for trace in selection.traces),
           "every inner model runs the complete fixed epoch budget")
    c.true(1 <= selection.selected_epoch <= 2,
           "integer median selected epoch stays within fixed budget")

    artifact = refit_outer_train(
        features, mask, timestamps, source_indices, labels,
        outer_train_indices=_fold().train_indices,
        outer_test_indices=_fold().test_indices,
        arm=ARM_FUSION,
        seed=0,
        selected_epoch=selection.selected_epoch,
        config=config,
    )
    c.eq(artifact.epochs_trained, selection.selected_epoch,
         "fresh outer-train model runs exactly the median inner epoch")
    c.eq(artifact.scaler.fit_indices, tuple(range(8)),
         "outer refit scaler uses complete outer train only")
    c.true("validation" not in inspect.signature(refit_outer_train).parameters,
           "outer refit API has no validation input")
    for invalid_epoch in (True, 1.5):
        c.raises(lambda invalid_epoch=invalid_epoch: refit_outer_train(
            features, mask, timestamps, source_indices, labels,
            outer_train_indices=_fold().train_indices,
            outer_test_indices=_fold().test_indices,
            arm=ARM_FUSION,
            seed=0,
            selected_epoch=invalid_epoch,
            config=config,
        ), ValueError, "selected epoch must be a non-bool integer")


def test_nested_fold_contract_rejects_repeated_validation_or_incomplete_outer_partition(c: Check):
    features, mask, timestamps, source_indices, labels = _data()
    config = BenchmarkConfig(max_epochs=1, mirror_probability=0.0)
    valid = _fold()
    repeated = NestedGroupFold(
        valid.train_indices,
        valid.test_indices,
        (valid.inner_folds[0],) * 4,
    )
    c.raises(lambda: select_inner_epoch(
        features, mask, timestamps, source_indices, labels,
        fold=repeated, arm=ARM_FUSION, seed=0, config=config,
    ), ValueError, "four repeated validation splits cannot masquerade as nested CV")

    missing_row = NestedGroupFold(
        valid.train_indices,
        np.asarray([8], dtype=np.int64),
        valid.inner_folds,
    )
    c.raises(lambda: select_inner_epoch(
        features, mask, timestamps, source_indices, labels,
        fold=missing_row, arm=ARM_FUSION, seed=0, config=config,
    ), ValueError, "outer train and test must cover every dataset row exactly once")


def test_outer_refit_moves_fresh_model_to_feature_device(c: Check):
    if not torch.backends.mps.is_available():
        c.true(True, "device test is conditional when no accelerator is available")
        return
    features, mask, timestamps, source_indices, labels = _data(n=4)
    device = torch.device("mps")
    artifact = refit_outer_train(
        features.to(device), mask.to(device), timestamps.to(device),
        source_indices.to(device), labels.to(device),
        outer_train_indices=np.asarray([0, 1], dtype=np.int64),
        outer_test_indices=np.asarray([2, 3], dtype=np.int64),
        arm=ARM_FUSION,
        seed=0,
        selected_epoch=1,
        config=BenchmarkConfig(max_epochs=1, mirror_probability=0.0),
    )
    c.eq(next(artifact.model.parameters()).device.type, "mps",
         "fresh refit model follows the feature device")


def test_probability_ensemble_averages_three_seed_probabilities_not_logits(c: Check):
    probabilities = np.asarray([
        [0.1, 0.9],
        [0.2, 0.8],
        [0.6, 0.3],
    ])
    got = ensemble_seed_probabilities(probabilities)
    c.true(bool(np.allclose(got, [0.3, 2.0 / 3.0])),
           "seed ensemble is arithmetic mean at probability level")
    c.raises(lambda: ensemble_seed_probabilities(probabilities[:2]), ValueError,
             "all three registered seeds are required")
    c.raises(lambda: ensemble_seed_probabilities(np.asarray([[2.0], [0.2], [0.3]])),
             ValueError, "inputs must already be probabilities")


def test_real_outer_runner_is_unconditionally_locked_until_task7(c: Check):
    c.raises(lambda: require_frozen_outer_registry("a" * 64),
             OuterEvaluationLockedError,
             "a plausible hash cannot bypass the absent Task 7 protocol")
    script = ROOT / "scripts" / "run_dynamic_landmark_benchmark.py"
    spec = importlib.util.spec_from_file_location("locked_dynamic_runner", script)
    if spec is None or spec.loader is None:
        raise AssertionError("runner cannot be imported")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    c.raises(lambda: module.main([]), OuterEvaluationLockedError,
             "real runner remains fail-closed")


if __name__ == "__main__":
    run_all("test_dynamic_landmark_benchmark", dict(globals()))
