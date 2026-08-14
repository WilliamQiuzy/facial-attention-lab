"""Contracts for label-free NeuroFace manual68 geometry calibration."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from _testlib import Check, run_all  # noqa: E402
from src.training.neuroface_geometry_calibration_v1 import (  # noqa: E402
    CalibratedTransferDataset,
    evaluate_calibrated_transfer,
    mirror_semantic23,
)


def test_semantic_mirror_is_an_exact_involution(c: Check):
    rng = np.random.default_rng(813)
    values = rng.normal(size=(10, 23))
    c.true(bool(np.array_equal(mirror_semantic23(mirror_semantic23(values)), values)))
    c.eq(float(mirror_semantic23(values)[0, 3]), -float(values[0, 3]))


def test_calibrated_comparison_is_development_only(c: Check):
    rng = np.random.default_rng(8)
    n = 18
    labels = np.asarray([index % 2 for index in range(n)])
    baseline = rng.normal(size=(n, 110))
    baseline[labels == 1, :4] += 2
    calibrated = baseline.copy()
    development = np.arange(16)
    protected = np.arange(16, 18)
    baseline[protected] = np.nan
    calibrated[protected] = np.nan
    folds = np.full(n, -1)
    folds[:16] = np.repeat(np.arange(4), 4)
    result = evaluate_calibrated_transfer(CalibratedTransferDataset(
        baseline, baseline.copy(), calibrated, calibrated.copy(), labels,
        np.asarray([f"g{i}" for i in range(n)], dtype=object),
        development, protected, folds,
    ))
    c.eq(tuple(result["metrics"]), ("landmark_110d", "manual68_calibrated_110d"))
    c.eq(result["protected_predictions"], 0)
    c.eq(result["development_groups"], 16)


if __name__ == "__main__":
    run_all("test_neuroface_geometry_calibration_v1", dict(globals()))
