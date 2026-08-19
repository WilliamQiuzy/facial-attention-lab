"""Development-only mirror-invariant Landmark 110D experiment."""
from __future__ import annotations

import copy
import json
import stat
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_mirror_invariant_110d import (  # noqa: E402
    BASELINE,
    BOOTSTRAP_REPEATS,
    CANDIDATE,
    CANDIDATE_REGISTRY,
    FIXED_C,
    INNER_FOLDS,
    MIRROR_TOLERANCE,
    OUTER_FOLD_NUMBER,
    _evaluate_gates,
    _parser,
    _write_private_report,
    build_development_matrices,
    mirror_dynamic_features,
    run_development_screen,
    run_fixed_inner_oof,
)
from scripts.run_dynamic_landmark_classical import ClassicalDataset  # noqa: E402
from _testlib import Check, run_all  # noqa: E402


def _synthetic_dataset() -> ClassicalDataset:
    count = 20
    features = np.zeros((count, 4, 32, 95), dtype=np.float32)
    masks = np.ones((count, 4, 32), dtype=bool)
    timestamps = np.stack([
        np.stack([
            window * 10.0 + np.arange(32, dtype=np.float64) * 0.1
            for window in range(4)
        ])
        for _ in range(count)
    ])
    source_indices = np.stack([
        np.stack([
            window * 100 + np.arange(32, dtype=np.int64)
            for window in range(4)
        ])
        for _ in range(count)
    ])
    labels = np.asarray([index % 2 for index in range(count)], dtype=np.int64)
    ramp = np.arange(32, dtype=np.float32)
    for index, label in enumerate(labels):
        features[index, 0, 0, 0] = float(index)
        first = 1.0 if label else 0.1
        second = 0.2 if label else 0.1
        features[index, :, :, 72] = first * ramp
        features[index, :, :, 73] = second * ramp
        features[index, :, :, 74] = np.abs(
            features[index, :, :, 72] - features[index, :, :, 73]
        )
        features[index, :, :, 90] = first * ramp
        features[index, :, :, 91] = second * ramp
    return ClassicalDataset(
        features=features,
        valid_masks=masks,
        timestamps=timestamps,
        source_frame_indices=source_indices,
        nuisance=np.zeros((count, 9), dtype=np.float64),
        labels=labels,
        group_ids=np.asarray([
            f"grp_{index:064x}" for index in range(count)
        ]),
        recording_ids=tuple(
            f"rec_{index:064x}" for index in range(count)
        ),
    )


def test_protocol_is_closed_and_has_no_tuning_surface(c: Check):
    c.eq(CANDIDATE_REGISTRY, {BASELINE: 110, CANDIDATE: 110},
         "only the fixed baseline and mirror-invariant candidate are present")
    c.eq(FIXED_C, 0.01, "regularization stays fixed")
    c.eq(OUTER_FOLD_NUMBER, 0, "the protected direction fold is frozen")
    c.eq(INNER_FOLDS, 4, "development OOF uses four grouped folds")
    c.eq(BOOTSTRAP_REPEATS, 5000, "paired bootstrap budget is frozen")
    c.eq(MIRROR_TOLERANCE, 1e-12, "mirror invariance tolerance is frozen")
    destinations = {
        action.dest for action in _parser()._actions if action.dest != "help"
    }
    c.eq(destinations, {"palsynet_cache_root"},
         "CLI exposes only the governed cache location")


def test_raw_mirror_is_an_exact_involution(c: Check):
    rng = np.random.default_rng(20260805)
    features = rng.normal(size=(4, 32, 95)).astype(np.float32)
    mirrored = mirror_dynamic_features(features)
    restored = mirror_dynamic_features(mirrored)
    c.true(np.array_equal(restored, features),
           "applying the frozen mirror twice restores the exact trajectory")


def test_matrix_extraction_is_restricted_to_development_rows(c: Check):
    dataset = _synthetic_dataset()
    seen: list[int] = []

    def observed_mirror(features: np.ndarray) -> np.ndarray:
        seen.append(int(features[0, 0, 0]))
        return mirror_dynamic_features(features)

    prepared = build_development_matrices(
        dataset, mirror_transform=observed_mirror
    )
    development = set(prepared.development_indices.tolist())
    protected = set(prepared.protected_indices.tolist())
    expected_seen = {
        int(dataset.features[index, 0, 0, 0]) for index in development
    }
    c.eq(set(seen), expected_seen, "mirror transform sees development rows only")
    c.eq(len(seen), 2 * len(development),
         "each development row is mirrored and then restored")
    c.true(development.isdisjoint(protected), "outer rows remain protected")
    c.eq(prepared.original.shape, (len(development), 110),
         "original matrix is development-local 110D")
    c.eq(prepared.mirrored.shape, prepared.original.shape,
         "mirrored matrix preserves the feature contract")
    c.true(np.array_equal(prepared.remirrored, prepared.original),
           "the complete mirror path restores each original 110D row")

    def non_involutive(features: np.ndarray) -> np.ndarray:
        return features + np.asarray(1.0, dtype=features.dtype)

    c.raises(lambda: build_development_matrices(
        dataset, mirror_transform=non_involutive
    ), ValueError, "a non-involutive mirror implementation fails closed")


def test_inner_oof_is_mirror_invariant_and_never_touches_outer(c: Check):
    dataset = _synthetic_dataset()
    prepared = build_development_matrices(dataset)
    result = run_fixed_inner_oof(dataset, prepared, CANDIDATE)
    c.eq(result.probabilities.shape, (prepared.development_indices.size,),
         "OOF probabilities contain development rows only")
    c.true(np.isfinite(result.probabilities).all(),
           "every development row receives one prediction")
    c.true(result.max_mirror_probability_error <= MIRROR_TOLERANCE,
           "test-time ensembling is exactly mirror invariant")
    protected = set(prepared.protected_indices.tolist())
    for event in result.audit_events:
        c.true(set(event.indices).isdisjoint(protected),
               "no fit or prediction event touches protected rows")
    predicted = [
        index for event in result.audit_events if event.operation == "predict"
        for index in event.indices
    ]
    c.eq(sorted(predicted), sorted(prepared.development_indices.tolist()),
         "inner validations cover development exactly once")
    reordered = replace(
        prepared,
        mirrored=np.roll(prepared.mirrored, shift=1, axis=0),
    )
    c.raises(
        lambda: run_fixed_inner_oof(dataset, reordered, CANDIDATE),
        ValueError,
        "reordered mirror rows fail before fitting or prediction",
    )


def test_report_is_aggregate_and_gates_are_fail_closed(c: Check):
    report = run_development_screen(_synthetic_dataset()).report
    c.eq(tuple(report["metrics"]), (BASELINE, CANDIDATE),
         "report preserves the preregistered comparison order")
    c.eq(report["audit"]["protected_candidate_feature_extractions"], 0,
         "protected candidate-feature use is explicit")
    c.eq(report["audit"]["protected_fits"], 0,
         "protected fit use is explicit")
    c.eq(report["audit"]["protected_predictions"], 0,
         "protected prediction use is explicit")
    serialized = json.dumps(report, allow_nan=False)
    c.true("grp_" not in serialized and "rec_" not in serialized,
           "aggregate report contains no opaque row identifiers")
    c.true("/Users/" not in serialized and "probabilities" not in serialized,
           "aggregate report contains no paths or per-record predictions")

    baseline = {"auroc": 0.9, "balanced_accuracy": 0.8, "brier": 0.2}
    candidate = {"auroc": 0.9, "balanced_accuracy": 0.8, "brier": 0.2}
    audit = {
        "protected_candidate_feature_extractions": 0,
        "protected_fits": 0,
        "protected_predictions": 0,
    }
    boundary = _evaluate_gates(
        baseline, candidate, MIRROR_TOLERANCE, audit
    )
    c.true(all(boundary.values()), "exact gate boundaries pass")
    contaminated = dict(audit, protected_predictions=1)
    failed = _evaluate_gates(
        baseline, candidate, MIRROR_TOLERANCE + 1e-15, contaminated
    )
    c.true(not failed["mirror_probability_error_at_most_1e_12"],
           "mirror error above tolerance fails")
    c.true(not failed[
        "zero_protected_candidate_feature_extractions_fits_predictions"
    ], "protected use fails")


def test_private_report_writer_is_atomic_and_rejects_identifiers(c: Check):
    report = run_development_screen(_synthetic_dataset()).report
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "private" / "report.json"
        _write_private_report(path, report)
        c.eq(stat.S_IMODE(path.stat().st_mode), 0o600,
             "report is owner-readable and owner-writable only")
        c.eq(json.loads(path.read_text())["schema_version"],
             "mirror_invariant_110d_v1_report", "complete JSON is published")
        c.raises(lambda: _write_private_report(path, report),
                 FileExistsError, "report is never overwritten")
        contaminated = copy.deepcopy(report)
        contaminated["dataset"]["group_ids"] = ["grp_" + "0" * 64]
        c.raises(lambda: _write_private_report(
            Path(directory) / "private" / "contaminated.json", contaminated
        ), ValueError, "writer rejects row identifiers")
        row_level = copy.deepcopy(report)
        row_level["metrics"][BASELINE]["auroc"] = [0.1] * 39
        c.raises(lambda: _write_private_report(
            Path(directory) / "private" / "row-level.json", row_level
        ), ValueError, "writer rejects numeric vectors in aggregate fields")
        unexpected_array = copy.deepcopy(report)
        unexpected_array["dataset"]["name"] = [0.1] * 39
        c.raises(lambda: _write_private_report(
            Path(directory) / "private" / "unexpected-array.json",
            unexpected_array,
        ), ValueError, "writer rejects arrays outside fixed aggregate fields")
        upgraded_claim = copy.deepcopy(report)
        upgraded_claim["dataset"]["claim_unit"] = "person_held_out"
        upgraded_claim["dataset"]["identity_status"] = "reviewed"
        c.raises(lambda: _write_private_report(
            Path(directory) / "private" / "upgraded-claim.json",
            upgraded_claim,
        ), ValueError, "writer rejects an unsupported validation upgrade")
        protected = copy.deepcopy(report)
        protected["audit"]["protected_predictions"] = 1
        c.raises(lambda: _write_private_report(
            Path(directory) / "private" / "protected.json", protected
        ), ValueError, "writer rejects protected-row use")


if __name__ == "__main__":
    run_all("test_mirror_invariant_110d", dict(globals()))
