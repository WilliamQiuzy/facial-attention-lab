"""Development-only nuisance challenge for clinical landmark dynamics v1."""
from __future__ import annotations

import json
import copy
import stat
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_action_clinical_geometry_v1 import (  # noqa: E402
    BOOTSTRAP_REPEATS,
    CANDIDATE_REGISTRY,
    FIXED_C,
    INNER_FOLDS,
    OUTER_FOLD_NUMBER,
    _parser,
    _evaluate_gates,
    _write_private_report,
    build_development_candidate_matrices,
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
        first_amplitude = 1.0 if label else 0.1
        second_amplitude = 0.2 if label else 0.1
        features[index, :, :, 86] = first_amplitude * ramp
        features[index, :, :, 87] = second_amplitude * ramp
        features[index, :, :, 88] = np.abs(
            features[index, :, :, 86] - features[index, :, :, 87]
        )
        features[index, :, :, 93] = first_amplitude * ramp
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
    c.eq(tuple(CANDIDATE_REGISTRY), (
        "nuisance", "landmark", "clinical_dynamics",
        "clinical_dynamics_plus_nuisance",
    ), "candidate order is frozen")
    c.eq(tuple(CANDIDATE_REGISTRY.values()), (9, 110, 58, 67),
         "candidate dimensions are frozen")
    c.eq(FIXED_C, 0.01, "regularization is fixed rather than selected")
    c.eq(OUTER_FOLD_NUMBER, 0, "only the frozen direction fold is used")
    c.eq(INNER_FOLDS, 4, "development OOF uses four grouped folds")
    c.eq(BOOTSTRAP_REPEATS, 5000, "paired interval has a fixed budget")
    destinations = {
        action.dest for action in _parser()._actions if action.dest != "help"
    }
    c.eq(destinations, {"palsynet_cache_root"},
         "CLI exposes data location only, not C, folds, candidates, or output")


def test_candidate_extraction_is_restricted_to_outer_train(c: Check):
    dataset = _synthetic_dataset()
    seen: list[int] = []

    def extractor(features, mask, timestamps, source_indices):
        seen.append(int(features[0, 0, 0]))
        return np.zeros(58, dtype=np.float64)

    prepared = build_development_candidate_matrices(
        dataset, clinical_extractor=extractor
    )
    development = set(prepared.development_indices.tolist())
    protected = set(prepared.protected_indices.tolist())
    c.eq(set(seen), development, "clinical extraction sees development rows only")
    c.eq(len(seen), len(development), "each development row is extracted once")
    c.true(development.isdisjoint(protected), "frozen outer rows stay protected")
    c.eq(set(prepared.extraction_indices), development,
         "the explicit extraction audit covers only development rows")
    c.eq(prepared.matrices["clinical_dynamics_plus_nuisance"].shape,
         (len(development), 67), "combined matrix is development-local")


def test_fixed_inner_oof_covers_development_once_and_never_protected(c: Check):
    dataset = _synthetic_dataset()
    prepared = build_development_candidate_matrices(dataset)
    result = run_fixed_inner_oof(dataset, prepared, "clinical_dynamics")
    c.eq(result.probabilities.shape, (prepared.development_indices.size,),
         "OOF array has no outer slots")
    c.true(np.isfinite(result.probabilities).all(),
           "every development row receives one prediction")
    c.eq(len(result.audit_events), INNER_FOLDS * 3,
         "each fold records scaler fit, model fit, and validation prediction")
    protected = set(prepared.protected_indices.tolist())
    for event in result.audit_events:
        c.true(set(event.indices).isdisjoint(protected),
               "no fit or prediction event touches protected indices")
    predicted = [
        index for event in result.audit_events if event.operation == "predict"
        for index in event.indices
    ]
    c.eq(sorted(predicted), sorted(prepared.development_indices.tolist()),
         "inner validations cover development exactly once")


def test_screen_reports_incremental_nuisance_challenge_without_identifiers(c: Check):
    screen = run_development_screen(_synthetic_dataset())
    report = screen.report
    c.eq(tuple(report["metrics"]), tuple(CANDIDATE_REGISTRY),
         "all candidates are reported in frozen order")
    comparison = report["primary_comparison"]
    c.eq(comparison["candidate"], "clinical_dynamics_plus_nuisance",
         "incremental model is the primary candidate")
    c.eq(comparison["baseline"], "nuisance",
         "recorded nuisance is the primary baseline")
    manual_delta = (
        report["metrics"]["clinical_dynamics_plus_nuisance"]["auroc"]
        - report["metrics"]["nuisance"]["auroc"]
    )
    c.true(np.isclose(comparison["delta_auroc"], manual_delta, atol=1e-12),
           "bootstrap direction agrees with an independent AUROC subtraction")
    c.eq(report["audit"]["protected_feature_extractions"], 0,
         "report makes protected extraction use explicit")
    c.eq(report["audit"]["protected_fits"], 0,
         "report makes protected fit use explicit")
    c.eq(report["audit"]["protected_predictions"], 0,
         "report makes protected prediction use explicit")
    serialized = json.dumps(report, allow_nan=False)
    c.true("grp_" not in serialized and "rec_" not in serialized,
           "aggregate report contains no opaque row identifiers")
    c.true("/Users/" not in serialized and "probabilities" not in serialized,
           "aggregate report contains no paths or per-record predictions")


def test_gate_boundaries_are_strict_and_derived_from_observed_audit(c: Check):
    metrics = {
        "clinical_dynamics": {"sensitivity": 0.70, "specificity": 0.70}
    }
    audit = {
        "protected_feature_extractions": 0,
        "protected_fits": 0,
        "protected_predictions": 0,
    }
    at_zero = _evaluate_gates(
        metrics, {"delta_auroc": 0.10, "ci95": [0.0, 0.2]}, audit
    )
    c.true(not at_zero["paired_ci95_lower_bound_above_zero"],
           "a zero interval bound fails rather than rounding up")
    above_zero = _evaluate_gates(
        metrics, {"delta_auroc": 0.10, "ci95": [1e-12, 0.2]}, audit
    )
    c.true(all(above_zero.values()), "exact pass boundaries are accepted")
    contaminated = dict(audit, protected_predictions=1)
    contaminated_gates = _evaluate_gates(
        metrics, {"delta_auroc": 0.10, "ci95": [1e-12, 0.2]}, contaminated
    )
    c.true(not contaminated_gates[
        "zero_protected_feature_extractions_fits_predictions"
    ], "protected-use gate is derived from observed audit counts")


def test_private_report_writer_is_atomic_mode_0600_and_no_overwrite(c: Check):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "private" / "report.json"
        report = run_development_screen(_synthetic_dataset()).report
        _write_private_report(path, report)
        c.eq(stat.S_IMODE(path.stat().st_mode), 0o600,
             "report is owner-readable and owner-writable only")
        c.eq(json.loads(path.read_text())["schema_version"],
             "action_clinical_geometry_v1_report",
             "writer commits complete valid JSON")
        c.raises(lambda: _write_private_report(path, report),
                 FileExistsError, "report is never overwritten")
        contaminated = copy.deepcopy(report)
        contaminated["dataset"]["group_ids"] = ["grp_" + "0" * 64]
        c.raises(lambda: _write_private_report(
            Path(directory) / "private" / "contaminated.json", contaminated
        ), ValueError, "writer rejects sensitive identifiers before serialization")
        relative_path = copy.deepcopy(report)
        relative_path["dataset"]["collection_manifest_sha256"] = "../private/data.json"
        c.raises(lambda: _write_private_report(
            Path(directory) / "private" / "relative.json", relative_path
        ), ValueError, "writer rejects relative paths and file-like values")
        probability = copy.deepcopy(report)
        probability["metrics"]["nuisance"]["per_record_probability"] = [0.1]
        c.raises(lambda: _write_private_report(
            Path(directory) / "private" / "probability.json", probability
        ), ValueError, "closed nested schema rejects probability-bearing variants")
        generic_identifier = copy.deepcopy(report)
        generic_identifier["dataset"]["patient_id"] = "P1"
        c.raises(lambda: _write_private_report(
            Path(directory) / "private" / "patient.json", generic_identifier
        ), ValueError, "closed nested schema rejects generic identifiers")
        protected_use = copy.deepcopy(report)
        protected_use["audit"]["protected_predictions"] = 1
        c.raises(lambda: _write_private_report(
            Path(directory) / "private" / "protected.json", protected_use
        ), ValueError, "writer refuses reports that used protected rows")
        c.true(not tuple(path.parent.glob(".*.tmp")),
               "temporary files are cleaned after publication")


if __name__ == "__main__":
    run_all("test_action_clinical_geometry_v1", dict(globals()))
