from __future__ import annotations

import inspect
import json
import tempfile
from pathlib import Path

from _testlib import run_all

from scripts import run_bilateral_reconstruction_shared_search_v10 as runner


def _source_metrics(
    *, accuracy=0.92, specificity=0.85, sensitivity=0.90, auroc=0.94,
):
    return {
        source: {
            "accuracy": accuracy,
            "specificity": specificity,
            "sensitivity": sensitivity,
            "auroc": auroc,
            "balanced_accuracy": 0.5 * (specificity + sensitivity),
            "brier": 0.10,
        }
        for source in runner.SOURCES
    }


def _evaluations():
    return {
        candidate.candidate_id: {
            str(seed): {
                "within_source": _source_metrics(
                    specificity=0.80 if candidate.candidate_id == "BRV10-000" else 0.85
                ),
                "leave_one_source_out": _source_metrics(
                    accuracy=0.70, specificity=0.65, sensitivity=0.75, auroc=0.72
                ),
                "model_fits": 9,
                "task_specific_parameter_fraction": 0.05,
                "active_candidate_id": candidate.candidate_id,
            }
            for seed in runner.SEEDS
        }
        for candidate in runner.candidate_registry_v10()
    }


def test_report_requires_six_candidates_three_seeds_and_no_patient_rows(c):
    evaluations = _evaluations()
    ensemble = {
        candidate.candidate_id: _source_metrics(
            specificity=0.80 if candidate.candidate_id == "BRV10-000" else 0.85
        )
        for candidate in runner.candidate_registry_v10()
    }
    report = runner.build_report(
        evaluations=evaluations,
        ensemble_metrics=ensemble,
        counts={"palsynet": 38, "neuroface": 36, "meei": 56},
        runtime={"gpu": "NVIDIA H200", "epochs": 20, "folds": 6},
        commitments={"implementation_sha256": "a" * 64},
    )
    c.eq(report["schema_version"], "bilateral_reconstruction_shared_v10_search")
    c.eq(report["selection"]["research_baseline"], "BLV9-009/BRV10-000")
    c.eq(report["selection"]["seeds"], [0, 1, 2])
    c.eq(len(report["candidate_registry"]), 6)
    c.eq(set(report["evaluations"]), set(evaluations))
    c.eq(report["audit"], {
        "palsynet_protected_reads": 0,
        "mayo_reads": 0,
        "mayo_predictions": 0,
    })
    emitted = json.dumps(report, sort_keys=True).lower()
    for forbidden in (
        "probabilities", "group_id", "participant_id", "patient_id",
        "recording_id", "/users/", "/home/",
    ):
        c.true(forbidden not in emitted)


def test_atomic_release_is_fresh_and_exact(c):
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "release-v1"
        payload = {"schema_version": "test", "value": 10}
        digest = runner.write_release_atomic(output, payload)
        c.eq(json.loads((output / "report.json").read_text()), payload)
        c.eq(len(digest), 64)
        c.raises(lambda: runner.write_release_atomic(output, payload), FileExistsError)


def test_cli_has_no_candidate_tuning_seed_or_mayo_knobs(c):
    source = inspect.getsource(runner._parser).lower()
    for required in (
        "palsynet-cache-root", "neuroface-cache", "meei-cache", "output",
    ):
        c.true(required in source)
    for forbidden in (
        "candidate-ids", "learning-rate", "width", "threshold", "seed",
        "dropout", "mayo", "outer-test", "protected-test",
    ):
        c.true(forbidden not in source)


if __name__ == "__main__":
    run_all("test_run_bilateral_reconstruction_shared_search_v10", dict(globals()))
