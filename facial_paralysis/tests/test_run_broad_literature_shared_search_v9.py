from __future__ import annotations

import inspect
import json
import tempfile
from pathlib import Path

from _testlib import run_all

from scripts import run_broad_literature_shared_search_v9 as runner


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
    result = {}
    for candidate in runner.candidate_registry_v9():
        result[candidate.candidate_id] = {
            str(seed): {
                "within_source": _source_metrics(
                    specificity=0.80 if candidate.candidate_id == "BLV9-000" else 0.85
                ),
                "leave_one_source_out": _source_metrics(
                    accuracy=0.70, specificity=0.65, sensitivity=0.75, auroc=0.72
                ),
                "model_fits": 9,
                "task_specific_parameter_fraction": 0.05,
                "active_mechanism": candidate.mechanism,
            }
            for seed in runner.SEEDS
        }
    return result


def test_report_requires_all_twenty_models_three_seeds_and_no_patient_rows(c):
    evaluations = _evaluations()
    ensemble = {
        candidate.candidate_id: _source_metrics(
            specificity=0.80 if candidate.candidate_id == "BLV9-000" else 0.85
        )
        for candidate in runner.candidate_registry_v9()
    }
    report = runner.build_report(
        evaluations=evaluations,
        ensemble_metrics=ensemble,
        counts={"palsynet": 38, "neuroface": 36, "meei": 56},
        runtime={"gpu": "NVIDIA H200", "epochs": 20, "folds": 6},
        commitments={"implementation_sha256": "a" * 64},
    )
    c.eq(report["schema_version"], "broad_literature_shared_v9_search")
    c.eq(report["selection"]["evaluated_new_models"], 20)
    c.eq(report["selection"]["seeds"], [0, 1, 2])
    c.eq(len(report["candidate_registry"]), 21)
    c.eq(set(report["evaluations"]), set(evaluations))
    c.eq(report["audit"], {
        "palsynet_protected_reads": 0,
        "mayo_reads": 0,
        "mayo_predictions": 0,
    })
    emitted = json.dumps(report, sort_keys=True).lower()
    for forbidden in ("probabilities", "group_id", "patient_id", "/users/", "private"):
        c.true(forbidden not in emitted)
    missing = dict(evaluations)
    missing.pop("BLV9-020")
    c.raises(lambda: runner.build_report(
        evaluations=missing, ensemble_metrics=ensemble,
        counts={"palsynet": 38, "neuroface": 36, "meei": 56},
        runtime={"gpu": "NVIDIA H200"},
        commitments={"implementation_sha256": "a" * 64},
    ), ValueError)


def test_atomic_release_is_fresh_and_exact(c):
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "release-v1"
        payload = {"schema_version": "test", "value": 3}
        digest = runner.write_release_atomic(output, payload)
        c.eq(json.loads((output / "report.json").read_text()), payload)
        c.eq(len(digest), 64)
        c.raises(lambda: runner.write_release_atomic(output, payload), FileExistsError)
        c.true(not any(path.name.startswith(".release-v1") for path in output.parent.iterdir()))


def test_cli_has_no_candidate_or_tuning_or_mayo_knobs(c):
    source = inspect.getsource(runner._parser).lower()
    for required in ("palsynet-cache-root", "neuroface-cache", "meei-cache", "output"):
        c.true(required in source)
    for forbidden in (
        "candidate-ids", "learning-rate", "width", "threshold", "seed",
        "dropout", "mayo", "outer-test", "protected-test",
    ):
        c.true(forbidden not in source)


if __name__ == "__main__":
    run_all("test_run_broad_literature_shared_search_v9", dict(globals()))
