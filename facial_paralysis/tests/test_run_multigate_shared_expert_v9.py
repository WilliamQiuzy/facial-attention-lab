from __future__ import annotations

import inspect

from _testlib import run_all

from scripts import run_multigate_shared_expert_v9 as runner


def _metrics(specificity=0.86):
    return {
        source: {
            "accuracy": 0.93,
            "auroc": 0.95,
            "balanced_accuracy": 0.5 * (0.92 + specificity),
            "sensitivity": 0.92,
            "specificity": specificity,
            "brier": 0.08,
        }
        for source in runner.SOURCES
    }


def _evaluations():
    return {
        candidate_id: {
            str(seed): {
                "within_source": _metrics(),
                "leave_one_source_out": _metrics(0.75),
                "model_fits": 9,
                "task_specific_parameter_fraction": 0.05,
            }
            for seed in runner.SEEDS
        }
        for candidate_id in ("MSE9-000", "MSE9-001")
    }


def test_report_is_exact_multiseed_shared_expert_evidence(c):
    report = runner.build_report(
        evaluations=_evaluations(),
        ensemble_metrics={"MSE9-000": _metrics(0.86), "MSE9-001": _metrics(0.90)},
        counts={"palsynet": 38, "neuroface": 36, "meei": 56},
        runtime={"gpu": "NVIDIA H200", "epochs": 20, "folds": 6},
        commitments={"implementation_sha256": "a" * 64},
    )
    c.eq(report["schema_version"], "multigate_shared_expert_v9_search")
    c.eq(report["selection"]["primary_metric"], "minimum_source_specificity")
    c.eq(report["selection"]["seeds"], [0, 1, 2])
    c.true(report["decision"]["promotion_authorized"])
    c.eq(report["audit"], {
        "palsynet_protected_reads": 0, "mayo_reads": 0, "mayo_predictions": 0,
    })
    emitted = str(report).lower()
    c.true("probabilities" not in emitted and "group_id" not in emitted)
    c.true("/users/" not in emitted and "private" not in emitted)


def test_cli_has_no_arbitrary_search_or_clinical_data_knobs(c):
    source = inspect.getsource(runner._parser).lower()
    for required in ("palsynet-cache-root", "neuroface-cache", "meei-cache", "output"):
        c.true(required in source)
    for forbidden in (
        "mayo", "candidate-ids", "expert-count", "expert-rank", "learning-rate",
        "dropout", "threshold", "outer-test", "protected-test",
    ):
        c.true(forbidden not in source)


if __name__ == "__main__":
    run_all("test_run_multigate_shared_expert_v9", dict(globals()))
