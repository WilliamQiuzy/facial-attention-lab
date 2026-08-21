from __future__ import annotations

import inspect

from _testlib import run_all

from scripts import run_literature_grounded_shared_search_v9 as runner


def _source_metrics(
    *, accuracy: float = 0.92, auroc: float = 0.94,
    sensitivity: float = 0.90, specificity: float = 0.85,
):
    return {
        source: {
            "accuracy": accuracy,
            "auroc": auroc,
            "balanced_accuracy": 0.5 * (sensitivity + specificity),
            "sensitivity": sensitivity,
            "specificity": specificity,
            "brier": 0.10,
        }
        for source in runner.SOURCES
    }


def _evaluation(metrics=None):
    observed = _source_metrics() if metrics is None else metrics
    return {
        str(seed): {
            "within_source": observed,
            "leave_one_source_out": _source_metrics(accuracy=0.75, specificity=0.70),
            "model_fits": 9,
            "task_specific_parameter_fraction": 0.05,
        }
        for seed in runner.SEEDS
    }


def test_combination_requires_both_single_mechanisms_on_every_seed(c):
    comparator = _evaluation()
    good = _evaluation(_source_metrics(specificity=0.87))
    bad = _evaluation(_source_metrics(accuracy=0.88, auroc=0.90))
    c.true(runner.combination_is_authorized({
        "LGS9-000": comparator, "LGS9-001": good, "LGS9-002": good,
    }))
    c.true(not runner.combination_is_authorized({
        "LGS9-000": comparator, "LGS9-001": good, "LGS9-002": bad,
    }))


def test_report_is_multiseed_aggregate_only_and_specificity_first(c):
    evaluations = {
        "LGS9-000": _evaluation(),
        "LGS9-001": _evaluation(_source_metrics(specificity=0.88)),
        "LGS9-002": _evaluation(_source_metrics(specificity=0.86)),
        "LGS9-003": _evaluation(_source_metrics(specificity=0.89)),
    }
    report = runner.build_report(
        evaluations=evaluations,
        counts={"palsynet": 38, "neuroface": 36, "meei": 56},
        runtime={"gpu": "NVIDIA H200", "epochs": 20, "folds": 6},
        commitments={"implementation_sha256": "a" * 64},
    )
    c.eq(report["schema_version"], "literature_grounded_shared_v9_search")
    c.eq(report["selection"]["primary_metric"], "minimum_source_specificity")
    c.eq(report["selection"]["seeds"], [0, 1, 2])
    c.true(report["combination_authorized"])
    c.eq(report["audit"], {
        "palsynet_protected_reads": 0,
        "mayo_reads": 0,
        "mayo_predictions": 0,
    })
    emitted = str(report).lower()
    c.true("group_id" not in emitted and "probabilities" not in emitted)
    c.true("/users/" not in emitted and "private" not in emitted)


def test_cli_has_only_authenticated_development_data_and_no_search_knobs(c):
    source = inspect.getsource(runner._parser).lower()
    for name in (
        "palsynet-cache-root", "neuroface-cache", "meei-cache", "output",
    ):
        c.true(name in source)
    for forbidden in (
        "mayo", "outer-test", "protected-test", "learning-rate", "dropout",
        "width", "threshold", "candidate-ids",
    ):
        c.true(forbidden not in source)


if __name__ == "__main__":
    run_all("test_run_literature_grounded_shared_search_v9", dict(globals()))
