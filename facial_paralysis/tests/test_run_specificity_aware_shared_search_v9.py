from __future__ import annotations

import inspect

from _testlib import run_all

from scripts import run_specificity_aware_shared_search_v9 as runner
from src.models.specificity_aware_candidate_registry_v9 import candidate_registry_v9


def _metrics(value: float = 0.91, specificity: float = 0.85):
    return {
        source: {
            "accuracy": value,
            "auroc": value + 0.02,
            "balanced_accuracy": (0.90 + specificity) / 2.0,
            "sensitivity": 0.90,
            "specificity": specificity,
            "brier": 1.0 - value,
        }
        for source in runner.SOURCES
    }


def _evaluations():
    return {
        row.candidate_id: {
            "fixed": _metrics(),
            "calibrated": _metrics(specificity=0.90),
            "thresholds": {source: [0.5] * 6 for source in runner.SOURCES},
        }
        for row in candidate_registry_v9()
    }


def _report():
    ids = tuple(row.candidate_id for row in candidate_registry_v9())
    return runner.build_report(
        phase="screen",
        seed=0,
        candidate_ids=ids,
        evaluations=_evaluations(),
        ranking=ids,
        counts={"palsynet": 38, "neuroface": 36, "meei": 56},
        runtime={"gpu": "NVIDIA H200", "epochs": 20, "folds": 6},
        commitments={"implementation_sha256": "a" * 64},
        screen_report_sha256=None,
    )


def test_phase_is_full_screen_then_exact_top_three_confirmation(c):
    ids = tuple(row.candidate_id for row in candidate_registry_v9())
    c.eq(runner.candidate_ids_for_phase("screen", 0, None), ids)
    screen = _report()
    c.eq(
        runner.candidate_ids_for_phase("confirm", 1, screen), ids[:3]
    )
    c.eq(
        runner.candidate_ids_for_phase("confirm", 2, screen), ids[:3]
    )
    c.raises(lambda: runner.candidate_ids_for_phase("screen", 1, None), ValueError)
    c.raises(lambda: runner.candidate_ids_for_phase("confirm", 0, screen), ValueError)
    corrupted = dict(screen)
    corrupted["ranking"] = list(reversed(ids))
    c.raises(
        lambda: runner.candidate_ids_for_phase("confirm", 1, corrupted),
        ValueError,
    )


def test_report_is_aggregate_specificity_first_and_boundary_closed(c):
    report = _report()
    c.eq(report["schema_version"], "specificity_aware_shared_router_v9_search")
    c.eq(report["selection"]["primary_metric"], "minimum_source_specificity")
    c.eq(report["selection"]["minimum_sensitivity"], 0.85)
    c.eq(report["promotion_gate"]["minimum_accuracy"], 0.90)
    c.eq(report["promotion_gate"]["minimum_specificity"], 0.80)
    c.eq(report["promotion_gate"]["minimum_auroc"], 0.92)
    c.eq(report["audit"], {
        "palsynet_protected_reads": 0,
        "mayo_reads": 0,
        "mayo_predictions": 0,
    })
    emitted = str(report).lower()
    c.true("group_id" not in emitted)
    c.true("probabilities" not in emitted)
    c.true("private" not in emitted and "/users/" not in emitted)


def test_cli_exposes_only_authenticated_development_sources(c):
    source = inspect.getsource(runner._parser).lower()
    for name in (
        "palsynet-cache-root", "neuroface-cache", "meei-cache", "phase",
        "screen-report", "output",
    ):
        c.true(name in source)
    c.true("mayo" not in source)
    c.true("outer-test" not in source and "protected-test" not in source)


if __name__ == "__main__":
    run_all("test_run_specificity_aware_shared_search_v9", dict(globals()))
