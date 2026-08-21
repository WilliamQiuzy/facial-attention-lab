from __future__ import annotations

import inspect

from _testlib import run_all

from scripts import run_distilled_shared_search_v9 as runner
from src.models.distilled_shared_candidate_registry_v9 import candidate_registry_v9


def _metrics(value: float = 0.93):
    return {
        source: {
            "accuracy": value,
            "auroc": value,
            "balanced_accuracy": value,
            "sensitivity": value,
            "specificity": value,
            "brier": 1.0 - value,
        }
        for source in runner.SOURCES
    }


def test_report_recomputes_specificity_first_ranking_and_stays_aggregate(c):
    registry = candidate_registry_v9()
    evaluations = {
        row.candidate_id: {
            "metrics": _metrics(0.93),
            "model_fits": 6,
            "teacher_self_training_rows": 0,
            "outer_held_teacher_reads": 0,
            "task_specific_parameter_fraction": 0.08,
        }
        for row in registry
    }
    ranking = tuple(row.candidate_id for row in registry)
    report = runner.build_report(
        evaluations=evaluations,
        ranking=ranking,
        counts={"palsynet": 38, "neuroface": 36, "meei": 56},
        runtime={"gpu": "NVIDIA H200", "deterministic": True},
        commitments={"implementation_sha256": "a" * 64},
    )
    c.eq(report["schema_version"], "distilled_shared_router_v9_search")
    c.eq(report["selection"]["primary_metric"], "minimum_source_specificity")
    c.eq(report["audit"]["mayo_reads"], 0)
    c.eq(report["audit"]["palsynet_protected_reads"], 0)
    emitted = str(report).lower()
    c.true("probabilities" not in emitted and "group_id" not in emitted)


def test_cli_has_no_mayo_protected_or_external_teacher_artifact_surface(c):
    source = inspect.getsource(runner._parser).lower()
    for name in (
        "palsynet-cache-root", "neuroface-cache", "meei-cache", "output",
    ):
        c.true(name in source)
    c.true("mayo" not in source)
    c.true("teacher-profile" not in source and "private-oof" not in source)


if __name__ == "__main__":
    run_all("test_run_distilled_shared_search_v9", dict(globals()))
