from __future__ import annotations

import inspect

from _testlib import run_all

from scripts import run_shared_ensemble_search_v9 as runner
from src.evaluation.shared_ensemble_search_v9 import ensemble_candidate_registry_v9


def _metrics():
    return {
        source: {
            "accuracy": 0.91,
            "auroc": 0.93,
            "balanced_accuracy": 0.90,
            "sensitivity": 0.90,
            "specificity": 0.90,
            "brier": 0.10,
        }
        for source in runner.SOURCES
    }


def test_report_is_full_aggregate_shared_ensemble_search(c):
    registry = ensemble_candidate_registry_v9()
    ids = tuple(row.candidate_id for row in registry)
    evaluations = {
        row.candidate_id: {
            "metrics": _metrics(),
            "member_models": len(row.member_candidate_ids) * len(row.seeds),
            "model_fits": 6 * len(row.member_candidate_ids) * len(row.seeds),
        }
        for row in registry
    }
    report = runner.build_report(
        evaluations=evaluations,
        ranking=ids,
        counts={"palsynet": 38, "neuroface": 36, "meei": 56},
        runtime={"gpu": "NVIDIA H200", "epochs": 20, "folds": 6},
        commitments={"implementation_sha256": "a" * 64},
    )
    c.eq(report["schema_version"], "shared_deep_ensemble_v9_search")
    c.eq(len(report["candidate_registry"]), 16)
    c.eq(report["selection"]["primary_metric"], "minimum_source_specificity")
    c.eq(report["audit"]["mayo_reads"], 0)
    c.eq(report["audit"]["palsynet_protected_reads"], 0)
    emitted = str(report).lower()
    c.true("probabilities" not in emitted and "group_id" not in emitted)


def test_cli_has_no_mayo_or_protected_surface(c):
    source = inspect.getsource(runner._parser).lower()
    for name in (
        "palsynet-cache-root", "neuroface-cache", "meei-cache", "output",
    ):
        c.true(name in source)
    c.true("mayo" not in source)
    c.true("outer-test" not in source and "protected-test" not in source)


if __name__ == "__main__":
    run_all("test_run_shared_ensemble_search_v9", dict(globals()))
