from __future__ import annotations

import inspect

from _testlib import run_all

from scripts import run_script_phenotype_search_v9 as runner
from src.models.script_phenotype_router_v9 import candidate_registry_v9


def _metrics():
    return {
        source: {
            "accuracy": 0.93, "auroc": 0.94, "balanced_accuracy": 0.93,
            "sensitivity": 0.93, "specificity": 0.93, "brier": 0.10,
        }
        for source in runner.SOURCES
    }


def test_report_is_aggregate_specificity_first_and_shared(c):
    registry = candidate_registry_v9()
    evaluations = {
        row.candidate_id: {
            "metrics": _metrics(), "model_fits": 6,
            "task_specific_parameter_fraction": 0.05,
        }
        for row in registry
    }
    report = runner.build_report(
        evaluations=evaluations,
        ranking=tuple(row.candidate_id for row in registry),
        counts={"palsynet": 38, "neuroface": 36, "meei": 56},
        runtime={"gpu": "NVIDIA H200", "deterministic": True},
        commitments={"implementation_sha256": "a" * 64},
    )
    c.eq(report["schema_version"], "script_phenotype_router_v9_search")
    c.eq(report["selection"]["primary_metric"], "minimum_source_specificity")
    c.true(report["model"]["full_478d_plus_110d_shared_encoder"])
    c.eq(report["audit"]["mayo_reads"], 0)
    c.true("probabilities" not in str(report).lower())


def test_cli_has_no_mayo_or_protected_test_surface(c):
    source = inspect.getsource(runner._parser).lower()
    c.true("palsynet-cache-root" in source and "neuroface-cache" in source)
    c.true("meei-cache" in source and "output" in source)
    c.true("mayo" not in source and "outer-test" not in source)


if __name__ == "__main__":
    run_all("test_run_script_phenotype_search_v9", dict(globals()))
