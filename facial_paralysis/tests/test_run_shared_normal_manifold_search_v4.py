from __future__ import annotations

import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _testlib import run_all

from scripts import run_shared_normal_manifold_search_v4 as runner
from src.models.normal_manifold_candidate_registry_v4 import candidate_registry_v4


def _metrics(value: float = 0.91):
    return {
        source: {
            "accuracy": value,
            "balanced_accuracy": value,
            "auroc": value + 0.02,
            "sensitivity": value,
            "specificity": value,
            "brier": 1.0 - value,
        }
        for source in runner.SOURCES
    }


def _cosines():
    return {
        "palsynet__neuroface": 0.1,
        "palsynet__meei": -0.2,
        "neuroface__meei": 0.3,
    }


def test_phase_contract_is_six_screen_and_two_confirm(c):
    ids = tuple(candidate.candidate_id for candidate in candidate_registry_v4())
    runner.validate_candidate_phase("screen", ids)
    runner.validate_candidate_phase("confirm", ids[:2])
    c.raises(lambda: runner.validate_candidate_phase("screen", ids[:-1]), ValueError)
    c.raises(lambda: runner.validate_candidate_phase("confirm", ids[:3]), ValueError)


def test_report_is_aggregate_gradient_bound_and_mayo_closed(c):
    ids = tuple(candidate.candidate_id for candidate in candidate_registry_v4())
    report = runner.build_report(
        phase="screen",
        evaluations={candidate_id: _metrics() for candidate_id in ids},
        gradient_cosines={candidate_id: _cosines() for candidate_id in ids},
        ranking=ids,
        counts={"palsynet": 38, "neuroface": 36, "meei": 56},
        runtime={"gpu": "NVIDIA H200", "epochs": 20, "seed": 0, "folds": 6},
        commitments={"implementation_sha256": "a" * 64},
    )
    c.eq(report["selection"]["primary_metric"], "minimum_source_balanced_accuracy")
    c.eq(len(report["candidate_registry"]), 6)
    c.true(report["model"]["shared_normal_anchor"])
    c.eq(report["audit"]["mayo_reads"], 0)
    c.eq(report["audit"]["palsynet_protected_reads"], 0)
    emitted = str(report).lower()
    c.true("group_id" not in emitted and "probabilities" not in emitted)


def test_cli_has_only_three_development_sources(c):
    source = inspect.getsource(runner._parser).lower()
    for name in (
        "palsynet-cache-root", "neuroface-cache", "meei-cache", "output",
        "candidate-ids",
    ):
        c.true(name in source)
    c.true("mayo" not in source)
    c.true("outer-test" not in source and "protected-test" not in source)


if __name__ == "__main__":
    run_all("test_run_shared_normal_manifold_search_v4", dict(globals()))
