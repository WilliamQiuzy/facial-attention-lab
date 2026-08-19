from __future__ import annotations

import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _testlib import run_all

from scripts import run_conflict_aware_shared_search_v5 as runner
from src.models.conflict_aware_candidate_registry_v5 import candidate_registry_v5


def _metrics():
    return {source: {
        "accuracy": 0.9, "balanced_accuracy": 0.9, "auroc": 0.92,
        "sensitivity": 0.9, "specificity": 0.9, "brier": 0.1,
    } for source in runner.SOURCES}


def _cosines(value: float):
    return {
        "palsynet__neuroface": value,
        "palsynet__meei": value,
        "neuroface__meei": value,
    }


def test_runner_phase_report_and_boundary(c):
    ids = tuple(item.candidate_id for item in candidate_registry_v5())
    runner.validate_candidate_phase("screen", ids)
    runner.validate_candidate_phase("confirm", ids[:2])
    report = runner.build_report(
        phase="screen", evaluations={item: _metrics() for item in ids},
        pre_cosines={item: _cosines(-0.1) for item in ids},
        post_cosines={item: _cosines(0.0) for item in ids}, ranking=ids,
        counts={"palsynet": 38, "neuroface": 36, "meei": 56},
        runtime={"gpu": "NVIDIA H200", "epochs": 20, "seed": 0, "folds": 6},
        commitments={"implementation_sha256": "a" * 64},
    )
    c.eq(report["model"]["locked_base_candidate"], "NMR4-001")
    c.eq(report["audit"]["mayo_reads"], 0)
    c.true("probabilities" not in str(report).lower())
    parser_source = inspect.getsource(runner._parser).lower()
    c.true("mayo" not in parser_source and "outer-test" not in parser_source)


if __name__ == "__main__":
    run_all("test_run_conflict_aware_shared_search_v5", dict(globals()))
