from __future__ import annotations

import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _testlib import run_all

from scripts import run_compact_shared_severity_v3 as runner
from src.models.compact_shared_severity_v3 import compact_candidate_registry


def _metrics(value: float):
    return {
        source: {
            "accuracy": value,
            "balanced_accuracy": value,
            "auroc": value,
            "sensitivity": value,
            "specificity": value,
            "brier": 1.0 - value,
        }
        for source in runner.SOURCES
    }


def test_cli_is_three_source_development_only(c):
    source = inspect.getsource(runner._parser).lower()
    for name in (
        "palsynet-cache-root", "reviewed-identity-manifest", "review-ledger",
        "split-registry", "neuroface-cache", "meei-cache", "candidate-ids",
        "phase", "output",
    ):
        c.true(name in source)
    c.true("mayo" not in source and "outer-test" not in source)


def test_phase_contract_is_exact_16_then_locked_four(c):
    identifiers = tuple(
        candidate.candidate_id for candidate in compact_candidate_registry()
    )
    runner.validate_candidate_phase("screen", identifiers)
    runner.validate_candidate_phase("confirm", identifiers[:4])
    c.raises(
        lambda: runner.validate_candidate_phase("screen", identifiers[:-1]),
        ValueError,
    )
    c.raises(
        lambda: runner.validate_candidate_phase("confirm", identifiers[:3]),
        ValueError,
    )


def test_report_is_aggregate_shared_and_medically_closed(c):
    identifiers = tuple(
        candidate.candidate_id for candidate in compact_candidate_registry()
    )
    evaluations = {
        candidate_id: _metrics(0.70 + index * 0.01)
        for index, candidate_id in enumerate(identifiers)
    }
    ranking = tuple(reversed(identifiers))
    report = runner.build_report(
        phase="screen",
        evaluations=evaluations,
        ranking=ranking,
        counts={"palsynet": 38, "neuroface": 36, "meei": 56},
        runtime={"gpu": "NVIDIA H200", "epochs": 40, "seed": 0, "folds": 6},
        commitments={"implementation_sha256": "a" * 64},
    )
    c.eq(report["model"]["shared_patient_embedding_dim"], 64)
    c.eq(report["model"]["dense_transform"], "fixed_regional_excursion_velocity")
    c.true(report["model"]["flip_used"] is False)
    c.true(report["model"]["source_identifier_input"] is False)
    c.eq(report["selection"]["primary_metric"], "minimum_source_accuracy")
    c.eq(report["audit"]["mayo_reads"], 0)
    c.eq(report["audit"]["palsynet_protected_reads"], 0)
    c.true("group_id" not in str(report).lower())
    c.true("probabilities" not in str(report).lower())
    c.true("v6" not in str(report).lower())


if __name__ == "__main__":
    run_all("test_run_compact_shared_severity_v3", dict(globals()))
