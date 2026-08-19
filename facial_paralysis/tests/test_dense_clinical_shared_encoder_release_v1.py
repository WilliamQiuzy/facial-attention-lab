from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _testlib import run_all


REPORT = (
    ROOT / "docs/results/artifacts/dense_clinical_shared_encoder_v1/report.json"
)


def test_release_is_three_seed_aggregate_and_not_promoted(c):
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    c.eq(report["schema_version"], "dense_clinical_shared_encoder_v1_three_seed")
    c.eq(report["protocol"]["seeds"], [0, 1, 2])
    c.eq(report["protocol"]["participant_disjoint_folds"], 6)
    c.eq(report["counts"], {"palsynet": 38, "neuroface": 36, "meei": 56})
    c.eq(report["decision"]["promotion_authorized"], False)
    c.eq(report["decision"]["v6_remains_primary"], True)
    c.eq(report["audit"], {
        "mayo_predictions": 0,
        "mayo_reads": 0,
        "palsynet_protected_reads": 0,
    })


def test_release_exposes_all_sources_without_identifiers_or_private_paths(c):
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    for candidate in ("110d_only", "dense_clinical"):
        c.eq(
            set(report["evaluations"][candidate]),
            {"palsynet", "neuroface", "meei"},
        )
        for source in report["evaluations"][candidate].values():
            for metric in ("accuracy", "auroc", "balanced_accuracy", "brier"):
                c.eq(len(source[metric]["values"]), 3)
    emitted = REPORT.read_text(encoding="utf-8").lower()
    for forbidden in ("group_id", "/home/", "/users/", "patient_id"):
        c.true(forbidden not in emitted)


if __name__ == "__main__":
    run_all("test_dense_clinical_shared_encoder_release_v1", dict(globals()))
