"""Participant-level contracts for exploratory NeuroFace SLP transfer."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src.evaluation.neuroface_slp_transfer_v1 import build_slp_transfer_report  # noqa: E402
from _testlib import Check, run_all  # noqa: E402


def _rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    cohorts = ["healthy_control"] * 4 + ["als"] * 4 + ["post_stroke"] * 4
    for index, cohort in enumerate(cohorts):
        score = 0.05 + index * 0.07
        severity = 1.0 + index * 0.3
        for task in ("NSM_KISS", "NSM_OPEN", "NSM_SPREAD"):
            rows.append({
                "participant_id": f"grp_{index + 1:064x}",
                "cohort": cohort,
                "label": 0 if cohort == "healthy_control" else 1,
                "task": task,
                "probability": score,
                "slp_scores": {
                    "symmetry": severity,
                    "rom": severity,
                    "speed": severity,
                    "variability": severity,
                    "fatigue": severity,
                    "total": severity * 5.0,
                },
            })
    return rows


def test_slp_report_is_participant_level_directional_and_identifier_free(c: Check):
    report = build_slp_transfer_report(
        _rows(), external_report_sha256="a" * 64, bootstrap_repeats=32
    )
    c.eq(report["scale_direction"], "higher_is_more_severe_dysfunction",
         "SLP scale direction is explicit")
    c.eq(report["counts"]["participants"], 12,
         "three task rows become one participant unit")
    c.eq(report["associations"]["all_participants_spectrum"]["total"]["rho"], 1.0,
         "monotone model and SLP severity have perfect rank agreement")
    c.eq(report["clinical_reference_discrimination"]["total_auroc"], 1.0,
         "SLP severity is checked against released cohorts")
    encoded = json.dumps(report, allow_nan=False)
    c.true(all(token not in encoded for token in (
        "grp_", "participant_id", "probability", "/Users/", ".avi"
    )), "aggregate association report contains no row-level data")


def test_missing_primary_task_or_slp_domain_fails_closed(c: Check):
    rows = _rows()
    c.raises(lambda: build_slp_transfer_report(
        rows[:-1], external_report_sha256="a" * 64, bootstrap_repeats=16
    ), ValueError, "all three primary tasks are required")
    changed = _rows()
    del changed[0]["slp_scores"]["fatigue"]
    c.raises(lambda: build_slp_transfer_report(
        changed, external_report_sha256="a" * 64, bootstrap_repeats=16
    ), ValueError, "every frozen SLP domain is required")


if __name__ == "__main__":
    run_all("test_neuroface_slp_transfer_v1", dict(globals()))
