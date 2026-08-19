from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _testlib import run_all

from scripts.run_compact_shared_severity_v3 import (
    _implementation_sha256 as v3_implementation_sha256,
)
from scripts.run_medically_gated_shared_search_v2 import (
    _implementation_sha256 as v2_implementation_sha256,
)

REPORT = ROOT / "docs/results/artifacts/medically_gated_shared_encoder_v2/report.json"
SUMMARY = ROOT / "docs/results/medically_gated_shared_encoder_v2.md"


def _report():
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_release_exhausts_48_shared_candidates_without_promotion(c):
    report = _report()
    c.eq(report["search"]["total_shared_candidates_evaluated"], 48)
    c.eq(report["search"]["v2_full_mesh_candidates"], 32)
    c.eq(report["search"]["v3_compact_regional_candidates"], 16)
    c.true(report["search"]["v2_locked_top_four_three_seed_complete"])
    c.true(report["gate"]["passed"] is False)
    c.true(report["decision"]["promote_as_current_model"] is False)
    c.true(report["decision"]["clinical_claim_authorized"] is False)


def test_selected_metrics_recompute_from_three_seeds(c):
    report = _report()
    selected = report["strongest_stable_shared_candidate"]
    c.eq(selected["candidate_id"], "MSC2-022")
    minimum = 1.0
    for source, metrics in selected["three_seed_metrics"].items():
        values = np.asarray(metrics["accuracy_values"], dtype=np.float64)
        c.eq(values.shape, (3,))
        c.true(np.isclose(values.mean(), metrics["accuracy_mean"]))
        c.true(np.isclose(values.std(), metrics["accuracy_population_sd"]))
        minimum = min(minimum, float(metrics["accuracy_mean"]))
    c.true(minimum < report["gate"]["required_accuracy_each_source"])


def test_implementation_and_data_commitments_match(c):
    report = _report()
    commitments = report["commitments"]
    c.eq(commitments["v2_implementation_sha256"], v2_implementation_sha256())
    c.eq(commitments["v3_implementation_sha256"], v3_implementation_sha256())
    for name, value in commitments.items():
        c.true(name.endswith("sha256"))
        c.eq(len(value), 64)
        int(value, 16)


def test_public_artifacts_keep_mayo_and_identifiers_closed(c):
    report = _report()
    c.eq(report["audit"]["mayo_reads"], 0)
    c.eq(report["audit"]["mayo_predictions"], 0)
    c.eq(report["audit"]["palsynet_protected_reads"], 0)
    c.true(report["audit"]["row_level_probabilities_public"] is False)
    c.true(report["audit"]["participant_identifiers_public"] is False)
    emitted = REPORT.read_text(encoding="utf-8").lower()
    for forbidden in ("group_id", "/users/", "/home/", '"probabilities":['):
        c.true(forbidden not in emitted)
    summary = SUMMARY.read_text(encoding="utf-8")
    c.true("48 genuinely shared models" in summary)
    c.true("Mayo was not read, trained on, or scored" in summary)


if __name__ == "__main__":
    run_all("test_medically_gated_shared_encoder_release_v2", dict(globals()))
