"""Release-runner contracts for Universal Orofacial Model v1."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from scripts.run_universal_orofacial_v1 import (  # noqa: E402
    _implementation_sha256,
    build_public_meei_diagnostic_report,
    implementation_component_paths,
    build_public_development_report,
    parser,
    validate_public_development_report,
)
from src.evaluation.universal_orofacial_v1 import (  # noqa: E402
    CANDIDATES,
    CandidateEvaluation,
)
from _testlib import Check, run_all  # noqa: E402


def _metrics(auroc: float, balanced_accuracy: float, brier: float):
    return {
        "accuracy": balanced_accuracy,
        "auroc": auroc,
        "balanced_accuracy": balanced_accuracy,
        "sensitivity": balanced_accuracy,
        "specificity": balanced_accuracy,
        "brier": brier,
    }


def _evaluations():
    output = {}
    values = {
        CANDIDATES[0]: (0.91, 0.90, 0.15),
        CANDIDATES[1]: (0.93, 0.91, 0.13),
        CANDIDATES[2]: (0.92, 0.92, 0.12),
    }
    for candidate, (auroc, balanced, brier) in values.items():
        output[candidate] = CandidateEvaluation(
            candidate=candidate,
            protocol="six_fold_source_class_stratified_participant_oof",
            probabilities=np.linspace(0.1, 0.9, 74),
            metrics={
                "overall": _metrics(auroc, balanced, brier),
                "palsynet": _metrics(auroc + 0.01, balanced + 0.01, brier),
                "neuroface": _metrics(auroc, balanced, brier),
            },
            model_fits=6 if candidate == CANDIDATES[0] else 18,
        )
    return output


def _transfers():
    output = {}
    for candidate in CANDIDATES:
        output[candidate] = {
            "palsynet_to_neuroface": {
                "training_source": "palsynet", "held_source": "neuroface",
                "training_participants": 38, "held_participants": 36,
                "model_fits": 1 if candidate == CANDIDATES[0] else 3,
                "metrics": _metrics(0.91, 0.88, 0.16),
            },
            "neuroface_to_palsynet": {
                "training_source": "neuroface", "held_source": "palsynet",
                "training_participants": 36, "held_participants": 38,
                "model_fits": 1 if candidate == CANDIDATES[0] else 3,
                "metrics": _metrics(0.92, 0.89, 0.15),
            },
        }
    return output


def test_parser_exposes_data_paths_but_no_model_selection_controls(c: Check):
    actions = {action.dest for action in parser()._actions}
    c.true({
        "mode", "palsynet_cache_root", "reviewed_identity_manifest",
        "review_ledger", "split_registry", "neuroface_private_manifest",
        "neuroface_cache_root", "output_root",
    }.issubset(actions), "development inputs are explicit")
    c.true(actions.isdisjoint({
        "candidate", "threshold", "c", "seed", "epochs", "learning_rate",
        "weight_decay", "hidden_dim",
    }), "CLI cannot tune or select after seeing outcomes")


def test_implementation_closure_is_present_and_hashable(c: Check):
    paths = {path.as_posix() for path in implementation_component_paths()}
    for required in (
        "src/evaluation/meei_external_v1.py",
        "src/preprocessing/action_capacity_features_v1.py",
        "src/preprocessing/script_action_segmentation_v1.py",
    ):
        c.true(any(path.endswith(required) for path in paths),
               f"implementation closure includes {required}")
    digest = _implementation_sha256()
    c.eq(len(digest), 64, "every direct 110D/model/loader dependency is present")


def test_public_report_contains_only_aggregate_recomputable_evidence(c: Check):
    report = build_public_development_report(
        _evaluations(), _transfers(),
        counts={
            "participants": 74, "palsynet_participants": 38,
            "neuroface_participants": 36,
        },
        provenance={
            "implementation_sha256": "a" * 64,
            "palsynet_split_registry_sha256": "b" * 64,
            "neuroface_private_manifest_sha256": "c" * 64,
            "neuroface_collection_manifest_sha256": "d" * 64,
        },
        audit={
            "palsynet_development_cache_records_loaded": 39,
            "palsynet_protected_cache_records_loaded": 0,
            "palsynet_protected_predictions": 0,
            "meei_reads": 0, "mayo_reads": 0, "yfp_reads": 0,
        },
        locked_artifact_sha256="e" * 64,
    )
    validate_public_development_report(report)
    encoded = json.dumps(report, sort_keys=True)
    c.true(all(token not in encoded for token in (
        "grp_", "rec_", "probabilities", "coefficient", "intercept", "/Users/",
    )), "public report contains no IDs, predictions, paths, or model weights")
    c.eq(report["decision"]["locked_candidate"], CANDIDATES[1],
         "winner follows worst-source AUROC before other metrics")
    c.true(report["decision"]["development_gate_passed"],
           "all OOF and transfer AUROC gates are recomputed")


def test_public_report_rejects_forged_winner_or_protected_access(c: Check):
    report = build_public_development_report(
        _evaluations(), _transfers(),
        counts={
            "participants": 74, "palsynet_participants": 38,
            "neuroface_participants": 36,
        },
        provenance={
            "implementation_sha256": "a" * 64,
            "palsynet_split_registry_sha256": "b" * 64,
            "neuroface_private_manifest_sha256": "c" * 64,
            "neuroface_collection_manifest_sha256": "d" * 64,
        },
        audit={
            "palsynet_development_cache_records_loaded": 39,
            "palsynet_protected_cache_records_loaded": 0,
            "palsynet_protected_predictions": 0,
            "meei_reads": 0, "mayo_reads": 0, "yfp_reads": 0,
        },
        locked_artifact_sha256="e" * 64,
    )
    forged = json.loads(json.dumps(report))
    forged["decision"]["locked_candidate"] = CANDIDATES[0]
    c.raises(lambda: validate_public_development_report(forged), ValueError,
             "winner is independently recomputed")
    leaked = json.loads(json.dumps(report))
    leaked["audit"]["palsynet_protected_predictions"] = 1
    c.raises(lambda: validate_public_development_report(leaked), ValueError,
             "protected evaluation fails closed")


def test_meei_report_is_locked_repeated_diagnostic_not_selection(c: Check):
    report = build_public_meei_diagnostic_report(
        candidate=CANDIDATES[0],
        metrics=_metrics(0.81, 0.72, 0.18),
        counts={"participants": 60, "affected": 50, "unaffected": 10},
        development_report_sha256="a" * 64,
        locked_artifact_sha256="b" * 64,
        participant_manifest_sha256="c" * 64,
        collection_manifest_sha256="d" * 64,
        cache_collection_sha256="e" * 64,
    )
    c.eq(report["protocol"], {
        "candidate_selection": False, "model_refit": False,
        "scaler_refit": False, "threshold_selection": False,
        "status": "repeated_already_exposed_external_diagnostic",
    }, "MEEI cannot alter the locked universal candidate")
    c.true(not report["decision"]["cross_institutionally_robust"],
           "both AUROC and balanced accuracy must reach 0.90")
    c.true(all(token not in json.dumps(report, sort_keys=True)
               for token in ("grp_", "rec_", "probabilities", "/Users/")),
           "MEEI public diagnostic remains participant aggregate")


if __name__ == "__main__":
    run_all("test_run_universal_orofacial_v1", dict(globals()))
