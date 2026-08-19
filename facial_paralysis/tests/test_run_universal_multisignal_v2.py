"""Runner/report contracts for Universal Multi-Signal v2."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from scripts.run_universal_multisignal_v2 import (  # noqa: E402
    _implementation_sha256,
    build_public_report,
    implementation_component_paths,
    parser,
    validate_public_report,
)
from src.evaluation.universal_orofacial_v1 import CandidateEvaluation  # noqa: E402
from _testlib import Check, run_all  # noqa: E402


def _metrics(auc, ba, brier):
    return {"accuracy": ba, "auroc": auc, "balanced_accuracy": ba,
            "sensitivity": ba, "specificity": ba, "brier": brier}


def _inputs():
    values = {
        "landmark_110": (0.70, 0.70, 0.20),
        "blendshape_288": (0.92, 0.91, 0.13),
        "fusion_398": (0.90, 0.93, 0.12),
    }
    evaluations = {}
    transfers = {}
    for name, (auc, ba, brier) in values.items():
        evaluations[name] = CandidateEvaluation(
            candidate=name,
            protocol="six_fold_source_class_stratified_participant_oof",
            probabilities=np.linspace(0.1, 0.9, 74), model_fits=6,
            metrics={
                "overall": _metrics(auc, ba, brier),
                "palsynet": _metrics(auc + 0.01, ba + 0.01, brier),
                "neuroface": _metrics(auc, ba, brier),
            },
        )
        transfers[name] = {
            "palsynet_to_neuroface": {
                "training_source": "palsynet", "held_source": "neuroface",
                "training_participants": 38, "held_participants": 36,
                "model_fits": 1, "metrics": _metrics(0.91, 0.8, 0.2),
            },
            "neuroface_to_palsynet": {
                "training_source": "neuroface", "held_source": "palsynet",
                "training_participants": 36, "held_participants": 38,
                "model_fits": 1, "metrics": _metrics(0.92, 0.8, 0.2),
            },
        }
    return evaluations, transfers


def test_parser_has_no_post_outcome_model_controls(c: Check):
    actions = {action.dest for action in parser()._actions}
    c.true({"palsynet_cache_root", "neuroface_cache_root", "output_root"}
           .issubset(actions), "v2 declares its data locations")
    c.true(actions.isdisjoint({"representation", "c", "threshold", "seed"}),
           "representation and Logistic protocol cannot be changed by CLI")


def test_v2_implementation_closure_includes_reused_runner_and_gates(c: Check):
    paths = {path.as_posix() for path in implementation_component_paths()}
    for required in (
        "scripts/run_universal_orofacial_v1.py",
        "scripts/run_dynamic_landmark_classical.py",
        "scripts/freeze_palsynet_person_split_registry.py",
        "src/preprocessing/action_capacity_features_v1.py",
        "src/preprocessing/script_action_segmentation_v1.py",
    ):
        c.true(any(path.endswith(required) for path in paths),
               f"v2 implementation closure includes {required}")
    c.eq(len(_implementation_sha256()), 64,
         "v2 transitive implementation closure is hashable")


def test_report_recomputes_winner_and_contains_no_private_rows(c: Check):
    evaluations, transfers = _inputs()
    report = build_public_report(
        evaluations, transfers,
        counts={"participants": 74, "palsynet_participants": 38,
                "neuroface_participants": 36},
        provenance={"implementation_sha256": "a" * 64,
                    "palsynet_split_registry_sha256": "b" * 64,
                    "neuroface_private_manifest_sha256": "c" * 64,
                    "neuroface_collection_manifest_sha256": "d" * 64},
        audit={"palsynet_protected_cache_records_loaded": 0,
               "palsynet_protected_predictions": 0,
               "meei_reads": 0, "mayo_reads": 0, "yfp_reads": 0},
        locked_artifact_sha256="e" * 64,
    )
    validate_public_report(report)
    c.eq(report["decision"]["locked_representation"], "blendshape_288",
         "winner uses worst-source AUROC")
    c.true(report["decision"]["development_gate_passed"],
           "synthetic aggregate clears every frozen development gate")
    encoded = json.dumps(report, sort_keys=True)
    c.true(all(token not in encoded for token in (
        "grp_", "rec_", "probabilities", "coefficient", "/Users/",
    )), "public v2 report excludes row-level and model-private evidence")
    forged = json.loads(encoded)
    forged["decision"]["locked_representation"] = "fusion_398"
    c.raises(lambda: validate_public_report(forged), ValueError,
             "published v2 winner is independently recomputed")


if __name__ == "__main__":
    run_all("test_run_universal_multisignal_v2", dict(globals()))
