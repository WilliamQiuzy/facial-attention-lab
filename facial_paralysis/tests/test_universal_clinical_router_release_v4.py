from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from _testlib import Check, run_all  # noqa: E402
from src.models.universal_clinical_router_v4 import (  # noqa: E402
    cue_aligned_upper_probability,
    scripted_multimechanism_probability,
)


MODEL = ROOT / "docs/results/artifacts/universal_clinical_router_v4/model.json"
REPORT = ROOT / "docs/results/artifacts/universal_clinical_router_v4/report.json"
PALSYNET = (
    ROOT / "outputs/dynamic_landmark/artifacts/110d-generalization-v1/"
    "final_palsynet_artifact.json"
)


def _load():
    return json.loads(MODEL.read_bytes())


def test_release_is_deidentified_and_keeps_claim_boundary(c: Check):
    payload = MODEL.read_bytes()
    document = json.loads(payload)
    c.eq(document["schema_version"], "universal_clinical_router_v4", "schema is frozen")
    c.true(not document["routing"]["dataset_identity_input"], "source name cannot route")
    c.eq(document["claim_boundary"]["protected_palsynet_reads"], 0,
         "universal refit did not open the protected PalsyNet partition")
    c.true(not document["claim_boundary"]["clinical_accuracy"],
           "development evidence is not relabelled clinical accuracy")
    lowered = payload.lower()
    for token in (
        b"/users/", b"/home/", b"participant_id", b"recording_id",
        b"source_sha256", b"grp_", b"rec_", b"aws_secret", b"runpod",
        b"nvapi-",
    ):
        c.true(token not in lowered, f"release excludes sensitive token {token!r}")


def test_release_binds_the_existing_frozen_110d_artifact(c: Check):
    document = _load()
    c.eq(
        hashlib.sha256(PALSYNET.read_bytes()).hexdigest(),
        document["palsynet"]["artifact_sha256"],
        "universal artifact references the exact frozen 110D model",
    )


def test_aggregate_report_binds_model_and_reports_all_sources(c: Check):
    report = json.loads(REPORT.read_bytes())
    c.eq(
        report["model_artifact"]["sha256"],
        hashlib.sha256(MODEL.read_bytes()).hexdigest(),
        "aggregate report binds the exact executable artifact",
    )
    c.eq(
        set(report["evaluations"]),
        {"palsynet_development", "palsynet_sealed_outer",
         "neuroface_development", "meei_development"},
        "aggregate report retains every participant-level evaluation",
    )
    c.true(all(
        report["evaluations"][name]["metrics"]["auroc"] >= 0.90
        for name in ("palsynet_development", "neuroface_development",
                     "meei_development")
    ), "all primary development evidence profiles exceed 0.90 AUROC")
    c.true(not report["claim_boundary"]["clinical_accuracy"],
           "aggregate metrics remain development evidence")


def test_neuroface_release_heads_execute_through_public_runtime(c: Check):
    document = _load()["neuroface"]
    c.eq(len(document["marlin_heads"]), 18, "fixed median uses all 18 heads")
    clinical = document["clinical_heads"]
    landmark_dim = clinical["post_stroke_asymmetry_mean110"]["input_dimension"]
    au_dim = clinical["als_oromotor_robust_pool1600"]["input_dimension"]
    dimensions = {}
    for candidate in document["marlin_heads"]:
        dimensions[candidate["representation"]] = candidate["targets"][0]["head"][
            "input_dimension"
        ]
    probability = scripted_multimechanism_probability(
        landmark_original=np.zeros((2, landmark_dim), dtype=np.float64),
        landmark_mirrored=np.zeros((2, landmark_dim), dtype=np.float64),
        au_values=np.zeros((2, au_dim), dtype=np.float64),
        marlin_representations={
            name: np.zeros((2, dimension), dtype=np.float64)
            for name, dimension in dimensions.items()
        },
        artifact={
            key: document[key] for key in ("clinical_heads", "marlin_heads", "gate")
        },
    )
    c.true(probability.shape == (2,) and np.isfinite(probability).all(),
           "frozen scripted artifact is directly executable")


def test_meei_release_heads_execute_through_public_runtime(c: Check):
    document = _load()["meei"]
    original = {}
    mirrored = {}
    for row in document["heads"]:
        dimension = row["head"]["input_dimension"]
        original[row["name"]] = np.zeros((2, dimension), dtype=np.float64)
        mirrored[row["name"]] = np.zeros((2, dimension), dtype=np.float64)
    probability = cue_aligned_upper_probability(
        original, mirrored,
        {key: document[key] for key in (
            "heads", "probability_weights", "decision_threshold"
        )},
    )
    c.true(probability.shape == (2,) and np.isfinite(probability).all(),
           "frozen cue-aligned artifact is directly executable")


if __name__ == "__main__":
    run_all("test_universal_clinical_router_release_v4", dict(globals()))
