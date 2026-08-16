from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from _testlib import Check, run_all  # noqa: E402


REGISTRY = ROOT / "docs/model_registry.json"
MODEL = ROOT / "docs/results/artifacts/universal_clinical_router_v4/model.json"
REPORT = ROOT / "docs/results/artifacts/universal_clinical_router_v4/report.json"
FROZEN_110D = (
    ROOT / "outputs/dynamic_landmark/artifacts/110d-generalization-v1/"
    "final_palsynet_artifact.json"
)
EXPECTED_MODEL_SHA256 = (
    "c8f8c217d508b15bf0d8626b42cead857192ecd738b1fffab94f364c6ed80495"
)
CURRENT_EXPORTS = {
    "CURRENT_MODEL_ARTIFACT_SHA256",
    "CURRENT_MODEL_NAME",
    "CURRENT_MODEL_SCHEMA_VERSION",
    "SCRIPTED_COMMON_TASKS",
    "TIMING_AUTHORITIES",
    "UPPER_PROMPT_TASKS",
    "cue_aligned_upper_probability",
    "evidence_profile",
    "linear_head_probability",
    "load_current_artifact",
    "median_low_confidence_gate",
    "scripted_multimechanism_probability",
    "serialized_head_probability",
}


def test_registry_and_default_package_bind_only_v4(c: Check):
    registry = json.loads(REGISTRY.read_bytes())
    c.eq(set(registry), {"schema_version", "current", "archived"})
    c.eq(registry["schema_version"], "facial_paralysis_model_registry_v1")
    current = registry["current"]
    c.eq(set(current), {
        "name", "schema_version", "status", "python_module",
        "runtime_module", "artifact_path", "artifact_sha256", "report_path",
    })
    c.eq(current["name"], "universal_clinical_router_v4")
    c.eq(current["schema_version"], "universal_clinical_router_v4")
    c.eq(current["status"], "development_candidate_not_clinically_validated")
    c.eq(current["python_module"], "src.models.current")
    c.eq(current["runtime_module"], "src.models.universal_clinical_router_v4")
    c.eq(current["artifact_path"], MODEL.relative_to(ROOT).as_posix())
    c.eq(current["artifact_sha256"], EXPECTED_MODEL_SHA256)
    c.eq(current["report_path"], REPORT.relative_to(ROOT).as_posix())
    c.true(type(registry["archived"]) is list and registry["archived"])
    c.true(all(
        type(row) is dict
        and row.get("status") == "archived_not_current"
        and row.get("default_import") is False
        for row in registry["archived"]
    ))

    package = importlib.import_module("src.models")
    current_module = importlib.import_module("src.models.current")
    c.eq(set(package.__all__), CURRENT_EXPORTS)
    c.eq(set(current_module.__all__), CURRENT_EXPORTS)
    c.eq(package.CURRENT_MODEL_NAME, "universal_clinical_router_v4")
    c.eq(package.CURRENT_MODEL_ARTIFACT_SHA256, EXPECTED_MODEL_SHA256)
    artifact = package.load_current_artifact()
    c.eq(artifact["schema_version"], package.CURRENT_MODEL_SCHEMA_VERSION)
    c.true(artifact["routing"]["dataset_identity_input"] is False)


def test_current_artifacts_and_frozen_dependency_are_exact(c: Check):
    model = json.loads(MODEL.read_bytes())
    report = json.loads(REPORT.read_bytes())
    c.eq(hashlib.sha256(MODEL.read_bytes()).hexdigest(), EXPECTED_MODEL_SHA256)
    c.eq(report["model_artifact"]["sha256"], EXPECTED_MODEL_SHA256)
    c.true(FROZEN_110D.is_file())
    c.eq(
        hashlib.sha256(FROZEN_110D.read_bytes()).hexdigest(),
        model["palsynet"]["artifact_sha256"],
    )


def test_active_docs_name_v4_and_do_not_restore_old_champion(c: Check):
    readme = (ROOT / "README.md").read_text()
    script_readme = (ROOT / "scripts/README.md").read_text()
    results_readme = (ROOT / "docs/results/README.md").read_text()
    current = (ROOT / "docs/CURRENT_MODEL.md").read_text()
    pipeline = (ROOT / "docs/PIPELINE.md").read_text()
    c.true("Universal Clinical Router v4" in readme[:1200])
    c.true("Universal Clinical Router v4" in script_readme[:1200])
    c.true("Universal Clinical Router v4" in results_readme[:1200])
    c.true("Universal Clinical Router v4" in current[:1200])
    c.true("Universal Clinical Router v4" in pipeline[:1600])
    c.true("No raw-video production CLI" in script_readme)
    for stale in (
        "current development champion is the **mirror-invariant 110D",
        "当前唯一的开发集 champion 是 **110维",
        "current development champion: mirror-invariant 110D",
    ):
        c.true(stale.lower() not in current.lower())
        c.true(stale.lower() not in pipeline.lower())

    for historical_doc in (
        ROOT / "docs/archive/manuscripts/web_model_transfer_draft_pre_v4.md",
        ROOT / "docs/archive/experiments/autoresearch_fp_pre_v4.md",
        ROOT / "docs/archive/experiments/training_runs_pre_v4.md",
        ROOT / "docs/archive/experiments/mayo_loop_findings_pre_v4.md",
    ):
        prefix = historical_doc.read_text()[:800].lower()
        c.true("archived" in prefix or "historical" in prefix)

    for retired_top_level in (
        "PAPER_DRAFT.md", "autoresearch_fp.md", "training_runs.md",
        "loop_findings.md",
    ):
        c.true(not (ROOT / "docs" / retired_top_level).exists())


def test_generated_legacy_outputs_are_not_tracked(c: Check):
    tracked = subprocess.check_output(
        ("git", "ls-files", "outputs", "autoresearch_fp"),
        cwd=ROOT, text=True,
    ).splitlines()
    allowed_output = (
        "outputs/dynamic_landmark/",
        "outputs/meei_external_v1/report.json",
    )
    retired = [
        path for path in tracked
        if path.startswith("outputs/") and not (
            path.startswith(allowed_output[0]) or path == allowed_output[1]
        )
    ]
    retired.extend(
        path for path in tracked
        if path.startswith("autoresearch_fp/experiments/")
        or path.endswith(".log")
        or path.endswith("_search.tsv")
        or path in {
            "autoresearch_fp/results.tsv",
            "autoresearch_fp/clinical_search.tsv",
            "autoresearch_fp/geo_landmark_search.tsv",
        }
    )
    c.eq(retired, [], "generated legacy outputs must stay outside source control")


def test_pre_v4_generic_training_and_prediction_entrypoints_are_retired(c: Check):
    tracked = set(subprocess.check_output(
        ("git", "ls-files", "scripts"), cwd=ROOT, text=True
    ).splitlines())
    retired = {
        "scripts/predict.py",
        "scripts/train_palsynet.py",
        "scripts/train_v2_pod.py",
        "scripts/train_v3_pod.py",
        "scripts/train_v4_pod.py",
        "scripts/run2_roboflow.py",
        "scripts/run3_fnp_region.py",
        "scripts/run4_cfd_controls.py",
        "scripts/run5_yfp_region.py",
        "scripts/run6_unified.py",
        "scripts/run7_yfp_clips.py",
        "scripts/run8_yfp_targeted.py",
        "scripts/run9_temporal_pool.py",
        "scripts/run10_mayo_severity.py",
        "scripts/fp_research.py",
        "scripts/mayo_score_per_action.py",
        "scripts/visualize.py",
    }
    c.eq(sorted(tracked & retired), [], "ambiguous pre-v4 entrypoints are retired")


if __name__ == "__main__":
    run_all("test_current_model_contract_v4", dict(globals()))
