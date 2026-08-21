from __future__ import annotations

import hashlib
import json
from pathlib import Path

from _testlib import run_all


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "docs/results/artifacts/shared_v9_specificity_first"
ARTIFACT_SHA256 = {
    "specificity_anchor_screen.json": "6f3eda5e5515e274d9bbc8a8eda5a99a6fac26876fbff377ddf452c6e1ff14fe",
    "ensemble_screen_nondeterministic_a.json": "669958074c3ec48bb89ee58e179f27d62eb0b37b67003247755408a4c608413a",
    "ensemble_screen_nondeterministic_b.json": "af47b45b12671080f793073f044691a0d342b00a48061e2f47f490159397f10a",
    "distillation_screen.json": "957ab1e3e30d6f171a30b29a16ab5af38d1accf0a443eb51e89fe16d555c038a",
    "mechanism_screen.json": "d3f547c3f81a840271cc9e714c12c1b75075a7be9aad670bee3a1a3f26238555",
    "phenotype_screen.json": "0c56f22bcfb10c6c3ef1dd4cdc5c3fdde4b1a1292eee599584298115e271f4ce",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _passes(metrics) -> bool:
    return all(
        metrics[source]["accuracy"] >= 0.90
        and metrics[source]["specificity"] >= 0.80
        and metrics[source]["auroc"] >= 0.92
        and metrics[source]["sensitivity"] >= 0.85
        for source in ("palsynet", "neuroface", "meei")
    )


def test_machine_evidence_is_exact_aggregate_and_no_candidate_passes(c):
    documents = {}
    for name, digest in ARTIFACT_SHA256.items():
        path = ARTIFACT_ROOT / name
        c.eq(_sha(path), digest)
        payload = path.read_text()
        c.true("grp_" not in payload and "participant_id" not in payload)
        c.true("/Users/" not in payload and "/home/" not in payload)
        documents[name] = json.loads(payload)
    anchor = documents["specificity_anchor_screen.json"]
    for evaluation in anchor["evaluations"].values():
        c.true(not _passes(evaluation["fixed"]))
        c.true(not _passes(evaluation["calibrated"]))
    for name in (
        "ensemble_screen_nondeterministic_a.json", "distillation_screen.json",
        "mechanism_screen.json", "phenotype_screen.json",
    ):
        for evaluation in documents[name]["evaluations"].values():
            c.true(not _passes(evaluation["metrics"]))
    c.eq(
        documents["distillation_screen.json"]["evaluations"]["DSR9-000"]["metrics"],
        documents["phenotype_screen.json"]["evaluations"]["SAP9-000"]["metrics"],
    )


def test_nondeterministic_v8_repeat_is_disclosed_not_selected(c):
    first = json.loads(
        (ARTIFACT_ROOT / "ensemble_screen_nondeterministic_a.json").read_text()
    )["evaluations"]["SEN9-000"]["metrics"]
    second = json.loads(
        (ARTIFACT_ROOT / "ensemble_screen_nondeterministic_b.json").read_text()
    )["evaluations"]["SEN9-000"]["metrics"]
    c.true(first != second)
    c.true(first["meei"]["accuracy"] != second["meei"]["accuracy"])


def test_v8_registry_and_deployment_remain_locked(c):
    c.eq(_sha(ROOT / "docs/model_registry.json"),
         "67ce23f2fb3155e181d5615c69e721ec19e253bb7797426587ed4bef5e63f489")
    c.eq(_sha(ROOT / "docs/CURRENT_DEPLOYMENT_MODEL.md"),
         "702e3da45e1cdd19a04526046635442a5394c8dac0abf4baaa4d81381f342bf4")


def test_public_decision_is_honest_and_complete(c):
    document = (ROOT / "docs/results/shared_v9_specificity_first.md").read_text()
    for phrase in (
        "No V9 candidate is promoted", "94", "deterministic",
        "Mayo reads: **0**", "V8 remains the canonical deployment model",
    ):
        c.true(phrase in document)
    c.true("Mayo accuracy" not in document)


if __name__ == "__main__":
    run_all("test_shared_v9_specificity_first_release", dict(globals()))
