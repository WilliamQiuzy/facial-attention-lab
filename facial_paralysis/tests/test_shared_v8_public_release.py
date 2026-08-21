from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

from _testlib import run_all


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "releases/shared-v8-deployment-v1"


def test_public_release_is_complete_but_excludes_restricted_weights(c):
    model = json.loads((RELEASE / "model_manifest.json").read_bytes())
    image = json.loads((RELEASE / "image_manifest.json").read_bytes())
    acceptance = json.loads((RELEASE / "acceptance_summary.json").read_bytes())
    c.eq(model["model_id"], "residual_shared_router_v8_rsr8_001")
    c.eq(model["weights_sha256"], "72e40ea7b127b6768e931665df622550f06cc5a1bbad20070a42614c5b9901ab")
    c.eq(image["image_id"], "sha256:d5f1de3c57ab5b080ab30907f114b764b67bc3acc897ff29ceda426cb44296ea")
    c.eq(acceptance["status"], "pass")
    c.true(acceptance["restart_deterministic"] is True)
    c.true(acceptance["clinical_validation"] is False)
    c.true(acceptance["cpu_gpu_maximum_absolute_difference"] < 1e-5)
    tracked = subprocess.check_output(("git", "ls-files"), cwd=ROOT, text=True)
    c.true("weights.npz" not in tracked and "model.pt" not in tracked)
    c.true("restricted" in (RELEASE / "README.md").read_text().lower())


def test_registry_keeps_v8_historical_after_v9_deployment_activation(c):
    registry = json.loads((ROOT / "docs/model_registry.json").read_bytes())
    c.eq(registry["schema_version"], "facial_paralysis_model_registry_v3")
    c.eq(
        registry["current"]["name"],
        "broad_literature_shared_v9_blv9_009_ensemble",
    )
    c.eq(registry["benchmark"]["name"], "universal_clinical_router_v4")
    deployment = registry["deployment"]
    c.eq(deployment["name"], "broad_literature_shared_v9_blv9_009_ensemble")
    c.eq(deployment["version"], "shared-v9-public-oci-v1")
    c.true(any(
        row.get("name") == "residual_shared_router_v8_rsr8_001"
        and row.get("status") == "archived_not_current"
        for row in registry["archived"]
    ))


def test_public_manifests_are_identifier_and_secret_free(c):
    payload = b"\n".join(path.read_bytes() for path in RELEASE.iterdir() if path.is_file())
    lowered = payload.lower()
    for forbidden in (
        b"participant_id", b"group_id", b"patient_id", b"/users/",
        b"/home/", b"aws_secret", b"runpod_api", b"nvapi-",
    ):
        c.true(forbidden not in lowered, forbidden.decode("utf-8"))
    c.eq(hashlib.sha256((RELEASE / "model_manifest.json").read_bytes()).hexdigest(),
         "d40721a8fbda0c37a7e30a49f68bc94aa943e34c35e21864a1e79a2597676cbc")


if __name__ == "__main__":
    run_all("test_shared_v8_public_release", dict(globals()))
