from __future__ import annotations

import json
from pathlib import Path
import re

from _testlib import run_all


ROOT = Path(__file__).resolve().parents[1]


def test_bundle_build_requires_the_exact_restricted_model(c):
    source = (ROOT / "environment/shared_v8_bundle_v1.Dockerfile").read_text()
    c.true("COPY --chown=1001:1001 manifest.json weights.npz /model/" in source)
    c.true("72e40ea7b127b6768e931665df622550f06cc5a1bbad20070a42614c5b9901ab" in source)
    c.true("USER 1001:1001" in source)
    c.true("org.opencontainers.image.source" in source)
    c.true("ADD " not in source and "COPY ." not in source)


def test_compose_is_digest_pinned_and_preserves_hardening(c):
    source = (ROOT / "deploy/shared-v8/compose.yaml").read_text()
    match = re.search(
        r"image: ghcr\.io/williamqiuzy/facial-attention-lab-shared-v8-bundle"
        r"@sha256:([0-9a-f]{64})",
        source,
    )
    c.true(match is not None)
    c.true(":latest" not in source)
    for token in (
        "read_only: true", 'user: "1001:1001"', "cap_drop:", "- ALL",
        "no-new-privileges:true", "127.0.0.1:18080:8080", "tmpfs:",
        "capabilities: [gpu]",
    ):
        c.true(token in source, token)


def test_oci_manifest_and_quickstart_are_closed(c):
    manifest = json.loads(
        (ROOT / "releases/shared-v8-deployment-v1/oci_manifest.json").read_bytes()
    )
    c.eq(manifest["schema_version"], "shared_v8_oci_distribution_v1")
    c.eq(manifest["bundle_visibility"], "private")
    c.eq(manifest["weights_sha256"],
         "72e40ea7b127b6768e931665df622550f06cc5a1bbad20070a42614c5b9901ab")
    c.true(re.fullmatch(r"sha256:[0-9a-f]{64}", manifest["runtime_digest"]) is not None)
    c.true(re.fullmatch(r"sha256:[0-9a-f]{64}", manifest["bundle_digest"]) is not None)
    c.true("latest" not in json.dumps(manifest))
    quickstart = (ROOT / "deploy/shared-v8/README.md").read_text().lower()
    for phrase in ("docker login ghcr.io", "docker compose pull", "docker compose up -d", "/readyz", "not clinical"):
        c.true(phrase in quickstart, phrase)


if __name__ == "__main__":
    run_all("test_shared_v8_portable_bundle", dict(globals()))
