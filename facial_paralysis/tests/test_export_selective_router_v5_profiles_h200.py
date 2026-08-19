from __future__ import annotations

import hashlib
import sys
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from _testlib import Check, run_all  # noqa: E402
from scripts.export_selective_router_v5_profiles_h200 import (  # noqa: E402
    HELPER_SHA256,
    build_private_profile_bytes,
    publish_private_payload,
    validate_development_threshold_aggregation,
)
from scripts.run_selective_universal_router_v5 import (  # noqa: E402
    load_private_profile_bytes,
)


def _values():
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    final = np.asarray([0.1, 0.2, 0.8, 0.9], dtype=np.float64)
    components = np.stack((final, final), axis=1)
    return labels, final, components


def test_private_profile_export_is_deterministic_and_deidentified(c: Check):
    labels, final, components = _values()
    first = build_private_profile_bytes(
        "free_asymmetry", labels, final, components,
        decision_threshold=np.full(4, 0.5, dtype=np.float64),
    )
    second = build_private_profile_bytes(
        "free_asymmetry", labels, final, components,
        decision_threshold=np.full(4, 0.5, dtype=np.float64),
    )
    c.eq(hashlib.sha256(first).hexdigest(), hashlib.sha256(second).hexdigest())
    loaded = load_private_profile_bytes(first)
    c.eq(loaded["anonymous_groups"], tuple(f"anonymous_{i:04d}" for i in range(4)))
    c.true(b"participant_id" not in first and b"recording_id" not in first)


def test_private_profile_export_rejects_alignment_or_dtype_drift(c: Check):
    labels, final, components = _values()
    c.raises(
        lambda: build_private_profile_bytes(
            "free_asymmetry", labels, final[:-1], components,
            decision_threshold=np.full(4, 0.5, dtype=np.float64),
        ), ValueError,
    )
    c.raises(
        lambda: build_private_profile_bytes(
            "free_asymmetry", labels, final.astype(np.float32), components,
            decision_threshold=np.full(4, 0.5, dtype=np.float64),
        ), ValueError,
    )


def test_helper_commitments_are_closed_lowercase_sha256(c: Check):
    c.eq(len(HELPER_SHA256), 18, "all execution-affecting legacy helpers are pinned")
    c.true(all(
        type(path) is str and path and type(digest) is str and len(digest) == 64
        and digest == digest.lower() and all(ch in "0123456789abcdef" for ch in digest)
        for path, digest in HELPER_SHA256.items()
    ))


def test_private_payload_publication_is_no_overwrite(c: Check):
    labels, final, components = _values()
    payload = build_private_profile_bytes(
        "free_asymmetry", labels, final, components,
        decision_threshold=np.full(4, 0.5, dtype=np.float64),
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "profile.npz"
        digest = publish_private_payload(path, payload)
        c.eq(digest, hashlib.sha256(payload).hexdigest())
        c.raises(lambda: publish_private_payload(path, payload), FileExistsError)


def test_final_threshold_may_aggregate_but_not_replace_nested_oof_rule(c: Check):
    thresholds = np.asarray([
        0.5118117069420783, 0.5376212668894103, 0.4931714554185852,
        0.512731595487173, 0.4951417070930201, 0.505786158207183,
    ], dtype=np.float64)
    diagnostic = validate_development_threshold_aggregation(
        0.5092985069157732, thresholds
    )
    c.eq(diagnostic["evaluation_scope"], "per_participant_oof")
    c.true(diagnostic["artifact_differs_from_oof_median"])
    c.raises(
        lambda: validate_development_threshold_aggregation(0.80, thresholds),
        ValueError, "a materially different artifact threshold is not the same model",
    )


if __name__ == "__main__":
    run_all("test_export_selective_router_v5_profiles_h200", dict(globals()))
