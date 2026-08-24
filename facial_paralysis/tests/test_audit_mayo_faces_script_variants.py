"""Aggregate-only tests for medically valid Mayo FACES script variants."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _testlib import Check, run_all  # noqa: E402
from test_faces_shared_v9_pipeline import _payloads  # noqa: E402
from scripts.audit_mayo_faces_script_variants import (  # noqa: E402
    audit_authenticated_capture_payloads,
)


def test_audit_counts_both_variants_and_exact_duplicates(c: Check):
    seven = _payloads(include_optional=False, timing_source="audio_forced_alignment")
    eight = _payloads(
        include_optional=True,
        video=b"different exact video bytes",
        timing_source="blinded_manual",
    )
    report = audit_authenticated_capture_payloads((seven, eight, seven))
    c.eq(report["authenticated_payloads"], 3)
    c.eq(report["unique_recordings"], 2)
    c.eq(report["exact_duplicate_payloads"], 1)
    c.eq(report["seven_step_recordings"], 1)
    c.eq(report["eight_step_recordings"], 1)
    c.eq(report["timing_sources"], {
        "audio_forced_alignment": 1,
        "blinded_manual": 1,
        "capture_event_log": 0,
    })
    c.eq(report["model_predictions"], 0)
    c.eq(report["movement_magnitude_used_for_variant_assignment"], False)


def test_audit_is_identifier_free_and_fails_closed_on_ambiguous_evidence(c: Check):
    report = audit_authenticated_capture_payloads(())
    c.eq(report["eligibility_gap"], "no_authenticated_capture_evidence")
    serialized = json.dumps(report, sort_keys=True).casefold()
    c.true(all(token not in serialized for token in (
        "path", "filename", "patient", "subject", "/users/",
    )))

    video, manifest, timeline = _payloads(include_optional=False)
    changed = json.loads(manifest)
    changed["video_sha256"] = "0" * 64
    c.raises(
        lambda: audit_authenticated_capture_payloads((
            (video, json.dumps(changed).encode("utf-8"), timeline),
        )),
        ValueError,
        "digest drift fails the entire audit rather than becoming an unknown row",
    )


if __name__ == "__main__":
    run_all("test_audit_mayo_faces_script_variants", dict(globals()))
