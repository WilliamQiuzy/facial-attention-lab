#!/usr/bin/env python3
"""Classify exact authenticated FACES payloads into aggregate 1–7/1–8 counts.

This module intentionally performs no inference and serializes no identifiers,
paths, hashes, or per-recording rows. Variant assignment comes only from the
SHA-bound external timeline and manifest, never from visible motion.
"""
from __future__ import annotations

from typing import Sequence

from src.preprocessing.faces_shared_v9_pipeline import parse_capture_evidence


_TIMING_SOURCES = (
    "audio_forced_alignment",
    "blinded_manual",
    "capture_event_log",
)


def audit_authenticated_capture_payloads(
    payloads: Sequence[tuple[bytes, bytes, bytes]],
) -> dict[str, object]:
    """Return a closed aggregate report after exact-byte evidence validation."""
    if (
        not isinstance(payloads, Sequence)
        or isinstance(payloads, (str, bytes, bytearray))
    ):
        raise ValueError("capture payloads must be a bounded sequence")
    authenticated: dict[str, tuple[bytes, bytes, int, str]] = {}
    duplicate_payloads = 0
    for index, item in enumerate(payloads):
        if (
            type(item) is not tuple
            or len(item) != 3
            or any(type(value) is not bytes for value in item)
        ):
            raise ValueError(f"capture payload {index} differs from the exact-byte contract")
        video, manifest, timeline = item
        evidence = parse_capture_evidence(video, manifest, timeline)
        steps = len(evidence.timeline.actions)
        source = evidence.timeline.timing_source.value
        prior = authenticated.get(evidence.video_sha256)
        if prior is not None:
            if prior != (manifest, timeline, steps, source):
                raise ValueError("one exact recording has ambiguous script evidence")
            duplicate_payloads += 1
            continue
        authenticated[evidence.video_sha256] = (manifest, timeline, steps, source)

    timing_counts = {source: 0 for source in _TIMING_SOURCES}
    seven = 0
    eight = 0
    for _manifest, _timeline, steps, source in authenticated.values():
        if steps == 7:
            seven += 1
        elif steps == 8:
            eight += 1
        else:  # parse_capture_evidence should make this unreachable.
            raise ValueError("authenticated FACES evidence has an unsupported variant")
        timing_counts[source] += 1

    unique = len(authenticated)
    return {
        "schema_version": "mayo_faces_script_variant_audit_v1",
        "authenticated_payloads": len(payloads),
        "unique_recordings": unique,
        "exact_duplicate_payloads": duplicate_payloads,
        "seven_step_recordings": seven,
        "eight_step_recordings": eight,
        "timing_sources": timing_counts,
        "movement_magnitude_used_for_variant_assignment": False,
        "model_predictions": 0,
        "eligibility_gap": None if unique else "no_authenticated_capture_evidence",
    }


__all__ = ["audit_authenticated_capture_payloads"]
