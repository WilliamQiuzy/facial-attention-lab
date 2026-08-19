"""Contracts for the aggregate-only Mayo action-anchor feasibility audit."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _testlib import Check, run_all  # noqa: E402
import scripts.audit_mayo_action_anchor_feasibility_v1 as audit  # noqa: E402
from scripts.audit_mayo_action_anchor_feasibility_v1 import (  # noqa: E402
    ACTIONS,
    MediaRecord,
    TimingEvent,
    build_aggregate_report,
    build_private_audit_registry,
    evaluate_timing_gate,
    inventory_media,
    reference_events_sha256,
    validate_output_boundaries,
    write_audit_release_pair_no_overwrite,
    write_private_registry_no_overwrite,
)


def test_inventory_is_content_deduplicated_and_probe_is_aggregate(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "patient-a").mkdir()
        (root / "patient-b").mkdir()
        first = root / "patient-a" / "take.mov"
        duplicate = root / "patient-b" / "copy.mov"
        second = root / "patient-b" / "other.mp4"
        first.write_bytes(b"same-video")
        duplicate.write_bytes(b"same-video")
        second.write_bytes(b"different-video")
        (root / "patient-a" / "depth.bin").write_bytes(b"not media")
        probed = []

        def probe(path: Path):
            probed.append(path)
            payload = path.read_bytes()
            return {
                "duration_seconds": 31.25 if payload == b"same-video" else 42.5,
                "has_audio": payload == b"same-video",
            }

        inventory = inventory_media(root, probe=probe)
    c.eq(inventory.source_files, 3)
    c.eq(inventory.unique_contents, 2)
    c.eq(inventory.exact_duplicate_files, 1)
    c.eq(inventory.audio_bearing_source_files, 2)
    c.eq(inventory.audio_free_source_files, 1)
    c.eq(inventory.audio_bearing_unique_contents, 1)
    c.eq(len(probed), 2, "ffprobe runs once per unique content")
    c.true(all(record.path.is_absolute() for record in inventory.records))

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / "take.mov"
        source.write_bytes(b"AAAA")

        def replacing_probe(fd_path: Path):
            source.write_bytes(b"BBBB")
            return {"duration_seconds": 30.0, "has_audio": True}

        c.raises(lambda: inventory_media(root, probe=replacing_probe), ValueError,
                 "hash and ffprobe must consume one stable file identity")

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        first = root / "first.mov"
        second = root / "second.mov"
        first.write_bytes(b"AAAA")
        second.write_bytes(b"AAAA")

        def replace_nonrepresentative(fd_path: Path):
            second.write_bytes(b"BBBB")
            return {"duration_seconds": 30.0, "has_audio": True}

        c.raises(lambda: inventory_media(
            root, probe=replace_nonrepresentative
        ), ValueError, "every duplicate member is revalidated after probe")

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        first = root / "first.mov"
        second = root / "second.mov"
        first.write_bytes(b"AAAA")
        second.write_bytes(b"CCCC")
        first_digest = hashlib.sha256(first.read_bytes()).hexdigest()
        second_digest = hashlib.sha256(second.read_bytes()).hexdigest()
        earlier = first if first_digest < second_digest else second
        probe_count = 0

        def later_probe_mutates_earlier(fd_path: Path):
            nonlocal probe_count
            probe_count += 1
            if probe_count == 2:
                earlier.write_bytes(b"ZZZZ")
            return {"duration_seconds": 30.0, "has_audio": True}

        c.raises(lambda: inventory_media(
            root, probe=later_probe_mutates_earlier
        ), ValueError, "all media members are revalidated after every probe")


def test_registry_selects_exact_smallest_twelve_audio_hashes_and_is_private(c: Check):
    records = tuple(
        MediaRecord(
            path=Path(f"/private/patient-{index}.mov"),
            source_sha256=f"{index:064x}",
            source_file_count=1,
            size_bytes=100 + index,
            duration_seconds=30.0,
            has_audio=index != 0,
        )
        for index in range(14)
    )
    registry, payload = build_private_audit_registry(records)
    expected = [f"{index:064x}" for index in range(1, 13)]
    c.eq(registry["selected_source_sha256"], expected)
    c.eq(len(payload), len(json.dumps(registry, sort_keys=True, indent=2)) + 1)
    c.true(b"/private/" not in payload and b"patient-" not in payload)
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "private-registry.json"
        digest = write_private_registry_no_overwrite(output, payload)
        c.eq(digest, hashlib.sha256(payload).hexdigest())
        c.eq(os.stat(output).st_mode & 0o777, 0o600)
        c.raises(lambda: write_private_registry_no_overwrite(output, payload), FileExistsError)
        c.eq(output.read_bytes(), payload,
             "no-overwrite failure preserves the existing exact registry")
    ambiguous = records + (MediaRecord(
        path=Path("/private/unknown.mov"),
        source_sha256="0" * 64,
        source_file_count=1,
        size_bytes=100,
        duration_seconds=None,
        has_audio=None,
    ),)
    c.raises(lambda: build_private_audit_registry(ambiguous), ValueError,
             "unknown audio metadata makes lexicographic selection ambiguous")


def _annotation_audit_payload(registry_payload: bytes, reference) -> bytes:
    value = {
        "schema_version": "mayo_action_anchor_blinded_reference_audit_v1",
        "registry_sha256": hashlib.sha256(registry_payload).hexdigest(),
        "annotator_count": 2,
        "annotators_blinded": True,
        "adjudication_complete": True,
        "boundary_difference_adjudication_threshold_ms": 500,
        "reference_events_sha256": reference_events_sha256(reference),
        "prompted_flat_manually_verified_count": sum(
            event.prompted_flat and event.prompted_flat_manually_verified
            for event in reference
        ),
    }
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("ascii")


def _perfect_events():
    records = tuple(
        MediaRecord(
            path=Path(f"/private/patient-{index}.mov"),
            source_sha256=f"{index + 100:064x}",
            source_file_count=1,
            size_bytes=100,
            duration_seconds=30.0,
            has_audio=True,
        )
        for index in range(12)
    )
    _, registry_payload = build_private_audit_registry(records)
    registry = json.loads(registry_payload)
    reference = []
    predicted = []
    for recording_slot in range(12):
        source_sha = registry["selected_source_sha256"][recording_slot]
        for action_index, action in enumerate(ACTIONS):
            start = action_index * 4_000
            flat = (recording_slot, action_index) in {(0, 0), (1, 1)}
            event = TimingEvent(
                recording_slot, source_sha, action, start, start + 3_000,
                flat, flat,
            )
            reference.append(event)
            predicted.append(TimingEvent(
                recording_slot, source_sha, action, start, start + 3_000,
                False, False,
            ))
    reference = tuple(reference)
    return (
        registry_payload,
        _annotation_audit_payload(registry_payload, reference),
        reference,
        tuple(predicted),
    )


def test_timing_gate_uses_all_72_events_and_prompted_flat_requirement(c: Check):
    registry_payload, audit_payload, reference, predicted = _perfect_events()
    passed = evaluate_timing_gate(
        registry_payload, audit_payload, reference, predicted
    )
    c.eq(passed.summary["reference_events"], 72)
    c.eq(passed.summary["precision"], 1.0)
    c.eq(passed.summary["recall"], 1.0)
    c.eq(passed.summary["median_temporal_iou"], 1.0)
    c.eq(passed.summary["prompted_flat_attempts"], 2)
    c.eq(passed.summary["eligible"], True)
    failed = evaluate_timing_gate(
        registry_payload, audit_payload, reference, predicted[:-4]
    )
    c.true(failed.summary["recall"] < 0.95)
    c.eq(failed.summary["eligible"], False)
    no_flat = tuple(
        TimingEvent(
            e.recording_slot, e.source_sha256, e.action, e.start_ms, e.end_ms,
            False, False,
        )
        for e in reference
    )
    c.raises(lambda: evaluate_timing_gate(
        registry_payload, audit_payload, no_flat, predicted
    ), ValueError, "reference events cannot drift from blinded audit bytes")
    wrong_source = list(reference)
    first = wrong_source[0]
    wrong_source[0] = TimingEvent(
        first.recording_slot, "f" * 64, first.action, first.start_ms,
        first.end_ms, first.prompted_flat, first.prompted_flat_manually_verified,
    )
    c.raises(lambda: evaluate_timing_gate(
        registry_payload, audit_payload, wrong_source, predicted
    ), ValueError, "72 events must belong to the exact hash-selected recordings")
    swapped = list(predicted)
    left, right = swapped[0], swapped[1]
    swapped[0] = TimingEvent(
        left.recording_slot, left.source_sha256, right.action,
        left.start_ms, left.end_ms, False, False,
    )
    swapped[1] = TimingEvent(
        right.recording_slot, right.source_sha256, left.action,
        right.start_ms, right.end_ms, False, False,
    )
    wrong_labels = evaluate_timing_gate(
        registry_payload, audit_payload, reference, swapped
    )
    c.eq(wrong_labels.summary["matched_events"], 70)
    c.eq(wrong_labels.summary["precision"], 70 / 72)
    c.eq(wrong_labels.summary["recall"], 70 / 72)


def test_public_report_has_no_paths_hashes_or_mayo_predictions(c: Check):
    records = tuple(
        MediaRecord(
            path=Path(f"/Users/private/patient-{index}.mov"),
            source_sha256=f"{index + 100:064x}",
            source_file_count=1,
            size_bytes=100,
            duration_seconds=30.0,
            has_audio=True,
        )
        for index in range(12)
    )
    registry, payload = build_private_audit_registry(records)
    report = build_aggregate_report(
        source_files=12,
        records=records,
        exact_duplicate_files=0,
        sidecar_counts={
            "capture_event_log": 0,
            "audio_forced_alignment": 0,
            "blinded_manual": 0,
        },
        registry_sha256=hashlib.sha256(payload).hexdigest(),
        timing_gate=None,
    )
    encoded = json.dumps(report, sort_keys=True)
    c.eq(report["timing_gate"]["eligible"], False)
    c.eq(report["timing_gate"]["reference_events"], 0)
    c.eq(report["scoring"]["mayo_action_expert_predictions"], 0)
    c.eq(report["scoring"]["mayo_accuracy_defined"], False)
    c.eq(report["audit_registry"]["selected_recordings"], 12)
    c.true(registry["selected_source_sha256"][0] not in encoded)
    for forbidden in ("/Users/", "patient-", ".mov", "source_sha256"):
        c.true(forbidden not in encoded, f"public report excludes {forbidden}")
    c.raises(lambda: build_aggregate_report(
        source_files=12,
        records=records,
        exact_duplicate_files=0,
        sidecar_counts={
            "capture_event_log": 0,
            "audio_forced_alignment": 0,
            "blinded_manual": 0,
        },
        registry_sha256=hashlib.sha256(payload).hexdigest(),
        timing_gate={"eligible": True, "reference_events": 72},
    ), ValueError, "an arbitrary caller dictionary cannot forge eligibility")


def test_output_boundaries_and_pair_publication_fail_closed(c: Check):
    with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
        root = Path(temporary)
        media = root / "media"
        private_parent = root / "private"
        public_parent = root / "public"
        media.mkdir()
        private_parent.mkdir()
        public_parent.mkdir()
        private = private_parent / "registry.json"
        public = public_parent / "report.json"
        validate_output_boundaries(media, private, public)
        c.raises(lambda: validate_output_boundaries(
            media, media / "mutated.json", public
        ), ValueError)
        (media.parent / "detour").mkdir()
        noncanonical_media = media.parent / "detour" / ".." / media.name
        c.raises(lambda: validate_output_boundaries(
            noncanonical_media, media / "mutated.json", public
        ), ValueError, "noncanonical media roots cannot bypass containment")
        c.raises(lambda: validate_output_boundaries(media, public, public), ValueError)
        symlink = root / "linked"
        symlink.symlink_to(public_parent, target_is_directory=True)
        c.raises(lambda: validate_output_boundaries(
            media, private, symlink / "report.json"
        ), ValueError)

        original_link = audit.os.link
        calls = []

        def fail_second_link(*args, **kwargs):
            calls.append((args, kwargs))
            if len(calls) == 2:
                raise OSError("injected second publication failure")
            return original_link(*args, **kwargs)

        audit.os.link = fail_second_link
        try:
            c.raises(lambda: write_audit_release_pair_no_overwrite(
                private, b"private\n", public, b"public\n"
            ), OSError)
        finally:
            audit.os.link = original_link
        c.true(not private.exists() and not public.exists(),
               "pair publication failure leaves no half release")
        original_write = audit.os.write
        writes = []

        def fail_short_write(descriptor, payload):
            writes.append(bytes(payload))
            if len(writes) == 2:
                raise OSError("injected staged write failure")
            return original_write(descriptor, payload[:2])

        audit.os.write = fail_short_write
        try:
            c.raises(lambda: write_private_registry_no_overwrite(
                private, b"long-private-payload\n"
            ), OSError)
        finally:
            audit.os.write = original_write
        c.true(not private.exists())
        c.eq([path.name for path in private_parent.iterdir()], [],
             "failed staged write leaves no partial final or temp artifact")

        links = []

        def replace_first_final(*args, **kwargs):
            result = original_link(*args, **kwargs)
            links.append((args, kwargs))
            if len(links) == 1:
                private.unlink()
                private.write_bytes(b"tampered")
            return result

        audit.os.link = replace_first_final
        try:
            c.raises(lambda: write_audit_release_pair_no_overwrite(
                private, b"private\n", public, b"public\n"
            ), ValueError, "final path replacement cannot return committed hashes")
        finally:
            audit.os.link = original_link
        c.true(not public.exists(), "matching published peer rolls back on tamper")
        c.eq(private.read_bytes(), b"tampered",
             "identity-aware rollback does not delete an attacker replacement")


if __name__ == "__main__":
    run_all("test_mayo_action_anchor_feasibility_v1", dict(globals()))
