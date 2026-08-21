"""Transactional PalsyNet Action-Aligned cache contracts."""
from __future__ import annotations

import os
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _testlib import Check, run_all  # noqa: E402
from scripts.build_palsynet_action_aligned_v1 import (  # noqa: E402
    ACTION_CACHE_SCHEMA,
    ActionAlignedExtraction,
    extract_action_aligned_source,
    enumerate_reviewed_sources,
    load_action_aligned_cache,
    select_development_sources,
    write_action_aligned_cache,
)
from scripts.build_palsynet_v2_windows import (  # noqa: E402
    IdentityBinding,
    SourceVideo,
)
from src.datasets.dynamic_landmark import DYNAMIC_FEATURE_NAMES  # noqa: E402
from src.preprocessing.action_aligned_110d import ACTION_SLOT_ORDER  # noqa: E402


class _Capture:
    def __init__(self, _path: str, *, frame_count: int = 640):
        self.frame_count = frame_count
        self.position = 0

    def isOpened(self):
        return True

    def get(self, prop):
        return {
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: float(self.frame_count),
            cv2.CAP_PROP_FRAME_WIDTH: 300.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 300.0,
            cv2.CAP_PROP_POS_FRAMES: float(self.position),
        }.get(prop, 0.0)

    def set(self, prop, value):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self.position = int(value)
            return True
        return False

    def read(self):
        if self.position >= self.frame_count:
            return False, None
        frame = np.full((300, 300, 3), self.position % 251, dtype=np.uint8)
        frame[0, 0, 0] = self.position // 251
        frame[0, 0, 1] = self.position % 251
        self.position += 1
        return True, frame

    def release(self):
        pass


class _Extractor:
    def extract_frame_with_nuisance(self, frame):
        index = int(frame[0, 0, 0]) * 251 + int(frame[0, 0, 1])
        values = np.zeros(95, dtype=np.float32)
        peak_rows = {
            50: (3, 4, 5),
            100: (9, 10),
            150: (7, 8, 19, 20),
            200: (44, 45),
            250: (38,),
            300: (25, 34, 35),
            400: (44, 45),
        }
        if index in peak_rows:
            values[list(peak_rows[index])] = 1.0
        values[72:] = np.float32(index / 1000.0)
        return values, {"face_scale": 0.1, "eye_line_roll_degrees": 0.0}


def _source(path: Path, *, label: str = "affected") -> SourceVideo:
    binding = IdentityBinding(
        source_sha256="a" * 64,
        recording_id="rec_" + "b" * 64,
        group_id="grp_" + "c" * 64,
        label=label,
        identity_status="reviewed",
        claim_unit="person_held_out",
    )
    return SourceVideo(path=path, source_sha256="a" * 64, binding=binding)


def test_extraction_scans_label_free_and_emits_seven_complete_windows(c: Check):
    source = _source(Path("unused.mp4"))
    result = extract_action_aligned_source(
        source,
        _Extractor(),
        capture_factory=lambda path: _Capture(path),
    )
    c.eq(result.features.shape, (7, 32, 95))
    c.eq(result.valid_mask.shape, (7, 32))
    c.eq(result.action_slots, ACTION_SLOT_ORDER)
    c.eq(result.source_frame_indices[:, 0].tolist(), [34, 84, 134, 184, 234, 284, 384])
    c.true(bool(result.valid_mask.all()))
    c.eq(result.binding.recording_id, source.binding.recording_id)
    c.eq(result.binding.label, "affected")


def test_reviewed_sources_join_only_by_content_hash(c: Check):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        rows = []
        for label, count in (("affected", 27), ("unaffected", 22)):
            (root / label).mkdir()
            for number in range(count):
                path = root / label / f"misleading-{number}.mp4"
                path.write_bytes(f"{label}:{number}".encode())
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                absolute = number if label == "affected" else 27 + number
                rows.append({
                    "adjudication_evidence_sha256": None,
                    "adjudication_outcome": "none",
                    "claim_unit": "person_held_out",
                    "group_id": "grp_" + f"{absolute:064x}",
                    "identity_status": "reviewed",
                    "label": label,
                    "recording_id": "rec_" + f"{absolute:064x}",
                    "source_label": label,
                    "source_sha256": digest,
                    "training_eligible": True,
                })
        manifest = root / "reviewed.json"
        manifest.write_text(json.dumps({
            "schema_version": "palsynet_identity_reviewed_v1",
            "dataset": "PalsyNet",
            "claim_unit": "person_held_out",
            "identity_review": {
                "status": "reviewed",
                "label_blinded": True,
                "exhaustive_pair_review": True,
                "uncertainties_resolved": True,
            },
            "counts": {
                "total_recordings": 49,
                "reviewed_groups": 49,
                "eligible_recordings": 49,
                "eligible_groups": 49,
                "excluded_recordings": 0,
                "excluded_groups": 0,
            },
            "fingerprints": {
                "contact_inventory_sha256": "1" * 64,
                "cross_label_adjudication_sha256": "2" * 64,
                "generated_manifest_sha256": "3" * 64,
                "review_ledger_sha256": "4" * 64,
                "reviewer_evidence_sha256": "5" * 64,
                "source_collection_sha256": "6" * 64,
            },
            "recordings": rows,
        }))
        sources = enumerate_reviewed_sources(root, manifest)
        c.eq(len(sources), 49)
        c.eq({source.binding.label for source in sources}, {"affected", "unaffected"})
        changed = root / "affected" / "misleading-0.mp4"
        changed.write_bytes(b"changed")
        c.raises(lambda: enumerate_reviewed_sources(root, manifest), ValueError)


def test_split_filter_returns_only_authenticated_development_sources(c: Check):
    sources = tuple(
        SourceVideo(
            path=Path(f"video-{number}.mp4"),
            source_sha256=f"{number:064x}",
            binding=IdentityBinding(
                source_sha256=f"{number:064x}",
                recording_id="rec_" + f"{number:064x}",
                group_id="grp_" + f"{number:064x}",
                label="affected" if number < 27 else "unaffected",
                identity_status="reviewed",
                claim_unit="person_held_out",
            ),
        )
        for number in range(49)
    )
    assignments = [
        {
            "recording_id": source.binding.recording_id,
            "group_id": source.binding.group_id,
            "semantic_group_key_sha256": f"{number + 100:064x}",
            "partition": "development" if number < 39 else "protected",
            "outer_fold": 1 if number < 39 else 0,
            "inner_fold": number % 4 if number < 39 else None,
        }
        for number, source in enumerate(sources)
    ]
    registry = {
        "schema_version": "palsynet_person_split_registry_v1",
        "dataset": "PalsyNet",
        "claim_unit": "person_held_out",
        "identity_status": "reviewed",
        "source_collection_sha256": "1" * 64,
        "reviewed_manifest_sha256": "2" * 64,
        "review_ledger_sha256": "3" * 64,
        "outer_fold_number": 0,
        "protocol": {},
        "counts": {
            "eligible_recordings": 49,
            "eligible_groups": 49,
            "development_recordings": 39,
            "development_groups": 39,
            "protected_recordings": 10,
            "protected_groups": 10,
        },
        "assignments": assignments,
    }
    selected = select_development_sources(sources, registry)
    c.eq(len(selected), 39)
    c.true(all(source.binding.recording_id.endswith(f"{number:064x}")
               for number, source in enumerate(selected)))
    drifted = json.loads(json.dumps(registry))
    drifted["assignments"][0]["group_id"] = "grp_" + "f" * 64
    c.raises(lambda: select_development_sources(sources, drifted), ValueError)


def test_cache_round_trip_is_private_exact_and_no_overwrite(c: Check):
    source = _source(Path("unused.mp4"), label="unaffected")
    result = extract_action_aligned_source(
        source,
        _Extractor(),
        capture_factory=lambda path: _Capture(path),
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "record.npz"
        write_action_aligned_cache(path, result)
        c.eq(os.stat(path).st_mode & 0o777, 0o600)
        loaded = load_action_aligned_cache(path)
        c.eq(loaded.schema_version, ACTION_CACHE_SCHEMA)
        c.eq(loaded.action_slots, ACTION_SLOT_ORDER)
        c.eq(loaded.binding.label, "unaffected")
        c.eq(loaded.feature_names, DYNAMIC_FEATURE_NAMES)
        c.true(np.array_equal(loaded.features, result.features))
        c.raises(lambda: write_action_aligned_cache(path, result), FileExistsError)


def test_cache_loader_fails_closed_on_field_or_identity_drift(c: Check):
    source = _source(Path("unused.mp4"))
    result = extract_action_aligned_source(
        source,
        _Extractor(),
        capture_factory=lambda path: _Capture(path),
    )
    with tempfile.TemporaryDirectory() as directory:
        missing = Path(directory) / "missing.npz"
        np.savez(missing, features=result.features)
        c.raises(lambda: load_action_aligned_cache(missing), ValueError)

        wrong = ActionAlignedExtraction(
            features=result.features,
            valid_mask=result.valid_mask,
            timestamps=result.timestamps,
            source_frame_indices=result.source_frame_indices,
            source_frame_count=result.source_frame_count,
            fps=result.fps,
            binding=IdentityBinding(
                source_sha256=result.binding.source_sha256,
                recording_id="not_opaque",
                group_id=result.binding.group_id,
                label=result.binding.label,
                identity_status=result.binding.identity_status,
                claim_unit=result.binding.claim_unit,
            ),
            action_slots=result.action_slots,
        )
        c.raises(
            lambda: write_action_aligned_cache(Path(directory) / "wrong.npz", wrong),
            ValueError,
        )


if __name__ == "__main__":
    run_all("test_build_palsynet_action_aligned_v1", dict(globals()))
