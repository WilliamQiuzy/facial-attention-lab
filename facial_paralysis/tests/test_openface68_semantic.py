"""Contracts for source-neutral semantic23 and the RAVDESS OpenFace adapter.

All fixtures are synthetic.  The production RAVDESS corpus is audited by the
preparation script, but this test module never reads or writes that corpus.
"""
from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import os
import stat
import sys
import tempfile
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.prepare_ravdess_semantic23 as prep  # noqa: E402
from scripts.prepare_ravdess_semantic23 import (  # noqa: E402
    RAVDESS_ARCHIVE_RELATIVE_PATH,
    RavdessInventoryExpectation,
    audit_ravdess_inventory,
    build_generation_from_audited_sources,
    opaque_actor_id,
    opaque_trial_id,
    parse_openface_csv,
)
from src.preprocessing.openface68_semantic import (  # noqa: E402
    OPENFACE68_ADAPTER_METADATA,
    OPENFACE68_MIDLINE,
    OPENFACE68_MOUTH_BOTTOM,
    OPENFACE68_MOUTH_TOP,
    OPENFACE68_REQUIRED_INDICES,
    OPENFACE68_SIDE_A_BROW,
    OPENFACE68_SIDE_A_CORNER,
    OPENFACE68_SIDE_A_EYE_RING,
    OPENFACE68_SIDE_A_LOWER,
    OPENFACE68_SIDE_A_UPPER,
    OPENFACE68_SIDE_B_BROW,
    OPENFACE68_SIDE_B_CORNER,
    OPENFACE68_SIDE_B_EYE_RING,
    OPENFACE68_SIDE_B_LOWER,
    OPENFACE68_SIDE_B_UPPER,
    openface68_to_semantic23,
)
from src.preprocessing.semantic_landmarks import (  # noqa: E402
    CLINICAL23_V2_ADAPTER_METADATA,
    SEMANTIC23_DEFINITIONS,
    SEMANTIC23_FEATURE_NAMES,
    SEMANTIC23_SCHEMA,
    clinical23_v2_to_semantic23,
)
from _testlib import Check, run_all  # noqa: E402


EXPECTED_NAMES = (
    "fissure_h_side_a", "fissure_h_side_b", "fissure_h_absdiff",
    "fissure_h_side_a_minus_side_b",
    "fissure_w_side_a", "fissure_w_side_b", "fissure_w_absdiff",
    "eye_measure_side_a", "eye_measure_side_b", "eye_measure_absdiff",
    "brow_h_side_a", "brow_h_side_b", "brow_h_absdiff",
    "brow_h_side_a_minus_side_b",
    "corner_y_side_a", "corner_y_side_b", "corner_y_absdiff",
    "corner_y_side_a_minus_side_b",
    "corner_x_side_a", "corner_x_side_b", "corner_x_absdiff",
    "mouth_width", "mouth_open",
)

TEST_ID_KEY = b"k" * 32


def _face() -> np.ndarray:
    """Return a symmetric synthetic OpenFace-68 face in pixel coordinates."""
    p = np.full((68, 2), (50.0, 50.0), dtype=np.float32)
    eye_a = {
        36: (30, 40), 37: (32, 38), 38: (38, 38),
        39: (40, 40), 40: (38, 42), 41: (32, 42),
    }
    eye_b = {
        42: (60, 40), 43: (62, 38), 44: (68, 38),
        45: (70, 40), 46: (68, 42), 47: (62, 42),
    }
    for idx, xy in {**eye_a, **eye_b}.items():
        p[idx] = xy
    for idx, x in zip(range(17, 22), np.linspace(30, 40, 5)):
        p[idx] = (x, 30)
    for idx, x in zip(range(22, 27), np.linspace(60, 70, 5)):
        p[idx] = (x, 30)
    for i, idx in enumerate(OPENFACE68_MIDLINE):
        p[idx] = (50, 25 + 4 * i)
    p[48] = (40, 70)
    p[54] = (60, 70)
    p[62] = (50, 68)
    p[66] = (50, 72)
    return p


def _rotate(points: np.ndarray, angle: float) -> np.ndarray:
    out = points.copy()
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.asarray(((c, -s), (s, c)), dtype=np.float32)
    out[:] = (out - np.asarray((50, 50), np.float32)) @ rotation.T + 50
    return out


def _mirror_and_swap(points: np.ndarray) -> np.ndarray:
    """Horizontal reflection with exact OpenFace left/right topology swap."""
    out = points.copy()
    out[:, 0] = 100 - points[:, 0]
    pairs = (
        (36, 45), (37, 44), (38, 43), (39, 42), (40, 47), (41, 46),
        (17, 26), (18, 25), (19, 24), (20, 23), (21, 22),
        (48, 54), (49, 53), (50, 52), (55, 59), (56, 58),
        (60, 64), (61, 63), (65, 67),
    )
    for a, b in pairs:
        out[a] = (100 - points[b, 0], points[b, 1])
        out[b] = (100 - points[a, 0], points[a, 1])
    return out


def _csv_header() -> list[str]:
    return ["frame", "timestamp", "confidence", *[f"x_{i}" for i in range(68)],
            *[f"y_{i}" for i in range(68)]]


def _csv_row(frame: int, timestamp: float, confidence: float,
             points: np.ndarray) -> list[object]:
    return [frame, timestamp, confidence, *points[:, 0], *points[:, 1]]


def _write_csv(path: Path, rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(_csv_header())
        writer.writerows(rows)


def _synthetic_tree(
    root: Path, *, duplicate_first_frame: bool = False
) -> tuple[RavdessInventoryExpectation, list[Path]]:
    first = root / "extracted" / "01-01-01-01-01-01-01.csv"
    second = root / "extracted" / "01-01-01-01-01-01-02.csv"
    _write_csv(first, [
        _csv_row(1, 0.000, 0.99, _face()),
        _csv_row(1 if duplicate_first_frame else 2, 0.033, 0.50, _face()),
    ])
    _write_csv(second, [_csv_row(1, 0.000, 0.80, _face())])
    archive = root / RAVDESS_ARCHIVE_RELATIVE_PATH
    archive.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        handle.write(first, arcname=first.name)
        handle.write(second, arcname=second.name)
    header_bytes = (",".join(_csv_header())).encode("utf-8")
    expected = RavdessInventoryExpectation(
        archive_size=archive.stat().st_size,
        archive_md5=hashlib.md5(archive.read_bytes()).hexdigest(),  # noqa: S324
        csv_files=2,
        actors=2,
        frames=3,
        header_sha256=hashlib.sha256(header_bytes).hexdigest(),
        empty_trials=0,
        repeated_headers=0,
    )
    return expected, [first, second]


def _zip_member_bytes(archive: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(archive, "r") as handle:
        return {name: handle.read(name) for name in sorted(handle.namelist())}


def _write_zip_bytes(destination: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for name, payload in sorted(members.items()):
            handle.writestr(name, payload)


def _transient_zipfile_path_swap(
    archive_path: Path, replacement_path: Path
):
    """Return a ZipFile wrapper that swaps only the archive's lexical path."""
    original_zipfile = prep.zipfile.ZipFile
    parked_original = archive_path.with_name(".parked-original.zip")

    def wrapped(file, *args, **kwargs):
        wrapped.opened_arguments.append(file)
        os.replace(archive_path, parked_original)
        os.replace(replacement_path, archive_path)
        try:
            opened = original_zipfile(file, *args, **kwargs)
        finally:
            os.replace(archive_path, replacement_path)
            os.replace(parked_original, archive_path)
        return opened

    wrapped.opened_arguments = []
    return original_zipfile, wrapped


def _assert_lock_reacquirable(c: Check, lock_path: Path, message: str) -> None:
    descriptor = os.open(lock_path, os.O_RDWR | os.O_NOFOLLOW)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)
    c.eq(stat.S_IMODE(lock_path.stat().st_mode), 0o600, message)


def test_semantic23_schema_is_exact_and_self_describing(c: Check):
    c.eq(SEMANTIC23_SCHEMA, "semantic23_v1", "generic schema version")
    c.eq(SEMANTIC23_FEATURE_NAMES, EXPECTED_NAMES, "source-neutral feature order")
    c.eq(len(SEMANTIC23_DEFINITIONS), 23, "one definition per feature")
    c.eq(tuple(item.name for item in SEMANTIC23_DEFINITIONS), EXPECTED_NAMES,
         "definition order matches vector order")
    for item in SEMANTIC23_DEFINITIONS:
        c.true(item.unit in {"interocular_distance", "interocular_distance_squared"},
               f"explicit unit for {item.name}")
        c.true(bool(item.definition), f"definition for {item.name}")
        c.true(item.sign in {"nonnegative", "side_a_minus_side_b", "eye_line_relative"},
               f"sign convention for {item.name}")


def test_clinical23_adapter_is_explicit_not_length_inference(c: Check):
    source = np.arange(23, dtype=np.float32)
    got = clinical23_v2_to_semantic23(source)
    c.true(bool(np.array_equal(got, source)), "clinical V2 reorder is currently identity")
    c.true(got is not source, "adapter returns an owned array")
    c.eq(CLINICAL23_V2_ADAPTER_METADATA["source_schema"], "clinical23_v2",
         "source schema is explicit")
    c.eq(CLINICAL23_V2_ADAPTER_METADATA["target_schema"], SEMANTIC23_SCHEMA,
         "target schema is explicit")
    c.eq(CLINICAL23_V2_ADAPTER_METADATA["eye_measure"], "height_times_width",
         "eye measure is not polygon area")
    c.raises(lambda: clinical23_v2_to_semantic23(np.zeros(22, np.float32)),
             ValueError, "same contract cannot be inferred from arbitrary length")
    bad = source.copy()
    bad[3] = np.nan
    c.raises(lambda: clinical23_v2_to_semantic23(bad), ValueError,
             "non-finite source vector fails closed")


def test_openface_topology_schema_eye_measure_and_dtype(c: Check):
    c.eq(OPENFACE68_SIDE_A_EYE_RING, (36, 37, 38, 39, 40, 41),
         "exact OpenFace side-A eye ring")
    c.eq(OPENFACE68_SIDE_B_EYE_RING, (42, 43, 44, 45, 46, 47),
         "exact OpenFace side-B eye ring")
    c.eq(OPENFACE68_SIDE_A_UPPER, (37, 38), "exact side-A upper lid")
    c.eq(OPENFACE68_SIDE_A_LOWER, (40, 41), "exact side-A lower lid")
    c.eq(OPENFACE68_SIDE_B_UPPER, (43, 44), "exact side-B upper lid")
    c.eq(OPENFACE68_SIDE_B_LOWER, (46, 47), "exact side-B lower lid")
    c.eq(OPENFACE68_SIDE_A_BROW, (17, 18, 19, 20, 21), "exact side-A brow")
    c.eq(OPENFACE68_SIDE_B_BROW, (22, 23, 24, 25, 26), "exact side-B brow")
    c.eq((OPENFACE68_SIDE_A_CORNER, OPENFACE68_SIDE_B_CORNER), (48, 54),
         "exact commissures")
    c.eq((OPENFACE68_MOUTH_TOP, OPENFACE68_MOUTH_BOTTOM), (62, 66),
         "exact inner central lip points")
    c.eq(OPENFACE68_REQUIRED_INDICES, tuple(sorted(set(OPENFACE68_REQUIRED_INDICES))),
         "required topology is exact, sorted and unique")
    c.true(all(0 <= i < 68 for i in OPENFACE68_REQUIRED_INDICES),
           "topology uses exact 0-based OpenFace-68 indices")
    vector = openface68_to_semantic23(_face())
    by_name = dict(zip(SEMANTIC23_FEATURE_NAMES, vector))
    c.eq(vector.shape, (23,), "vector shape")
    c.eq(vector.dtype, np.float32, "vector dtype")
    c.true(bool(np.isfinite(vector).all()), "finite vector")
    for side in ("a", "b"):
        want = by_name[f"fissure_h_side_{side}"] * by_name[f"fissure_w_side_{side}"]
        c.true(abs(float(by_name[f"eye_measure_side_{side}"] - want)) < 1e-7,
               f"side {side} eye measure equals height times width")
    c.eq(OPENFACE68_ADAPTER_METADATA["source_topology"], "openface_68_2d",
         "source topology metadata")
    c.eq(OPENFACE68_ADAPTER_METADATA["scale_normalization"], "interocular_distance",
         "source scaling metadata")


def test_openface_translation_scale_and_roll_invariance(c: Check):
    base = _face()
    reference = openface68_to_semantic23(base)
    translated = base + np.asarray((17.5, -9.25), np.float32)
    scaled = base * np.float32(3.7)
    rolled = _rotate(base, 0.31)
    for name, transformed in (("translation", translated), ("scale", scaled),
                              ("roll", rolled)):
        c.true(bool(np.allclose(reference, openface68_to_semantic23(transformed),
                                atol=2e-5)), f"{name} invariant")


def test_openface_symmetry_signed_and_absolute_perturbations(c: Check):
    symmetric = dict(zip(SEMANTIC23_FEATURE_NAMES,
                         openface68_to_semantic23(_face())))
    for name in ("fissure_h_absdiff", "fissure_h_side_a_minus_side_b",
                 "fissure_w_absdiff", "eye_measure_absdiff", "brow_h_absdiff",
                 "brow_h_side_a_minus_side_b", "corner_y_absdiff",
                 "corner_y_side_a_minus_side_b", "corner_x_absdiff"):
        c.true(abs(float(symmetric[name])) < 1e-6, f"symmetric {name} is zero")

    changed = _face()
    changed[54, 1] += 4
    original = dict(zip(SEMANTIC23_FEATURE_NAMES,
                        openface68_to_semantic23(changed)))
    mirrored = dict(zip(SEMANTIC23_FEATURE_NAMES,
                        openface68_to_semantic23(_mirror_and_swap(changed))))
    c.true(float(original["corner_y_absdiff"]) > 0, "absolute asymmetry grows")
    c.true(float(original["corner_y_side_a_minus_side_b"]) < 0,
           "side-A minus side-B sign is retained")
    for name in ("fissure_h_absdiff", "fissure_w_absdiff", "eye_measure_absdiff",
                 "brow_h_absdiff", "corner_y_absdiff", "corner_x_absdiff"):
        c.true(abs(float(original[name] - mirrored[name])) < 2e-5,
               f"mirror keeps absolute feature {name}")
    for name in ("fissure_h_side_a_minus_side_b", "brow_h_side_a_minus_side_b",
                 "corner_y_side_a_minus_side_b"):
        c.true(abs(float(original[name] + mirrored[name])) < 2e-5,
               f"mirror changes sign for {name}")


def test_openface_malformed_geometry_fails_closed(c: Check):
    c.raises(lambda: openface68_to_semantic23(np.zeros((67, 2), np.float32)),
             ValueError, "requires exact 68-point topology")
    c.raises(lambda: openface68_to_semantic23(np.zeros((68, 3), np.float32)),
             ValueError, "requires exact 2D coordinates")
    bad = _face()
    bad[36, 0] = np.nan
    c.raises(lambda: openface68_to_semantic23(bad), ValueError,
             "non-finite required coordinate")
    unused_nan = _face()
    unused_nan[0, 0] = np.nan
    c.true(bool(np.isfinite(openface68_to_semantic23(unused_nan)).all()),
           "only coordinates required by the semantic adapter are mandatory")
    degenerate = _face()
    degenerate[42:48] = degenerate[36:42]
    c.raises(lambda: openface68_to_semantic23(degenerate), ValueError,
             "degenerate interocular distance")


def test_csv_parser_keeps_timestamps_and_detector_gaps(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "01-01-01-01-01-01-01.csv"
        invalid_geometry = _face()
        invalid_geometry[36, 0] = np.nan
        _write_csv(path, [
            _csv_row(10, 1.000, 0.80, _face()),
            _csv_row(11, 1.033, 0.79, _face()),
            _csv_row(12, 1.066, 0.99, invalid_geometry),
            _csv_row(13, 1.099, 0.95, _face()),
        ])
        trial = parse_openface_csv(path)
    c.eq(trial.features.shape, (4, 23), "no frame is dropped")
    c.true(bool(np.array_equal(trial.frame_indices, [10, 11, 12, 13])),
           "source frame indices preserved")
    c.true(bool(np.allclose(trial.timestamps, [1.000, 1.033, 1.066, 1.099])),
           "source timestamps preserved")
    c.true(bool(np.array_equal(trial.valid_mask, [True, False, False, True])),
           "threshold is inclusive and malformed geometry remains a gap")
    c.true(bool(np.all(trial.features[~trial.valid_mask] == 0)),
           "masked gaps use neutral storage, never interpolation")

    for confidence in (-0.01, 1.01):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "01-01-01-01-01-01-01.csv"
            _write_csv(path, [_csv_row(1, 0.0, confidence, _face())])
            c.raises(lambda: parse_openface_csv(path), ValueError,
                     "detector confidence outside [0, 1] fails closed")


def test_opaque_provenance_is_stable_and_does_not_expose_names(c: Check):
    actor = opaque_actor_id("07", key=TEST_ID_KEY)
    same_actor = opaque_actor_id("07", key=TEST_ID_KEY)
    other_actor = opaque_actor_id("08", key=TEST_ID_KEY)
    other_key_actor = opaque_actor_id("07", key=b"z" * 32)
    trial = opaque_trial_id("source-sha256-sentinel", key=TEST_ID_KEY)
    c.eq(actor, same_actor, "actor ID stable")
    c.true(actor != other_actor, "actors remain distinguishable")
    c.true(actor != other_key_actor, "private HMAC key prevents public enumeration")
    c.true(actor.startswith("actor_") and len(actor) == 22, "bounded opaque actor ID")
    c.true(trial.startswith("trial_") and len(trial) == 22, "bounded opaque trial ID")
    c.true("07" not in actor and "source" not in trial, "raw provenance not exposed")
    c.raises(lambda: opaque_actor_id("07", key=b"short"), ValueError,
             "HMAC pseudonyms require a private 256-bit key")

    with tempfile.TemporaryDirectory() as temporary:
        key_path = Path(temporary) / "private-id.key"
        first = prep.load_or_create_private_id_key(key_path)
        second = prep.load_or_create_private_id_key(key_path)
        c.eq(first, second, "private ID key is stable across runs")
        c.eq(len(first), 32, "private ID key has 256 bits")
        mode = stat.S_IMODE(key_path.stat().st_mode)
        c.eq(mode, 0o600, "private ID key is owner-only")


def test_archive_audit_uses_one_nofollow_fd_across_transient_path_swap(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        data_root = Path(temporary) / "source"
        expected, _ = _synthetic_tree(data_root)
        archive_path = data_root / RAVDESS_ARCHIVE_RELATIVE_PATH
        original_members = _zip_member_bytes(archive_path)
        replacement_members = dict(original_members)
        first_name = sorted(replacement_members)[0]
        lines = replacement_members[first_name].splitlines(keepends=True)
        replacement_members[first_name] = lines[0] + lines[0] + b"".join(lines[1:])
        replacement_path = archive_path.with_name("transient-replacement.zip")
        _write_zip_bytes(replacement_path, replacement_members)

        archive_opens: list[int] = []
        original_open = prep.os.open
        original_zipfile, wrapped_zipfile = _transient_zipfile_path_swap(
            archive_path, replacement_path
        )

        def tracked_open(path, flags, *args, **kwargs):
            if Path(path).resolve() == archive_path.resolve():
                archive_opens.append(flags)
            return original_open(path, flags, *args, **kwargs)

        prep.os.open = tracked_open
        prep.zipfile.ZipFile = wrapped_zipfile
        try:
            c.raises(lambda: audit_ravdess_inventory(
                data_root, expectation=expected
            ), ValueError, "transient archive path replacement fails closed")
        finally:
            prep.zipfile.ZipFile = original_zipfile
            prep.os.open = original_open

        c.eq(len(archive_opens), 1, "audit opens the archive exactly once")
        c.true(bool(archive_opens[0] & os.O_NOFOLLOW),
               "audit archive fd rejects a final symlink")
        c.true(hasattr(wrapped_zipfile.opened_arguments[0], "fileno"),
               "ZipFile receives the already verified archive file object")


def test_audited_inventory_retains_deterministic_member_digests(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        data_root = Path(temporary) / "source"
        expected, _ = _synthetic_tree(data_root)
        archive_path = data_root / RAVDESS_ARCHIVE_RELATIVE_PATH
        original_members = _zip_member_bytes(archive_path)
        inventory = audit_ravdess_inventory(data_root, expectation=expected)
        c.eq(
            inventory.member_sha256,
            {name: hashlib.sha256(payload).hexdigest()
             for name, payload in sorted(original_members.items())},
            "audited inventory retains deterministic member byte digests",
        )


def test_generation_uses_one_verified_fd_and_audited_member_digests(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        data_root = base / "source"
        expected, extracted = _synthetic_tree(data_root)
        archive_path = data_root / RAVDESS_ARCHIVE_RELATIVE_PATH
        original_members = _zip_member_bytes(archive_path)
        inventory = audit_ravdess_inventory(data_root, expectation=expected)

        changed = _face()
        changed[54, 1] += 7.0
        changed_csv = base / extracted[0].name
        _write_csv(changed_csv, [
            _csv_row(1, 0.000, 0.99, changed),
            _csv_row(2, 0.033, 0.50, changed),
        ])
        replacement_members = dict(original_members)
        replacement_members[extracted[0].name] = changed_csv.read_bytes()
        replacement_path = archive_path.with_name("transient-replacement.zip")
        _write_zip_bytes(replacement_path, replacement_members)

        archive_opens: list[int] = []
        original_open = prep.os.open
        original_zipfile, wrapped_zipfile = _transient_zipfile_path_swap(
            archive_path, replacement_path
        )

        def tracked_open(path, flags, *args, **kwargs):
            if Path(path).resolve() == archive_path.resolve():
                archive_opens.append(flags)
            return original_open(path, flags, *args, **kwargs)

        prep.os.open = tracked_open
        prep.zipfile.ZipFile = wrapped_zipfile
        output = base / "derived_semantic23"
        try:
            c.raises(lambda: build_generation_from_audited_sources(
                data_root, output, inventory, expectation=expected,
                id_key=TEST_ID_KEY,
            ), ValueError, "transient archive path replacement fails generation")
        finally:
            prep.zipfile.ZipFile = original_zipfile
            prep.os.open = original_open

        c.true(not output.exists(),
               "transient path replacement cannot publish consumed bytes")
        c.eq(len(archive_opens), 1, "generation opens the archive exactly once")
        c.true(bool(archive_opens[0] & os.O_NOFOLLOW),
               "generation archive fd rejects a final symlink")
        c.true(hasattr(wrapped_zipfile.opened_arguments[0], "fileno"),
               "generation ZipFile consumes the already verified file object")


def test_generation_rejects_member_digest_outside_audited_inventory(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        data_root = base / "source"
        expected, _ = _synthetic_tree(data_root)
        inventory = audit_ravdess_inventory(data_root, expectation=expected)
        tampered = dict(inventory.member_sha256)
        tampered[sorted(tampered)[0]] = "0" * 64
        mismatched_inventory = replace(inventory, member_sha256=tampered)
        output = base / "derived_semantic23"
        c.raises(lambda: build_generation_from_audited_sources(
            data_root, output, mismatched_inventory, expectation=expected,
            id_key=TEST_ID_KEY), ValueError,
            "single-read member bytes must match the audited digest map")
        c.true(not output.exists(), "member digest mismatch publishes nothing")


def test_output_paths_reject_lexical_symlink_bypasses(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        data_root = base / "source"
        expected, _ = _synthetic_tree(data_root)
        inventory = audit_ravdess_inventory(data_root, expectation=expected)

        external_parent = base / "external-parent"
        external_parent.mkdir()
        linked_parent = data_root / "linked-parent"
        linked_parent.symlink_to(external_parent, target_is_directory=True)
        escaped_output = linked_parent / "derived_semantic23"
        c.raises(lambda: build_generation_from_audited_sources(
            data_root, escaped_output, inventory, expectation=expected,
            id_key=TEST_ID_KEY), ValueError,
            "symlinked descendant output parent is rejected")
        c.true(not (external_parent / "derived_semantic23").exists(),
               "descendant symlink cannot redirect derived output")

        canonical = data_root / "derived_semantic23"
        external_output = base / "external-output"
        canonical.symlink_to(external_output, target_is_directory=True)
        key_path = data_root / ".must-not-exist.key"
        original = prep.FROZEN_RAVDESS_INVENTORY
        prep.FROZEN_RAVDESS_INVENTORY = expected
        try:
            c.raises(lambda: prep.prepare_ravdess_semantic23(
                data_root, id_key_path=key_path
            ), FileExistsError, "canonical final symlink is rejected lexically")
        finally:
            prep.FROZEN_RAVDESS_INVENTORY = original
        c.true(canonical.is_symlink() and not external_output.exists(),
               "final symlink is neither followed nor replaced")
        c.true(not key_path.exists(),
               "invalid canonical output fails before private key creation")


def test_production_rejects_resolved_but_noncanonical_output_alias(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        data_root = Path(temporary) / "source"
        expected, _ = _synthetic_tree(data_root)
        alias = data_root / "alias"
        alias.symlink_to(data_root, target_is_directory=True)
        key_path = data_root / ".must-not-exist.key"
        original = prep.FROZEN_RAVDESS_INVENTORY
        prep.FROZEN_RAVDESS_INVENTORY = expected
        try:
            c.raises(lambda: prep.prepare_ravdess_semantic23(
                data_root, output_root=alias / "derived_semantic23",
                id_key_path=key_path,
            ), ValueError, "production output must be the exact lexical canonical path")
        finally:
            prep.FROZEN_RAVDESS_INVENTORY = original
        c.true(not key_path.exists(), "noncanonical output fails without key creation")


def test_private_key_read_is_same_fd_nofollow_and_identity_bound(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        key_path = base / "private-id.key"
        key_path.write_bytes(b"a" * 32)
        key_path.chmod(0o600)
        original_open = prep.os.open
        read_flags: list[int] = []

        def tracked_open(path, flags, *args, **kwargs):
            if Path(path) == key_path:
                read_flags.append(flags)
            return original_open(path, flags, *args, **kwargs)

        prep.os.open = tracked_open
        try:
            c.eq(prep.load_or_create_private_id_key(key_path), b"a" * 32,
                 "existing key bytes")
        finally:
            prep.os.open = original_open
        c.eq(len(read_flags), 1, "existing key is opened exactly once")
        c.true(bool(read_flags[0] & os.O_NOFOLLOW),
               "existing key read rejects a final symlink")
        c.eq(read_flags[0] & os.O_ACCMODE, os.O_RDONLY,
             "existing key is read through an O_RDONLY fd")

        replacement = base / "replacement.key"
        replacement.write_bytes(b"b" * 32)
        replacement.chmod(0o600)
        parked = base / "parked.key"
        swapped = False

        def swap_after_open(path, flags, *args, **kwargs):
            nonlocal swapped
            descriptor = original_open(path, flags, *args, **kwargs)
            if Path(path) == key_path and not swapped:
                os.replace(key_path, parked)
                os.replace(replacement, key_path)
                swapped = True
            return descriptor

        prep.os.open = swap_after_open
        try:
            c.raises(lambda: prep.load_or_create_private_id_key(key_path),
                     ValueError, "key lexical identity change after open is rejected")
        finally:
            prep.os.open = original_open


def test_private_key_creation_is_exclusive_nofollow_and_owner_only(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        key_path = Path(temporary) / "private-id.key"
        original_open = prep.os.open
        creation: list[tuple[int, int]] = []

        def tracked_open(path, flags, mode=0o777, *args, **kwargs):
            if Path(path) == key_path:
                creation.append((flags, mode))
            return original_open(path, flags, mode, *args, **kwargs)

        prep.os.open = tracked_open
        try:
            payload = prep.load_or_create_private_id_key(key_path)
        finally:
            prep.os.open = original_open
        c.eq(len(payload), 32, "created key size")
        c.eq(len(creation), 1, "new key is created in one exclusive open")
        flags, mode = creation[0]
        c.true(bool(flags & os.O_EXCL) and bool(flags & os.O_NOFOLLOW),
               "key creation is exclusive and nofollow")
        c.eq(mode, 0o600, "key creation requests exact owner-only mode")
        c.eq(stat.S_IMODE(key_path.stat().st_mode), 0o600,
             "created key remains exact owner-only mode")


def test_generation_is_bound_to_verified_archive_member_bytes(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        data_root = base / "source"
        expected, extracted = _synthetic_tree(data_root)
        inventory = audit_ravdess_inventory(data_root, expectation=expected)
        archive_path = data_root / RAVDESS_ARCHIVE_RELATIVE_PATH
        with zipfile.ZipFile(archive_path, "r") as archive:
            member_bytes = archive.read(extracted[0].name)

        changed = _face()
        changed[54, 1] += 4.0
        _write_csv(extracted[0], [
            _csv_row(1, 0.000, 0.99, changed),
            _csv_row(2, 0.033, 0.50, changed),
        ])
        c.true(hashlib.sha256(extracted[0].read_bytes()).hexdigest()
               != hashlib.sha256(member_bytes).hexdigest(),
               "synthetic extracted copy is value-tampered")

        output = base / "derived_semantic23"
        build_generation_from_audited_sources(
            data_root, output, inventory, expectation=expected,
            id_key=TEST_ID_KEY,
        )
        manifest = json.loads((output / "manifest.json").read_text("utf-8"))
        source_digest = hashlib.sha256(member_bytes).hexdigest()
        expected_trial_id = opaque_trial_id(source_digest, key=TEST_ID_KEY)
        two_frame = next(item for item in manifest["trials"]
                         if item["trial_id"] == expected_trial_id)
        c.eq(two_frame["trial_id"], opaque_trial_id(source_digest, key=TEST_ID_KEY),
             "keyed trial identity is computed from verified ZIP member bytes")
        archive_trial = prep.parse_openface_csv_bytes(
            member_bytes, source_name=extracted[0].name
        )
        with np.load(output / "trials" / f"{two_frame['trial_id']}.npz") as cache:
            c.true(bool(np.array_equal(cache["features"], archive_trial.features)),
                   "cache values come from the verified ZIP, not extracted copy")


def test_public_manifest_never_exposes_audited_raw_member_digests(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        data_root = base / "source"
        expected, _ = _synthetic_tree(data_root)
        inventory = audit_ravdess_inventory(data_root, expectation=expected)
        output = base / "derived_semantic23"
        build_generation_from_audited_sources(
            data_root, output, inventory, expectation=expected,
            id_key=TEST_ID_KEY,
        )
        manifest_text = (output / "manifest.json").read_text("utf-8")
        manifest = json.loads(manifest_text)
        for raw_digest in inventory.member_sha256.values():
            c.true(raw_digest not in manifest_text,
                   "public manifest cannot expose an enumerable raw member digest")
        c.true(all("source_sha256" not in record for record in manifest["trials"]),
               "public trial records omit unkeyed source digests")


def test_public_cache_integrity_ids_are_private_key_scoped(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        data_root = base / "source"
        expected, _ = _synthetic_tree(data_root)
        inventory = audit_ravdess_inventory(data_root, expectation=expected)
        outputs = (base / "derived-a", base / "derived-b")
        keys = (b"a" * 32, b"b" * 32)
        manifests: list[dict] = []

        for output, key in zip(outputs, keys):
            build_generation_from_audited_sources(
                data_root, output, inventory, expectation=expected,
                id_key=key,
            )
            text = (output / "manifest.json").read_text("utf-8")
            manifests.append(json.loads(text))
            raw_cache_digests = {
                hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (output / "trials").glob("*.npz")
            }
            for raw_digest in raw_cache_digests:
                c.true(raw_digest not in text,
                       "public manifest cannot expose a raw cache SHA-256")

        for manifest in manifests:
            c.true(all(set(record) == {
                "trial_id", "actor_id", "cache_integrity_id",
            } for record in manifest["trials"]),
                "public trial records expose only keyed cache integrity IDs")
        first_ids = {
            record["cache_integrity_id"] for record in manifests[0]["trials"]
        }
        second_ids = {
            record["cache_integrity_id"] for record in manifests[1]["trials"]
        }
        c.true(first_ids.isdisjoint(second_ids),
               "different private keys cannot yield intersectable cache fingerprints")


def test_public_manifest_order_is_keyed_not_raw_name_order(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        data_root = base / "source"
        expected, _ = _synthetic_tree(data_root)
        inventory = audit_ravdess_inventory(data_root, expectation=expected)
        output = base / "derived_semantic23"
        build_generation_from_audited_sources(
            data_root, output, inventory, expectation=expected,
            id_key=TEST_ID_KEY,
        )
        manifest = json.loads((output / "manifest.json").read_text("utf-8"))
        public_order = [record["trial_id"] for record in manifest["trials"]]
        raw_name_order = [
            opaque_trial_id(digest, key=TEST_ID_KEY)
            for digest in inventory.member_sha256.values()
        ]
        c.true(public_order != raw_name_order,
               "public record positions cannot preserve raw filename order")
        c.eq(public_order, sorted(public_order),
             "keyed opaque trial ID determines public record order")


def test_public_manifest_keeps_only_aggregate_frame_qc(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        data_root = base / "source"
        expected, _ = _synthetic_tree(data_root)
        inventory = audit_ravdess_inventory(data_root, expectation=expected)
        output = base / "derived_semantic23"
        build_generation_from_audited_sources(
            data_root, output, inventory, expectation=expected,
            id_key=TEST_ID_KEY,
        )
        manifest = json.loads((output / "manifest.json").read_text("utf-8"))
        records = manifest["trials"]
        c.true(all("source_frames" not in record and "valid_frames" not in record
                   for record in records),
               "per-trial frame counts cannot fingerprint public source members")
        c.eq(manifest["quality_control"], {
            "source_frames": 3,
            "valid_frames": 2,
            "invalid_frames": 1,
        }, "frame quality control remains available only in aggregate")
        c.true(all(record["actor_id"].startswith("actor_") for record in records),
               "keyed actor grouping remains available for training splits")


def test_manifest_deidentification_guard_rejects_raw_member_digest(c: Check):
    raw_digest = hashlib.sha256(b"public-archive-member").hexdigest()
    c.raises(lambda: prep._assert_manifest_deidentified(
        json.dumps({"leaked_digest": raw_digest}),
        source_root=Path("/private/raw/ravdess"),
        source_paths=[Path("01-01-01-01-01-01-01.csv")],
        raw_source_sha256s={raw_digest},
    ), ValueError, "aggregate deidentification guard rejects raw source digests")


def test_output_parent_swap_after_validation_fails_closed(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        data_root = base / "source"
        expected, _ = _synthetic_tree(data_root)
        inventory = audit_ravdess_inventory(data_root, expectation=expected)
        safe_parent = data_root / "safe-output-parent"
        safe_parent.mkdir()
        parked_parent = data_root / "parked-safe-output-parent"
        attack_target = base / "attack-target"
        attack_target.mkdir()
        output = safe_parent / "derived_semantic23"

        original_acquire = prep._acquire_output_lock
        swapped = False

        def swap_parent_then_acquire(*args, **kwargs):
            nonlocal swapped
            if not swapped:
                os.replace(safe_parent, parked_parent)
                safe_parent.symlink_to(attack_target, target_is_directory=True)
                swapped = True
            return original_acquire(*args, **kwargs)

        prep._acquire_output_lock = swap_parent_then_acquire
        try:
            c.raises(lambda: build_generation_from_audited_sources(
                data_root, output, inventory, expectation=expected,
                id_key=TEST_ID_KEY,
            ), ValueError, "output parent identity swap after validation fails closed")
        finally:
            prep._acquire_output_lock = original_acquire
            if safe_parent.is_symlink():
                safe_parent.unlink()
            if parked_parent.exists():
                os.replace(parked_parent, safe_parent)

        c.true(not any(attack_target.iterdir()),
               "parent swap creates no lock, staging, or output in attack target")
        c.true(not any("staging" in path.name for path in safe_parent.iterdir()),
               "failed anchored transaction removes sensitive staging")
        c.true(not output.exists(), "parent swap publishes no derived output")


def test_generation_holds_nofollow_output_parent_directory_fd(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        data_root = base / "source"
        expected, _ = _synthetic_tree(data_root)
        inventory = audit_ravdess_inventory(data_root, expectation=expected)
        output = base / "derived_semantic23"
        original_open = prep.os.open
        parent_flags: list[int] = []

        def tracked_open(path, flags, *args, **kwargs):
            if Path(path) == output.parent and kwargs.get("dir_fd") is None:
                parent_flags.append(flags)
            return original_open(path, flags, *args, **kwargs)

        prep.os.open = tracked_open
        try:
            build_generation_from_audited_sources(
                data_root, output, inventory, expectation=expected,
                id_key=TEST_ID_KEY,
            )
        finally:
            prep.os.open = original_open
        c.eq(len(parent_flags), 1, "generation opens the output parent exactly once")
        c.true(bool(parent_flags[0] & os.O_DIRECTORY)
               and bool(parent_flags[0] & os.O_NOFOLLOW),
               "output parent is held with O_DIRECTORY and O_NOFOLLOW")


def test_output_parent_swap_during_staging_fails_closed(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        data_root = base / "source"
        expected, _ = _synthetic_tree(data_root)
        inventory = audit_ravdess_inventory(data_root, expectation=expected)
        safe_parent = data_root / "safe-output-parent"
        safe_parent.mkdir()
        parked_parent = data_root / "parked-safe-output-parent"
        attack_target = base / "attack-target"
        attack_target.mkdir()
        output = safe_parent / "derived_semantic23"

        original_guard = prep._assert_manifest_deidentified
        swapped = False

        def swap_parent_after_staging(*args, **kwargs):
            nonlocal swapped
            result = original_guard(*args, **kwargs)
            if not swapped:
                stage_name = next(
                    path.name for path in safe_parent.iterdir()
                    if "staging" in path.name
                )
                os.replace(safe_parent, parked_parent)
                safe_parent.symlink_to(attack_target, target_is_directory=True)
                (attack_target / stage_name).mkdir()
                swapped = True
            return result

        prep._assert_manifest_deidentified = swap_parent_after_staging
        try:
            c.raises(lambda: build_generation_from_audited_sources(
                data_root, output, inventory, expectation=expected,
                id_key=TEST_ID_KEY,
            ), ValueError, "output parent swap during staging fails closed")
        finally:
            prep._assert_manifest_deidentified = original_guard
            if safe_parent.is_symlink():
                safe_parent.unlink()
            if parked_parent.exists():
                os.replace(parked_parent, safe_parent)

        c.true(not (attack_target / "derived_semantic23").exists(),
               "staging-time swap publishes no attacker output")
        c.true(not any((path / "manifest.json").exists()
                       for path in attack_target.iterdir() if path.is_dir()),
               "staging-time swap leaks no manifest into attacker staging")
        c.true(not any("staging" in path.name for path in safe_parent.iterdir()),
               "staging-time swap cleans the anchored sensitive stage")
        c.true(not output.exists(), "staging-time swap publishes no trusted output")


def test_output_parent_swap_at_publish_fails_closed(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        data_root = base / "source"
        expected, _ = _synthetic_tree(data_root)
        inventory = audit_ravdess_inventory(data_root, expectation=expected)
        safe_parent = data_root / "safe-output-parent"
        safe_parent.mkdir()
        parked_parent = data_root / "parked-safe-output-parent"
        attack_target = base / "attack-target"
        attack_target.mkdir()
        output = safe_parent / "derived_semantic23"

        original_publish = prep._publish_directory_no_replace
        swapped = False

        def swap_parent_then_publish(*args, **kwargs):
            nonlocal swapped
            if not swapped:
                if isinstance(args[0], Path):
                    stage_name = args[0].name
                else:
                    stage_name = str(args[1])
                (attack_target / stage_name).mkdir()
                os.replace(safe_parent, parked_parent)
                safe_parent.symlink_to(attack_target, target_is_directory=True)
                swapped = True
            return original_publish(*args, **kwargs)

        prep._publish_directory_no_replace = swap_parent_then_publish
        try:
            c.raises(lambda: build_generation_from_audited_sources(
                data_root, output, inventory, expectation=expected,
                id_key=TEST_ID_KEY,
            ), ValueError, "output parent swap at publish fails closed")
        finally:
            prep._publish_directory_no_replace = original_publish
            if safe_parent.is_symlink():
                safe_parent.unlink()
            if parked_parent.exists():
                os.replace(parked_parent, safe_parent)

        c.true(not (attack_target / "derived_semantic23").exists(),
               "publish-time swap creates no attacker output")
        c.true(not any("staging" in path.name for path in safe_parent.iterdir()),
               "publish-time swap cleans the anchored stage")
        c.true(not output.exists(), "publish-time swap rolls back trusted output")


def test_output_parent_swap_at_lock_release_rolls_back_generation(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        data_root = base / "source"
        expected, _ = _synthetic_tree(data_root)
        inventory = audit_ravdess_inventory(data_root, expectation=expected)
        safe_parent = data_root / "safe-output-parent"
        safe_parent.mkdir()
        parked_parent = data_root / "parked-safe-output-parent"
        attack_target = base / "attack-target"
        attack_target.mkdir()
        output = safe_parent / "derived_semantic23"

        original_release = prep._release_output_lock
        swapped = False

        def swap_parent_then_release(*args, **kwargs):
            nonlocal swapped
            result = original_release(*args, **kwargs)
            if not swapped:
                os.replace(safe_parent, parked_parent)
                safe_parent.symlink_to(attack_target, target_is_directory=True)
                swapped = True
            return result

        prep._release_output_lock = swap_parent_then_release
        caught: BaseException | None = None
        try:
            try:
                build_generation_from_audited_sources(
                    data_root, output, inventory, expectation=expected,
                    id_key=TEST_ID_KEY,
                )
            except BaseException as exc:  # noqa: BLE001 - assert exact fail-closed state
                caught = exc
            c.true(isinstance(caught, ValueError),
                   "lock-release parent swap must fail the transaction")
            c.true(not (attack_target / output.name).exists(),
                   "release-time swap publishes no attacker generation")
            c.true(not (parked_parent / output.name).exists(),
                   "release-time swap rolls back the anchored generation")
            c.true(not any("staging" in path.name for path in attack_target.iterdir()),
                   "release-time swap leaves no attacker staging")
            c.true(not any("staging" in path.name for path in parked_parent.iterdir()),
                   "release-time swap removes anchored staging")
        finally:
            prep._release_output_lock = original_release
            if safe_parent.is_symlink():
                safe_parent.unlink()
            if parked_parent.exists():
                os.replace(parked_parent, safe_parent)

        c.true(not output.exists(),
               "failed release transaction returns no lexical generation")
        lock = safe_parent / f".{output.name}.lock"
        c.true(lock.is_file() and not lock.is_symlink(),
               "failed release preserves the persistent owner-only lock")
        _assert_lock_reacquirable(c, lock, "failed release lock is reacquirable")


def test_archive_postcheck_output_lock_and_no_replace_fail_closed(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        data_root = base / "source"
        expected, _ = _synthetic_tree(data_root)
        inventory = audit_ravdess_inventory(data_root, expectation=expected)
        output = base / "derived_semantic23"
        lock = output.parent / f".{output.name}.lock"
        lock.touch(mode=0o600)
        lock.chmod(0o600)
        foreign_descriptor = os.open(lock, os.O_RDWR | os.O_NOFOLLOW)
        fcntl.flock(foreign_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            c.raises(lambda: build_generation_from_audited_sources(
                data_root, output, inventory, expectation=expected,
                id_key=TEST_ID_KEY), BlockingIOError,
                "held advisory output lock blocks concurrent producer")
        finally:
            fcntl.flock(foreign_descriptor, fcntl.LOCK_UN)
            os.close(foreign_descriptor)
        c.true(lock.is_file() and not lock.is_symlink(),
               "foreign persistent lock is never removed")
        c.true(not output.exists(), "locked transaction publishes nothing")
        _assert_lock_reacquirable(c, lock, "foreign lock is safely reacquirable")

        original_parser = prep.parse_openface_csv_bytes
        archive_path = data_root / RAVDESS_ARCHIVE_RELATIVE_PATH
        mutated = False

        def mutate_archive_after_first_read(data: bytes, *, source_name: str):
            nonlocal mutated
            trial = original_parser(data, source_name=source_name)
            if not mutated:
                replacement = archive_path.with_suffix(".replacement.zip")
                replacement.write_bytes(archive_path.read_bytes() + b"post-audit mutation")
                os.replace(replacement, archive_path)
                mutated = True
            return trial

        prep.parse_openface_csv_bytes = mutate_archive_after_first_read
        try:
            c.raises(lambda: build_generation_from_audited_sources(
                data_root, output, inventory, expectation=expected,
                id_key=TEST_ID_KEY), ValueError,
                "archive mutation during generation fails the pre-promotion postcheck")
        finally:
            prep.parse_openface_csv_bytes = original_parser
        c.true(not output.exists(), "mutated archive publishes no generation")
        c.true(lock.is_file() and not lock.is_symlink(),
               "failed transaction preserves its safe persistent lock")
        _assert_lock_reacquirable(c, lock, "failed transaction lock is reacquirable")
        c.true(not any("staging" in path.name for path in base.iterdir()),
               "archive mutation leaves no staging directory")

        stage = base / ".manual-stage"
        destination = base / "manual-output"
        stage.mkdir()
        destination.mkdir()
        parent_descriptor = os.open(
            base, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        )
        try:
            c.raises(lambda: prep._publish_directory_no_replace(
                parent_descriptor, stage.name, destination.name
            ), FileExistsError,
                "publication never replaces an existing empty path")
        finally:
            os.close(parent_descriptor)
        c.true(stage.is_dir() and destination.is_dir(),
               "failed no-replace publication preserves both paths")


def test_production_entrypoint_uses_private_key_and_canonical_output(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        invalid_root = base / "invalid-source"
        invalid_expected, _ = _synthetic_tree(invalid_root)
        invalid_archive = invalid_root / RAVDESS_ARCHIVE_RELATIVE_PATH
        invalid_archive.write_bytes(invalid_archive.read_bytes() + b"drift")
        invalid_key = invalid_root / ".must-not-exist.key"
        original = prep.FROZEN_RAVDESS_INVENTORY
        prep.FROZEN_RAVDESS_INVENTORY = invalid_expected
        try:
            c.raises(lambda: prep.prepare_ravdess_semantic23(
                invalid_root, id_key_path=invalid_key
            ), ValueError, "invalid archive fails before private state is created")
        finally:
            prep.FROZEN_RAVDESS_INVENTORY = original
        c.true(not invalid_key.exists(),
               "failed source audit leaves no private key side effect")

        data_root = base / "source"
        expected, _ = _synthetic_tree(data_root)
        key_path = data_root / ".test-private-id-key"
        original = prep.FROZEN_RAVDESS_INVENTORY
        prep.FROZEN_RAVDESS_INVENTORY = expected
        try:
            manifest = prep.prepare_ravdess_semantic23(
                data_root, id_key_path=key_path
            )
        finally:
            prep.FROZEN_RAVDESS_INVENTORY = original
        c.eq(manifest["schema"], SEMANTIC23_SCHEMA)
        c.true((data_root / "derived_semantic23" / "manifest.json").is_file(),
               "production entrypoint publishes only at canonical output")
        c.eq(stat.S_IMODE(key_path.stat().st_mode), 0o600,
             "production entrypoint uses an owner-only persistent key")


def test_inventory_and_transactional_generation_fail_closed(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)

        failing_source = base / "failing_source"
        failing_expected, _ = _synthetic_tree(
            failing_source, duplicate_first_frame=True
        )
        failing_inventory = audit_ravdess_inventory(
            failing_source, expectation=failing_expected
        )
        failed_output = base / "failed_generation"
        c.raises(lambda: build_generation_from_audited_sources(
            failing_source, failed_output, failing_inventory,
            expectation=failing_expected, id_key=TEST_ID_KEY), ValueError,
            "parse failure aborts the staged transaction")
        c.true(not failed_output.exists(), "failed transaction publishes no output")
        c.true(not any("staging" in path.name for path in base.iterdir()),
               "failed transaction removes its staging directory")

        data_root = base / "source"
        expected, files = _synthetic_tree(data_root)
        inventory = audit_ravdess_inventory(data_root, expectation=expected)
        c.eq(inventory.csv_files, 2, "audited CSV count")
        c.eq(inventory.frames, 3, "audited frame count")

        output = base / "derived_semantic23"
        build_generation_from_audited_sources(data_root, output, inventory,
                                              expectation=expected,
                                              id_key=TEST_ID_KEY)
        manifest_path = output / "manifest.json"
        manifest_text = manifest_path.read_text(encoding="utf-8")
        manifest = json.loads(manifest_text)
        c.eq(manifest["schema"], SEMANTIC23_SCHEMA, "manifest target schema")
        c.eq(manifest["adapter"]["source_topology"], "openface_68_2d",
             "manifest carries explicit adapter metadata")
        c.eq(manifest["adapter"]["scale_normalization"], "interocular_distance",
             "manifest carries source scaling")
        c.eq(len(manifest["trials"]), 2, "all audited trials emitted")
        c.true(all(item["trial_id"].startswith("trial_") for item in manifest["trials"]),
               "trial provenance is opaque")
        c.eq(manifest["provenance_policy"]["actor_id"],
             "private_hmac_sha256_base32",
             "manifest identifies keyed pseudonymization")
        c.true(TEST_ID_KEY.hex() not in manifest_text,
               "private HMAC key is never serialized")
        for source in files:
            c.true(source.name not in manifest_text and str(source) not in manifest_text,
                   "aggregate manifest contains no raw path or filename")
        c.eq(len(list((output / "trials").glob("*.npz"))), 2,
             "one cache per trial")
        two_frame_trial_id = opaque_trial_id(
            inventory.member_sha256[files[0].name], key=TEST_ID_KEY
        )
        two_frame_record = next(item for item in manifest["trials"]
                                if item["trial_id"] == two_frame_trial_id)
        with np.load(output / "trials" / f"{two_frame_record['trial_id']}.npz") as cache:
            c.true(bool(np.array_equal(cache["valid_mask"], [True, False])),
                   "published cache retains the low-confidence detector gap")
            c.true(bool(np.allclose(cache["timestamps"], [0.000, 0.033])),
                   "published cache retains source timestamps")
            c.eq(str(cache["schema"]), SEMANTIC23_SCHEMA,
                 "published cache names the exact target schema")
        c.true(not any("staging" in p.name for p in base.iterdir()),
               "successful transaction leaves no staging directory")
        lock = output.parent / f".{output.name}.lock"
        c.true(lock.is_file() and not lock.is_symlink(),
               "successful transaction preserves a safe persistent lock")
        _assert_lock_reacquirable(c, lock, "successful transaction lock is reacquirable")

        c.raises(lambda: build_generation_from_audited_sources(
            data_root, output, inventory, expectation=expected,
            id_key=TEST_ID_KEY), FileExistsError,
            "existing generation is not silently replaced")
        c.true(manifest_path.exists(), "failed replacement leaves old generation intact")

        archive = data_root / RAVDESS_ARCHIVE_RELATIVE_PATH
        with archive.open("ab") as handle:
            handle.write(b"unexpected archive mutation")
        c.raises(lambda: audit_ravdess_inventory(data_root, expectation=expected),
                 ValueError, "inventory drift fails closed before generation")


if __name__ == "__main__":
    run_all("test_openface68_semantic", dict(globals()))
