"""Fail-closed YFP regional manifest and static feature contract."""
from __future__ import annotations

import hashlib
import json
import os
import struct
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.datasets.yfp_region_manifest as yfp_manifest  # noqa: E402

from src.datasets.yfp_region_manifest import (  # noqa: E402
    AUDIT_SCHEMA,
    EYE_FEATURE_NAMES,
    MOUTH_FEATURE_NAMES,
    ManifestError,
    build_audit_manifest,
    clinical23_region_features,
    finalize_eligible_manifest,
    parse_regional_xml,
    require_eligible_manifest,
    write_manifest_once,
)
from scripts.extract_yfp_clinical23 import extract_yfp_clinical23  # noqa: E402
from scripts.run_yfp_region_ordinal import run_yfp_region_ordinal  # noqa: E402
from _testlib import Check, run_all  # noqa: E402


def _bmp(path: Path, width: int = 2, height: int = 2, truncate: bool = False,
         compression: int = 0, pixel_bytes: int | None = None,
         pixel_offset: int = 54, reserved1: int = 0, reserved2: int = 0,
         clr_used: int = 0, clr_important: int = 0) -> None:
    row = ((width * 3 + 3) // 4) * 4
    pixels = bytes(row * height if pixel_bytes is None else pixel_bytes)
    size = 54 + len(pixels)
    header = (
        b"BM" + struct.pack("<IHHI", size, reserved1, reserved2, pixel_offset)
        + struct.pack("<IiiHHIIiiII", 40, width, height, 1, 24, compression,
                      len(pixels), 2835, 2835, clr_used, clr_important)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((header + pixels)[:-1] if truncate else header + pixels)


def _xml_text(filename: str = "1.bmp", width: int = 2, height: int = 2,
              eye: str = "SlightPalsy_Eyes", mouth: str = "StrongPalsy_Mouth",
              terminal: bool = True) -> str:
    end = "</annotation>" if terminal else ""
    return f"""<annotation><filename>{filename}</filename>
<size><width>{width}</width><height>{height}</height><depth>3</depth></size>
<object><name>{eye}</name></object><object><name>{mouth}</name></object>{end}"""


def _fixture(root: Path, xml_text: str | None = None, truncate: bool = False) -> tuple[Path, Path]:
    image = root / "Image" / "1" / "1.bmp"
    xml = root / "Image_large_XML" / "1" / "1.xml"
    _bmp(image, truncate=truncate)
    xml.parent.mkdir(parents=True, exist_ok=True)
    xml.write_text(xml_text if xml_text is not None else _xml_text(), encoding="utf-8")
    return image, xml


def _authorized_artifacts(base: Path, audit_override: dict | None = None):
    root = base / "data"
    _fixture(root)
    audit = build_audit_manifest(root) if audit_override is None else audit_override
    audit_path = base / "audit.json"
    write_manifest_once(audit, audit_path)
    license_path = base / "license"
    subject_map = base / "map.json"
    authorization = base / "auth.json"
    license_path.write_text("research access grant", encoding="utf-8")
    audit_digest = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    license_digest = hashlib.sha256(license_path.read_bytes()).hexdigest()
    mapping = {
        "schema_version": "yfp_reviewed_subject_map_v1",
        "audit_manifest_sha256": audit_digest,
        "mappings": [{"subject_proxy": audit["rows"][0]["subject_proxy"],
                       "reviewed_group": "group_" + "1" * 64}],
    }
    subject_map.write_text(json.dumps(mapping, sort_keys=True), encoding="utf-8")
    map_digest = hashlib.sha256(subject_map.read_bytes()).hexdigest()
    auth = {
        "schema_version": "yfp_region_eligibility_authorization_v1",
        "authorized": True,
        "audit_manifest_sha256": audit_digest,
        "license_artifact_sha256": license_digest,
        "reviewed_subject_map_sha256": map_digest,
        "purpose": "110d-generalization-v1-yfp-static-region-ordinal",
    }
    authorization.write_text(json.dumps(auth, sort_keys=True), encoding="utf-8")
    return root, audit, audit_path, license_path, subject_map, authorization


def _error_text(fn) -> str:
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 - tests inspect fail-closed reason
        return str(exc)
    raise AssertionError("expected an exception")


def test_native_and_single_terminal_repair_are_the_only_parse_paths(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _image, xml = _fixture(root)
        parsed = parse_regional_xml(xml)
        c.eq(parsed.parse_status, "native")
        c.eq((parsed.eye, parsed.mouth), (1, 2))
        original = xml.read_bytes()
        xml.write_text(_xml_text(terminal=False), encoding="utf-8")
        repaired = parse_regional_xml(xml)
        c.eq(repaired.parse_status, "repaired_missing_terminal_annotation")
        c.eq(xml.read_bytes(), _xml_text(terminal=False).encode(), "repair is in memory only")
        xml.write_bytes(original)


def test_malformed_dtd_entity_and_unknown_or_conflicting_labels_fail(c: Check):
    bad = {
        "mismatch": "<annotation><object></annotation>",
        "dtd": "<!DOCTYPE annotation><annotation></annotation>",
        "entity": "<!DOCTYPE annotation [<!ENTITY x 'x'>]><annotation>&x;</annotation>",
        "unknown": _xml_text(eye="Maybe_Eyes"),
        "conflict": _xml_text().replace(
            "</annotation>", "<object><name>Normal_Eyes</name></object></annotation>"),
    }
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "x.xml"
        for name, text in bad.items():
            path.write_text(text, encoding="utf-8")
            c.raises(lambda p=path: parse_regional_xml(p), ManifestError, name)


def test_utf16_entity_declaration_cannot_bypass_xml_guard(c: Check):
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "utf16.xml"
        text = """<?xml version="1.0" encoding="UTF-16"?>
<!DOCTYPE annotation [<!ENTITY bmp "1.bmp">]>
<annotation><filename>&bmp;</filename>
<size><width>2</width><height>2</height><depth>3</depth></size>
<object><name>SlightPalsy_Eyes</name></object>
<object><name>StrongPalsy_Mouth</name></object></annotation>"""
        path.write_bytes(text.encode("utf-16"))
        c.raises(lambda: parse_regional_xml(path), ManifestError,
                 "UTF-16 DTD/entity declarations are forbidden")


def test_bomless_utf16_dtd_with_leading_whitespace_cannot_bypass_guard(c: Check):
    with tempfile.TemporaryDirectory() as td:
        text = """ \n<!DOCTYPE annotation [<!ENTITY bmp "1.bmp">]>
<annotation><filename>&bmp;</filename>
<size><width>2</width><height>2</height><depth>3</depth></size>
<object><name>SlightPalsy_Eyes</name></object>
<object><name>StrongPalsy_Mouth</name></object></annotation>"""
        for encoding in ("utf-16-le", "utf-16-be"):
            path = Path(td) / f"{encoding}.xml"
            path.write_bytes(text.encode(encoding))
            c.raises(lambda p=path: parse_regional_xml(p), ManifestError,
                     f"BOM-less {encoding} DTD/entity declarations are forbidden")


def test_truncated_compressed_bmp_is_never_accepted(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        image, _xml = _fixture(root)
        _bmp(image, compression=1, pixel_bytes=1)
        manifest = build_audit_manifest(root)
        c.eq(manifest["aggregate"]["eligible_anchor_count"], 0)
        c.eq(manifest["aggregate"]["quarantine_reasons"],
             {"unsupported_bmp_compression": 1})


def test_bi_rgb_header_layout_is_fully_validated(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        image, _xml = _fixture(root)
        _bmp(image, pixel_offset=1)
        manifest = build_audit_manifest(root)
        c.eq(manifest["aggregate"]["eligible_anchor_count"], 0)
        c.eq(manifest["aggregate"]["quarantine_reasons"],
             {"unsupported_bmp_layout": 1})

        for index, malformed in enumerate((
            {"reserved1": 1}, {"reserved2": 1},
            {"clr_used": 1}, {"clr_important": 1},
        )):
            malformed_root = root / f"malformed-{index}"
            malformed_image, _ = _fixture(malformed_root)
            _bmp(malformed_image, **malformed)
            malformed_manifest = build_audit_manifest(malformed_root)
            c.eq(malformed_manifest["aggregate"]["eligible_anchor_count"], 0)
            c.eq(malformed_manifest["aggregate"]["quarantine_reasons"],
                 {"unsupported_bmp_layout": 1})


def test_truncated_bmp_dimension_conflict_and_path_escape_are_quarantined(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _fixture(root, truncate=True)
        manifest = build_audit_manifest(root)
        c.eq(manifest["aggregate"]["eligible_anchor_count"], 0)
        c.eq(manifest["aggregate"]["quarantine_reasons"], {"truncated_bmp": 1})

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _fixture(root, _xml_text(width=3))
        manifest = build_audit_manifest(root)
        c.eq(manifest["aggregate"]["quarantine_reasons"], {"dimension_mismatch": 1})

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        outside = root.parent / f"escape-{root.name}.bmp"
        _bmp(outside)
        xml = root / "Image_large_XML" / "1" / "1.xml"
        xml.parent.mkdir(parents=True)
        xml.write_text(_xml_text(), encoding="utf-8")
        image = root / "Image" / "1" / "1.bmp"
        image.parent.mkdir(parents=True)
        image.symlink_to(outside)
        c.raises(lambda: build_audit_manifest(root), ManifestError, "symlink escape")
        outside.unlink()


def test_duplicate_keys_and_source_digests_are_rejected(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        image, xml = _fixture(root)
        duplicate = root / "Image2" / "1" / "1.bmp"
        duplicate.parent.mkdir(parents=True)
        duplicate.write_bytes(image.read_bytes())
        manifest = build_audit_manifest(root)
        c.eq(manifest["aggregate"]["quarantine_reasons"], {"duplicate_image_key": 1})

        duplicate.unlink()
        xml2 = root / "Image_large_XML" / "2" / "2.xml"
        image2 = root / "Image" / "2" / "2.bmp"
        xml2.parent.mkdir(parents=True)
        image2.parent.mkdir(parents=True)
        xml2.write_text(_xml_text(filename="2.bmp"), encoding="utf-8")
        image2.write_bytes(image.read_bytes())
        manifest = build_audit_manifest(root)
        c.eq(manifest["aggregate"]["eligible_anchor_count"], 0)
        c.true(manifest["aggregate"]["quarantine_reasons"]["duplicate_source_digest"] >= 2)


def test_audit_manifest_is_immutable_ineligible_and_region_labels_stay_separate(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _fixture(root)
        manifest = build_audit_manifest(root)
        c.eq(manifest["schema_version"], AUDIT_SCHEMA)
        c.eq(manifest["stage"], "audit")
        c.eq(manifest["training_eligible"], False)
        c.eq(manifest["counters"], {"extractions": 0, "fits": 0, "predictions": 0})
        c.eq(len(manifest["rows"]), 1)
        row = manifest["rows"][0]
        c.eq(row["targets"], {"eye": 1, "mouth": 2, "brow": None,
                               "action": None, "phase": None})
        c.eq(row["source_commitment"], row["targets_commitment"])
        c.eq(row["group_status"], "unreviewed_subject_folder_proxy")
        c.raises(lambda: require_eligible_manifest(manifest), ManifestError)
        forged = deepcopy(manifest)
        forged.update({"schema_version": "yfp_region_eligible_manifest_v1",
                       "stage": "eligible", "training_eligible": True})
        forged["rows"][0]["reviewed_group"] = "group_" + "1" * 64
        c.raises(lambda: require_eligible_manifest(forged), ManifestError,
                 "flipping eligibility fields cannot bypass evidence commitments")


def test_manifest_write_is_no_overwrite(c: Check):
    with tempfile.TemporaryDirectory() as td:
        output = Path(td) / "manifest.json"
        write_manifest_once({"a": 1}, output)
        c.raises(lambda: write_manifest_once({"a": 2}, output), FileExistsError)
        c.eq(json.loads(output.read_text()), {"a": 1})


def test_eligible_finalizer_requires_all_three_bound_evidence_artifacts(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "data"
        _fixture(root)
        audit = build_audit_manifest(root)
        audit_path = Path(td) / "audit.json"
        write_manifest_once(audit, audit_path)
        license_path = Path(td) / "license"
        subject_map = Path(td) / "map.json"
        authorization = Path(td) / "auth.json"
        license_path.write_text("research access grant", encoding="utf-8")
        audit_digest = hashlib.sha256(audit_path.read_bytes()).hexdigest()
        license_digest = hashlib.sha256(license_path.read_bytes()).hexdigest()
        mapping = {
            "schema_version": "yfp_reviewed_subject_map_v1",
            "audit_manifest_sha256": audit_digest,
            "mappings": [{"subject_proxy": audit["rows"][0]["subject_proxy"],
                           "reviewed_group": "group_" + "1" * 64}],
        }
        subject_map.write_text(json.dumps(mapping, sort_keys=True), encoding="utf-8")
        map_digest = hashlib.sha256(subject_map.read_bytes()).hexdigest()
        auth = {
            "schema_version": "yfp_region_eligibility_authorization_v1",
            "authorized": True,
            "audit_manifest_sha256": audit_digest,
            "license_artifact_sha256": license_digest,
            "reviewed_subject_map_sha256": map_digest,
            "purpose": "110d-generalization-v1-yfp-static-region-ordinal",
        }
        authorization.write_text(json.dumps(auth, sort_keys=True), encoding="utf-8")
        for missing in (license_path, subject_map, authorization):
            backup = missing.read_bytes()
            missing.unlink()
            c.raises(lambda: finalize_eligible_manifest(
                audit_path, license_path, subject_map, authorization), ManifestError)
            missing.write_bytes(backup)
        eligible = finalize_eligible_manifest(
            audit_path, license_path, subject_map, authorization)
        c.eq(eligible["training_eligible"], True)
        c.eq(eligible["stage"], "eligible")
        c.eq(eligible["rows"][0]["reviewed_group"], "group_" + "1" * 64)
        require_eligible_manifest(eligible)


def test_finalizer_rebuilds_the_entire_audit_and_rejects_bound_aggregate_forgery(c: Check):
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        root = base / "data"
        _fixture(root)
        forged = build_audit_manifest(root)
        forged["aggregate"]["eye_anchor_count"] = 999
        _root, _audit, audit_path, license_path, subject_map, authorization = (
            _authorized_artifacts(base / "evidence", forged)
        )
        c.raises(lambda: finalize_eligible_manifest(
            audit_path, license_path, subject_map, authorization), ManifestError,
            "authorization cannot legitimize an audit that differs from source rebuild")


def test_runtime_reauthenticates_exact_evidence_and_rejects_replacement_or_symlink(c: Check):
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        _root, _audit, audit_path, license_path, subject_map, authorization = (
            _authorized_artifacts(base)
        )
        eligible = finalize_eligible_manifest(
            audit_path, license_path, subject_map, authorization)
        eligible_path = base / "eligible.json"
        write_manifest_once(eligible, eligible_path)
        c.raises(lambda: yfp_manifest.authenticate_eligible_manifest(eligible_path),
                 ManifestError, "runtime stays locked without an out-of-band authorization pin")
        locked_features = base / "locked-features"
        locked_report = base / "locked-report.json"
        c.raises(lambda: extract_yfp_clinical23(
            eligible_path, locked_features, base / "missing-model.task"), ManifestError,
            "extraction stays locked before model or output access")
        c.raises(lambda: run_yfp_region_ordinal(
            eligible_path, base / "missing-cache", locked_report), ManifestError,
            "training stays locked before cache or output access")
        c.true(not locked_features.exists() and not locked_report.exists())

        original_pin = yfp_manifest.PINNED_YFP_AUTHORIZATION_SHA256
        yfp_manifest.PINNED_YFP_AUTHORIZATION_SHA256 = hashlib.sha256(
            authorization.read_bytes()).hexdigest()
        try:
            authenticated = yfp_manifest.authenticate_eligible_manifest(eligible_path)
            c.eq(authenticated, eligible)

            license_path.write_text("replacement grant", encoding="utf-8")
            error = _error_text(
                lambda: yfp_manifest.authenticate_eligible_manifest(eligible_path))
            c.true("license" in error.lower() or "evidence" in error.lower())
            extraction_output = base / "features"
            error = _error_text(lambda: extract_yfp_clinical23(
                eligible_path, extraction_output, base / "missing-model.task"))
            c.true("license" in error.lower() or "evidence" in error.lower())
            c.true(not extraction_output.exists())
            report_output = base / "report.json"
            error = _error_text(lambda: run_yfp_region_ordinal(
                eligible_path, base / "missing-cache", report_output))
            c.true("license" in error.lower() or "evidence" in error.lower())
            c.true(not report_output.exists())

            license_path.write_text("research access grant", encoding="utf-8")
            real_license = base / "real-license"
            license_path.rename(real_license)
            license_path.symlink_to(real_license)
            c.raises(lambda: yfp_manifest.authenticate_eligible_manifest(eligible_path),
                     ManifestError,
                     "evidence symlinks are forbidden even with matching bytes")
        finally:
            yfp_manifest.PINNED_YFP_AUTHORIZATION_SHA256 = original_pin


def test_out_of_band_authorization_pin_rejects_self_consistent_forged_bundle(c: Check):
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        evidence_a = _authorized_artifacts(base / "trusted")
        evidence_b = _authorized_artifacts(base / "forged")
        paths = []
        for name, evidence in (("trusted", evidence_a), ("forged", evidence_b)):
            _root, _audit, audit_path, license_path, subject_map, authorization = evidence
            eligible = finalize_eligible_manifest(
                audit_path, license_path, subject_map, authorization)
            eligible_path = base / f"{name}-eligible.json"
            write_manifest_once(eligible, eligible_path)
            paths.append((eligible_path, authorization))

        original_pin = yfp_manifest.PINNED_YFP_AUTHORIZATION_SHA256
        yfp_manifest.PINNED_YFP_AUTHORIZATION_SHA256 = hashlib.sha256(
            paths[0][1].read_bytes()).hexdigest()
        try:
            c.eq(yfp_manifest.authenticate_eligible_manifest(paths[0][0])["stage"],
                 "eligible")
            c.raises(lambda: yfp_manifest.authenticate_eligible_manifest(paths[1][0]),
                     ManifestError,
                     "a complete self-consistent bundle cannot replace the pinned authorization")
        finally:
            yfp_manifest.PINNED_YFP_AUTHORIZATION_SHA256 = original_pin


def test_all_zero_and_self_referential_eligible_commitments_fail_closed(c: Check):
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        _root, _audit, audit_path, license_path, subject_map, authorization = (
            _authorized_artifacts(base)
        )
        eligible = finalize_eligible_manifest(
            audit_path, license_path, subject_map, authorization)
        for field in (
            "audit_manifest_sha256", "license_artifact_sha256",
            "reviewed_subject_map_sha256", "eligibility_authorization_sha256",
            "source_collection_sha256",
        ):
            eligible[field] = "0" * 64
        c.raises(lambda: require_eligible_manifest(eligible), ManifestError,
                 "all-zero digests are not commitments")

        valid = finalize_eligible_manifest(
            audit_path, license_path, subject_map, authorization)
        zero_row = deepcopy(valid)
        zero_row["rows"][0]["anchor_key"] = "anchor_" + "0" * 64
        zero_row["rows"][0]["source_commitment"] = "0" * 64
        zero_row["rows"][0]["targets_commitment"] = "0" * 64
        c.raises(lambda: require_eligible_manifest(zero_row), ManifestError,
                 "row-level all-zero commitments are forbidden")

        eligible_path = base / "eligible.json"
        write_manifest_once(valid, eligible_path)
        forged = deepcopy(valid)
        forged["evidence_artifacts"]["license_artifact"]["path"] = str(eligible_path)
        forged_path = base / "self-forged.json"
        write_manifest_once(forged, forged_path)
        c.raises(lambda: yfp_manifest.authenticate_eligible_manifest(forged_path),
                 ManifestError, "eligible manifest cannot serve as its own evidence")


def test_static_six_feature_order_and_capture_swap_invariance(c: Check):
    c.eq(EYE_FEATURE_NAMES, (
        "fissure_height_bilateral_mean", "fissure_height_absolute_difference",
        "fissure_width_bilateral_mean", "fissure_width_absolute_difference",
        "eye_area_bilateral_mean", "eye_area_absolute_difference",
    ))
    c.eq(MOUTH_FEATURE_NAMES, (
        "commissure_height_bilateral_mean", "commissure_height_absolute_difference",
        "commissure_radius_bilateral_mean", "commissure_radius_absolute_difference",
        "mouth_width", "mouth_open",
    ))
    clinical = np.arange(23, dtype=np.float64)
    eye = clinical23_region_features(clinical, "eye")
    mouth = clinical23_region_features(clinical, "mouth")
    c.true(np.allclose(eye, [0.5, 1, 4.5, 1, 7.5, 1]))
    c.true(np.allclose(mouth, [14.5, 1, 18.5, 1, 21, 22]))
    swapped = clinical.copy()
    for left, right in ((0, 1), (4, 5), (7, 8), (14, 15), (18, 19)):
        swapped[left], swapped[right] = clinical[right], clinical[left]
    c.true(np.array_equal(eye, clinical23_region_features(swapped, "eye")))
    c.true(np.array_equal(mouth, clinical23_region_features(swapped, "mouth")))
    c.raises(lambda: clinical23_region_features(np.zeros((4, 32, 95)), "eye"),
             ManifestError, "static frames cannot be tiled into dynamic 110D")


def test_audit_only_manifest_stops_before_extraction_fit_or_prediction(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "data"
        _fixture(root)
        manifest_path = Path(td) / "audit.json"
        write_manifest_once(build_audit_manifest(root), manifest_path)
        extraction_output = Path(td) / "features"
        report_output = Path(td) / "report.json"
        c.raises(lambda: extract_yfp_clinical23(
            manifest_path, extraction_output), ManifestError)
        c.true(not extraction_output.exists(), "gate precedes MediaPipe output creation")
        c.raises(lambda: run_yfp_region_ordinal(
            manifest_path, Path(td) / "absent-cache", report_output), ManifestError)
        c.true(not report_output.exists(), "gate precedes fit and prediction output")
        audit = json.loads(manifest_path.read_text())
        c.eq(audit["counters"], {"extractions": 0, "fits": 0, "predictions": 0})


if __name__ == "__main__":
    run_all(__name__, globals())
