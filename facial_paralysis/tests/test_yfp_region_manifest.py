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


def _bmp(path: Path, width: int = 2, height: int = 2, truncate: bool = False) -> None:
    row = ((width * 3 + 3) // 4) * 4
    pixels = bytes(row * height)
    size = 54 + len(pixels)
    header = (
        b"BM" + struct.pack("<IHHI", size, 0, 0, 54)
        + struct.pack("<IiiHHIIiiII", 40, width, height, 1, 24, 0,
                      len(pixels), 2835, 2835, 0, 0)
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
