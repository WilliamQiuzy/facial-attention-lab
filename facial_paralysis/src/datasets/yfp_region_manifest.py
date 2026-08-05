"""Provenance-locked YFP static regional-severity manifests.

This module deliberately treats subject folders as unreviewed grouping proxies
and never promotes an audit manifest in place.  Source XML/BMP bytes are
read-only; the sole tolerated XML repair is one missing terminal
``</annotation>`` appended in memory.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import struct
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

AUDIT_SCHEMA = "yfp_region_audit_manifest_v1"
ELIGIBLE_SCHEMA = "yfp_region_eligible_manifest_v1"
SUBJECT_MAP_SCHEMA = "yfp_reviewed_subject_map_v1"
AUTHORIZATION_SCHEMA = "yfp_region_eligibility_authorization_v1"
AUTHORIZATION_PURPOSE = "110d-generalization-v1-yfp-static-region-ordinal"

EYE_LABELS = {"Normal_Eyes": 0, "SlightPalsy_Eyes": 1, "StrongPalsy_Eyes": 2}
MOUTH_LABELS = {"Normal_Mouth": 0, "SlightPalsy_Mouth": 1, "StrongPalsy_Mouth": 2}
KNOWN_LABELS = frozenset((*EYE_LABELS, *MOUTH_LABELS))

EYE_FEATURE_NAMES = (
    "fissure_height_bilateral_mean",
    "fissure_height_absolute_difference",
    "fissure_width_bilateral_mean",
    "fissure_width_absolute_difference",
    "eye_area_bilateral_mean",
    "eye_area_absolute_difference",
)
MOUTH_FEATURE_NAMES = (
    "commissure_height_bilateral_mean",
    "commissure_height_absolute_difference",
    "commissure_radius_bilateral_mean",
    "commissure_radius_absolute_difference",
    "mouth_width",
    "mouth_open",
)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_GROUP = re.compile(r"^group_[0-9a-f]{64}$")
_SOURCE_TOPS = ("Image", "Image2", "Image3", "Image4")


class ManifestError(RuntimeError):
    """A fail-closed YFP provenance or eligibility error."""


@dataclass(frozen=True)
class ParsedRegionalXML:
    filename: str
    width: int
    height: int
    eye: int | None
    mouth: int | None
    parse_status: str


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True).encode("utf-8")


def _safe_regular_file(path: Path, root: Path) -> Path:
    root_resolved = root.resolve(strict=True)
    if root.is_symlink():
        raise ManifestError(f"root must not be a symlink: {root}")
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ManifestError(f"path is outside YFP root: {path}") from exc
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ManifestError(f"symlink is forbidden in YFP source tree: {cursor}")
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root_resolved)
    except (FileNotFoundError, ValueError) as exc:
        raise ManifestError(f"source path escapes or is missing: {path}") from exc
    if not resolved.is_file():
        raise ManifestError(f"source is not a regular file: {path}")
    return resolved


def parse_regional_xml(path: str | Path) -> ParsedRegionalXML:
    """Parse one combined regional XML with exactly one allowed EOF repair."""
    path = Path(path)
    data = path.read_bytes()
    upper = data.upper()
    if b"<!DOCTYPE" in upper or b"<!ENTITY" in upper:
        raise ManifestError("DTD and entity declarations are forbidden")
    try:
        root = ET.fromstring(data)
        status = "native"
    except ET.ParseError as native_error:
        stripped = data.rstrip()
        if (not stripped.startswith(b"<annotation")
                or stripped.endswith(b"</annotation>")):
            raise ManifestError("XML is malformed beyond the allowed terminal repair") from native_error
        repaired = stripped + b"</annotation>"
        try:
            root = ET.fromstring(repaired)
        except ET.ParseError as repair_error:
            raise ManifestError("XML is malformed beyond the allowed terminal repair") from repair_error
        status = "repaired_missing_terminal_annotation"
    if root.tag != "annotation":
        raise ManifestError("XML root must be annotation")
    filename = (root.findtext("filename") or "").strip()
    if not filename or Path(filename).name != filename or not filename.lower().endswith(".bmp"):
        raise ManifestError("XML filename must be one BMP basename")
    try:
        width = int((root.findtext("size/width") or "").strip())
        height = int((root.findtext("size/height") or "").strip())
    except ValueError as exc:
        raise ManifestError("XML dimensions must be integers") from exc
    if width <= 0 or height <= 0:
        raise ManifestError("XML dimensions must be positive")

    values: dict[str, set[int]] = {"eye": set(), "mouth": set()}
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        if name in EYE_LABELS:
            values["eye"].add(EYE_LABELS[name])
        elif name in MOUTH_LABELS:
            values["mouth"].add(MOUTH_LABELS[name])
        else:
            raise ManifestError(f"unknown regional label: {name!r}")
    if any(len(item) > 1 for item in values.values()):
        raise ManifestError("conflicting labels for one region")
    eye = next(iter(values["eye"]), None)
    mouth = next(iter(values["mouth"]), None)
    if eye is None and mouth is None:
        raise ManifestError("XML has no eye or mouth target")
    return ParsedRegionalXML(
        filename=filename,
        width=width,
        height=height,
        eye=eye,
        mouth=mouth,
        parse_status=status,
    )


def _bmp_dimensions(path: Path) -> tuple[int, int]:
    actual_size = path.stat().st_size
    with path.open("rb") as stream:
        header = stream.read(54)
    if len(header) < 54 or header[:2] != b"BM":
        raise ManifestError("truncated_bmp")
    declared_size, pixel_offset = struct.unpack_from("<I", header, 2)[0], struct.unpack_from("<I", header, 10)[0]
    dib_size = struct.unpack_from("<I", header, 14)[0]
    if dib_size < 40:
        raise ManifestError("unsupported_bmp")
    width, raw_height = struct.unpack_from("<ii", header, 18)
    planes, bit_count = struct.unpack_from("<HH", header, 26)
    compression = struct.unpack_from("<I", header, 30)[0]
    if width <= 0 or raw_height == 0 or planes != 1:
        raise ManifestError("invalid_bmp_dimensions")
    height = abs(raw_height)
    if declared_size > actual_size or pixel_offset >= actual_size:
        raise ManifestError("truncated_bmp")
    if compression == 0 and bit_count in (1, 4, 8, 16, 24, 32):
        row_bytes = ((width * bit_count + 31) // 32) * 4
        if pixel_offset + row_bytes * height > actual_size:
            raise ManifestError("truncated_bmp")
    return width, height


def _source_files(root: Path) -> tuple[dict[tuple[str, str], list[Path]], list[Path]]:
    images: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for top in _SOURCE_TOPS:
        directory = root / top
        if not directory.exists():
            continue
        if directory.is_symlink():
            raise ManifestError(f"symlink is forbidden: {directory}")
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                raise ManifestError(f"symlink is forbidden: {path}")
            if path.is_file() and path.suffix.lower() == ".bmp":
                _safe_regular_file(path, root)
                images[(path.parent.name, path.stem)].append(path)

    xml_root = root / "Image_large_XML"
    if not xml_root.exists() or xml_root.is_symlink():
        raise ManifestError("Image_large_XML is missing or unsafe")
    xmls: list[Path] = []
    for subject_dir in sorted(xml_root.iterdir(), key=lambda p: p.name):
        if subject_dir.name in {"eyes", "mouth"}:
            continue
        if subject_dir.is_symlink():
            raise ManifestError(f"symlink is forbidden: {subject_dir}")
        if not subject_dir.is_dir():
            continue
        for path in sorted(subject_dir.glob("*.xml"), key=lambda p: p.name):
            _safe_regular_file(path, root)
            xmls.append(path)
    return images, xmls


def build_audit_manifest(yfp_root: str | Path) -> dict[str, Any]:
    """Inventory valid combined anchors; invalid anchors remain quarantined."""
    root = Path(yfp_root)
    if not root.exists() or not root.is_dir() or root.is_symlink():
        raise ManifestError("YFP root must be an existing non-symlink directory")
    images, xmls = _source_files(root)
    quarantine: Counter[str] = Counter()
    candidates: list[dict[str, Any]] = []
    inventory_commitments: list[str] = []

    for xml_path in xmls:
        subject, stem = xml_path.parent.name, xml_path.stem
        key = (subject, stem)
        matches = images.get(key, [])
        if len(matches) != 1:
            quarantine["duplicate_image_key" if len(matches) > 1 else "missing_image"] += 1
            continue
        image_path = matches[0]
        try:
            parsed = parse_regional_xml(xml_path)
        except ManifestError as exc:
            message = str(exc)
            if "DTD" in message or "entity" in message:
                reason = "unsafe_xml"
            elif "unknown" in message:
                reason = "unknown_label"
            elif "conflicting" in message:
                reason = "conflicting_label"
            else:
                reason = "malformed_xml"
            quarantine[reason] += 1
            continue
        if parsed.filename.lower() != image_path.name.lower():
            quarantine["filename_mismatch"] += 1
            continue
        try:
            image_width, image_height = _bmp_dimensions(image_path)
        except ManifestError as exc:
            quarantine[str(exc)] += 1
            continue
        if (image_width, image_height) != (parsed.width, parsed.height):
            quarantine["dimension_mismatch"] += 1
            continue
        xml_digest = _sha256_file(xml_path)
        image_digest = _sha256_file(image_path)
        inventory_commitments.extend((f"xml:{xml_digest}", f"bmp:{image_digest}"))
        source_commitment = _sha256_bytes(
            f"yfp-region-anchor-v1\0{xml_digest}\0{image_digest}".encode("ascii"))
        subject_proxy = "subject_" + _sha256_bytes(
            f"yfp-subject-folder-proxy-v1\0{subject}".encode("utf-8"))
        candidates.append({
            "anchor_key": "anchor_" + source_commitment,
            "subject_proxy": subject_proxy,
            "group_status": "unreviewed_subject_folder_proxy",
            "source_commitment": source_commitment,
            "targets_commitment": source_commitment,
            "xml": {
                "relative_path": xml_path.relative_to(root).as_posix(),
                "sha256": xml_digest,
                "bytes": xml_path.stat().st_size,
                "parse_status": parsed.parse_status,
            },
            "image": {
                "relative_path": image_path.relative_to(root).as_posix(),
                "sha256": image_digest,
                "bytes": image_path.stat().st_size,
                "width": image_width,
                "height": image_height,
            },
            "targets": {"eye": parsed.eye, "mouth": parsed.mouth,
                        "brow": None, "action": None, "phase": None},
        })

    duplicate_indices: set[int] = set()
    for field in ("xml", "image"):
        by_digest: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(candidates):
            by_digest[row[field]["sha256"]].append(index)
        for indices in by_digest.values():
            if len(indices) > 1:
                duplicate_indices.update(indices)
    if duplicate_indices:
        quarantine["duplicate_source_digest"] += len(duplicate_indices)
        candidates = [row for index, row in enumerate(candidates)
                      if index not in duplicate_indices]

    rows = sorted(candidates, key=lambda row: row["anchor_key"])
    source_commitment = _sha256_bytes(
        "\n".join(sorted(inventory_commitments)).encode("ascii"))
    return {
        "schema_version": AUDIT_SCHEMA,
        "dataset": "YFP",
        "stage": "audit",
        "training_eligible": False,
        "source_root": str(root.resolve()),
        "source_collection_sha256": source_commitment,
        "grouping_claim": "subject_folder_proxy_unreviewed_not_patient_disjoint",
        "targets": {"eye": [0, 1, 2], "mouth": [0, 1, 2],
                    "brow": None, "action": None, "phase": None},
        "feature_contract": {
            "kind": "static_clinical23_region_6d",
            "eye": list(EYE_FEATURE_NAMES),
            "mouth": list(MOUTH_FEATURE_NAMES),
            "dynamic_110d_allowed": False,
        },
        "counters": {"extractions": 0, "fits": 0, "predictions": 0},
        "aggregate": {
            "combined_xml_count": len(xmls),
            "indexed_bmp_key_count": len(images),
            "eligible_anchor_count": len(rows),
            "quarantined_anchor_count": int(sum(quarantine.values())),
            "quarantine_reasons": dict(sorted(quarantine.items())),
            "subject_proxy_count": len({row["subject_proxy"] for row in rows}),
            "eye_anchor_count": sum(row["targets"]["eye"] is not None for row in rows),
            "mouth_anchor_count": sum(row["targets"]["mouth"] is not None for row in rows),
        },
        "rows": rows,
    }


def write_manifest_once(manifest: dict[str, Any], output: str | Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    payload = json.dumps(manifest, sort_keys=True, indent=2) + "\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(output, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
    except BaseException:
        output.unlink(missing_ok=True)
        raise


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read {label}") from exc
    if not isinstance(value, dict):
        raise ManifestError(f"{label} must be a JSON object")
    return value


def _validate_audit_shape(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != AUDIT_SCHEMA or manifest.get("stage") != "audit":
        raise ManifestError("not a YFP audit manifest")
    if manifest.get("training_eligible") is not False:
        raise ManifestError("audit manifest can never be training eligible")
    if manifest.get("counters") != {"extractions": 0, "fits": 0, "predictions": 0}:
        raise ManifestError("audit counters must remain zero")
    rows = manifest.get("rows")
    if not isinstance(rows, list) or manifest.get("aggregate", {}).get("eligible_anchor_count") != len(rows):
        raise ManifestError("audit row count mismatch")


def _revalidate_audit_sources(manifest: dict[str, Any]) -> None:
    _validate_audit_shape(manifest)
    root = Path(manifest.get("source_root", ""))
    if not root.exists() or root.is_symlink():
        raise ManifestError("audit source root is unavailable or unsafe")
    seen_anchors: set[str] = set()
    seen_digests: set[str] = set()
    for row in manifest["rows"]:
        if row.get("anchor_key") in seen_anchors:
            raise ManifestError("duplicate anchor key")
        seen_anchors.add(row.get("anchor_key"))
        xml_path = _safe_regular_file(root / row["xml"]["relative_path"], root)
        image_path = _safe_regular_file(root / row["image"]["relative_path"], root)
        xml_digest, image_digest = _sha256_file(xml_path), _sha256_file(image_path)
        if xml_digest != row["xml"]["sha256"] or image_digest != row["image"]["sha256"]:
            raise ManifestError("source digest changed after audit")
        if xml_digest in seen_digests or image_digest in seen_digests:
            raise ManifestError("duplicate source digest")
        seen_digests.update((xml_digest, image_digest))
        parsed = parse_regional_xml(xml_path)
        dimensions = _bmp_dimensions(image_path)
        if dimensions != (row["image"]["width"], row["image"]["height"]):
            raise ManifestError("source image dimensions changed")
        if dimensions != (parsed.width, parsed.height):
            raise ManifestError("source XML/image dimensions mismatch")
        expected_targets = {"eye": parsed.eye, "mouth": parsed.mouth,
                            "brow": None, "action": None, "phase": None}
        if row.get("targets") != expected_targets:
            raise ManifestError("target values changed after audit")


def finalize_eligible_manifest(
    audit_manifest: str | Path,
    license_artifact: str | Path,
    reviewed_subject_map: str | Path,
    eligibility_authorization: str | Path,
) -> dict[str, Any]:
    """Authenticate three independent evidence files and create a successor."""
    audit_path = Path(audit_manifest)
    license_path = Path(license_artifact)
    map_path = Path(reviewed_subject_map)
    auth_path = Path(eligibility_authorization)
    if not all(path.exists() and path.is_file() and not path.is_symlink()
               for path in (audit_path, license_path, map_path, auth_path)):
        raise ManifestError("license, reviewed map, and authorization are all required")
    manifest = _load_json_object(audit_path, "audit manifest")
    _revalidate_audit_sources(manifest)
    audit_digest = _sha256_file(audit_path)
    license_digest = _sha256_file(license_path)
    map_digest = _sha256_file(map_path)
    subject_map = _load_json_object(map_path, "reviewed subject map")
    authorization = _load_json_object(auth_path, "eligibility authorization")
    if set(subject_map) != {"schema_version", "audit_manifest_sha256", "mappings"}:
        raise ManifestError("reviewed subject map has unexpected fields")
    if (subject_map["schema_version"] != SUBJECT_MAP_SCHEMA
            or subject_map["audit_manifest_sha256"] != audit_digest
            or not isinstance(subject_map["mappings"], list)):
        raise ManifestError("reviewed subject map is not bound to the audit")
    expected_auth_fields = {
        "schema_version", "authorized", "audit_manifest_sha256",
        "license_artifact_sha256", "reviewed_subject_map_sha256", "purpose",
    }
    if set(authorization) != expected_auth_fields:
        raise ManifestError("eligibility authorization has unexpected fields")
    if authorization != {
        "schema_version": AUTHORIZATION_SCHEMA,
        "authorized": True,
        "audit_manifest_sha256": audit_digest,
        "license_artifact_sha256": license_digest,
        "reviewed_subject_map_sha256": map_digest,
        "purpose": AUTHORIZATION_PURPOSE,
    }:
        raise ManifestError("eligibility authorization evidence digests do not match")
    proxy_set = {row["subject_proxy"] for row in manifest["rows"]}
    mapping: dict[str, str] = {}
    for item in subject_map["mappings"]:
        if not isinstance(item, dict) or set(item) != {"subject_proxy", "reviewed_group"}:
            raise ManifestError("reviewed mapping row is malformed")
        proxy, group = item["subject_proxy"], item["reviewed_group"]
        if proxy in mapping or not _GROUP.fullmatch(group):
            raise ManifestError("reviewed mapping is duplicate or has an invalid group")
        mapping[proxy] = group
    if set(mapping) != proxy_set:
        raise ManifestError("reviewed subject map must cover every subject proxy exactly")

    eligible_rows = [dict(row, reviewed_group=mapping[row["subject_proxy"]])
                     for row in manifest["rows"]]
    return {
        **{key: value for key, value in manifest.items()
           if key not in {"schema_version", "stage", "training_eligible", "rows"}},
        "schema_version": ELIGIBLE_SCHEMA,
        "stage": "eligible",
        "training_eligible": True,
        "audit_manifest_sha256": audit_digest,
        "license_artifact_sha256": license_digest,
        "reviewed_subject_map_sha256": map_digest,
        "eligibility_authorization_sha256": _sha256_file(auth_path),
        "grouping_claim": "independently_reviewed_group_map",
        "rows": eligible_rows,
    }


def require_eligible_manifest(manifest: dict[str, Any]) -> None:
    if (manifest.get("schema_version") != ELIGIBLE_SCHEMA
            or manifest.get("stage") != "eligible"
            or manifest.get("training_eligible") is not True):
        raise ManifestError("only the separately finalized eligible manifest is permitted")
    if manifest.get("counters") != {"extractions": 0, "fits": 0, "predictions": 0}:
        raise ManifestError("eligibility manifest counters must begin at zero")
    evidence_fields = (
        "audit_manifest_sha256", "license_artifact_sha256",
        "reviewed_subject_map_sha256", "eligibility_authorization_sha256",
        "source_collection_sha256",
    )
    if any(not _HEX64.fullmatch(str(manifest.get(field, "")))
           for field in evidence_fields):
        raise ManifestError("eligible manifest lacks authenticated evidence commitments")
    if manifest.get("grouping_claim") != "independently_reviewed_group_map":
        raise ManifestError("eligible manifest lacks an independently reviewed group claim")
    if manifest.get("targets") != {
            "eye": [0, 1, 2], "mouth": [0, 1, 2],
            "brow": None, "action": None, "phase": None}:
        raise ManifestError("eligible target contract changed")
    if manifest.get("feature_contract") != {
            "kind": "static_clinical23_region_6d",
            "eye": list(EYE_FEATURE_NAMES),
            "mouth": list(MOUTH_FEATURE_NAMES),
            "dynamic_110d_allowed": False}:
        raise ManifestError("eligible feature contract changed")
    rows = manifest.get("rows")
    aggregate = manifest.get("aggregate")
    if (not isinstance(rows, list) or not rows
            or not isinstance(aggregate, dict)
            or aggregate.get("eligible_anchor_count") != len(rows)):
        raise ManifestError("eligible rows and aggregate count are inconsistent")
    seen: set[str] = set()
    for row in rows:
        anchor = str(row.get("anchor_key", ""))
        commitment = str(row.get("source_commitment", ""))
        if (anchor in seen or not anchor.startswith("anchor_")
                or not _HEX64.fullmatch(anchor[7:])
                or not _HEX64.fullmatch(commitment)
                or row.get("targets_commitment") != commitment
                or row.get("group_status") != "unreviewed_subject_folder_proxy"
                or not _GROUP.fullmatch(str(row.get("reviewed_group", "")))):
            raise ManifestError("eligible row provenance or reviewed group is invalid")
        seen.add(anchor)


def clinical23_region_features(clinical23: np.ndarray, target: str) -> np.ndarray:
    """Return the fixed capture-swap-invariant six-dimensional static vector."""
    values = np.asarray(clinical23, dtype=np.float64)
    if values.shape != (23,) or not np.isfinite(values).all():
        raise ManifestError("clinical23 input must be one finite static 23D frame")
    if target == "eye":
        output = (
            0.5 * (values[0] + values[1]), abs(values[0] - values[1]),
            0.5 * (values[4] + values[5]), abs(values[4] - values[5]),
            0.5 * (values[7] + values[8]), abs(values[7] - values[8]),
        )
    elif target == "mouth":
        output = (
            0.5 * (values[14] + values[15]), abs(values[14] - values[15]),
            0.5 * (values[18] + values[19]), abs(values[18] - values[19]),
            values[21], values[22],
        )
    else:
        raise ManifestError("target must be eye or mouth")
    result = np.asarray(output, dtype=np.float64)
    if result.shape != (6,) or not np.isfinite(result).all():
        raise ManifestError("regional feature transform produced invalid values")
    return result


__all__ = [
    "AUDIT_SCHEMA", "ELIGIBLE_SCHEMA", "EYE_FEATURE_NAMES",
    "MOUTH_FEATURE_NAMES", "ManifestError", "ParsedRegionalXML",
    "build_audit_manifest", "clinical23_region_features",
    "finalize_eligible_manifest", "parse_regional_xml",
    "require_eligible_manifest", "write_manifest_once",
]
