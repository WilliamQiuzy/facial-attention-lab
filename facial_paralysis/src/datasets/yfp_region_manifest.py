"""Provenance-locked YFP static regional-severity manifests.

This module deliberately treats subject folders as unreviewed grouping proxies
and never promotes an audit manifest in place.  Source XML/BMP bytes are
read-only; the sole tolerated XML repair is one missing terminal
``</annotation>`` appended in memory.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
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

# Fail closed until a separately reviewed activation commit pins the exact
# authorization artifact. Runtime trust must not come from the eligible JSON.
PINNED_YFP_AUTHORIZATION_SHA256: str | None = None

EYE_LABELS = {"Normal_Eyes": 0, "SlightPalsy_Eyes": 1, "StrongPalsy_Eyes": 2}
MOUTH_LABELS = {"Normal_Mouth": 0, "SlightPalsy_Mouth": 1, "StrongPalsy_Mouth": 2}

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


@dataclass(frozen=True)
class _SecureArtifact:
    canonical_path: str
    data: bytes
    sha256: str
    identity: tuple[int, int]


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


def _valid_digest(value: Any) -> bool:
    text = str(value)
    return bool(_HEX64.fullmatch(text)) and text != "0" * 64


def _secure_read_artifact(path: str | Path, label: str) -> _SecureArtifact:
    """Read and hash one regular non-symlink file from a stable descriptor."""
    path = Path(path).expanduser()
    try:
        leaf = path.lstat()
    except OSError as exc:
        raise ManifestError(f"cannot open {label}") from exc
    if stat.S_ISLNK(leaf.st_mode) or not stat.S_ISREG(leaf.st_mode):
        raise ManifestError(f"{label} must be a regular non-symlink file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ManifestError(f"cannot securely open {label}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ManifestError(f"{label} descriptor is not a regular file")
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
        raise ManifestError(f"{label} changed while it was authenticated")
    try:
        current = os.stat(path, follow_symlinks=False)
        canonical = str(path.resolve(strict=True))
    except OSError as exc:
        raise ManifestError(f"{label} changed after it was authenticated") from exc
    if ((current.st_dev, current.st_ino) != (before.st_dev, before.st_ino)
            or stat.S_ISLNK(current.st_mode)):
        raise ManifestError(f"{label} path was replaced during authentication")
    return _SecureArtifact(
        canonical_path=canonical,
        data=b"".join(chunks),
        sha256=digest.hexdigest(),
        identity=(before.st_dev, before.st_ino),
    )


def _secure_json_artifact(path: str | Path, label: str) -> tuple[_SecureArtifact, dict[str, Any]]:
    artifact = _secure_read_artifact(path, label)
    try:
        value = json.loads(artifact.data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestError(f"{label} must be a UTF-8 JSON object") from exc
    if not isinstance(value, dict):
        raise ManifestError(f"{label} must be a JSON object")
    return artifact, value


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


def _decoded_xml_and_encoding(data: bytes) -> tuple[str, str, bytes]:
    """Decode once so security inspection and parsing consume identical text."""
    encodings = (
        (b"\x00\x00\xfe\xff", "utf-32-be"),
        (b"\xff\xfe\x00\x00", "utf-32-le"),
        (b"\xfe\xff", "utf-16-be"),
        (b"\xff\xfe", "utf-16-le"),
        (b"\xef\xbb\xbf", "utf-8"),
    )
    bom = b""
    encoding: str | None = None
    for prefix, candidate in encodings:
        if data.startswith(prefix):
            bom, encoding = prefix, candidate
            break
    text: str | None = None
    if encoding is None and b"\x00" in data:
        candidates: list[tuple[str, str]] = []
        for candidate in ("utf-32-be", "utf-32-le", "utf-16-be", "utf-16-le"):
            try:
                candidate_text = data.decode(candidate, errors="strict")
            except UnicodeDecodeError:
                continue
            if "\x00" not in candidate_text and candidate_text.lstrip().startswith("<"):
                candidates.append((candidate, candidate_text))
        if len(candidates) != 1:
            raise ManifestError("unsupported, ambiguous, or invalid wide XML encoding")
        encoding, text = candidates[0]
    if encoding is None:
        encoding = "utf-8"
        declaration = re.match(
            br"\s*<\?xml\s+[^>]*encoding\s*=\s*['\"]([^'\"]+)['\"]",
            data[:512], flags=re.IGNORECASE)
        if declaration:
            try:
                requested = declaration.group(1).decode("ascii", errors="strict").lower()
            except UnicodeDecodeError as exc:
                raise ManifestError("unsupported or unsafe XML encoding") from exc
            allowed = {
                "utf-8": "utf-8", "us-ascii": "ascii", "ascii": "ascii",
                "iso-8859-1": "iso-8859-1", "latin-1": "iso-8859-1",
            }
            if requested not in allowed:
                raise ManifestError("unsupported or unsafe XML encoding")
            encoding = allowed[requested]
    if text is None:
        try:
            text = data[len(bom):].decode(encoding, errors="strict")
        except (UnicodeDecodeError, LookupError) as exc:
            raise ManifestError("XML encoding is invalid") from exc
    if re.search(r"<!\s*(?:doctype|entity)\b", text, flags=re.IGNORECASE):
        raise ManifestError("DTD and entity declarations are forbidden")
    return text, encoding, bom


def parse_regional_xml(path: str | Path) -> ParsedRegionalXML:
    """Parse one combined regional XML with exactly one allowed EOF repair."""
    path = Path(path)
    data = path.read_bytes()
    decoded, _encoding, _bom = _decoded_xml_and_encoding(data)
    try:
        root = ET.fromstring(decoded)
        status = "native"
    except ET.ParseError as native_error:
        stripped = decoded.rstrip()
        root_text = stripped.lstrip()
        if root_text.startswith("<?xml"):
            declaration_end = root_text.find("?>")
            if declaration_end < 0:
                raise ManifestError(
                    "XML is malformed beyond the allowed terminal repair"
                ) from native_error
            root_text = root_text[declaration_end + 2:].lstrip()
        if (not root_text.startswith("<annotation")
                or root_text.endswith("</annotation>")):
            raise ManifestError("XML is malformed beyond the allowed terminal repair") from native_error
        repaired = stripped + "</annotation>"
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
    reserved1, reserved2 = struct.unpack_from("<HH", header, 6)
    dib_size = struct.unpack_from("<I", header, 14)[0]
    clr_used, clr_important = struct.unpack_from("<II", header, 46)
    if (dib_size != 40 or pixel_offset != 54
            or reserved1 != 0 or reserved2 != 0
            or clr_used != 0 or clr_important != 0):
        raise ManifestError("unsupported_bmp_layout")
    width, raw_height = struct.unpack_from("<ii", header, 18)
    planes, bit_count = struct.unpack_from("<HH", header, 26)
    compression = struct.unpack_from("<I", header, 30)[0]
    declared_image_size = struct.unpack_from("<I", header, 34)[0]
    if width <= 0 or raw_height == 0 or planes != 1:
        raise ManifestError("invalid_bmp_dimensions")
    height = abs(raw_height)
    if compression != 0:
        raise ManifestError("unsupported_bmp_compression")
    if bit_count not in (24, 32):
        raise ManifestError("unsupported_bmp_bit_depth")
    if declared_size != actual_size or pixel_offset >= actual_size:
        raise ManifestError("truncated_bmp")
    row_bytes = ((width * bit_count + 31) // 32) * 4
    required_image_size = row_bytes * height
    if (declared_image_size != required_image_size
            or pixel_offset + required_image_size != actual_size):
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
    """Rebuild the complete audit and require canonical content equality."""
    _validate_audit_shape(manifest)
    root = Path(manifest.get("source_root", ""))
    if not root.exists() or root.is_symlink():
        raise ManifestError("audit source root is unavailable or unsafe")
    rebuilt = build_audit_manifest(root)
    if _canonical_json_bytes(rebuilt) != _canonical_json_bytes(manifest):
        raise ManifestError(
            "audit does not exactly match the rebuilt source inventory, aggregate, "
            "collection commitment, anchors, targets, filenames, parse states, and grouping"
        )


def _artifact_reference(artifact: _SecureArtifact) -> dict[str, str]:
    return {"path": artifact.canonical_path, "sha256": artifact.sha256}


def _construct_eligible_successor(
    audit_artifact: _SecureArtifact,
    manifest: dict[str, Any],
    license_artifact: _SecureArtifact,
    map_artifact: _SecureArtifact,
    subject_map: dict[str, Any],
    authorization_artifact: _SecureArtifact,
    authorization: dict[str, Any],
) -> dict[str, Any]:
    artifacts = (audit_artifact, license_artifact, map_artifact, authorization_artifact)
    if len({artifact.identity for artifact in artifacts}) != len(artifacts):
        raise ManifestError("audit, license, reviewed map, and authorization must be distinct files")
    if not license_artifact.data:
        raise ManifestError("license artifact must be nonempty")
    if any(not _valid_digest(artifact.sha256) for artifact in artifacts):
        raise ManifestError("evidence artifact digest is invalid")
    _revalidate_audit_sources(manifest)
    audit_digest = audit_artifact.sha256
    license_digest = license_artifact.sha256
    map_digest = map_artifact.sha256
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
        "eligibility_authorization_sha256": authorization_artifact.sha256,
        "evidence_artifacts": {
            "audit_manifest": _artifact_reference(audit_artifact),
            "license_artifact": _artifact_reference(license_artifact),
            "reviewed_subject_map": _artifact_reference(map_artifact),
            "eligibility_authorization": _artifact_reference(authorization_artifact),
        },
        "grouping_claim": "independently_reviewed_group_map",
        "rows": eligible_rows,
    }


def finalize_eligible_manifest(
    audit_manifest: str | Path,
    license_artifact: str | Path,
    reviewed_subject_map: str | Path,
    eligibility_authorization: str | Path,
) -> dict[str, Any]:
    """Authenticate four independent evidence files and create a successor."""
    audit_file, manifest = _secure_json_artifact(audit_manifest, "audit manifest")
    license_file = _secure_read_artifact(license_artifact, "license artifact")
    map_file, subject_map = _secure_json_artifact(
        reviewed_subject_map, "reviewed subject map")
    authorization_file, authorization = _secure_json_artifact(
        eligibility_authorization, "eligibility authorization")
    return _construct_eligible_successor(
        audit_file, manifest, license_file, map_file, subject_map,
        authorization_file, authorization,
    )


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
    if any(not _valid_digest(manifest.get(field, ""))
           for field in evidence_fields):
        raise ManifestError("eligible manifest lacks authenticated evidence commitments")
    references = manifest.get("evidence_artifacts")
    expected_reference_keys = {
        "audit_manifest", "license_artifact", "reviewed_subject_map",
        "eligibility_authorization",
    }
    if not isinstance(references, dict) or set(references) != expected_reference_keys:
        raise ManifestError("eligible manifest lacks exact evidence artifact references")
    for reference in references.values():
        if (not isinstance(reference, dict) or set(reference) != {"path", "sha256"}
                or not isinstance(reference["path"], str)
                or not Path(reference["path"]).is_absolute()
                or not _valid_digest(reference["sha256"])):
            raise ManifestError("eligible evidence artifact reference is malformed")
    if (references["audit_manifest"]["sha256"] != manifest["audit_manifest_sha256"]
            or references["license_artifact"]["sha256"] != manifest["license_artifact_sha256"]
            or references["reviewed_subject_map"]["sha256"] != manifest["reviewed_subject_map_sha256"]
            or references["eligibility_authorization"]["sha256"] != manifest["eligibility_authorization_sha256"]):
        raise ManifestError("eligible evidence references disagree with top-level commitments")
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
                or not _valid_digest(anchor[7:])
                or not _valid_digest(commitment)
                or row.get("targets_commitment") != commitment
                or row.get("group_status") != "unreviewed_subject_folder_proxy"
                or not _GROUP.fullmatch(str(row.get("reviewed_group", "")))):
            raise ManifestError("eligible row provenance or reviewed group is invalid")
        seen.add(anchor)


def authenticate_eligible_manifest(
    manifest_path: str | Path,
    *,
    return_digest: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], str]:
    """Reopen all evidence, rebuild the audit, and reconstruct the successor."""
    pinned_authorization = PINNED_YFP_AUTHORIZATION_SHA256
    if not _valid_digest(pinned_authorization):
        raise ManifestError(
            "YFP extraction and training are locked until a separately reviewed "
            "authorization digest is pinned"
        )
    eligible_file, manifest = _secure_json_artifact(manifest_path, "eligible manifest")
    require_eligible_manifest(manifest)
    if manifest["eligibility_authorization_sha256"] != pinned_authorization:
        raise ManifestError("eligible manifest is not bound to the pinned authorization")
    references = manifest["evidence_artifacts"]
    audit_file, audit = _secure_json_artifact(
        references["audit_manifest"]["path"], "audit manifest")
    license_file = _secure_read_artifact(
        references["license_artifact"]["path"], "license artifact")
    map_file, subject_map = _secure_json_artifact(
        references["reviewed_subject_map"]["path"], "reviewed subject map")
    authorization_file, authorization = _secure_json_artifact(
        references["eligibility_authorization"]["path"],
        "eligibility authorization",
    )
    evidence = (audit_file, license_file, map_file, authorization_file)
    if eligible_file.identity in {artifact.identity for artifact in evidence}:
        raise ManifestError("eligible manifest cannot serve as its own evidence")
    for key, artifact in zip(
        ("audit_manifest", "license_artifact", "reviewed_subject_map",
         "eligibility_authorization"),
        evidence,
    ):
        reference = references[key]
        if (artifact.canonical_path != reference["path"]
                or artifact.sha256 != reference["sha256"]):
            raise ManifestError(f"{key.replace('_', ' ')} evidence digest or path changed")
    reconstructed = _construct_eligible_successor(
        audit_file, audit, license_file, map_file, subject_map,
        authorization_file, authorization,
    )
    if _canonical_json_bytes(reconstructed) != _canonical_json_bytes(manifest):
        raise ManifestError("eligible manifest is not the exact reconstructed successor")
    if return_digest:
        return manifest, eligible_file.sha256
    return manifest


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
    "MOUTH_FEATURE_NAMES", "ManifestError", "PINNED_YFP_AUTHORIZATION_SHA256",
    "ParsedRegionalXML",
    "authenticate_eligible_manifest", "build_audit_manifest", "clinical23_region_features",
    "finalize_eligible_manifest", "parse_regional_xml",
    "require_eligible_manifest", "write_manifest_once",
]
