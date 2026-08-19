"""Regenerate a deidentified, provenance-locked PalsyNet MARLIN cache.

This command is intentionally fail-closed: it accepts only the complete
27-affected/22-unaffected collection and never serializes source names or paths.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import random
import secrets
import shutil
import stat
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Callable, Iterator, Mapping, NamedTuple

import numpy as np


EXPECTED_LABEL_COUNTS = {"affected": 27, "unaffected": 22}
N_CLIPS = 4
MARLIN_WIDTH = 768
BUNDLE_SCHEMA_VERSION = "palsynet_marlin_identity_v1"
OUTPUT_DIRECTORY_NAME = "identity_marlin_v1"
MANIFEST_SCHEMA_VERSION = "palsynet_marlin_extraction_manifest_v1"
PROVENANCE_SCHEMA_VERSION = "palsynet_bundle_provenance_v1"
SHA256_KEYS = {
    "marlin_model_sha256",
    "marlin_config_sha256",
    "face_landmarker_model_sha256",
}
BUNDLE_FIELDS = {
    "marlin",
    "schema_version",
    "source_sha256",
    "n_clips",
    "capture_mirrored",
    *SHA256_KEYS,
}
MANIFEST_FIELDS = {
    "schema_version",
    "dataset",
    "counts",
    "config",
    "collection_fingerprints",
    "dependency_versions",
    "asset_hashes",
    "implementation_source_tree_sha256",
    "records",
}
DEPENDENCY_KEYS = {
    "python",
    "numpy",
    "torch",
    "opencv-python",
    "mediapipe",
    "safetensors",
}
RUNTIME_NAMESPACE_PREFIXES = ("src", "marlin_vit_base_ytf", "utils")


class SourceRecord(NamedTuple):
    path: Path
    label: str
    source_sha256: str


class InputSnapshot(NamedTuple):
    implementation_root: Path
    marlin_dir: Path
    face_landmarker_model: Path
    asset_hashes: dict[str, str]
    implementation_source_tree_sha256: str
    dependency_versions: dict[str, str]


def _lexical_absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _require_real_directory(path: str | Path, what: str) -> Path:
    candidate = _lexical_absolute(path)
    try:
        info = os.lstat(candidate)
    except FileNotFoundError as exc:
        raise ValueError(f"{what} is missing") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"{what} must be a real directory")
    return candidate


def _require_real_file(path: str | Path, what: str) -> Path:
    candidate = _lexical_absolute(path)
    try:
        info = os.lstat(candidate)
    except FileNotFoundError as exc:
        raise ValueError(f"{what} is missing") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ValueError(f"{what} must be a real regular file")
    if info.st_size <= 0:
        raise ValueError(f"{what} must not be empty")
    return candidate


def _strict_tree_entries(root: str | Path, what: str) -> tuple[Path, list[tuple[Path, int]]]:
    """Inventory every descendant, surfacing unreadable trees and all links."""
    candidate = _require_real_directory(root, what)
    result: list[tuple[Path, int]] = []
    stack = [candidate]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as scanner:
                entries = sorted(scanner, key=lambda entry: entry.name)
        except OSError as exc:
            raise ValueError(f"{what} contains an unreadable directory") from exc
        child_directories = []
        for entry in entries:
            path = Path(entry.path)
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise ValueError(f"{what} contains an unreadable entry") from exc
            if stat.S_ISLNK(info.st_mode):
                raise ValueError(f"{what} must not contain symlinks")
            result.append((path, info.st_mode))
            if stat.S_ISDIR(info.st_mode):
                child_directories.append(path)
        stack.extend(reversed(child_directories))
    return candidate, result


def _assert_tree_has_no_symlinks(root: str | Path, what: str) -> Path:
    candidate, _ = _strict_tree_entries(root, what)
    return candidate


def _assert_no_symlink_components(
    path: str | Path,
    anchor: str | Path,
    what: str,
) -> Path:
    """Reject symlinks below a caller-selected real trust anchor."""
    candidate = _lexical_absolute(path)
    trusted = _require_real_directory(anchor, f"{what} trust anchor")
    try:
        relative = candidate.relative_to(trusted)
    except ValueError as exc:
        raise ValueError(f"{what} must stay within its trusted root") from exc
    current = trusted
    for part in relative.parts:
        current = current / part
        try:
            info = os.lstat(current)
        except FileNotFoundError as exc:
            raise ValueError(f"{what} is missing") from exc
        if stat.S_ISLNK(info.st_mode):
            raise ValueError(f"{what} path must not contain symlinks")
    return candidate


def _sha256_regular_file(path: str | Path) -> str:
    candidate = _lexical_absolute(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(candidate, flags)
    except (FileNotFoundError, OSError) as exc:
        raise ValueError("input file is missing or unsafe") from exc
    digest = hashlib.sha256()
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("input must be a regular file")
        with os.fdopen(fd, "rb", closefd=False) as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    finally:
        os.close(fd)
    return digest.hexdigest()


def collect_sources(video_root: str | Path) -> list[SourceRecord]:
    """Return the complete collection sorted only by content digest."""
    root, tree_entries = _strict_tree_entries(video_root, "video input tree")
    records: list[SourceRecord] = []
    for label, expected_count in EXPECTED_LABEL_COUNTS.items():
        label_root = _require_real_directory(root / label, f"{label} class directory")
        paths = []
        for child in label_root.iterdir():
            info = os.lstat(child)
            if stat.S_ISLNK(info.st_mode):
                raise ValueError("video input tree must not contain symlinks")
            if child.suffix.lower() == ".mp4":
                if not stat.S_ISREG(info.st_mode):
                    raise ValueError("video input must be a regular file")
                paths.append(child)
        if len(paths) != expected_count:
            raise ValueError(
                f"PalsyNet count gate failed for {label}: "
                f"expected {expected_count}, observed {len(paths)}"
            )
        records.extend(
            SourceRecord(path=path, label=label, source_sha256=_sha256_regular_file(path))
            for path in paths
        )
    selected_paths = {row.path for row in records}
    observed_video_paths = {
        path
        for path, mode in tree_entries
        if stat.S_ISREG(mode) and path.suffix.lower() == ".mp4"
    }
    if observed_video_paths != selected_paths:
        raise ValueError("PalsyNet must contain exactly the 49 locked class videos")
    records.sort(key=lambda row: (row.source_sha256, row.label))
    hashes = [row.source_sha256 for row in records]
    if len(set(hashes)) != len(hashes):
        raise ValueError("PalsyNet source videos must have unique SHA-256 digests")
    return records


def _hash_source_tree(implementation_root: Path, marlin_dir: Path) -> str:
    """Hash executable Python sources without serializing their local paths."""
    roots = (
        (b"implementation", implementation_root / "src"),
        (b"marlin", marlin_dir),
    )
    digest = hashlib.sha256()
    n_sources = 0
    for namespace, root in roots:
        root, tree_entries = _strict_tree_entries(root, "implementation source tree")
        source_paths = [
            path for path, mode in tree_entries
            if stat.S_ISREG(mode) and path.suffix == ".py"
        ]
        for path in sorted(source_paths, key=lambda item: item.relative_to(root).as_posix()):
            relative = path.relative_to(root).as_posix().encode("utf-8")
            file_digest = bytes.fromhex(_sha256_regular_file(path))
            digest.update(len(namespace).to_bytes(2, "big"))
            digest.update(namespace)
            digest.update(len(relative).to_bytes(4, "big"))
            digest.update(relative)
            digest.update(file_digest)
            n_sources += 1
    if n_sources < 3:
        raise ValueError("implementation source tree is incomplete")
    return digest.hexdigest()


def _default_dependency_versions() -> dict[str, str]:
    distributions = {
        "numpy": "numpy",
        "torch": "torch",
        "mediapipe": "mediapipe",
        "safetensors": "safetensors",
    }
    versions = {"python": platform.python_version()}
    for key, distribution in distributions.items():
        try:
            versions[key] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise ValueError(f"required dependency metadata is missing: {key}") from exc

    opencv_candidates = (
        "opencv-python",
        "opencv-contrib-python",
        "opencv-python-headless",
        "opencv-contrib-python-headless",
    )
    installed_opencv: list[tuple[str, str]] = []
    for distribution in opencv_candidates:
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
        installed_opencv.append((distribution, version))
    if len(installed_opencv) != 1:
        raise ValueError(
            "runtime must contain exactly one supported OpenCV distribution"
        )
    distribution, version = installed_opencv[0]
    versions["opencv-python"] = f"{distribution}=={version}"
    return versions


def _validated_dependency_versions(values: Mapping[str, object]) -> dict[str, str]:
    if not isinstance(values, Mapping) or set(values) != DEPENDENCY_KEYS:
        raise ValueError("dependency versions have an unexpected schema")
    result: dict[str, str] = {}
    for key in sorted(DEPENDENCY_KEYS):
        value = values[key]
        if (
            not isinstance(value, str)
            or not value
            or "/" in value
            or "\\" in value
            or "\x00" in value
        ):
            raise ValueError("dependency versions must be non-path strings")
        result[key] = value
    return result


def _snapshot_inputs(
    implementation_root: str | Path,
    marlin_dir: str | Path,
    face_landmarker_model: str | Path,
    dependency_versions_fn: Callable[[], Mapping[str, object]],
) -> InputSnapshot:
    implementation = _require_real_directory(implementation_root, "implementation root")
    marlin = _require_real_directory(marlin_dir, "MARLIN directory")
    expected_marlin = implementation / "data" / "external" / "marlin_vit_base_ytf"
    if marlin != expected_marlin:
        raise ValueError("MARLIN directory must belong to the explicit implementation root")
    _assert_no_symlink_components(marlin, implementation, "MARLIN directory")

    encoder_source = implementation / "src" / "models" / "backbones" / "marlin_video.py"
    landmarker_source = (
        implementation / "src" / "baselines" / "oo_multimodal" / "utils" / "__init__.py"
    )
    _assert_no_symlink_components(encoder_source, implementation, "MARLIN encoder implementation")
    _assert_no_symlink_components(
        landmarker_source, implementation, "MediaPipe landmarker implementation"
    )
    _require_real_file(encoder_source, "MARLIN encoder implementation")
    _require_real_file(landmarker_source, "MediaPipe landmarker implementation")
    config = _require_real_file(marlin / "config.json", "MARLIN config")
    model = _require_real_file(marlin / "model.safetensors", "MARLIN model")
    landmarker_path = _lexical_absolute(face_landmarker_model)
    _assert_no_symlink_components(landmarker_path, implementation, "face landmarker model")
    landmarker = _require_real_file(landmarker_path, "face landmarker model")
    try:
        parsed_config = json.loads(config.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("MARLIN config must be valid JSON") from exc
    if not isinstance(parsed_config, dict):
        raise ValueError("MARLIN config must contain a JSON object")

    asset_hashes = {
        "marlin_model_sha256": _sha256_regular_file(model),
        "marlin_config_sha256": _sha256_regular_file(config),
        "face_landmarker_model_sha256": _sha256_regular_file(landmarker),
    }
    return InputSnapshot(
        implementation_root=implementation,
        marlin_dir=marlin,
        face_landmarker_model=landmarker,
        asset_hashes=asset_hashes,
        implementation_source_tree_sha256=_hash_source_tree(implementation, marlin),
        dependency_versions=_validated_dependency_versions(dependency_versions_fn()),
    )


def _default_deterministic_setup() -> None:
    import torch

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)


def _module_file(module: object) -> Path:
    value = getattr(module, "__file__", None)
    if not isinstance(value, str):
        raise ValueError("runtime module has no verifiable source file")
    return _lexical_absolute(value)


def _namespace_module_names(prefix: str) -> list[str]:
    return sorted(
        name for name in sys.modules
        if name == prefix or name.startswith(prefix + ".")
    )


def _verify_namespace_root(prefix: str, root: Path) -> None:
    names = _namespace_module_names(prefix)
    if not names:
        raise ValueError(f"explicit runtime namespace {prefix} was not imported")
    for name in names:
        module = sys.modules[name]
        try:
            _module_file(module).relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"runtime namespace {prefix} escaped the explicit implementation root"
            ) from exc


@contextmanager
def _isolated_runtime_namespaces() -> Iterator[None]:
    """Keep explicit runtime imports installed through lazy video encoding."""
    previous_path = list(sys.path)
    previous_dont_write_bytecode = sys.dont_write_bytecode
    previous_pycache_prefix = getattr(sys, "pycache_prefix", None)
    saved = {
        name: module for name, module in list(sys.modules.items())
        if any(
            name == prefix or name.startswith(prefix + ".")
            for prefix in RUNTIME_NAMESPACE_PREFIXES
        )
    }
    for name in saved:
        sys.modules.pop(name, None)
    sys.dont_write_bytecode = True
    if hasattr(sys, "pycache_prefix"):
        sys.pycache_prefix = os.path.join(
            os.sep,
            ".facial-paralysis-source-only",
            secrets.token_hex(16),
        )
    importlib.invalidate_caches()
    try:
        yield
    finally:
        sys.path[:] = previous_path
        sys.dont_write_bytecode = previous_dont_write_bytecode
        if hasattr(sys, "pycache_prefix"):
            sys.pycache_prefix = previous_pycache_prefix
        for name in list(sys.modules):
            if any(
                name == prefix or name.startswith(prefix + ".")
                for prefix in RUNTIME_NAMESPACE_PREFIXES
            ):
                sys.modules.pop(name, None)
        sys.modules.update(saved)
        importlib.invalidate_caches()


def _default_encoder_factory(implementation_root: Path, marlin_dir: Path):
    if _namespace_module_names("src") or _namespace_module_names("marlin_vit_base_ytf"):
        raise ValueError("MARLIN runtime namespace was not freshly isolated")
    expected_module = implementation_root / "src" / "models" / "backbones" / "marlin_video.py"
    previous_path = list(sys.path)
    sys.path[:0] = [str(implementation_root), str(marlin_dir.parent)]
    try:
        module = importlib.import_module("src.models.backbones.marlin_video")
    finally:
        sys.path[:] = previous_path
    if _module_file(module) != expected_module:
        raise ValueError("MARLIN encoder was not imported from the explicit implementation root")
    importlib.import_module("marlin_vit_base_ytf")
    _verify_namespace_root("src", implementation_root)
    _verify_namespace_root("marlin_vit_base_ytf", marlin_dir)
    encoder_class = getattr(module, "MarlinVideoEncoder", None)
    if encoder_class is None:
        raise ValueError("explicit implementation has no MARLIN encoder")
    return encoder_class.from_default_weights(marlin_dir)


def _default_landmarker_factory(implementation_root: Path, model_path: Path):
    if _namespace_module_names("utils"):
        raise ValueError("landmarker runtime namespace was not freshly isolated")
    oo_root = implementation_root / "src" / "baselines" / "oo_multimodal"
    sys.path.insert(0, str(oo_root))
    try:
        module = importlib.import_module("utils")
    finally:
        if sys.path and sys.path[0] == str(oo_root):
            sys.path.pop(0)
    try:
        _module_file(module).relative_to(oo_root)
    except ValueError as exc:
        raise ValueError(
            "landmarker was not imported from the explicit implementation root"
        ) from exc
    landmarker_class = getattr(module, "MediaPipeFaceLandmarker", None)
    if landmarker_class is None:
        raise ValueError("explicit implementation has no MediaPipe landmarker")
    wrapper = importlib.import_module("utils.mediapipe_wrapper")
    expected_wrapper = oo_root / "utils" / "mediapipe_wrapper.py"
    if _module_file(wrapper) != expected_wrapper:
        raise ValueError("landmarker wrapper was not imported from the explicit root")
    if landmarker_class is not getattr(wrapper, "MediaPipeFaceLandmarker", None):
        raise ValueError("landmarker class binding does not match the explicit wrapper")
    _verify_namespace_root("utils", oo_root)

    # The upstream wrapper can provision a missing default asset.  This command
    # forbids that fallback even under a delete/race after initial validation:
    # the exact caller-provided model must still be a real local file.
    explicit_model = _lexical_absolute(model_path)

    def require_explicit_model(candidate):
        if _lexical_absolute(candidate) != explicit_model:
            raise ValueError("landmarker attempted to substitute another model")
        return _require_real_file(explicit_model, "face landmarker model")

    setattr(wrapper, "_ensure_model", require_explicit_model)
    return landmarker_class(model_path=str(model_path))


def _require_sha256(value: object, what: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{what} must be a lowercase SHA-256 digest")
    return value


def _embedding_sha256(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(b"float32:4x768\x00")
    digest.update(np.ascontiguousarray(array, dtype=np.float32).tobytes(order="C"))
    return digest.hexdigest()


def _validate_marlin_array(value: object) -> np.ndarray:
    if value is None:
        raise ValueError("MARLIN encoding is missing one or more required windows")
    array = np.asarray(value)
    if array.shape != (N_CLIPS, MARLIN_WIDTH):
        raise ValueError("MARLIN encoding must have exact shape (4, 768)")
    if array.dtype != np.dtype(np.float32):
        raise ValueError("MARLIN encoding must use exact float32 dtype")
    if not np.isrealobj(array) or not np.isfinite(array).all():
        raise ValueError("MARLIN encoding must be finite and real")
    return np.ascontiguousarray(array)


def _scalar_value(bundle: Mapping[str, np.ndarray], key: str) -> object:
    value = np.asarray(bundle[key])
    if value.shape != ():
        raise ValueError(f"bundle field {key} must be scalar")
    return value.item()


def validate_bundle(
    path: str | Path,
    expected_source_sha256: str,
    expected_asset_hashes: Mapping[str, str],
) -> tuple[str, str]:
    """Strictly reread one completed bundle and return file/embedding digests."""
    candidate = _require_real_file(path, "MARLIN bundle")
    try:
        with np.load(candidate, allow_pickle=False) as loaded:
            if set(loaded.files) != BUNDLE_FIELDS:
                raise ValueError("MARLIN bundle has an unexpected schema")
            arrays = {key: np.asarray(loaded[key]) for key in loaded.files}
    except (OSError, KeyError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("MARLIN bundle"):
            raise
        raise ValueError("cannot read a valid MARLIN bundle") from exc
    marlin = _validate_marlin_array(arrays["marlin"])
    if _scalar_value(arrays, "schema_version") != BUNDLE_SCHEMA_VERSION:
        raise ValueError("MARLIN bundle schema version mismatch")
    if _scalar_value(arrays, "source_sha256") != expected_source_sha256:
        raise ValueError("MARLIN bundle source digest mismatch")
    n_clips = np.asarray(arrays["n_clips"])
    if n_clips.shape != () or n_clips.dtype != np.dtype(np.int64) or n_clips.item() != N_CLIPS:
        raise ValueError("MARLIN bundle n_clips must be scalar int64 four")
    if _scalar_value(arrays, "capture_mirrored") != "unknown":
        raise ValueError("MARLIN bundle capture_mirrored must be unknown")
    if set(expected_asset_hashes) != SHA256_KEYS:
        raise ValueError("expected asset hashes have an unexpected schema")
    for key in sorted(SHA256_KEYS):
        expected = _require_sha256(expected_asset_hashes[key], key)
        if _scalar_value(arrays, key) != expected:
            raise ValueError(f"MARLIN bundle {key} mismatch")
    return _sha256_regular_file(candidate), _embedding_sha256(marlin)


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_write_bundle(
    bundle_path: Path,
    marlin: np.ndarray,
    source_sha256: str,
    asset_hashes: Mapping[str, str],
) -> tuple[str, str]:
    bundle_path.parent.mkdir(mode=0o700)
    temporary = bundle_path.parent / f".clip.npz.tmp.{secrets.token_hex(16)}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as handle:
            np.savez(
                handle,
                marlin=marlin,
                schema_version=np.asarray(BUNDLE_SCHEMA_VERSION),
                source_sha256=np.asarray(source_sha256),
                n_clips=np.asarray(N_CLIPS, dtype=np.int64),
                capture_mirrored=np.asarray("unknown"),
                marlin_model_sha256=np.asarray(asset_hashes["marlin_model_sha256"]),
                marlin_config_sha256=np.asarray(asset_hashes["marlin_config_sha256"]),
                face_landmarker_model_sha256=np.asarray(
                    asset_hashes["face_landmarker_model_sha256"]
                ),
            )
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    finally:
        os.close(fd)
    os.replace(temporary, bundle_path)
    _fsync_directory(bundle_path.parent)
    return validate_bundle(bundle_path, source_sha256, asset_hashes)


def _encode_all_sources(
    sources: list[SourceRecord],
    bundles_root: Path,
    snapshot: InputSnapshot,
    encoder_factory: Callable,
    landmarker_factory: Callable,
    deterministic_setup_fn: Callable,
    *,
    use_default_encoder: bool,
    use_default_landmarker: bool,
) -> list[dict[str, str]]:
    isolate = use_default_encoder or use_default_landmarker
    runtime_context = _isolated_runtime_namespaces() if isolate else nullcontext()
    generated_records: list[dict[str, str]] = []
    with runtime_context:
        deterministic_setup_fn()
        encoder = encoder_factory(snapshot.implementation_root, snapshot.marlin_dir)
        if not hasattr(encoder, "to") or not hasattr(encoder, "eval"):
            raise ValueError("MARLIN encoder does not support explicit CPU eval mode")
        encoder = encoder.to("cpu")
        encoder = encoder.eval()
        landmarker = landmarker_factory(
            snapshot.implementation_root, snapshot.face_landmarker_model
        )

        for source in sources:
            before = _sha256_regular_file(source.path)
            if before != source.source_sha256:
                raise ValueError("source changed after inventory and before encoding")
            try:
                encoded = encoder.encode_video_path(
                    source.path,
                    n_clips=N_CLIPS,
                    landmarker=landmarker,
                )
            except Exception:
                raise RuntimeError(
                    "MARLIN encoding failed for source_sha256="
                    f"{source.source_sha256}"
                ) from None
            marlin = _validate_marlin_array(encoded)
            after = _sha256_regular_file(source.path)
            if after != before:
                raise ValueError("source changed while MARLIN was encoding it")

            bundle_key = f"bundles/{source.source_sha256}/clip.npz"
            bundle_path = bundles_root / source.source_sha256 / "clip.npz"
            bundle_sha256, embedding_sha256 = _atomic_write_bundle(
                bundle_path,
                marlin,
                source.source_sha256,
                snapshot.asset_hashes,
            )
            generated_records.append(
                {
                    "source_sha256": source.source_sha256,
                    "label": source.label,
                    "bundle_key": bundle_key,
                    "bundle_sha256": bundle_sha256,
                    "embedding_sha256": embedding_sha256,
                }
            )

        if use_default_encoder:
            _verify_namespace_root("src", snapshot.implementation_root)
            _verify_namespace_root("marlin_vit_base_ytf", snapshot.marlin_dir)
        if use_default_landmarker:
            oo_root = (
                snapshot.implementation_root / "src" / "baselines" / "oo_multimodal"
            )
            _verify_namespace_root("utils", oo_root)
    return generated_records


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _atomic_write_json(path: Path, value: object) -> None:
    temporary = path.parent / f".{path.name}.tmp.{secrets.token_hex(16)}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as handle:
            handle.write(_canonical_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    finally:
        os.close(fd)
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _fingerprint_rows(rows: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(rows)).hexdigest()


def _build_manifests(
    sources: list[SourceRecord],
    generated_records: list[dict[str, str]],
    snapshot: InputSnapshot,
) -> tuple[dict, dict]:
    provenance_records = [
        {
            "source_sha256": row["source_sha256"],
            "bundle_key": row["bundle_key"],
            "bundle_sha256": row["bundle_sha256"],
        }
        for row in generated_records
    ]
    provenance = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "dataset": "PalsyNet",
        "records": provenance_records,
    }
    source_fingerprint_rows = [
        {"source_sha256": row.source_sha256, "label": row.label} for row in sources
    ]
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset": "PalsyNet",
        "counts": {"affected": 27, "unaffected": 22, "total": 49},
        "config": {
            "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
            "capture_mirrored": "unknown",
            "device": "cpu",
            "embedding_dim": MARLIN_WIDTH,
            "n_clips": N_CLIPS,
        },
        "collection_fingerprints": {
            "source_collection_sha256": _fingerprint_rows(source_fingerprint_rows),
            "bundle_collection_sha256": _fingerprint_rows(generated_records),
        },
        "dependency_versions": snapshot.dependency_versions,
        "asset_hashes": snapshot.asset_hashes,
        "implementation_source_tree_sha256": snapshot.implementation_source_tree_sha256,
        "records": generated_records,
    }
    return provenance, manifest


def _assert_inputs_unchanged(
    video_root: Path,
    sources: list[SourceRecord],
    snapshot: InputSnapshot,
) -> None:
    if collect_sources(video_root) != sources:
        raise ValueError("source collection changed during extraction")
    current = _snapshot_inputs(
        snapshot.implementation_root,
        snapshot.marlin_dir,
        snapshot.face_landmarker_model,
        lambda: snapshot.dependency_versions,
    )
    if (
        current.asset_hashes != snapshot.asset_hashes
        or current.implementation_source_tree_sha256
        != snapshot.implementation_source_tree_sha256
    ):
        raise ValueError("model, config, landmarker, or implementation changed")


def _load_json_exact(path: Path) -> object:
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("JSON object contains duplicate keys")
            result[key] = value
        return result

    try:
        return json.loads(path.read_text(), object_pairs_hook=unique_object)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("generation contains invalid JSON") from exc


def _validate_generation(
    generation_root: Path,
    sources: list[SourceRecord],
    snapshot: InputSnapshot,
    expected_provenance: dict,
    expected_manifest: dict,
) -> None:
    _assert_tree_has_no_symlinks(generation_root, "generated cache")
    if {path.name for path in generation_root.iterdir()} != {
        "bundles", "provenance.json", "extraction_manifest.json"
    }:
        raise ValueError("generation root has unexpected files")
    if _load_json_exact(generation_root / "provenance.json") != expected_provenance:
        raise ValueError("provenance reread mismatch")
    manifest = _load_json_exact(generation_root / "extraction_manifest.json")
    if manifest != expected_manifest or set(manifest) != MANIFEST_FIELDS:
        raise ValueError("extraction manifest reread mismatch")

    bundles = _require_real_directory(generation_root / "bundles", "bundle root")
    expected_hashes = {row.source_sha256 for row in sources}
    observed_hashes = {path.name for path in bundles.iterdir()}
    if observed_hashes != expected_hashes:
        raise ValueError("generation bundle coverage mismatch")
    manifest_by_hash = {row["source_sha256"]: row for row in expected_manifest["records"]}
    for source_sha256 in sorted(expected_hashes):
        directory = _require_real_directory(bundles / source_sha256, "bundle directory")
        if {path.name for path in directory.iterdir()} != {"clip.npz"}:
            raise ValueError("bundle directory has unexpected files")
        bundle_sha256, embedding_sha256 = validate_bundle(
            directory / "clip.npz", source_sha256, snapshot.asset_hashes
        )
        row = manifest_by_hash[source_sha256]
        if bundle_sha256 != row["bundle_sha256"] or embedding_sha256 != row["embedding_sha256"]:
            raise ValueError("bundle digest reread mismatch")


def _remove_path(path: Path) -> None:
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return
    if stat.S_ISLNK(info.st_mode):
        raise ValueError("transaction path must not be a symlink")
    if stat.S_ISDIR(info.st_mode):
        shutil.rmtree(path)
    else:
        os.unlink(path)


@contextmanager
def _generation_lock(parent: Path, name: str) -> Iterator[None]:
    lock = parent / f".{name}.lock"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(lock, flags, 0o600)
    except FileExistsError as exc:
        raise RuntimeError("another cache generation holds the lock") from exc
    try:
        os.write(fd, b"locked\n")
        os.fsync(fd)
        os.close(fd)
        _fsync_directory(parent)
        yield
    finally:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            info = os.lstat(lock)
        except FileNotFoundError:
            pass
        else:
            if stat.S_ISLNK(info.st_mode):
                raise ValueError("generation lock was replaced by a symlink")
            os.unlink(lock)
            _fsync_directory(parent)


def _create_output_parent(video_root: Path) -> Path:
    trusted_parent = _require_real_directory(video_root.parent, "dataset parent")
    output_parent = trusted_parent / "derived"
    try:
        os.mkdir(output_parent, mode=0o700)
        _fsync_directory(trusted_parent)
    except FileExistsError:
        _require_real_directory(output_parent, "derived output parent")
    return output_parent


def _assert_existing_target_safe(target: Path) -> None:
    try:
        info = os.lstat(target)
    except FileNotFoundError:
        return
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ValueError("existing generation target must be a real directory")
    _assert_tree_has_no_symlinks(target, "existing generation target")


def _promote_generation(staging: Path, target: Path) -> None:
    parent = target.parent
    backup = parent / f".{target.name}.backup.{secrets.token_hex(16)}"
    failed_new = parent / f".{target.name}.failed.{secrets.token_hex(16)}"
    had_previous = target.exists()
    backup_moved = False
    staging_promoted = False
    try:
        if had_previous:
            os.replace(target, backup)
            backup_moved = True
            _fsync_directory(parent)
        os.replace(staging, target)
        staging_promoted = True
        _fsync_directory(parent)
    except BaseException as promotion_error:
        try:
            if backup_moved:
                if target.exists():
                    os.replace(target, failed_new)
                if backup.exists():
                    os.replace(backup, target)
            elif staging_promoted and target.exists():
                os.replace(target, staging)
            if backup_moved or staging_promoted:
                _fsync_directory(parent)
        except BaseException as rollback_error:
            raise RuntimeError("cache promotion rollback failed") from rollback_error
        finally:
            if failed_new.exists():
                try:
                    _remove_path(failed_new)
                except OSError:
                    pass
        raise promotion_error
    if had_previous:
        # The new target is already durable.  Backup deletion is post-commit
        # cleanup; a cleanup-only filesystem error must not turn success into a
        # reported failure after the old target can no longer be reconstructed.
        try:
            _remove_path(backup)
            _fsync_directory(parent)
        except OSError:
            pass


def generate_identity_cache(
    *,
    video_root: str | Path,
    implementation_root: str | Path,
    marlin_dir: str | Path,
    face_landmarker_model: str | Path,
    output_root: str | Path,
    encoder_factory: Callable | None = None,
    landmarker_factory: Callable | None = None,
    dependency_versions_fn: Callable | None = None,
    deterministic_setup_fn: Callable | None = None,
) -> dict:
    """Build and atomically promote one complete, validated cache generation."""
    video_root = _lexical_absolute(video_root)
    sources = collect_sources(video_root)
    expected_output = video_root.parent / "derived" / OUTPUT_DIRECTORY_NAME
    output_root = _lexical_absolute(output_root)
    if output_root != expected_output:
        raise ValueError("output root must be the canonical PalsyNet derived cache location")

    use_default_encoder = encoder_factory is None
    use_default_landmarker = landmarker_factory is None
    dependency_versions_fn = dependency_versions_fn or _default_dependency_versions
    deterministic_setup_fn = deterministic_setup_fn or _default_deterministic_setup
    encoder_factory = encoder_factory or _default_encoder_factory
    landmarker_factory = landmarker_factory or _default_landmarker_factory

    # Everything that can influence extraction is validated and fingerprinted
    # before the first lock/staging write.
    snapshot = _snapshot_inputs(
        implementation_root,
        marlin_dir,
        face_landmarker_model,
        dependency_versions_fn,
    )
    _assert_existing_target_safe(output_root)
    output_parent = _create_output_parent(video_root)
    if output_root.parent != output_parent:
        raise ValueError("canonical output parent mismatch")

    staging = output_parent / f".{OUTPUT_DIRECTORY_NAME}.staging.{secrets.token_hex(16)}"
    with _generation_lock(output_parent, OUTPUT_DIRECTORY_NAME):
        staging_created = False
        try:
            os.mkdir(staging, mode=0o700)
            staging_created = True
            _fsync_directory(output_parent)
            bundles_root = staging / "bundles"
            os.mkdir(bundles_root, mode=0o700)
            _fsync_directory(staging)

            generated_records = _encode_all_sources(
                sources,
                bundles_root,
                snapshot,
                encoder_factory,
                landmarker_factory,
                deterministic_setup_fn,
                use_default_encoder=use_default_encoder,
                use_default_landmarker=use_default_landmarker,
            )

            # Catch late source edits and every extraction asset/code drift before
            # manifests are trusted or the old generation is touched.
            _assert_inputs_unchanged(video_root, sources, snapshot)

            provenance, manifest = _build_manifests(sources, generated_records, snapshot)
            _atomic_write_json(staging / "provenance.json", provenance)
            _atomic_write_json(staging / "extraction_manifest.json", manifest)
            _fsync_directory(bundles_root)
            _fsync_directory(staging)
            _validate_generation(staging, sources, snapshot, provenance, manifest)
            _fsync_directory(staging)
            _assert_inputs_unchanged(video_root, sources, snapshot)
            _promote_generation(staging, output_root)
            return manifest
        finally:
            if staging_created and staging.exists():
                _remove_path(staging)
                _fsync_directory(output_parent)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--implementation-root", required=True)
    parser.add_argument("--marlin-dir", required=True)
    parser.add_argument("--face-landmarker-model", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)
    try:
        result = generate_identity_cache(
            video_root=args.video_root,
            implementation_root=args.implementation_root,
            marlin_dir=args.marlin_dir,
            face_landmarker_model=args.face_landmarker_model,
            output_root=args.output_root,
        )
    except Exception:
        parser.exit(
            status=1,
            message=(
                "PalsyNet MARLIN regeneration failed; "
                "canonical output state requires verification.\n"
            ),
        )
    print(f"Generated {result['counts']['total']} deidentified PalsyNet bundles.")


if __name__ == "__main__":
    main()
