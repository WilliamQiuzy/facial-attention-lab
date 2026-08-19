"""Strict readers for first-party MARLIN + MediaPipe action-bundle scripts.

Historical caches predate the ordered feature-schema contract.  Callers must
opt in to those caches explicitly; this reader never invents schema metadata for
them.  Schema-versioned caches are validated against the central registry before
any array is exposed to a training or validation script.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class BundleArrays:
    marlin: np.ndarray
    mp_seq: np.ndarray
    mp_mask: np.ndarray
    mp_feature_schema: str | None
    mp_feature_names: tuple[str, ...] | None
    mp_side_convention: str | None
    mp_capture_mirrored: str | None


def _core_imports():
    """Support both ``python scripts/foo.py`` and package-style imports."""
    try:
        from src.datasets.patient_multistream import ActionBundle
        from src.preprocessing.action_bundle import _assert_existing_cache_schema
    except ModuleNotFoundError:  # imported as facial_paralysis.scripts.*
        from facial_paralysis.src.datasets.patient_multistream import ActionBundle
        from facial_paralysis.src.preprocessing.action_bundle import (
            _assert_existing_cache_schema,
        )
    return ActionBundle, _assert_existing_cache_schema


def _scalar_text(value: np.ndarray, field: str, path: Path) -> str:
    try:
        return str(np.asarray(value).item())
    except ValueError as exc:
        raise RuntimeError(f"{path}: {field} must be a scalar string") from exc


def load_bundle_arrays(
    path: str | Path,
    *,
    allow_legacy_schema: bool,
    expected_feat_dim: int | None = None,
) -> BundleArrays:
    """Load one two-stream action bundle without weakening its disk contract.

    ``allow_legacy_schema=True`` accepts an audited old cache whose MediaPipe
    stream has no schema id.  Such a cache remains metadata-free in the returned
    value; only schema-aware files receive schema/name/side/mirror provenance.
    """
    path = Path(path)
    try:
        with np.load(path, allow_pickle=False) as cached:
            fields = set(cached.files)
            missing = {"marlin", "mp_seq", "mp_mask"} - fields
            if missing:
                raise RuntimeError(f"{path}: missing bundle fields {sorted(missing)}")
            schema = (
                _scalar_text(cached["mp_feature_schema"], "mp_feature_schema", path)
                if "mp_feature_schema" in fields else None
            )
            # A legacy cache may carry only mp_feat_dim.  Names/side/mirror,
            # however, are versioned provenance and are meaningless without the
            # schema id that fixes their interpretation.
            orphaned = {
                "mp_feature_names", "mp_side_convention", "mp_capture_mirrored"
            } & fields if schema is None else set()
            if orphaned:
                raise RuntimeError(
                    f"{path}: schema provenance {sorted(orphaned)} has no "
                    "mp_feature_schema"
                )
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"cannot inspect action bundle {path}") from exc

    if schema is None:
        if not allow_legacy_schema:
            raise RuntimeError(
                f"{path}: metadata-free legacy cache requires "
                "allow_legacy_schema=True"
            )
    else:
        _, assert_schema = _core_imports()
        # Use the id actually stored on disk.  The validator binds that id to
        # exact ordered names, feature width, side convention, and mirror value.
        assert_schema(path, schema)

    try:
        with np.load(path, allow_pickle=False) as cached:
            marlin = np.asarray(cached["marlin"])
            seq = np.asarray(cached["mp_seq"])
            mask = np.asarray(cached["mp_mask"])
            stored_dim = (
                int(np.asarray(cached["mp_feat_dim"]).item())
                if "mp_feat_dim" in cached.files else None
            )
            names = (
                tuple(str(x) for x in np.asarray(cached["mp_feature_names"]).tolist())
                if schema is not None else None
            )
            side = (
                _scalar_text(cached["mp_side_convention"], "mp_side_convention", path)
                if schema is not None else None
            )
            mirror = (
                _scalar_text(cached["mp_capture_mirrored"],
                             "mp_capture_mirrored", path)
                if schema is not None else None
            )
    except (OSError, ValueError, KeyError) as exc:
        raise RuntimeError(f"cannot load action bundle {path}") from exc

    if not np.issubdtype(marlin.dtype, np.number) or not np.isfinite(marlin).all():
        raise RuntimeError(f"{path}: MARLIN embeddings must be numeric and finite")
    if marlin.ndim not in (1, 2):
        raise RuntimeError(f"{path}: marlin must be 1D or 2D, got {marlin.shape}")
    if not np.issubdtype(seq.dtype, np.number) or not np.isfinite(seq).all():
        raise RuntimeError(f"{path}: MediaPipe values must be numeric and finite")
    if seq.ndim != 2:
        raise RuntimeError(f"{path}: mp_seq must be 2D, got {seq.shape}")
    if mask.dtype != np.bool_:
        raise RuntimeError(f"{path}: mp_mask must be stored as bool, got {mask.dtype}")
    if mask.shape != (seq.shape[0],):
        raise RuntimeError(
            f"{path}: mp_mask shape {mask.shape} does not match mp_seq {seq.shape}"
        )
    if stored_dim is not None and stored_dim != seq.shape[1]:
        raise RuntimeError(
            f"{path}: mp_feat_dim={stored_dim} but mp_seq has {seq.shape[1]} columns"
        )
    if expected_feat_dim is not None and seq.shape[1] != expected_feat_dim:
        raise RuntimeError(
            f"{path}: MediaPipe width {seq.shape[1]} != expected {expected_feat_dim}"
        )

    return BundleArrays(
        marlin=marlin.astype(np.float32, copy=True),
        mp_seq=seq.astype(np.float32, copy=True),
        mp_mask=mask.copy(),
        mp_feature_schema=schema,
        mp_feature_names=names,
        mp_side_convention=side,
        mp_capture_mirrored=mirror,
    )


def load_action_bundle(
    path: str | Path,
    *,
    allow_legacy_schema: bool,
    expected_feat_dim: int | None = None,
):
    """Strictly load arrays and retain exact disk provenance in ActionBundle."""
    ActionBundle, _ = _core_imports()
    arrays = load_bundle_arrays(
        path,
        allow_legacy_schema=allow_legacy_schema,
        expected_feat_dim=expected_feat_dim,
    )
    return ActionBundle(
        marlin=arrays.marlin,
        mp_seq=arrays.mp_seq,
        mp_mask=arrays.mp_mask,
        mp_feature_schema=arrays.mp_feature_schema,
        mp_feature_names=arrays.mp_feature_names,
        mp_side_convention=arrays.mp_side_convention,
        mp_capture_mirrored=arrays.mp_capture_mirrored,
    )
