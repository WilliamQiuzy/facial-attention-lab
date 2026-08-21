"""Multi-stream per-patient dataset for the MARLIN + MediaPipe pipeline.

Each item bundles, per action: cached frozen-MARLIN clip embeddings and a
MediaPipe per-frame feature sequence (docs/model_design.md §2/§8). Supports the
heterogeneous-label setup: every record carries a `task` name (e.g. "hb",
"binary") and a label valid for that task, so a batch can mix datasets and
`multitask_loss` routes each sample to its own head.

On-disk `.npz` per `<cache>/<patient_id>/<action>.npz`:
    marlin   (W, 768)  float32   — MARLIN clip embeddings (>=1 window)
    mp_seq   (T, F)     float32   — MediaPipe per-frame features
    mp_mask  (T,)       bool      — True = real frame
Back-compatible: if only the legacy `embeddings` (Wf, 768) key is present, it is
read as `marlin` and the MediaPipe stream is treated as empty.

Batch (from `collate_multistream`):
    marlin_emb  (B, A, W, 768) float32   marlin_mask (B, A, W) bool
    mp_seq      (B, A, T, F)   float32   mp_mask     (B, A, T) bool
    action_present (B, A) bool  label (B,) long  task_ids list[str]  patient_id list[str]
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from ..preprocessing.clinical_landmarks import (
    CLINICAL_LANDMARK_NAMES,
    CLINICAL_SIDE_CONVENTION,
)


_MEDIAPIPE_BLENDSHAPE_NAMES: tuple[str, ...] = (
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight", "cheekPuff", "cheekSquintLeft",
    "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft",
    "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft",
    "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft",
    "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft",
    "jawOpen", "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight",
    "mouthFrownLeft", "mouthFrownRight", "mouthFunnel", "mouthLeft",
    "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft",
    "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower",
    "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft",
    "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight",
)
_MEDIAPIPE_ASYMMETRY_NAMES = tuple(
    f"delta_left_minus_right_{name[:-4]}"
    for name in _MEDIAPIPE_BLENDSHAPE_NAMES
    if name.endswith("Left") and name[:-4] + "Right" in _MEDIAPIPE_BLENDSHAPE_NAMES
)
_MEDIAPIPE_BASE_NAMES = _MEDIAPIPE_BLENDSHAPE_NAMES + _MEDIAPIPE_ASYMMETRY_NAMES
_LEGACY_GEOMETRY_NAMES = (
    "ear_right", "ear_left", "ear_asym", "brow_asym", "mouthcorner_asym",
)

# A schema version is a complete column-order contract, not merely a dimension.
MP_FEATURE_NAMES_BY_SCHEMA: dict[str, tuple[str, ...]] = {
    "mediapipe_bs_lr_v1": _MEDIAPIPE_BASE_NAMES,
    "mediapipe_bs_lr_v1+legacy_geometry5_v1": (
        _MEDIAPIPE_BASE_NAMES + _LEGACY_GEOMETRY_NAMES
    ),
    "mediapipe_bs_lr_v1+clinical23_v2": (
        _MEDIAPIPE_BASE_NAMES + CLINICAL_LANDMARK_NAMES
    ),
}
MP_SIDE_CONVENTION_BY_SCHEMA: dict[str, str] = {
    "mediapipe_bs_lr_v1": "mediapipe_left_right_labels_capture_mirror_required",
    "mediapipe_bs_lr_v1+legacy_geometry5_v1": (
        "mediapipe_labels_plus_legacy_mesh_topology_capture_mirror_required"
    ),
    "mediapipe_bs_lr_v1+clinical23_v2": CLINICAL_SIDE_CONVENTION,
}
_CAPTURE_MIRRORED_VALUES = {"true", "false", "unknown"}

STANDARD_ACTIONS: tuple[str, ...] = (
    "rest", "brow_raise", "light_eye_closure", "forced_eye_closure", "smile",
)


@dataclass
class ActionBundle:
    """One action's two streams. None means the action/stream is absent."""
    marlin: np.ndarray | None = None     # (W, 768)
    mp_seq: np.ndarray | None = None     # (T, F)
    mp_mask: np.ndarray | None = None    # (T,) bool
    mp_feature_schema: str | None = None
    mp_feature_names: tuple[str, ...] | None = None
    mp_side_convention: str | None = None
    mp_capture_mirrored: str | None = None


@dataclass
class MultiStreamRecord:
    patient_id: str
    label: int
    task: str = "hb"
    actions: list[ActionBundle] = field(default_factory=list)   # len == n_actions


def _read_labels_csv(path: Path) -> dict[str, dict]:
    """Returns {patient_id: {label, task}}. Columns: patient_id, hb_grade (1..6);
    optional `task` (default 'hb') and `label` (0-indexed, used when task != hb)."""
    out: dict[str, dict] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            pid = row["patient_id"].strip()
            task = (row.get("task") or "hb").strip()
            if "label" in row and row["label"] not in (None, ""):
                label = int(row["label"])
            else:
                g = int(row["hb_grade"])
                if not 1 <= g <= 6:
                    raise ValueError(f"hb_grade out of range in {path}: {row}")
                label = g - 1
            out[pid] = {"label": label, "task": task}
    return out


def _pad_stack(arrays: list[np.ndarray | None], feat_dim: int):
    """List (len A) of (n_i, feat_dim) or None -> (A, max_n, feat_dim) float32 +
    (A, max_n) bool mask. Empty/None rows stay all-False."""
    A = len(arrays)
    max_n = max((a.shape[0] for a in arrays if a is not None and a.size), default=1)
    out = np.zeros((A, max_n, feat_dim), dtype=np.float32)
    mask = np.zeros((A, max_n), dtype=bool)
    for i, a in enumerate(arrays):
        if a is None or a.size == 0:
            continue
        n = a.shape[0]
        out[i, :n] = a.astype(np.float32, copy=False)
        mask[i, :n] = True
    return out, mask


class MultiStreamPatientDataset(Dataset):
    def __init__(
        self,
        records: Sequence[MultiStreamRecord],
        actions: Sequence[str] = STANDARD_ACTIONS,
        marlin_dim: int = 768,
        mp_feat_dim: int = 72,
        mp_feature_schema: str | None = None,
    ):
        self.records = list(records)
        self.actions = list(actions)
        self.marlin_dim = marlin_dim
        self.mp_feat_dim = mp_feat_dim
        self.mp_feature_schema = mp_feature_schema
        observed_schemas: set[str] = set()
        observed_names: set[tuple[str, ...]] = set()
        observed_side_conventions: set[str] = set()
        observed_capture_mirrored: set[str] = set()
        mp_schema_values: list[str | None] = []
        for r in self.records:
            if len(r.actions) != len(self.actions):
                raise ValueError(
                    f"patient {r.patient_id}: {len(r.actions)} action slots, "
                    f"expected {len(self.actions)}"
                )
            for action_i, bundle in enumerate(r.actions):
                if bundle.marlin is not None:
                    marlin = np.asarray(bundle.marlin)
                    if marlin.ndim != 2 or marlin.shape[1] != self.marlin_dim:
                        raise ValueError(
                            f"patient {r.patient_id} action {action_i}: marlin must have "
                            f"shape (W, {self.marlin_dim}), got {marlin.shape}"
                        )
                    if (not np.issubdtype(marlin.dtype, np.number)
                            or not np.isfinite(marlin).all()):
                        raise ValueError(
                            f"patient {r.patient_id} action {action_i}: "
                            "marlin embeddings must be finite"
                        )
                if bundle.mp_seq is not None:
                    mp_schema_values.append(bundle.mp_feature_schema)
                    if bundle.mp_seq.ndim != 2:
                        raise ValueError(
                            f"patient {r.patient_id} action {action_i}: mp_seq must be 2D, "
                            f"got {bundle.mp_seq.shape}"
                        )
                    if bundle.mp_seq.shape[1] != self.mp_feat_dim:
                        raise ValueError(
                            f"patient {r.patient_id} action {action_i}: mp_seq feature dim "
                            f"{bundle.mp_seq.shape[1]} != requested {self.mp_feat_dim}"
                        )
                    expected_mask_shape = (bundle.mp_seq.shape[0],)
                    if bundle.mp_mask is None:
                        raise ValueError(
                            f"patient {r.patient_id} action {action_i}: "
                            "mp_mask is required when mp_seq is present"
                        )
                    mask = np.asarray(bundle.mp_mask)
                    if mask.shape != expected_mask_shape:
                        raise ValueError(
                            f"patient {r.patient_id} action {action_i}: mp_mask shape "
                            f"{mask.shape} != {expected_mask_shape}"
                        )
                    if mask.dtype != np.bool_:
                        raise ValueError(
                            f"patient {r.patient_id} action {action_i}: mp_mask must be bool"
                        )
                    if (not np.issubdtype(bundle.mp_seq.dtype, np.number)
                            or not np.isfinite(bundle.mp_seq[mask]).all()):
                        raise ValueError(
                            f"patient {r.patient_id} action {action_i}: "
                            "valid MediaPipe frames must be finite"
                        )
                    if bundle.mp_feature_schema is not None:
                        observed_schemas.add(bundle.mp_feature_schema)
                        expected_names = MP_FEATURE_NAMES_BY_SCHEMA.get(
                            bundle.mp_feature_schema)
                        if expected_names is None:
                            raise ValueError(
                                f"patient {r.patient_id} action {action_i}: unknown "
                                f"MediaPipe feature schema {bundle.mp_feature_schema!r}"
                            )
                        if bundle.mp_feature_names != expected_names:
                            raise ValueError(
                                f"patient {r.patient_id} action {action_i}: feature names do "
                                f"not match schema {bundle.mp_feature_schema!r}"
                            )
                        observed_names.add(bundle.mp_feature_names)
                        expected_side = MP_SIDE_CONVENTION_BY_SCHEMA[
                            bundle.mp_feature_schema]
                        if bundle.mp_side_convention != expected_side:
                            raise ValueError(
                                f"patient {r.patient_id} action {action_i}: side convention "
                                f"{bundle.mp_side_convention!r} does not match schema "
                                f"{bundle.mp_feature_schema!r}"
                            )
                        if bundle.mp_capture_mirrored not in _CAPTURE_MIRRORED_VALUES:
                            raise ValueError(
                                f"patient {r.patient_id} action {action_i}: "
                                "mp_capture_mirrored must be true, false, or unknown"
                            )
                        observed_side_conventions.add(bundle.mp_side_convention)
                        observed_capture_mirrored.add(bundle.mp_capture_mirrored)
                    elif (bundle.mp_feature_names is not None
                          or bundle.mp_side_convention is not None
                          or bundle.mp_capture_mirrored is not None):
                        raise ValueError(
                            f"patient {r.patient_id} action {action_i}: MediaPipe schema "
                            "metadata requires an mp_feature_schema"
                        )
        if observed_schemas and any(schema is None for schema in mp_schema_values):
            raise ValueError(
                "cannot mix schema-versioned and metadata-free MediaPipe feature streams"
            )
        if len(observed_schemas) > 1:
            raise ValueError(f"mixed MediaPipe feature schemas: {sorted(observed_schemas)}")
        if (mp_schema_values and self.mp_feature_schema is not None
                and observed_schemas != {self.mp_feature_schema}):
            raise ValueError(
                f"observed schema {sorted(observed_schemas)} != requested {self.mp_feature_schema!r}"
            )
        if self.mp_feature_schema is None and observed_schemas:
            self.mp_feature_schema = next(iter(observed_schemas))
        if len(observed_names) > 1:
            raise ValueError("mixed MediaPipe feature-name layouts")
        self.mp_feature_names = next(iter(observed_names)) if observed_names else None
        if len(observed_side_conventions) > 1:
            raise ValueError("mixed MediaPipe side conventions")
        if len(observed_capture_mirrored) > 1:
            raise ValueError(
                "mixed capture-mirror provenance; canonicalize sides before training"
            )
        self.mp_side_convention = (
            next(iter(observed_side_conventions)) if observed_side_conventions else None
        )
        self.mp_capture_mirrored = (
            next(iter(observed_capture_mirrored)) if observed_capture_mirrored else None
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> dict:
        r = self.records[i]
        marlin = [b.marlin for b in r.actions]
        mp: list[np.ndarray | None] = []
        for action_i, bundle in enumerate(r.actions):
            if bundle.mp_seq is None:
                mp.append(None)
                continue
            safe = np.asarray(bundle.mp_seq, dtype=np.float32).copy()
            mask = np.asarray(bundle.mp_mask, dtype=bool)
            safe[~mask] = 0.0
            if not np.isfinite(safe).all():
                raise ValueError(
                    f"patient {r.patient_id} action {action_i}: "
                    "non-finite valid MediaPipe value observed after dataset construction"
                )
            mp.append(safe)
        marlin_emb, marlin_mask = _pad_stack(marlin, self.marlin_dim)
        mp_seq, mp_mask_from_seq = _pad_stack(mp, self.mp_feat_dim)
        # Prefer an explicit mp_mask if provided; else derive from presence.
        mp_mask = mp_mask_from_seq.copy()
        for a_i, b in enumerate(r.actions):
            if b.mp_mask is not None and b.mp_seq is not None and b.mp_seq.size:
                n = b.mp_seq.shape[0]
                mp_mask[a_i, :n] = b.mp_mask.astype(bool, copy=False)
        action_present = marlin_mask.any(axis=1) | mp_mask.any(axis=1)
        return {
            "patient_id": r.patient_id,
            "task": r.task,
            "label": int(r.label),
            "marlin_emb": torch.from_numpy(marlin_emb),
            "marlin_mask": torch.from_numpy(marlin_mask),
            "mp_seq": torch.from_numpy(mp_seq),
            "mp_mask": torch.from_numpy(mp_mask),
            "action_present": torch.from_numpy(action_present),
        }

    # ------------------------------------------------------------------
    @classmethod
    def from_disk(
        cls,
        cache_root: str | Path,
        labels_csv: str | Path,
        actions: Sequence[str] = STANDARD_ACTIONS,
        marlin_dim: int = 768,
        mp_feat_dim: int = 72,
        mp_feature_schema: str | None = None,
        allow_legacy_schema: bool = False,
    ) -> "MultiStreamPatientDataset":
        cache_root = Path(cache_root)
        labels = _read_labels_csv(Path(labels_csv))
        records: list[MultiStreamRecord] = []
        for pid in sorted(labels):
            bundles: list[ActionBundle] = []
            any_present = False
            for action in actions:
                p = cache_root / pid / f"{action}.npz"
                if not p.exists():
                    bundles.append(ActionBundle())
                    continue
                with np.load(p, allow_pickle=False) as d:
                    marlin = d["marlin"] if "marlin" in d.files else (
                        d["embeddings"] if "embeddings" in d.files else None)
                    mp_seq = d["mp_seq"] if "mp_seq" in d.files else None
                    mp_mask = d["mp_mask"] if "mp_mask" in d.files else None
                    stored_dim = int(np.asarray(d["mp_feat_dim"]).item()) \
                        if "mp_feat_dim" in d.files else None
                    stored_schema = str(np.asarray(d["mp_feature_schema"]).item()) \
                        if "mp_feature_schema" in d.files else None
                    stored_names = tuple(
                        str(x) for x in np.asarray(d["mp_feature_names"]).tolist()
                    ) if "mp_feature_names" in d.files else None
                    stored_side_convention = str(
                        np.asarray(d["mp_side_convention"]).item()
                    ) if "mp_side_convention" in d.files else None
                    stored_capture_mirrored = str(
                        np.asarray(d["mp_capture_mirrored"]).item()
                    ) if "mp_capture_mirrored" in d.files else None
                if (marlin is not None
                        and not np.issubdtype(marlin.dtype, np.number)):
                    raise ValueError(f"{p}: marlin must have a numeric dtype")
                if (mp_seq is not None
                        and not np.issubdtype(mp_seq.dtype, np.number)):
                    raise ValueError(f"{p}: mp_seq must have a numeric dtype")
                if mp_mask is not None and np.asarray(mp_mask).dtype != np.bool_:
                    raise ValueError(
                        f"{p}: mp_mask must be stored as bool, got "
                        f"{np.asarray(mp_mask).dtype}"
                    )
                actual_dim = mp_seq.shape[1] if mp_seq is not None and mp_seq.ndim == 2 else None
                if stored_dim is not None and actual_dim is not None and stored_dim != actual_dim:
                    raise ValueError(
                        f"{p}: stored mp_feat_dim={stored_dim}, mp_seq has {actual_dim}"
                    )
                if actual_dim is not None and actual_dim != mp_feat_dim:
                    raise ValueError(
                        f"{p}: mp_seq feature dim {actual_dim} != requested {mp_feat_dim}"
                    )
                if mp_seq is not None and stored_schema is None:
                    if not allow_legacy_schema:
                        raise ValueError(
                            f"{p}: missing mp_feature_schema; pass allow_legacy_schema=True "
                            "only for an audited legacy cache"
                        )
                elif (mp_seq is not None and mp_feature_schema is not None
                      and stored_schema != mp_feature_schema):
                    raise ValueError(
                        f"{p}: schema {stored_schema!r} != requested {mp_feature_schema!r}"
                    )
                if stored_names is not None and len(stored_names) != mp_feat_dim:
                    raise ValueError(
                        f"{p}: {len(stored_names)} feature names != requested {mp_feat_dim}"
                    )
                if marlin is not None and marlin.ndim == 1:
                    marlin = marlin[None, :]
                bundles.append(ActionBundle(
                    marlin=None if marlin is None else marlin.astype(np.float32),
                    mp_seq=None if mp_seq is None else mp_seq.astype(np.float32),
                    mp_mask=None if mp_mask is None else mp_mask.astype(bool),
                    mp_feature_schema=stored_schema,
                    mp_feature_names=stored_names,
                    mp_side_convention=stored_side_convention,
                    mp_capture_mirrored=stored_capture_mirrored,
                ))
                any_present = any_present or (marlin is not None) or (mp_seq is not None)
            if not any_present:
                print(f"[warn] patient {pid}: no cached action streams; skipping")
                continue
            records.append(MultiStreamRecord(
                patient_id=pid, label=labels[pid]["label"],
                task=labels[pid]["task"], actions=bundles))
        mp_bundles = [
            bundle
            for record in records
            for bundle in record.actions
            if bundle.mp_seq is not None
        ]
        if (mp_feature_schema is not None and mp_bundles
                and all(bundle.mp_feature_schema is None for bundle in mp_bundles)):
            expected_names = MP_FEATURE_NAMES_BY_SCHEMA.get(mp_feature_schema)
            if expected_names is None:
                raise ValueError(f"unknown MediaPipe feature schema {mp_feature_schema!r}")
            for bundle in mp_bundles:
                bundle.mp_feature_schema = mp_feature_schema
                bundle.mp_feature_names = expected_names
                bundle.mp_side_convention = MP_SIDE_CONVENTION_BY_SCHEMA[
                    mp_feature_schema]
                bundle.mp_capture_mirrored = "unknown"
        return cls(
            records, actions=actions, marlin_dim=marlin_dim,
            mp_feat_dim=mp_feat_dim, mp_feature_schema=mp_feature_schema,
        )


def collate_multistream(batch: list[dict]) -> dict:
    """Pad a batch over the window (W) and frame (T) axes; A is fixed."""
    if not batch:
        raise ValueError("empty batch")
    B = len(batch)
    A = batch[0]["marlin_emb"].shape[0]
    marlin_dim = batch[0]["marlin_emb"].shape[-1]
    mp_feat_dim = batch[0]["mp_seq"].shape[-1]
    W = max(b["marlin_emb"].shape[1] for b in batch)
    T = max(b["mp_seq"].shape[1] for b in batch)

    marlin_emb = torch.zeros(B, A, W, marlin_dim, dtype=torch.float32)
    marlin_mask = torch.zeros(B, A, W, dtype=torch.bool)
    mp_seq = torch.zeros(B, A, T, mp_feat_dim, dtype=torch.float32)
    mp_mask = torch.zeros(B, A, T, dtype=torch.bool)
    action_present = torch.zeros(B, A, dtype=torch.bool)
    labels = torch.zeros(B, dtype=torch.long)
    task_ids: list[str] = []
    patient_ids: list[str] = []

    for i, it in enumerate(batch):
        w = it["marlin_emb"].shape[1]; t = it["mp_seq"].shape[1]
        marlin_emb[i, :, :w] = it["marlin_emb"]
        marlin_mask[i, :, :w] = it["marlin_mask"]
        mp_seq[i, :, :t] = it["mp_seq"]
        mp_mask[i, :, :t] = it["mp_mask"]
        action_present[i] = it["action_present"]
        labels[i] = it["label"]
        task_ids.append(it["task"])
        patient_ids.append(it["patient_id"])

    return {
        "marlin_emb": marlin_emb, "marlin_mask": marlin_mask,
        "mp_seq": mp_seq, "mp_mask": mp_mask,
        "action_present": action_present, "label": labels,
        "task_ids": task_ids, "patient_id": patient_ids,
    }
