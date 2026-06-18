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

STANDARD_ACTIONS: tuple[str, ...] = (
    "rest", "brow_raise", "light_eye_closure", "forced_eye_closure", "smile",
)


@dataclass
class ActionBundle:
    """One action's two streams. None means the action/stream is absent."""
    marlin: np.ndarray | None = None     # (W, 768)
    mp_seq: np.ndarray | None = None     # (T, F)
    mp_mask: np.ndarray | None = None    # (T,) bool


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
        mp_feat_dim: int = 73,
    ):
        self.records = list(records)
        self.actions = list(actions)
        self.marlin_dim = marlin_dim
        self.mp_feat_dim = mp_feat_dim
        for r in self.records:
            if len(r.actions) != len(self.actions):
                raise ValueError(
                    f"patient {r.patient_id}: {len(r.actions)} action slots, "
                    f"expected {len(self.actions)}"
                )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> dict:
        r = self.records[i]
        marlin = [b.marlin for b in r.actions]
        mp = [b.mp_seq for b in r.actions]
        marlin_emb, marlin_mask = _pad_stack(marlin, self.marlin_dim)
        mp_seq, mp_mask_from_seq = _pad_stack(mp, self.mp_feat_dim)
        # Prefer an explicit mp_mask if provided; else derive from presence.
        mp_mask = mp_mask_from_seq.copy()
        for a_i, b in enumerate(r.actions):
            if b.mp_mask is not None and b.mp_seq is not None and b.mp_seq.size:
                n = min(len(b.mp_mask), mp_mask.shape[1])
                mp_mask[a_i, :n] = b.mp_mask[:n].astype(bool)
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
        mp_feat_dim: int = 73,
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
                d = np.load(p)
                marlin = d["marlin"] if "marlin" in d.files else (
                    d["embeddings"] if "embeddings" in d.files else None)
                mp_seq = d["mp_seq"] if "mp_seq" in d.files else None
                mp_mask = d["mp_mask"] if "mp_mask" in d.files else None
                if marlin is not None and marlin.ndim == 1:
                    marlin = marlin[None, :]
                bundles.append(ActionBundle(
                    marlin=None if marlin is None else marlin.astype(np.float32),
                    mp_seq=None if mp_seq is None else mp_seq.astype(np.float32),
                    mp_mask=None if mp_mask is None else mp_mask.astype(bool),
                ))
                any_present = any_present or (marlin is not None) or (mp_seq is not None)
            if not any_present:
                print(f"[warn] patient {pid}: no cached action streams; skipping")
                continue
            records.append(MultiStreamRecord(
                patient_id=pid, label=labels[pid]["label"],
                task=labels[pid]["task"], actions=bundles))
        return cls(records, actions=actions, marlin_dim=marlin_dim, mp_feat_dim=mp_feat_dim)


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
