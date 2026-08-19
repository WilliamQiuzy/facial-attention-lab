"""PyTorch Dataset over patient-foldered HB recordings.

Expected on-disk layout (per clinical team's plan):

    <data_root>/
      <patient_id>/
        <action_1>.mov
        <action_2>.mov
        ...
      <patient_id>/
        ...
      labels.csv          # columns: patient_id, hb_grade  (HB grade 1..6)

Each `<action_N>.mov` is a short clip of one Sunnybrook / HB pose.
The action filename stem (e.g. `rest`, `brow_raise`) becomes the action key.
We accept missing actions per patient: the model receives an `action_mask` so
the loss only counts actions that are present.

Embedding cache:
  Per-action embeddings are cached at:
    <embedding_cache_root>/<patient_id>/<action_name>.npz
  with the same schema as `outputs/embeddings/<take>.npz` produced by
  scripts/extract_video_embeddings.py. If the cache exists, we load from it
  (fast). Otherwise we extract on demand using `OoMLPMixerEncoder` (slow,
  prints a warning).

Items returned by __getitem__:
    {
      "patient_id":    str,
      "frame_emb":     (n_actions, n_frames, embed_dim) float32 — padded
      "frame_mask":    (n_actions, n_frames)            bool   — True = real frame
      "action_present": (n_actions,)                    bool   — True = action exists
      "label":         int                                     — 0..5 (HB Grade I..VI)
    }

This dataset can also be constructed from a pre-built list of `PatientRecord`s
(useful for the smoke test, where we synthesize fake data without touching disk).
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


# Standard 5-action House-Brackmann (HB) protocol — see Technical Report §3.2.
# Order matters: it defines the channel index of each action in the
# (n_actions, ...) tensors.
#
# HB-specific notes:
#   - "forced_eye_closure" is HB's discriminator for Grade II vs III (it exposes
#     synkinesis). Sunnybrook only has light eye closure; HB needs both.
#   - HB does NOT include snarl or pucker (those are Sunnybrook-only). Earlier
#     iterations of this code used Sunnybrook's 6 actions; we switched to HB
#     after the clinical team committed to HB grading.
STANDARD_ACTIONS: tuple[str, ...] = (
    "rest",
    "brow_raise",
    "light_eye_closure",
    "forced_eye_closure",
    "smile",
)


@dataclass
class PatientRecord:
    """One patient's data, fully resolved (either from disk or synthetic)."""

    patient_id: str
    label: int                                            # 0..n_classes-1
    # frame_emb_per_action[i] is (n_frames_i, embed_dim) — variable n_frames OK
    frame_emb_per_action: list[np.ndarray | None] = field(default_factory=list)


def _read_labels_csv(path: Path) -> dict[str, int]:
    """Returns {patient_id: hb_grade_zero_indexed}.
    `labels.csv` should have columns `patient_id,hb_grade` where hb_grade ∈ {1..6}."""
    labels: dict[str, int] = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row["patient_id"].strip()
            grade = int(row["hb_grade"])  # clinical 1..6
            if grade < 1 or grade > 6:
                raise ValueError(f"hb_grade out of range in {path}: {row}")
            labels[pid] = grade - 1
    return labels


def _stack_padded(arrays: Sequence[np.ndarray | None], embed_dim: int):
    """Pad a list of (n_frames_i, embed_dim) arrays to a single
    (len, max_n_frames, embed_dim) float32 + bool mask."""
    max_f = max((a.shape[0] for a in arrays if a is not None), default=1)
    n = len(arrays)
    out = np.zeros((n, max_f, embed_dim), dtype=np.float32)
    mask = np.zeros((n, max_f), dtype=bool)
    for i, a in enumerate(arrays):
        if a is None or a.shape[0] == 0:
            continue
        nf = a.shape[0]
        out[i, :nf] = a.astype(np.float32, copy=False)
        mask[i, :nf] = True
    return out, mask


class PatientVideoDataset(Dataset):
    def __init__(
        self,
        records: Sequence[PatientRecord],
        actions: Sequence[str] = STANDARD_ACTIONS,
        embed_dim: int = 768,
    ):
        self.records = list(records)
        self.actions = list(actions)
        self.embed_dim = embed_dim
        # sanity: every record's frame_emb_per_action must align with self.actions length
        for r in self.records:
            if len(r.frame_emb_per_action) != len(self.actions):
                raise ValueError(
                    f"patient {r.patient_id} has {len(r.frame_emb_per_action)} action "
                    f"slots, expected {len(self.actions)}"
                )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> dict:
        r = self.records[i]
        frame_emb, frame_mask = _stack_padded(r.frame_emb_per_action, self.embed_dim)
        action_present = frame_mask.any(axis=1)
        return {
            "patient_id": r.patient_id,
            "frame_emb": torch.from_numpy(frame_emb),
            "frame_mask": torch.from_numpy(frame_mask),
            "action_present": torch.from_numpy(action_present),
            "label": int(r.label),
        }

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_disk(
        cls,
        data_root: str | Path,
        embedding_cache_root: str | Path,
        actions: Sequence[str] = STANDARD_ACTIONS,
        labels_csv: str | Path | None = None,
        embed_dim: int = 768,
    ) -> "PatientVideoDataset":
        """Build from <data_root>/<patient_id>/<action>.mov + a labels CSV.

        Embeddings are loaded from `<embedding_cache_root>/<patient_id>/<action>.npz`
        if present. Missing caches mean the patient/action is excluded; we do
        NOT silently run encoding here (extraction is a heavy separate step,
        see `scripts/extract_video_embeddings.py`). A clear warning is printed.
        """
        data_root = Path(data_root)
        cache_root = Path(embedding_cache_root)
        labels_csv = Path(labels_csv) if labels_csv else data_root / "labels.csv"
        if not labels_csv.exists():
            raise FileNotFoundError(f"labels CSV not found at {labels_csv}")
        labels = _read_labels_csv(labels_csv)

        records: list[PatientRecord] = []
        for pid in sorted(labels):
            patient_dir = data_root / pid
            if not patient_dir.is_dir():
                print(f"[warn] labels.csv references patient {pid} but no folder exists; skipping")
                continue
            per_action: list[np.ndarray | None] = []
            for action in actions:
                cache_path = cache_root / pid / f"{action}.npz"
                if not cache_path.exists():
                    per_action.append(None)
                    continue
                data = np.load(cache_path)
                if "embeddings" not in data.files:
                    print(f"[warn] {cache_path} missing 'embeddings' key; skipping action")
                    per_action.append(None)
                    continue
                per_action.append(data["embeddings"].astype(np.float32, copy=False))
            if all(a is None for a in per_action):
                print(f"[warn] patient {pid}: no cached action embeddings found; skipping patient")
                continue
            records.append(PatientRecord(
                patient_id=pid,
                label=labels[pid],
                frame_emb_per_action=per_action,
            ))
        return cls(records, actions=actions, embed_dim=embed_dim)


def collate_patients(batch: list[dict]) -> dict:
    """Pad a batch of variable (n_actions, n_frames, D) tensors so we can stack
    into (B, n_actions, max_n_frames, D). n_actions is fixed across the dataset,
    so only the frame axis varies."""
    if not batch:
        raise ValueError("empty batch")
    B = len(batch)
    n_actions = batch[0]["frame_emb"].shape[0]
    embed_dim = batch[0]["frame_emb"].shape[2]
    max_f = max(b["frame_emb"].shape[1] for b in batch)

    frame_emb = torch.zeros(B, n_actions, max_f, embed_dim, dtype=torch.float32)
    frame_mask = torch.zeros(B, n_actions, max_f, dtype=torch.bool)
    action_present = torch.zeros(B, n_actions, dtype=torch.bool)
    labels = torch.zeros(B, dtype=torch.long)
    patient_ids: list[str] = []

    for i, item in enumerate(batch):
        f = item["frame_emb"].shape[1]
        frame_emb[i, :, :f] = item["frame_emb"]
        frame_mask[i, :, :f] = item["frame_mask"]
        action_present[i] = item["action_present"]
        labels[i] = item["label"]
        patient_ids.append(item["patient_id"])

    return {
        "patient_id": patient_ids,
        "frame_emb": frame_emb,
        "frame_mask": frame_mask,
        "action_present": action_present,
        "label": labels,
    }
