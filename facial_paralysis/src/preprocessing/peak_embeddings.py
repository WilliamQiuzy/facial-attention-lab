"""Stage 2: MediaPipe outputs (per-frame blendshapes) → keyframe selection →
Oo MLP-Mixer 768-d embeddings.

Refactor of `scripts/extract_video_embeddings.py` into reusable functions.

Per-take output: `<embedding_root>/<slot_id>.npz` with:
  - embeddings     : (n_picked, 768) float32
  - frame_idxs     : (n_picked,)     int64
  - frame_types    : (n_picked,)     str
  - activity       : (n_picked,)     float32  (blendshape L2 at that frame)
  - take_id        : str
  - fps            : float
  - n_frames_mov   : int
  - strategy       : str  (which selection strategy was used)
  - k_peaks        : int  (only meaningful for peaks-family strategies)
  - neutral_window_s : float

# Frame selection strategies
  - "auto"               : pick `uniform_fps` for short videos (< short_video_threshold_frames),
                           pick `peaks` otherwise. RECOMMENDED for mixed datasets.
  - "all"                : every frame from blendshapes_wide.csv. Use for short clips
                           when you want full temporal resolution.
  - "uniform_fps"        : downsample evenly to target_fps (e.g. 6 fps matches Oo's
                           training rate). Use for short action clips.
  - "peaks"              : 1 neutral (lowest activity in first neutral_window_s)
                           + top-K activity peaks (scipy.signal.find_peaks).
                           Original behavior; good for long mixed-pose sessions.
  - "peaks_plus_uniform" : peaks ∪ uniform_fps, deduplicated by frame index.

The encoder used is `OoMLPMixerEncoder` (see Plan B). To swap the backbone
later (e.g. MARLIN, FaRL), only `extract_keyframe_embeddings` needs touching.
"""
from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch
from scipy.signal import find_peaks


FrameStrategy = Literal["auto", "all", "uniform_fps", "peaks", "peaks_plus_uniform"]


@dataclass
class Stage2Config:
    mediapipe_root: Path                   # where Stage 1 wrote landmarks/blendshapes
    embedding_root: Path                   # where to write per-slot .npz
    strategy: FrameStrategy = "auto"
    # peaks-family knobs (used by "peaks", "peaks_plus_uniform", and by "auto"
    # when it falls back to peaks):
    k_peaks: int = 8
    neutral_window_s: float = 10.0
    peak_distance_s: float = 2.0
    peak_prominence_frac: float = 0.05
    # uniform-family knobs (used by "uniform_fps", "peaks_plus_uniform", "auto" short branch):
    target_fps: float = 6.0
    # auto routing threshold: videos with FEWER blendshape rows than this go to uniform_fps
    short_video_threshold_frames: int = 600
    # quality gate for the .mov file (skip stub / corrupted recordings):
    min_mov_size_mb: float = 50.0


# ----------------------------------------------------------------------
# Blendshape activity
# ----------------------------------------------------------------------
def load_activity(blendshapes_wide_csv: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse blendshapes_wide.csv. Returns (frame_idxs, activity).

    activity[i] = L2 norm of the 52 blendshape coefficients at frame i.
    Empty cells (frames where MediaPipe did not detect a face) are treated
    as zero — no detected face → no measurable facial movement.
    """
    frames: list[int] = []
    acts: list[float] = []
    with blendshapes_wide_csv.open() as f:
        reader = csv.reader(f)
        header = next(reader)
        bs_cols = list(range(2, len(header)))  # columns: frame, timestamp_ms, <52 cols>
        for row in reader:
            if not row:
                continue
            frames.append(int(row[0]))
            vals = np.array(
                [float(row[c]) if c < len(row) and row[c] != "" else 0.0 for c in bs_cols],
                dtype=np.float32,
            )
            acts.append(float(np.linalg.norm(vals)))
    return np.array(frames, dtype=np.int64), np.array(acts, dtype=np.float32)


# ----------------------------------------------------------------------
# Keyframe selection — multi-strategy
# ----------------------------------------------------------------------
def _resolve_strategy(strategy: FrameStrategy, n_frames: int, cfg: Stage2Config) -> str:
    """Materialize 'auto' into a concrete strategy based on video length."""
    if strategy == "auto":
        return "uniform_fps" if n_frames < cfg.short_video_threshold_frames else "peaks"
    return strategy


def _select_uniform(frame_idx: np.ndarray, activity: np.ndarray, fps: float,
                    cfg: Stage2Config) -> list[tuple[int, str, float]]:
    step = max(1, int(round(fps / max(cfg.target_fps, 1e-6))))
    return [(int(frame_idx[i]), "uniform", float(activity[i]))
            for i in range(0, len(frame_idx), step)]


def _select_peaks(frame_idx: np.ndarray, activity: np.ndarray, fps: float,
                  cfg: Stage2Config) -> list[tuple[int, str, float]]:
    picks: list[tuple[int, str, float]] = []
    window_n = max(1, int(cfg.neutral_window_s * fps))
    early = activity[:window_n]
    if early.size:
        i_neutral = int(np.argmin(early))
        picks.append((int(frame_idx[i_neutral]), "neutral", float(activity[i_neutral])))

    distance = max(1, int(cfg.peak_distance_s * fps))
    prominence = cfg.peak_prominence_frac * (activity.max() - activity.min() + 1e-6)
    peak_idx, _ = find_peaks(activity, distance=distance, prominence=prominence)
    if peak_idx.size == 0:
        peak_idx = np.argsort(-activity)[: cfg.k_peaks * 4]
    heights = activity[peak_idx]
    top = np.argsort(-heights)[: cfg.k_peaks]
    for i in np.sort(peak_idx[top]):
        picks.append((int(frame_idx[i]), "peak", float(activity[i])))
    return picks


def _dedupe_by_frame(items: list[tuple[int, str, float]]) -> list[tuple[int, str, float]]:
    """When the same frame appears in both peaks and uniform, prefer 'neutral' >
    'peak' > 'uniform' so the more-informative label sticks."""
    rank = {"neutral": 0, "peak": 1, "uniform": 2, "all": 3}
    by_frame: dict[int, tuple[int, str, float]] = {}
    for f, t, a in items:
        if f not in by_frame or rank[t] < rank[by_frame[f][1]]:
            by_frame[f] = (f, t, a)
    return sorted(by_frame.values(), key=lambda x: x[0])


def select_keyframes(
    frame_idx: np.ndarray,
    activity: np.ndarray,
    fps: float,
    cfg: Stage2Config,
) -> list[tuple[int, str, float]]:
    """Pick representative frames per `cfg.strategy`. Returns
    list of (frame_idx, frame_type, activity)."""
    if len(frame_idx) == 0:
        return []
    strat = _resolve_strategy(cfg.strategy, len(frame_idx), cfg)

    if strat == "all":
        return [(int(f), "all", float(a)) for f, a in zip(frame_idx, activity)]
    if strat == "uniform_fps":
        return _select_uniform(frame_idx, activity, fps, cfg)
    if strat == "peaks":
        return _select_peaks(frame_idx, activity, fps, cfg)
    if strat == "peaks_plus_uniform":
        return _dedupe_by_frame(
            _select_peaks(frame_idx, activity, fps, cfg)
            + _select_uniform(frame_idx, activity, fps, cfg)
        )
    raise ValueError(f"unknown strategy: {strat!r}")


# ----------------------------------------------------------------------
# Video frame I/O
# ----------------------------------------------------------------------
def _read_frames(mov_path: Path, frame_idxs: list[int]) -> dict[int, np.ndarray]:
    """Read the requested frame indices from a .mov.

    Single forward pass: sort the requested indices ascending and grab-forward.
    Returns {frame_idx: BGR uint8 ndarray}.
    """
    cap = cv2.VideoCapture(str(mov_path))
    if not cap.isOpened():
        raise IOError(f"cannot open {mov_path}")
    out: dict[int, np.ndarray] = {}
    wanted = sorted(set(frame_idxs))
    cur = 0
    try:
        for target in wanted:
            if target < cur:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                cur = target
            while cur < target:
                cap.grab()
                cur += 1
            ok, frame = cap.read()
            if not ok or frame is None:
                raise IOError(f"failed reading frame {target} from {mov_path}")
            out[target] = frame
            cur += 1
    finally:
        cap.release()
    return out


# ----------------------------------------------------------------------
# Per-slot extraction
# ----------------------------------------------------------------------
def extract_keyframe_embeddings(
    slot_id: str,
    mov_path: Path,
    encoder,
    landmarker,
    cfg: Stage2Config,
    fps_hint: float | None = None,
) -> dict | None:
    """Run Stage 2 for one video. Returns the saved-payload dict or None if
    no usable frames were obtained (e.g. all peaks failed MediaPipe detection).

    Idempotent: if `<embedding_root>/<slot_id>.npz` already exists, returns
    its contents without re-running.
    """
    out_path = cfg.embedding_root / f"{slot_id}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        data = np.load(out_path, allow_pickle=True)
        return {k: data[k] for k in data.files}

    bs_csv = cfg.mediapipe_root / slot_id / "blendshapes_wide.csv"
    if not bs_csv.exists():
        raise FileNotFoundError(f"missing Stage 1 output: {bs_csv}")

    frame_idx, activity = load_activity(bs_csv)
    cap = cv2.VideoCapture(str(mov_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) if fps_hint is None else float(fps_hint)
    n_frames_mov = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    picks = select_keyframes(frame_idx, activity, fps=fps, cfg=cfg)
    wanted = [min(p[0], n_frames_mov - 1) for p in picks]
    t0 = time.perf_counter()
    frames = _read_frames(mov_path, wanted)
    t_read = time.perf_counter() - t0

    t0 = time.perf_counter()
    kept: list[tuple[int, str, float]] = []
    embeddings: list[np.ndarray] = []
    for (orig_idx, ftype, act), clamped in zip(picks, wanted):
        emb = encoder.encode_image_bgr(frames[clamped], landmarker=landmarker)
        if emb is None:
            print(f"   skip frame {orig_idx} ({ftype}): MediaPipe detected no face on crop input")
            continue
        kept.append((orig_idx, ftype, act))
        embeddings.append(emb)
    t_encode = time.perf_counter() - t0

    if not embeddings:
        print(f"   {slot_id}: no usable frames; not saving .npz")
        return None

    E = np.stack(embeddings, axis=0).astype(np.float32)
    strat_used = _resolve_strategy(cfg.strategy, len(frame_idx), cfg)
    payload = {
        "embeddings": E,
        "frame_idxs": np.array([p[0] for p in kept], dtype=np.int64),
        "frame_types": np.array([p[1] for p in kept], dtype="U10"),
        "activity": np.array([p[2] for p in kept], dtype=np.float32),
        "take_id": slot_id,
        "fps": fps,
        "n_frames_mov": n_frames_mov,
        "strategy": strat_used,
        "k_peaks": cfg.k_peaks,
        "neutral_window_s": cfg.neutral_window_s,
        "target_fps": cfg.target_fps,
    }
    np.savez(out_path, **payload)
    type_counts: dict[str, int] = {}
    for _, t, _ in kept:
        type_counts[t] = type_counts.get(t, 0) + 1
    parts = ", ".join(f"{n} {t}" for t, n in sorted(type_counts.items()))
    print(f"   {slot_id}: {E.shape[0]} frames ({parts}) via '{strat_used}'  "
          f"read {t_read:.1f}s  encode {t_encode:.1f}s  -> {out_path.name}")
    return payload


def run_stage2_batch(
    items: list[tuple[str, Path]],
    encoder,
    landmarker,
    cfg: Stage2Config,
) -> list[dict]:
    """items: list of (slot_id, mov_path). Returns list of saved payload dicts."""
    cfg.embedding_root.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    print(f"\nStage 2: extracting embeddings for {len(items)} slots\n")
    for i, (slot_id, mov_path) in enumerate(items, 1):
        if mov_path.stat().st_size < cfg.min_mov_size_mb * 1024 * 1024:
            print(f"[{i}/{len(items)}] {slot_id}: skip (mov < {cfg.min_mov_size_mb} MB, "
                  f"actual {mov_path.stat().st_size / 1e6:.0f} MB)")
            continue
        print(f"[{i}/{len(items)}] {slot_id}")
        payload = extract_keyframe_embeddings(
            slot_id, mov_path, encoder, landmarker, cfg,
        )
        if payload is not None:
            results.append(payload)
    return results
