"""Stage 1: video → MediaPipe FaceLandmarker outputs.

Refactor of `data/mediapipe_face_landmarks.ipynb` into a reusable module.
Behavior is intended to be IDENTICAL to the notebook on existing data — same
output files, same done.json resume gate, same CSV float format.

Output layout (rooted at `Stage1Config.output_root`):

  <output_root>/
    _models/face_landmarker.task         # downloaded once
    per_video_meta.csv                   # manifest, one row per processed video
    <slot_id>/
      landmarks.csv                      # frame, point_idx, x, y, z      (478 pts/frame)
      blendshapes.csv                    # frame, blendshape, score       (long)
      blendshapes_wide.csv               # frame, timestamp_ms, 52 cols   (wide)
      transform_matrices.npy             # (n_frames, 4, 4) float32
      <stem>_landmarks.mp4               # annotated preview
      done.json                          # written LAST; resume marker

`slot_id` is the per-output-folder relative key. Conventions:
  - Single-video take folder    : slot_id = "<take_dir_name>"
                                  (current livelinkface_data layout)
  - Multi-video take folder     : slot_id = "<take_dir_name>/<video_stem>"
                                  (will be the future patient-folder layout,
                                   where take_dir = patient_id and each video
                                   inside is one HB pose / action)
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
VIDEO_EXTS = {".mov", ".mp4", ".m4v"}
DONE_MARKER = "done.json"
REQUIRED_FILES = (
    "landmarks.csv",
    "blendshapes.csv",
    "blendshapes_wide.csv",
    "transform_matrices.npy",
)
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


@dataclass
class Stage1Config:
    input_root: Path                       # e.g. data/livelinkface_data or data/patients
    output_root: Path                      # e.g. data/mediapipe_out
    write_annotated: bool = True
    force_reprocess: bool = False
    show_progress: bool = True
    progress_interval_s: float = 5.0
    csv_float_format: str = "%.4f"         # see notebook for rationale
    num_workers: int = field(default_factory=lambda: min(4, max(1, (os.cpu_count() or 2) - 1)))


# ----------------------------------------------------------------------
# Annotated-video drawing
# ----------------------------------------------------------------------
_MESH_COLOR = (0, 255, 0)
_IRIS_COLOR = (0, 0, 255)
_IRIS_START = 468


def draw_landmarks_on_frame(bgr_frame: np.ndarray, face_landmarks_list) -> np.ndarray:
    annotated = bgr_frame.copy()
    h, w = annotated.shape[:2]
    for face_landmarks in face_landmarks_list:
        for i, lm in enumerate(face_landmarks):
            cx, cy = int(lm.x * w), int(lm.y * h)
            if i >= _IRIS_START:
                cv2.circle(annotated, (cx, cy), 2, _IRIS_COLOR, -1, cv2.LINE_AA)
            else:
                cv2.circle(annotated, (cx, cy), 1, _MESH_COLOR, -1, cv2.LINE_AA)
    return annotated


# ----------------------------------------------------------------------
# Video discovery
# ----------------------------------------------------------------------
def discover_videos(root: Path) -> tuple[list[tuple[str, str, Path]], list[tuple[str, str]], list[tuple[str, str]]]:
    """Walk `<root>/<take_dir>/*.{mov,mp4,m4v}`.

    Returns (items, skipped, warnings):
      items   : list of (slot_id, take_name, video_path)
      skipped : list of (take_name, reason)
      warnings: list of (take_name, reason)

    `slot_id` is the per-output relative folder:
      - take_dir with 1 video  → slot_id = take_dir.name
      - take_dir with N videos → slot_id = "<take_dir.name>/<video_stem>"
    """
    items: list[tuple[str, str, Path]] = []
    skipped: list[tuple[str, str]] = []
    warnings: list[tuple[str, str]] = []
    if not root.exists():
        raise FileNotFoundError(f"input_root does not exist: {root.resolve()}")

    for take_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        vids = sorted(
            p for p in take_dir.iterdir()
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS
        )
        if not vids:
            skipped.append((take_dir.name, "no video file"))
            continue
        if len(vids) == 1:
            items.append((take_dir.name, take_dir.name, vids[0]))
        else:
            warnings.append((take_dir.name,
                             f"{len(vids)} videos, processing all into <take>/<stem>/"))
            for v in vids:
                items.append((f"{take_dir.name}/{v.stem}", take_dir.name, v))
    return items, skipped, warnings


# ----------------------------------------------------------------------
# Model handling
# ----------------------------------------------------------------------
def ensure_model(output_root: Path) -> Path:
    """Download face_landmarker.task once into `<output_root>/_models/`."""
    model_dir = output_root / "_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "face_landmarker.task"
    if not model_path.exists():
        print(f"downloading model: {model_path}")
        urllib.request.urlretrieve(MODEL_URL, model_path)
    return model_path


def make_landmarker(model_path: Path):
    """Fresh CPU FaceLandmarker in VIDEO mode (timestamps are per-stream)."""
    return mp_vision.FaceLandmarker.create_from_options(
        mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )


# ----------------------------------------------------------------------
# Progress reporter (consolidates parallel-worker progress into one line)
# ----------------------------------------------------------------------
class _ProgressReporter:
    def __init__(self, interval_s: float = 5.0):
        self._state: dict[str, tuple[int, int]] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0 = 0.0
        self._interval = interval_s

    def update(self, slot_id: str, current: int, total: int):
        with self._lock:
            self._state[slot_id] = (current, total)

    def remove(self, slot_id: str):
        with self._lock:
            self._state.pop(slot_id, None)

    def _format_line(self) -> str | None:
        with self._lock:
            items = sorted(self._state.items())
        if not items:
            return None
        parts = []
        for sid, (cur, total) in items:
            label = sid.split("/")[-1]
            if len(label) > 9 and label[:8].isdigit() and label[8] == "_":
                label = label[9:]
            pct = int(100 * cur / total) if total > 0 else 0
            parts.append(f"{label}:{cur}/{total}({pct:>3d}%)")
        elapsed = time.perf_counter() - self._t0
        return f"  [{elapsed:5.1f}s active={len(items)}] " + " | ".join(parts)

    def _run(self):
        while not self._stop.wait(self._interval):
            line = self._format_line()
            if line:
                print(line, flush=True)

    def start(self):
        self._t0 = time.perf_counter()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None


# ----------------------------------------------------------------------
# Resume logic
# ----------------------------------------------------------------------
def _is_already_done(take_out: Path) -> bool:
    """Done iff done.json exists AND all required output files exist."""
    if not (take_out / DONE_MARKER).exists():
        return False
    return all((take_out / f).exists() for f in REQUIRED_FILES)


def _load_done_meta(take_out: Path) -> dict | None:
    try:
        return json.loads((take_out / DONE_MARKER).read_text())
    except Exception:
        return None


# ----------------------------------------------------------------------
# Per-video processing
# ----------------------------------------------------------------------
def process_video(
    slot_id: str,
    video_path: Path,
    cfg: Stage1Config,
    model_path: Path,
    reporter: _ProgressReporter | None = None,
) -> dict:
    """Run MediaPipe on one video; write outputs into <output_root>/<slot_id>/.

    Returns a manifest meta dict (also written as done.json). Idempotent: if
    a prior successful run left a done.json + all required files, this returns
    the cached meta with `resumed=True` (unless `cfg.force_reprocess`).
    """
    take_out = cfg.output_root / slot_id
    if not cfg.force_reprocess and _is_already_done(take_out):
        cached = _load_done_meta(take_out)
        if cached is not None:
            cached = dict(cached)
            cached["resumed"] = True
            cached["elapsed_s"] = 0.0
            return cached

    take_out.mkdir(parents=True, exist_ok=True)
    stale = take_out / DONE_MARKER
    if stale.exists():
        try:
            stale.unlink()
        except Exception:
            pass

    landmarks_csv = take_out / "landmarks.csv"
    blendshapes_csv = take_out / "blendshapes.csv"
    blendshapes_wide = take_out / "blendshapes_wide.csv"
    transform_npy = take_out / "transform_matrices.npy"
    annotated_path = take_out / f"{video_path.stem}_landmarks.mp4"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")

    t0 = time.perf_counter()
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if not (math.isfinite(fps) and fps > 0):
            raise RuntimeError(f"invalid fps={fps} (header probably corrupt)")
        if width <= 0 or height <= 0:
            raise RuntimeError(f"invalid frame size {width}x{height}")

        writer = None
        if cfg.write_annotated:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(annotated_path), fourcc, fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError(f"cannot open writer for {annotated_path}")

        landmarker = make_landmarker(model_path)
        lm_rows: list[tuple] = []
        bs_rows: list[tuple] = []
        wide_rows: list[dict] = []
        mats: list[np.ndarray] = []
        n_face_frames = 0
        frame_idx = 0
        if reporter is not None:
            reporter.update(slot_id, 0, total_frames)
        try:
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                ts_ms = int(round(frame_idx * 1000.0 / fps))
                result = landmarker.detect_for_video(mp_image, ts_ms)

                fl_list = result.face_landmarks
                bs_list = result.face_blendshapes
                tm_list = result.facial_transformation_matrixes

                if fl_list:
                    n_face_frames += 1
                    for i, lm in enumerate(fl_list[0]):
                        lm_rows.append((frame_idx, i, lm.x, lm.y, lm.z))

                wide_row: dict = {"frame": frame_idx, "timestamp_ms": ts_ms}
                if bs_list:
                    for cat in bs_list[0]:
                        bs_rows.append((frame_idx, cat.category_name, float(cat.score)))
                        wide_row[cat.category_name] = float(cat.score)
                wide_rows.append(wide_row)

                if tm_list:
                    mats.append(np.asarray(tm_list[0], dtype=np.float32))
                else:
                    mats.append(np.full((4, 4), np.nan, dtype=np.float32))

                if writer is not None:
                    writer.write(draw_landmarks_on_frame(bgr, fl_list))
                frame_idx += 1
                if reporter is not None:
                    reporter.update(slot_id, frame_idx, total_frames)
        finally:
            if writer is not None:
                writer.release()
            landmarker.close()

        if frame_idx == 0:
            try:
                annotated_path.unlink(missing_ok=True)
            except Exception:
                pass
            raise RuntimeError(f"no readable frames in {video_path} (likely corrupt)")

        partial = total_frames > 0 and frame_idx < total_frames
        elapsed = time.perf_counter() - t0
        rate = frame_idx / elapsed if elapsed > 0 else float("nan")

        # Write data files BEFORE done.json. Float format trims CSV size while
        # keeping landmark sub-pixel precision and sub-noise blendshape precision.
        pd.DataFrame(lm_rows, columns=["frame", "point_idx", "x", "y", "z"]).to_csv(
            landmarks_csv, index=False, float_format=cfg.csv_float_format)
        pd.DataFrame(bs_rows, columns=["frame", "blendshape", "score"]).to_csv(
            blendshapes_csv, index=False, float_format=cfg.csv_float_format)

        df_wide = pd.DataFrame(wide_rows)
        if not df_wide.empty:
            cols = ["frame", "timestamp_ms"] + sorted(
                c for c in df_wide.columns if c not in {"frame", "timestamp_ms"}
            )
            df_wide = df_wide[cols]
        df_wide.to_csv(blendshapes_wide, index=False, float_format=cfg.csv_float_format)

        np.save(transform_npy, np.stack(mats, axis=0) if mats else np.empty((0, 4, 4)))

        meta = {
            "slot_id": slot_id,
            "video_path": str(video_path.resolve()),  # absolute, so manifest is portable across cwds
            "fps": fps,
            "width": width,
            "height": height,
            "n_frames_header": total_frames,
            "n_frames_read": frame_idx,
            "n_face_frames": n_face_frames,
            "partial": partial,
            "elapsed_s": round(elapsed, 2),
            "fps_proc": round(rate, 1),
            "wrote_annotated": cfg.write_annotated,
            "out_dir": str(take_out),
            "resumed": False,
        }
        (take_out / DONE_MARKER).write_text(json.dumps(meta, indent=2))
        return meta
    finally:
        cap.release()
        if reporter is not None:
            reporter.remove(slot_id)


# ----------------------------------------------------------------------
# Batch driver
# ----------------------------------------------------------------------
def run_batch(cfg: Stage1Config) -> list[dict]:
    """Discover videos under cfg.input_root, process each, write per_video_meta.csv.

    Returns the list of meta dicts (one per video, processed or resumed).
    """
    cfg.output_root.mkdir(parents=True, exist_ok=True)
    model_path = ensure_model(cfg.output_root)

    items, skipped, warnings = discover_videos(cfg.input_root)
    print(f"found {len(items)} videos to process, "
          f"{len(skipped)} folders skipped, {len(warnings)} warnings")
    for sid, _, vid in items:
        print(f"  + {sid} -> {vid.name}")
    for name, reason in skipped:
        print(f"  - SKIP {name} ({reason})")
    for name, reason in warnings:
        print(f"  ! WARN {name} ({reason})")

    if not items:
        return []

    reporter = _ProgressReporter(cfg.progress_interval_s) if cfg.show_progress else None
    metas: list[dict] = []
    t_batch = time.perf_counter()
    print(f"\n{'parallel' if cfg.num_workers > 1 else 'sequential'} mode "
          f"(num_workers={cfg.num_workers}, {len(items)} videos, "
          f"force_reprocess={cfg.force_reprocess})\n")

    def _run_one(item):
        slot_id, take_name, video_path = item
        try:
            meta = process_video(slot_id, video_path, cfg, model_path, reporter)
            meta["take"] = take_name
            return slot_id, meta, None
        except Exception as e:
            return slot_id, {
                "slot_id": slot_id, "take": take_name,
                "video_path": str(video_path), "error": str(e),
                "resumed": False,
            }, e

    def _format_line(meta, err):
        sid = meta.get("slot_id", "?")
        if err is not None:
            return f"! {sid} FAILED: {err}"
        if meta.get("resumed"):
            return (f"= {sid}: SKIPPED (already done; "
                    f"{meta.get('n_frames_read','?')} frames cached)")
        note = " (PARTIAL)" if meta.get("partial") else ""
        return (f"+ {sid}: {meta['n_frames_read']}/{meta['n_frames_header']} frames{note}, "
                f"face on {meta['n_face_frames']}, "
                f"{meta['elapsed_s']}s ({meta['fps_proc']} fps)")

    if reporter is not None:
        reporter.start()
    try:
        if cfg.num_workers <= 1 or len(items) <= 1:
            for it in items:
                sid, meta, err = _run_one(it)
                print(_format_line(meta, err), flush=True)
                metas.append(meta)
        else:
            with ThreadPoolExecutor(max_workers=cfg.num_workers) as pool:
                futures = [pool.submit(_run_one, it) for it in items]
                for f in as_completed(futures):
                    sid, meta, err = f.result()
                    print(_format_line(meta, err), flush=True)
                    metas.append(meta)
    finally:
        if reporter is not None:
            reporter.stop()

    metas.sort(key=lambda m: m.get("slot_id", ""))
    wall = time.perf_counter() - t_batch
    done = [m for m in metas if "error" not in m]
    processed = [m for m in done if not m.get("resumed")]
    resumed = [m for m in done if m.get("resumed")]
    total_frames = sum(m.get("n_frames_read", 0) for m in processed)
    print(f"\nbatch wall time: {wall:.1f}s | "
          f"processed: {len(processed)} ({total_frames} frames) | "
          f"resumed: {len(resumed)} | failed: {len(metas) - len(done)}")

    meta_csv = cfg.output_root / "per_video_meta.csv"
    pd.DataFrame(metas).to_csv(meta_csv, index=False)
    print(f"manifest: {meta_csv}")
    return metas
