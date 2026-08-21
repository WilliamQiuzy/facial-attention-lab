"""MediaPipe-free face cropping for MARLIN (OpenCV Haar cascade).

The original MARLIN encode path (`MarlinVideoEncoder.encode_clip_bgr`) crops/aligns
faces with MediaPipe. When MediaPipe is unavailable (e.g. no wheel for the local
Python), this module provides a drop-in cv2-only crop so MARLIN can still encode
raw video. It is NOT landmark-aligned — it is a margin-padded square around the
Haar face box — so embeddings differ slightly from the MediaPipe-aligned ones;
validate equivalence in the mean-centered space before relying on it (see
`scripts/mayo_cv2crop_validate.py`).

    crop_face_cv2(bgr) -> (224,224,3) uint8 RGB, or None if no face found.
    encode_video_cv2(enc, path, n_clips) -> (n_encoded, 768) or None.
"""
from __future__ import annotations

import cv2
import numpy as np

_HAAR = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def crop_face_cv2(
    bgr: np.ndarray, size: int = 224, margin: float = 0.35, min_size: int = 120
) -> np.ndarray | None:
    """Largest Haar face → margin-padded square crop → (size,size,3) RGB uint8."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = _HAAR.detectMultiScale(gray, 1.1, 5, minSize=(min_size, min_size))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    cx, cy = x + w / 2, y + h / 2
    s = max(int(max(w, h) * (1 + margin)), 1)
    x0, y0 = max(0, int(cx - s / 2)), max(0, int(cy - s / 2))
    x1, y1 = min(bgr.shape[1], x0 + s), min(bgr.shape[0], y0 + s)
    crop = bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    crop = cv2.resize(crop, (size, size))
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def _sample_idx(lo: int, hi: int, n: int = 16) -> np.ndarray:
    return np.linspace(lo, max(lo, hi - 1), n).round().astype(int)


def encode_clip_cv2(enc, frames_bgr: list[np.ndarray], normalizer=None):
    """Crop 16 frames with cv2, forward through a (frozen) MarlinVideoEncoder.
    Returns (768,) np.float32 or None if no face in any frame."""
    import torch

    crops = []
    for img in frames_bgr:
        c = crop_face_cv2(img)
        if c is None:
            continue
        if normalizer is not None:
            c = normalizer(c)
        crops.append(c)
    if not crops:
        return None
    while len(crops) < enc.CLIP_FRAMES:
        crops.append(crops[-1])
    arr = np.stack(crops[: enc.CLIP_FRAMES]).astype(np.float32) / 255.0
    clip = torch.from_numpy(arr).permute(3, 0, 1, 2).unsqueeze(0)
    device = next(enc.parameters()).device
    with torch.no_grad():
        return enc(clip.to(device)).squeeze(0).cpu().numpy()


def n_frames(path: str) -> int:
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def collect_frames(path: str, indices) -> dict[int, np.ndarray]:
    """Stream a video and keep ONLY the requested frame indices (memory-light —
    avoids holding all ~7k frames). Sequential decode; cv2 frame-seek is codec-
    unreliable, so we read through and grab on the way."""
    want = sorted({int(i) for i in indices})
    cap = cv2.VideoCapture(str(path))
    out, i, ptr = {}, 0, 0
    while ptr < len(want):
        ok, fr = cap.read()
        if not ok:
            break
        if i == want[ptr]:
            out[i] = fr
            ptr += 1
        i += 1
    cap.release()
    return out


def encode_video_cv2(enc, path: str, n_clips: int = 4, normalizer=None):
    """Split a video into n_clips contiguous windows, encode each (cv2 crop),
    streaming only the sampled frames. Returns (n_encoded, 768) or None."""
    n = n_frames(path)
    if n < 1:
        return None
    bounds = np.linspace(0, n, n_clips + 1).round().astype(int)
    windows, all_idx = [], []
    for i in range(n_clips):
        lo, hi = int(bounds[i]), int(bounds[i + 1])
        if hi - lo < 1:
            windows.append(None); continue
        idx = _sample_idx(lo, hi)
        windows.append(idx); all_idx.extend(int(j) for j in idx)
    frames = collect_frames(path, all_idx)
    vecs = []
    for idx in windows:
        if idx is None:
            continue
        window = [frames[j] for j in idx if j in frames]
        if not window:
            continue
        v = encode_clip_cv2(enc, window, normalizer=normalizer)
        if v is not None:
            vecs.append(v)
    return np.stack(vecs) if vecs else None
