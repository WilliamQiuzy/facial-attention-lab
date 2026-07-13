"""GPU MediaPipe + MARLIN extraction of the Mayo takes (RunPod A100).

Locally we could not extract the MediaPipe geometric stream (no mediapipe wheel),
and the cached mayo_bundles were UN-normalized (mismatched with v1, trained on
quality-NORMALIZED crops). This re-extracts every Mayo .mov with the REAL MediaPipe
FaceLandmarker + MARLIN, applying v1's normalizer (mode=normalize, work_size=112),
so the re-score is preprocessing-consistent. MARLIN on CUDA.

SPEED: the videos are 60fps ~2min (5–8k frames); sequential cv2 decode over the
network FS is ~10 fps (=9 min/take). We instead copy each mov to /dev/shm and
SEEK only the sampled frames (cap.set POS_FRAMES) — ~20x faster — and reuse a
single MediaPipe landmarker. ~2 min/take.

Whole-take bundle -> outputs/mayo_bundles_norm/<take>/clip.npz
  marlin (W,768) + schema-versioned MediaPipe stream (T,72)

Run on pod:  .venv/bin/python scripts/mayo_extract_pod.py
"""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.backbones.marlin_video import MarlinVideoEncoder, _face_crop_tools  # noqa: E402
from src.preprocessing.action_bundle import (  # noqa: E402
    MediaPipeFeatureExtractor,
    _assert_existing_cache_schema,
    _bundle_npz_payload,
)
from src.preprocessing.image_quality import QualityConfig, QualityNormalizer  # noqa: E402

LLF = ROOT / "data" / "livelinkface_data"
OUT = ROOT / "outputs" / "mayo_bundles_norm"
N_WINDOWS = 4
CLIP = 16
N_MP = 48
WORK_SIZE = 112
SHM = Path("/dev/shm/_mayo_take.mov")


def seek_frames(path: str, indices) -> dict[int, np.ndarray]:
    """Grab only the requested frame indices via seeking (no full decode)."""
    cap = cv2.VideoCapture(str(path))
    out = {}
    for i in sorted({int(x) for x in indices}):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        if ok:
            out[i] = fr
    cap.release()
    return out


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}", flush=True)
    enc = MarlinVideoEncoder.from_default_weights().to(dev).eval()
    Landmarker, _ = _face_crop_tools()
    crop_landmarker = Landmarker()                       # one MARLIN-crop landmarker, reused
    # Source capture orientation is not documented, so provenance must remain
    # unknown rather than guessing a patient-side convention.
    mp_ext = MediaPipeFeatureExtractor(with_geometry=False, capture_mirrored=None)
    normalizer = QualityNormalizer(QualityConfig(mode="normalize", work_size=WORK_SIZE))

    movs = sorted(LLF.glob("*/*.mov"))
    print(f"{len(movs)} movs", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    done = skipped = 0
    for vp in movs:
        take = vp.parent.name
        out = OUT / take / "clip.npz"
        if out.exists():
            _assert_existing_cache_schema(
                out,
                mp_ext.feature_schema,
                expected_side_convention=mp_ext.side_convention,
                expected_capture_mirrored="unknown",
            )
            done += 1; continue
        t0 = time.time()
        try:
            shutil.copy(vp, SHM)                          # network FS -> RAM
            cap = cv2.VideoCapture(str(SHM)); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
            if n < CLIP:
                print(f"  [skip] {take}: only {n} frames", flush=True); skipped += 1; continue
            bounds = np.linspace(0, n, N_WINDOWS + 1).round().astype(int)
            windows = [np.linspace(bounds[i], bounds[i + 1] - 1, CLIP).round().astype(int)
                       for i in range(N_WINDOWS)]
            mp_idx = np.linspace(0, n - 1, N_MP).round().astype(int)
            need = set(mp_idx.tolist())
            for w in windows:
                need.update(int(x) for x in w)
            frames = seek_frames(SHM, need)

            marlin_vecs = []
            for w in windows:
                wf = [frames[int(j)] for j in w if int(j) in frames]
                if not wf:
                    continue
                v = enc.encode_clip_bgr(wf, landmarker=crop_landmarker, normalizer=normalizer)
                if v is not None:
                    marlin_vecs.append(v)
            mpf = [frames[int(j)] for j in mp_idx if int(j) in frames]
            seq, mask = mp_ext.extract_sequence(mpf) if mpf else (None, None)
        except Exception as e:
            print(f"  [skip] {take}: {e}", flush=True); skipped += 1; continue
        finally:
            SHM.unlink(missing_ok=True)
        if not marlin_vecs or seq is None or not mask.any():
            print(f"  [skip] {take}: unusable (marlin={len(marlin_vecs)})", flush=True); skipped += 1; continue
        marlin = np.stack(marlin_vecs)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, **_bundle_npz_payload({
            "marlin": marlin, "mp_seq": seq, "mp_mask": mask,
        }, mp_ext))
        done += 1
        print(f"  {take}: marlin{marlin.shape} mp_seq{seq.shape} ({time.time()-t0:.1f}s)", flush=True)
    print(f"\nDONE: {done} bundles, {skipped} skipped. feat_dim={mp_ext.feat_dim} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
