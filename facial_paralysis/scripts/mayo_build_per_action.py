"""Build PER-ACTION Mayo bundles from the blendshape segmentation (RunPod A100).

Consumes outputs/mayo_blendshapes/segments.json (from mayo_blendshape_segment.py)
and, for each take, extracts one MARLIN+MediaPipe bundle per detected action,
placed in its CANONICAL slot (memory: mayo-action-protocol). This is the model's
intended per-action input (docs/model_design.md §2) — built on real Mayo data for
the first time. Frames are seeked from a /dev/shm copy of the mov; MARLIN uses the
mediapipe face crop + v1's quality normalizer.

Output: outputs/mayo_action_bundles/<take>/<ActionName>.npz
  marlin (W,768) + schema-versioned MediaPipe stream (T,72)

Run on pod:  .venv/bin/python scripts/mayo_build_per_action.py
"""
from __future__ import annotations

import glob
import json
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
    _bundle_npz_payload,
)
from src.preprocessing.image_quality import QualityConfig, QualityNormalizer  # noqa: E402

LLF = ROOT / "data" / "livelinkface_data"
SEG = ROOT / "outputs" / "mayo_blendshapes" / "segments.json"
OUTB = ROOT / "outputs" / "mayo_action_bundles"
ACTION_ORDER = ["EyebrowRise", "GentleEyeClosure", "TightEyeSqueeze",
                "RelaxedSmile", "LipPucker", "LowerTeethShow", "ReanimatedSmile"]
CLIP = 16
N_MP = 16
WORK_SIZE = 112
SHM = Path("/dev/shm/_pa.mov")


def seek_frames(path, indices):
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
    crop_lm = Landmarker()
    # The Mayo transfer does not currently establish capture mirroring.
    mp_ext = MediaPipeFeatureExtractor(with_geometry=False, capture_mirrored=None)
    norm = QualityNormalizer(QualityConfig(mode="normalize", work_size=WORK_SIZE))
    segs = json.loads(SEG.read_text())
    OUTB.mkdir(parents=True, exist_ok=True)

    for take, actions in segs.items():
        movs = glob.glob(str(LLF / take / "*.mov"))
        if not movs or not actions:
            print(f"[skip] {take}", flush=True); continue
        t0 = time.time()
        shutil.copy(movs[0], SHM)
        cap = cv2.VideoCapture(str(SHM)); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 60.0; cap.release()
        # gather all needed frame indices across this take's actions
        plan = []
        need = set()
        for a in actions:
            f0, f1 = int(a["t_start"] * fps), int(a["t_end"] * fps)
            f1 = min(max(f1, f0 + 1), n - 1)
            if f1 - f0 < 3:
                continue
            mar = np.linspace(f0, f1, CLIP).round().astype(int)
            mp = np.linspace(f0, f1, N_MP).round().astype(int)
            plan.append((a["action"], mar, mp))
            need.update(int(x) for x in mar); need.update(int(x) for x in mp)
        frames = seek_frames(SHM, need)
        SHM.unlink(missing_ok=True)
        made = []
        for action, mar, mp in plan:
            wf = [frames[int(j)] for j in mar if int(j) in frames]
            mpf = [frames[int(j)] for j in mp if int(j) in frames]
            if len(wf) < 3 or len(mpf) < 2:
                continue
            v = enc.encode_clip_bgr(wf, landmarker=crop_lm, normalizer=norm)
            seq, mask = mp_ext.extract_sequence(mpf)
            if v is None or seq is None or not mask.any():
                continue
            d = OUTB / take
            d.mkdir(parents=True, exist_ok=True)
            np.savez(d / f"{action}.npz", **_bundle_npz_payload({
                "marlin": v[None, :], "mp_seq": seq, "mp_mask": mask,
            }, mp_ext))
            made.append(action)
        print(f"{take}: {len(made)} action bundles {made} ({time.time()-t0:.1f}s)", flush=True)
    print(f"\nDONE -> {OUTB}", flush=True)


if __name__ == "__main__":
    main()
