"""Stage 2 (pod): extract MARLIN + MediaPipe bundles for the expanded-plan images.

Reads outputs/expanded_plan.json (from build_expanded_plan.py) and, for each image,
produces the same degenerate-clip bundle as the FNP/YFP image sets (Runs #3/#5):
MARLIN on the face-cropped image (tiled to 16) + a length-1 MediaPipe feature seq,
with v1's quality normalizer (normalize, work_size 112), mp_feat_dim=72.

Output: outputs/expanded_bundles/<id>.npz  (marlin (1,768), mp_seq (1,72), mp_mask (1,))

Run on pod:  .venv/bin/python scripts/extract_image_bundles_pod.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.backbones.marlin_video import MarlinVideoEncoder, _face_crop_tools  # noqa: E402
from src.preprocessing.action_bundle import MediaPipeFeatureExtractor  # noqa: E402
from src.preprocessing.image_quality import QualityConfig, QualityNormalizer  # noqa: E402

PLAN = ROOT / "outputs" / "expanded_plan.json"
OUTB = ROOT / "outputs" / "expanded_bundles"
WORK_SIZE = 112


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}", flush=True)
    enc = MarlinVideoEncoder.from_default_weights().to(dev).eval()
    Landmarker, _ = _face_crop_tools()
    crop_lm = Landmarker()
    mp_ext = MediaPipeFeatureExtractor(with_geometry=False)
    norm = QualityNormalizer(QualityConfig(mode="normalize", work_size=WORK_SIZE))

    plan = json.loads(PLAN.read_text())
    OUTB.mkdir(parents=True, exist_ok=True)
    done = skip = 0
    t0 = time.time()
    for k, e in enumerate(plan):
        out = OUTB / f"{e['id']}.npz"
        if out.exists():
            done += 1; continue
        # plan src_path is absolute (from the build host); remap to THIS root
        sp = e["src_path"]
        sp = str(ROOT / "data" / "external" / sp.split("data/external/", 1)[1]) if "data/external/" in sp else sp
        img = cv2.imread(sp)
        if img is None:
            skip += 1; continue
        v = enc.encode_clip_bgr([img], landmarker=crop_lm, normalizer=norm)   # crop+tile+MARLIN
        seq, mask = mp_ext.extract_sequence([img])
        if v is None or seq is None or not mask.any():
            skip += 1; continue
        np.savez(out, marlin=v[None, :].astype(np.float32),
                 mp_seq=seq.astype(np.float32), mp_mask=mask.astype(bool),
                 mp_feat_dim=mp_ext.feat_dim, task=e["task"], label=e["label"], dataset=e["dataset"])
        done += 1
        if (k + 1) % 200 == 0:
            print(f"  {k+1}/{len(plan)} done={done} skip={skip} ({time.time()-t0:.0f}s)", flush=True)
    print(f"\nDONE: {done} bundles, {skip} skipped -> {OUTB}", flush=True)


if __name__ == "__main__":
    main()
