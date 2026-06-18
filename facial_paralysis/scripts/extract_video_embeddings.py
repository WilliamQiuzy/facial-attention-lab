"""Legacy entry point — kept for backward compatibility.

This script's logic was moved to `src/preprocessing/peak_embeddings.py`.
Prefer the unified pipeline at `scripts/preprocess.py`, which runs both
Stage 1 (MediaPipe extraction) and Stage 2 (peak-frame encoding) with a
single invocation.

This file now simply delegates to the module. Identical output to before.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing.peak_embeddings import (  # noqa: E402
    Stage2Config, extract_keyframe_embeddings,
)
from src.models.backbones import OoMLPMixerEncoder  # noqa: E402
from src.baselines.oo_multimodal.utils import MediaPipeFaceLandmarker  # type: ignore  # noqa: E402

LL_DIR = ROOT / "data" / "livelinkface_data"
MP_DIR = ROOT / "data" / "mediapipe_out"
OUT_DIR = ROOT / "outputs" / "embeddings"


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"device: {device}")
    encoder = OoMLPMixerEncoder.from_default_weights().to(device).eval()
    landmarker = MediaPipeFaceLandmarker()
    cfg = Stage2Config(mediapipe_root=MP_DIR, embedding_root=OUT_DIR)

    candidates: list[tuple[str, Path]] = []
    for take_dir in sorted(LL_DIR.iterdir()):
        if not take_dir.is_dir():
            continue
        movs = list(take_dir.glob("*.mov"))
        bs_csv = MP_DIR / take_dir.name / "blendshapes_wide.csv"
        if not movs or not bs_csv.exists():
            continue
        if movs[0].stat().st_size < cfg.min_mov_size_mb * 1024 * 1024:
            print(f"  skipping {take_dir.name}: mov suspiciously small "
                  f"({movs[0].stat().st_size/1e6:.0f} MB)")
            continue
        candidates.append((take_dir.name, movs[0]))

    print(f"{len(candidates)} usable takes\n")
    for i, (slot_id, mov_path) in enumerate(candidates, 1):
        print(f"[{i}/{len(candidates)}] {slot_id}")
        extract_keyframe_embeddings(slot_id, mov_path, encoder, landmarker, cfg)


if __name__ == "__main__":
    main()
