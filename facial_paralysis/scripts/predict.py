"""Inference with the unified warm-start model: a video OR image -> facial-palsy
scores (binary palsy probability + eyes/mouth severity).

This is the deployable scoring path for the "test on iPhone" stage. It runs the
exact same preprocessing as training (frozen MARLIN + MediaPipe + quality
normalizer), so a Mayo .mov or any face image gets scored consistently. When real
Mayo/MEEI labels arrive, the same checkpoint is the fine-tuning starting point.

Usage:
  KMP_DUPLICATE_LIB_OK=TRUE \
  /opt/homebrew/Caskroom/miniforge/base/envs/dev/bin/python scripts/predict.py \
      --input path/to/face.{mov,mp4,jpg,png,bmp} [--checkpoint outputs/checkpoints/warmstart_v1.pt]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXT = {".mov", ".mp4", ".avi", ".m4v"}


def load_model(ckpt_path: Path, device: str):
    from src.models.facial_palsy_model import FacialPalsyModel, FacialPalsyConfig
    from src.models.multitask import TaskSpec
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    tasks = [TaskSpec(n, k, coupled=c) for (n, k, c) in ck["tasks"]]
    cfg = FacialPalsyConfig(tasks=tasks, **ck["model_cfg"])
    model = FacialPalsyModel(cfg).to(device).eval()
    model.load_state_dict(ck["state_dict"])
    return model, ck


def build_bundle(input_path: Path, enc, mp_ext, normalizer, n_windows: int):
    """Return (marlin (W,768), mp_seq (T,F), mp_mask (T,)) for image or video."""
    ext = input_path.suffix.lower()
    if ext in IMAGE_EXT:
        img = cv2.imread(str(input_path))
        if img is None:
            raise IOError(f"cannot read image: {input_path}")
        marlin = enc.encode_clip_bgr([img], normalizer=normalizer)
        seq, mask = mp_ext.extract_sequence([img])
        marlin = None if marlin is None else marlin[None, :]
    elif ext in VIDEO_EXT:
        from src.preprocessing.action_bundle import extract_action_bundle
        b = extract_action_bundle(input_path, enc, mp_ext,
                                  n_marlin_windows=n_windows, normalizer=normalizer)
        if b is None:
            raise RuntimeError("no usable face frames in video")
        marlin, seq, mask = b["marlin"], b["mp_seq"], b["mp_mask"]
    else:
        raise ValueError(f"unsupported input type: {ext}")
    if marlin is None or marlin.shape[0] == 0:
        raise RuntimeError("no face detected for the MARLIN stream")
    return marlin.astype(np.float32), seq.astype(np.float32), mask.astype(bool)


@torch.no_grad()
def score(model, marlin, seq, mask, device):
    from src.models.ordinal import predict_grade, cum_probs
    W = marlin.shape[0]
    T = seq.shape[0]
    me = torch.from_numpy(marlin)[None, None].to(device)          # (1,1,W,768)
    mm = torch.ones(1, 1, W, dtype=torch.bool, device=device)
    ms = torch.from_numpy(seq)[None, None].to(device)             # (1,1,T,F)
    mk = torch.from_numpy(mask)[None, None].to(device)            # (1,1,T)
    ap = torch.ones(1, 1, dtype=torch.bool, device=device)
    out = model(me, mm, ms, mk, ap)

    def region(task):
        cp = cum_probs(out[task])                                 # (1, K-1) = P(y>k)
        level = int(predict_grade(out[task])[0])
        expected = float(cp.sum(dim=1)[0])                        # E[y] = Σ P(y>k)
        return {"level": level, "expected": round(expected, 3),
                "p_gt": [round(float(x), 3) for x in cp[0].cpu().numpy()]}

    res = {}
    if "binary" in out:
        res["palsy_probability"] = round(float(cum_probs(out["binary"])[0, 0]), 3)
    for t in ("eyes", "mouth"):
        if t in out:
            res[t] = region(t)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--checkpoint", default=str(ROOT / "outputs" / "checkpoints" / "warmstart_v1.pt"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-windows", type=int, default=4, help="MARLIN windows for video")
    args = ap.parse_args()

    from src.models.backbones.marlin_video import MarlinVideoEncoder
    from src.preprocessing.action_bundle import MediaPipeFeatureExtractor
    from src.preprocessing.image_quality import QualityConfig, QualityNormalizer

    model, ck = load_model(Path(args.checkpoint), args.device)
    enc = MarlinVideoEncoder.from_default_weights().to(args.device).eval()
    mp_ext = MediaPipeFeatureExtractor()
    q = ck.get("quality", {"mode": "normalize", "work_size": 112})
    normalizer = QualityNormalizer(QualityConfig(mode=q["mode"], work_size=q["work_size"]))

    marlin, seq, mask = build_bundle(Path(args.input), enc, mp_ext, normalizer, args.n_windows)
    res = score(model, marlin, seq, mask, args.device)
    scheme = ck.get("label_scheme", {})
    for t in ("eyes", "mouth"):
        if t in res and t in scheme:
            res[t]["label"] = scheme[t][res[t]["level"]]

    print(json.dumps({"input": args.input, "scores": res}, indent=2))


if __name__ == "__main__":
    main()
