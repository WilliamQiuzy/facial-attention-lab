"""Shagdar diagnostic: predictions collapsed to always-'stroke' with p=1.000
on Obama (healthy). Hypothesis: the model takes RAW pixel coordinates as
node features and is therefore resolution-sensitive. Training was on
Toronto NeuroFace (likely standardized resolution); our test images span
820x1024 (Obama) to 2513x3696 (Bells) to 720x1280 (Mayo).

This script tries four input normalizations and reports per-split softmax:
  raw           : original pixel coords (baseline; expected degenerate)
  resize_h720   : resize image to height=720 first, then take pixel coords
  resize_h1080  : resize image to height=1080 first, then take pixel coords
  bbox_norm_512 : translate so face bbox centroid is at (256, 256), then
                  scale so longer side spans 512 pixels (face-frame invariant)
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import scipy.spatial
import torch
import torch.nn.functional as F
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from torch_geometric.data import Data, Batch

ROOT = Path(__file__).resolve().parent.parent
SHAGDAR_DIR = ROOT / "src" / "baselines" / "shagdar_gnn"
sys.path.insert(0, str(SHAGDAR_DIR))
from graph_model import GCNMid  # noqa: E402

MP_MODEL = SHAGDAR_DIR / "face_landmarker_v2_with_blendshapes.task"


def _make_detector():
    return mp_vision.FaceLandmarker.create_from_options(
        mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(MP_MODEL)),
            num_faces=1,
        )
    )


def _detect_landmarks(image_bgr: np.ndarray, detector):
    """Returns 478×2 normalized (x, y) in [0,1] or None."""
    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    det = detector.detect(mp_image)
    if not det.face_landmarks:
        return None, (h, w)
    return np.array([(lm.x, lm.y) for lm in det.face_landmarks[0]],
                    dtype=np.float32), (h, w)


def _to_data(landmarks_pix: np.ndarray) -> Data:
    """Build a torch_geometric Data from pixel-space landmarks (478, 2)."""
    pts_int = landmarks_pix.astype(np.int64)
    tri = scipy.spatial.Delaunay(pts_int).simplices
    edges = np.concatenate(
        [tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]], axis=0
    )
    edges = np.sort(edges, axis=1)
    edge_index = np.unique(edges, axis=0)
    x = torch.tensor(pts_int, dtype=torch.float32)
    ei = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=ei)


def build_variants(image_bgr: np.ndarray, detector) -> dict[str, Data | None]:
    """Same face image → four input-normalization variants."""
    out: dict[str, Data | None] = {}

    # 1) raw — original resolution pixel coords
    lms_norm, (h, w) = _detect_landmarks(image_bgr, detector)
    if lms_norm is None:
        return {k: None for k in ("raw", "resize_h720", "resize_h1080", "bbox_norm_512")}
    raw_pix = lms_norm * np.array([w, h], dtype=np.float32)
    out["raw"] = _to_data(raw_pix)

    # 2/3) resize the IMAGE to target height first, then re-detect (so MediaPipe
    # gives us pixel coords in the resized frame).
    for tag, target_h in (("resize_h720", 720), ("resize_h1080", 1080)):
        scale = target_h / h
        new_w = max(1, int(round(w * scale)))
        resized = cv2.resize(image_bgr, (new_w, target_h), interpolation=cv2.INTER_LINEAR)
        lms_r, _ = _detect_landmarks(resized, detector)
        out[tag] = _to_data(lms_r * np.array([new_w, target_h], dtype=np.float32)) if lms_r is not None else None

    # 4) bbox-normalized: center face bbox at (256, 256), scale longer side to 512.
    x0, y0 = raw_pix.min(axis=0)
    x1, y1 = raw_pix.max(axis=0)
    longer = max(x1 - x0, y1 - y0) or 1.0
    scale = 512.0 / longer
    centered = (raw_pix - np.array([(x0 + x1) / 2, (y0 + y1) / 2])) * scale + np.array([256, 256])
    out["bbox_norm_512"] = _to_data(centered)

    return out


def load_ensemble(weights_dir: Path, prefix: str, n: int, device) -> list[GCNMid]:
    models = []
    for i in range(n):
        m = GCNMid().to(device).eval()
        m.load_state_dict(torch.load(weights_dir / f"{prefix}_split_{i}.pt",
                                      map_location=device, weights_only=False),
                          strict=True)
        models.append(m)
    return models


@torch.no_grad()
def _predict(graph: Data, models: list[GCNMid], device) -> np.ndarray:
    batch = Batch.from_data_list([graph]).to(device)
    per_split = []
    for m in models:
        logits = m(batch.x, batch.edge_index, batch.batch)
        per_split.append(F.softmax(logits, dim=1).squeeze(0).cpu().numpy())
    return np.mean(per_split, axis=0)


def main():
    device = torch.device("cpu")
    detector = _make_detector()
    clean = load_ensemble(SHAGDAR_DIR / "model_weights", "clean", 4, device)
    bad = load_ensemble(SHAGDAR_DIR / "model_weights", "bad-pose-0", 3, device)
    print(f"loaded {len(clean)} clean + {len(bad)} bad-pose checkpoints\n")

    samples = [
        ("OBAMA_healthy", ROOT / "src/baselines/oo_multimodal/input_images/healthy_portrait.jpg"),
        ("BELLS_palsy",   ROOT / "assets/bellspalsy_wikimedia.jpg"),
        ("FACES_018",     ROOT / "data/livelinkface_data/20260305_FACES018/thumbnail.jpg"),
        ("MySlate_29",    ROOT / "data/livelinkface_data/20260414_MySlate_29/thumbnail.jpg"),
    ]

    for name, path in samples:
        if not path.exists():
            print(f"{name}: MISSING")
            continue
        bgr = cv2.imread(str(path))
        h, w = bgr.shape[:2]
        print(f"\n=== {name}  (orig {w}x{h}) ===")
        variants = build_variants(bgr, detector)
        print(f"{'variant':<18s}  {'clean p(stroke)':>16s}  {'bad-pose p(stroke)':>18s}")
        for tag, data in variants.items():
            if data is None:
                print(f"{tag:<18s}  NO_FACE")
                continue
            p_clean = _predict(data, clean, device)
            p_bad = _predict(data, bad, device)
            xy = data.x
            xmin, ymin = xy.min(0).values.tolist()
            xmax, ymax = xy.max(0).values.tolist()
            print(f"{tag:<18s}  {p_clean[1]:>16.3f}  {p_bad[1]:>18.3f}   "
                  f"coord-range x[{xmin:.0f},{xmax:.0f}] y[{ymin:.0f},{ymax:.0f}]")


if __name__ == "__main__":
    main()
