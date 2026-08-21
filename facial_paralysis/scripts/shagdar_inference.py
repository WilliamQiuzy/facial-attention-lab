"""Inference for the Shagdar 2024 GNN facial-palsy baseline.

Loads one of the released GCNMid checkpoints from `src/baselines/shagdar_gnn/`,
runs it on a small mix of inputs:
  - OBAMA (public-domain healthy portrait)
  - BELLS-palsy (Wikimedia, CC BY-SA 3.0)
  - A few Mayo take thumbnails

For each image: MediaPipe → 478 (x, y) landmarks → Delaunay edges → GCN → 2-class
softmax. Target convention from Shagdar's training: class 1 = "stroke" (palsy).

Ensemble note: Shagdar ships 4 clean-pose splits (clean_split_0..3.pt) and 3
bad-pose splits. We average softmax over all 4 clean splits for robustness.
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

WEIGHTS_DIR = SHAGDAR_DIR / "model_weights"
# Use Shagdar's bundled MediaPipe model so inputs match their training preprocessing
MP_MODEL = SHAGDAR_DIR / "face_landmarker_v2_with_blendshapes.task"


# ----------------------------------------------------------------------
# MediaPipe + graph building (matches graphizer478.py exactly)
# ----------------------------------------------------------------------
def _make_detector():
    options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MP_MODEL)),
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1,
    )
    return mp_vision.FaceLandmarker.create_from_options(options)


def image_to_graph(image_path: Path, detector) -> Data | None:
    """Same recipe as graphizer478.py: pixel-space (x, y) landmarks +
    Delaunay-triangulation edges. Returns a torch_geometric Data or None
    if MediaPipe finds no face."""
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise IOError(f"cannot read {image_path}")
    h, w = bgr.shape[:2]

    # mp.Image expects SRGB; MediaPipe accepts a file path too, but we already
    # loaded via cv2 so we go through the array path.
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    detection = detector.detect(mp_image)
    if not detection.face_landmarks:
        return None

    landmarks = np.array(
        [(int(lm.x * w), int(lm.y * h)) for lm in detection.face_landmarks[0]],
        dtype=np.int64,
    )  # (478, 2)

    tri = scipy.spatial.Delaunay(landmarks).simplices  # (T, 3)
    edges = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]], axis=0)
    edges = np.sort(edges, axis=1)
    edge_index = np.unique(edges, axis=0)  # (E, 2)

    x = torch.tensor(landmarks, dtype=torch.float32)
    ei = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # (2, E)
    return Data(x=x, edge_index=ei)


# ----------------------------------------------------------------------
# Ensemble inference
# ----------------------------------------------------------------------
def load_clean_ensemble(device: torch.device) -> list[GCNMid]:
    """Load all 4 clean-pose checkpoints."""
    models = []
    for i in range(4):
        path = WEIGHTS_DIR / f"clean_split_{i}.pt"
        model = GCNMid().to(device).eval()
        sd = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(sd, strict=True)
        models.append(model)
    return models


@torch.no_grad()
def predict(graph: Data, models: list[GCNMid], device: torch.device) -> dict:
    """Return per-split + mean softmax over (nonstroke, stroke)."""
    batch = Batch.from_data_list([graph]).to(device)
    per_split: list[np.ndarray] = []
    for m in models:
        logits = m(batch.x, batch.edge_index, batch.batch)  # (1, 2)
        prob = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        per_split.append(prob)
    mean_prob = np.mean(per_split, axis=0)
    return {
        "per_split_p_stroke": [float(p[1]) for p in per_split],
        "mean_p_nonstroke": float(mean_prob[0]),
        "mean_p_stroke": float(mean_prob[1]),
        "vote": "stroke" if mean_prob[1] >= 0.5 else "nonstroke",
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    device = torch.device("cpu")  # tiny model; CPU keeps results deterministic
    detector = _make_detector()
    models = load_clean_ensemble(device)
    print(f"loaded ensemble: {len(models)} clean-pose splits on {device}")

    samples = [
        ("OBAMA_healthy",
         ROOT / "src/baselines/oo_multimodal/input_images/healthy_portrait.jpg"),
        ("BELLS_palsy",
         ROOT / "assets/bellspalsy_wikimedia.jpg"),
        ("FACES_006",
         ROOT / "data/livelinkface_data/20260203_FACES006/thumbnail.jpg"),
        ("FACES_014",
         ROOT / "data/livelinkface_data/20260219_FACES014/thumbnail.jpg"),
        ("FACES_018",
         ROOT / "data/livelinkface_data/20260305_FACES018/thumbnail.jpg"),
        ("FACES_021",
         ROOT / "data/livelinkface_data/20260313_FACES021/thumbnail.jpg"),
        ("MySlate_6",
         ROOT / "data/livelinkface_data/20260109_MySlate_6/thumbnail.jpg"),
        ("MySlate_23",
         ROOT / "data/livelinkface_data/20260313_MySlate_23/thumbnail.jpg"),
        ("MySlate_28",
         ROOT / "data/livelinkface_data/20260410_MySlate_28/thumbnail.jpg"),
        ("MySlate_29",
         ROOT / "data/livelinkface_data/20260414_MySlate_29/thumbnail.jpg"),
    ]

    print(f"\n{'name':<18s}  {'graph':<10s}  {'p(stroke) per split':<40s}  {'mean p(stroke)':>14s}  verdict")
    print("-" * 110)
    for name, path in samples:
        if not path.exists():
            print(f"{name:<18s}  MISSING: {path}")
            continue
        graph = image_to_graph(path, detector)
        if graph is None:
            print(f"{name:<18s}  NO_FACE")
            continue
        result = predict(graph, models, device)
        per_split_str = " ".join(f"{p:.2f}" for p in result["per_split_p_stroke"])
        n_nodes = graph.x.shape[0]
        n_edges = graph.edge_index.shape[1]
        print(f"{name:<18s}  N{n_nodes} E{n_edges:<5d}  "
              f"{per_split_str:<40s}  {result['mean_p_stroke']:>14.3f}  {result['vote']}")


if __name__ == "__main__":
    main()
