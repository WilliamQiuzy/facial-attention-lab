"""Generate a Shagdar 2024 GNN baseline schematic.

The paper's PDF is paywalled (Springer LNCS 15618; see
`src/baselines/shagdar_gnn/AUDIT.md` for the 13 retrieval routes we tried).
This script draws a replacement schematic for the technical report by
running the same pipeline the paper describes on a public face image.

Layout (one row, four panels):
    1. Input face image
    2. 478 MediaPipe FaceLandmarker v2 keypoints overlaid
    3. Delaunay-triangulation graph (Shagdar's input representation)
    4. GCN architecture box + binary output

Output: `assets/shagdar_schematic.png` and `assets/shagdar_schematic.pdf`.

Usage:
    python scripts/draw_shagdar_schematic.py
"""
from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyBboxPatch
from scipy.spatial import Delaunay

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


REPO_ROOT = Path(__file__).resolve().parents[1]
IMG_PATH = REPO_ROOT / "src" / "baselines" / "oo_multimodal" / "input_images" / "healthy_portrait.jpg"
MODEL_PATH = REPO_ROOT / "src" / "baselines" / "shagdar_gnn" / "face_landmarker_v2_with_blendshapes.task"
OUT_DIR = REPO_ROOT / "assets"


def extract_landmarks(image_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (BGR image, Nx2 pixel-space landmarks)."""
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(image_path)
    h, w = bgr.shape[:2]

    base_options = mp_python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=1,
        output_face_blendshapes=False,
    )
    with mp_vision.FaceLandmarker.create_from_options(options) as detector:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)
    if not result.face_landmarks:
        raise RuntimeError("no face detected")
    lms = result.face_landmarks[0]
    coords = np.array([[lm.x * w, lm.y * h] for lm in lms])
    return bgr, coords


def delaunay_edges(coords: np.ndarray) -> np.ndarray:
    """Return Ex2 array of unique undirected edges from Delaunay triangulation."""
    tri = Delaunay(coords)
    edge_set: set[tuple[int, int]] = set()
    for simplex in tri.simplices:
        a, b, c = sorted(simplex)
        edge_set.update({(a, b), (a, c), (b, c)})
    return np.array(sorted(edge_set))


def _draw_face_panel(ax, bgr, title):
    ax.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)


def _draw_landmarks_panel(ax, bgr, coords, title):
    ax.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), alpha=0.35)
    ax.scatter(coords[:, 0], coords[:, 1], s=2.5, c="#d62728", alpha=0.9, linewidths=0)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)


def _draw_graph_panel(ax, coords, edges, title):
    # Edges as a LineCollection on a white background, then nodes on top.
    segs = coords[edges]  # (E, 2, 2)
    lc = LineCollection(segs, colors="#1f77b4", linewidths=0.35, alpha=0.65)
    ax.add_collection(lc)
    ax.scatter(coords[:, 0], coords[:, 1], s=2.5, c="#d62728", linewidths=0, zorder=3)
    # Frame the panel by image extent (matplotlib needs this with collections only).
    pad = 8
    ax.set_xlim(coords[:, 0].min() - pad, coords[:, 0].max() + pad)
    ax.set_ylim(coords[:, 1].max() + pad, coords[:, 1].min() - pad)  # invert y
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)


def _draw_gcn_panel(ax, title):
    """Vertical GCN block diagram + binary head."""
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)

    blocks = [
        ("GCNConv  2 → 64",  "#cfe2f3"),
        ("GCNConv  64 → 64", "#cfe2f3"),
        ("GCNConv  64 → 128", "#9fc5e8"),
        ("GCNConv  128 → 256","#9fc5e8"),
        ("GCNConv  256 → 256","#6fa8dc"),
        ("global_max_pool",       "#d9d9d9"),
        ("Linear  256 → 10", "#fce5cd"),
        ("Linear  10 → 2",   "#f6b26b"),
    ]

    box_w, box_h = 7.5, 0.85
    x0 = 1.25
    y_top = 9.5
    gap = 0.20
    for i, (label, color) in enumerate(blocks):
        y = y_top - i * (box_h + gap)
        box = FancyBboxPatch((x0, y - box_h), box_w, box_h,
                             boxstyle="round,pad=0.0,rounding_size=0.12",
                             linewidth=0.6, edgecolor="#444", facecolor=color)
        ax.add_patch(box)
        ax.text(x0 + box_w / 2, y - box_h / 2, label,
                ha="center", va="center", fontsize=8.5)
        if i < len(blocks) - 1:
            arrow_y = y - box_h - gap / 2
            ax.annotate("", xy=(x0 + box_w / 2, arrow_y - 0.04),
                        xytext=(x0 + box_w / 2, arrow_y + 0.04),
                        arrowprops=dict(arrowstyle="-|>", color="#444", lw=0.5))

    # Output labels
    y_out = y_top - len(blocks) * (box_h + gap) - 0.4
    ax.text(x0 + box_w / 4,     y_out, "non-stroke", ha="center", va="top", fontsize=8.5, color="#333")
    ax.text(x0 + box_w * 3 / 4, y_out, "stroke",     ha="center", va="top", fontsize=8.5, color="#333")


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    bgr, coords = extract_landmarks(IMG_PATH)
    edges = delaunay_edges(coords)
    print(f"[ok] {len(coords)} landmarks, {len(edges)} unique edges")

    fig, axes = plt.subplots(1, 4, figsize=(14.5, 4.0),
                             gridspec_kw=dict(width_ratios=[1, 1, 1, 1.05]))
    _draw_face_panel(axes[0], bgr, "Input image")
    _draw_landmarks_panel(axes[1], bgr, coords, "MediaPipe 478 landmarks")
    _draw_graph_panel(axes[2], coords, edges,
                      f"Delaunay graph\n({len(coords)} nodes, {len(edges)} edges)")
    _draw_gcn_panel(axes[3], "GCNMid → binary head")

    fig.suptitle("Shagdar 2024 GNN baseline pipeline", fontsize=13, y=1.02)
    fig.tight_layout()

    png_path = OUT_DIR / "shagdar_schematic.png"
    pdf_path = OUT_DIR / "shagdar_schematic.pdf"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"[write] {png_path}")
    print(f"[write] {pdf_path}")


if __name__ == "__main__":
    main()
