"""Render the facial-palsy model architecture as a clean PNG for the slides.
Run: python3 scripts/viz_architecture.py  -> outputs/viz/architecture.png
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Hiragino Sans GB", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "viz"; OUT.mkdir(parents=True, exist_ok=True)

C = {"in": "#dfe9f5", "frozen": "#cfe8cf", "train": "#ffe2b3", "head": "#f5cccc", "rep": "#e6d6f2"}


def box(ax, x, y, w, h, text, color, fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.02",
                                fc=color, ec="#444", lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, wrap=True)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                                 lw=1.4, color="#555"))


fig, ax = plt.subplots(figsize=(13, 7.5))
ax.set_xlim(0, 13); ax.set_ylim(0, 7.5); ax.axis("off")
ax.text(6.5, 7.2, "Facial-Palsy Severity Model — two-stream + shared-severity multi-task",
        ha="center", fontsize=13, weight="bold")

# Input
box(ax, 0.2, 5.2, 2.1, 1.0, "每个动作的片段\nPer-action clip\n(16 帧 / 单图)", C["in"])
# Appearance stream (frozen MARLIN)
box(ax, 3.0, 6.0, 2.6, 0.95, "MARLIN (冻结)\nViT-MAE 视频编码器\n→ 外观向量 768-d", C["frozen"])
# Geometric stream (mediapipe -> GRU)
box(ax, 3.0, 4.55, 2.6, 0.95, "MediaPipe\n52 blendshape + 左右不对称\n→ 逐帧序列 (T×F)", C["frozen"])
box(ax, 6.0, 4.55, 2.2, 0.95, "时序编码器 GRU\n(可训练, 注意力池化)\n→ 动态向量", C["train"])
# concat
box(ax, 8.7, 5.2, 1.5, 1.0, "拼接\nconcat\n(外观⊕动态)", C["rep"])
# per-action MLP + pooling
box(ax, 8.7, 3.5, 1.5, 1.0, "每动作 MLP\n→ 动作池化\n(mean/attn)", C["train"])
# patient repr + severity
box(ax, 10.6, 4.4, 2.1, 1.0, "患者表示 h\n→ 全局严重度 s\n(标量)", C["rep"])
# heads
box(ax, 10.6, 2.45, 2.1, 1.4,
    "多任务有序阈值头\n(读同一个 s):\n HB I–VI · 二分类 · 3级\n+ 区域头(自有严重度):\n 眼部 · 嘴部", C["head"], fs=8)

# arrows
arrow(ax, 2.3, 5.7, 3.0, 6.4)        # input -> marlin
arrow(ax, 2.3, 5.7, 3.0, 5.0)        # input -> mediapipe
arrow(ax, 5.6, 5.0, 6.0, 5.0)        # mediapipe -> gru
arrow(ax, 5.6, 6.45, 8.7, 5.9)       # marlin -> concat
arrow(ax, 8.2, 5.0, 8.7, 5.5)        # gru -> concat
arrow(ax, 9.45, 5.2, 9.45, 4.5)      # concat -> mlp
arrow(ax, 10.2, 4.0, 10.6, 4.6)      # mlp -> repr
arrow(ax, 11.65, 4.4, 11.65, 3.85)   # repr -> heads

# legend
leg = [("输入", C["in"]), ("冻结(预训练)", C["frozen"]), ("可训练", C["train"]),
       ("表示/严重度", C["rep"]), ("任务头", C["head"])]
for i, (t, c) in enumerate(leg):
    box(ax, 0.3 + i * 2.5, 0.5, 0.4, 0.4, "", c)
    ax.text(0.8 + i * 2.5, 0.7, t, va="center", fontsize=9)
ax.text(6.5, 1.5, "核心思想:不统一标签,统一潜变量 s;每个数据集是 s 上的一组有序切点(异构标签可共训)",
        ha="center", fontsize=9, style="italic", color="#333")

fig.savefig(OUT / "architecture.png", dpi=130, bbox_inches="tight")
print(f"wrote {OUT/'architecture.png'}")
