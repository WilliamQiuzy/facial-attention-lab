"""Explainability figure: WHY the model/score calls a patient severe — the per-region
left-right asymmetry (the clinically-trustworthy basis, per Run #14), drawn as a
schematic 'facogram' (no patient identity) + per-action asymmetry bars.

Run: python3 scripts/viz_decision_basis.py  -> outputs/viz/decision_basis_<take>.png + facogram_panel.png
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle, Arc, Rectangle
import matplotlib.cm as cm
from matplotlib.colors import Normalize
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Hiragino Sans GB", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent
ASYM = ROOT / "outputs" / "mayo_asymmetry" / "asymmetry_severity.json"
OUT = ROOT / "outputs" / "viz"; OUT.mkdir(parents=True, exist_ok=True)
CMAP = cm.get_cmap("RdYlGn_r"); NORM = Normalize(vmin=0.0, vmax=0.4)   # 0=对称(绿) → 0.4+=重度不对称(红)
ACT_CN = {"EyebrowRise": "抬眉", "GentleEyeClosure": "轻闭眼", "TightEyeSqueeze": "用力闭眼",
          "RelaxedSmile": "微笑", "LipPucker": "撅嘴", "LowerTeethShow": "露下齿", "ReanimatedSmile": "再微笑"}


def draw_face(ax, regions, weak):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off"); ax.set_aspect("equal")
    def col(r): return CMAP(NORM(regions.get(r, 0.0)))
    # weaker-half shading
    if weak in ("left", "right"):
        x0 = 0.5 if weak == "right" else 0.18      # subject-right shown on image-left convention kept simple
        ax.add_patch(Rectangle((x0, 0.12), 0.32, 0.78, fc="#000", alpha=0.05, ec="none", zorder=0))
    ax.add_patch(Ellipse((0.5, 0.5), 0.62, 0.82, fc="#fff7ee", ec="#888", lw=1.5, zorder=1))
    # brow (one region; color both)
    for cx in (0.36, 0.64):
        ax.add_patch(Arc((cx, 0.68), 0.18, 0.12, theta1=20, theta2=160, lw=6, color=col("brow"), zorder=2))
    # eyes
    for cx in (0.36, 0.64):
        ax.add_patch(Circle((cx, 0.58), 0.06, fc=col("eye"), ec="#555", lw=1, zorder=2))
    # mouth
    ax.add_patch(Arc((0.5, 0.33), 0.30, 0.16, theta1=200, theta2=340, lw=8, color=col("mouth"), zorder=2))
    # nose (neutral)
    ax.plot([0.5, 0.5], [0.55, 0.42], color="#aaa", lw=2, zorder=2)
    # region labels with values
    ax.text(0.5, 0.80, f"额/眉 {regions.get('brow', float('nan')):.2f}", ha="center", fontsize=8)
    ax.text(0.5, 0.50, f"眼 {regions.get('eye', float('nan')):.2f}", ha="center", fontsize=8)
    ax.text(0.5, 0.22, f"嘴 {regions.get('mouth', float('nan')):.2f}", ha="center", fontsize=8)
    if weak in ("left", "right"):
        ax.text(0.5, 0.04, f"较弱侧:{'患者左' if weak=='left' else '患者右'}", ha="center",
                fontsize=9, color="#b00", weight="bold")


def draw_bars(ax, per_action):
    items = [(ACT_CN.get(k, k), v) for k, v in per_action.items()]
    if not items:
        ax.axis("off"); return
    labels, vals = zip(*items)
    y = np.arange(len(labels))
    ax.barh(y, vals, color=[CMAP(NORM(v)) for v in vals], ec="#555")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9); ax.invert_yaxis()
    ax.set_xlim(0, max(0.45, max(vals) * 1.15)); ax.set_xlabel("左右不对称指数 (0=对称)", fontsize=9)
    for yi, v in zip(y, vals):
        ax.text(v + 0.005, yi, f"{v:.2f}", va="center", fontsize=8)


def one(take, rec):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4.2), gridspec_kw={"width_ratios": [1, 1.3]})
    draw_face(a1, rec["regions"], rec.get("weak_side"))
    a1.set_title("分区左右不对称(模型判断依据)", fontsize=10)
    draw_bars(a2, rec["per_action"])
    a2.set_title("逐动作不对称(越红=该动作越不对称)", fontsize=10)
    fig.suptitle(f"{take} — 总体不对称严重度 {rec['asym_overall']:.2f}", fontsize=12, weight="bold")
    sm = cm.ScalarMappable(norm=NORM, cmap=CMAP); sm.set_array([])
    fig.colorbar(sm, ax=[a1, a2], fraction=0.025, pad=0.02, label="不对称 (绿=健康 → 红=重)")
    fig.savefig(OUT / f"decision_basis_{take}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    data = {r["take"]: r for r in json.loads(ASYM.read_text())["ranked"]}
    # examples: most asymmetric + a moderate one
    ranked = sorted(data.values(), key=lambda r: r["asym_overall"], reverse=True)
    examples = [ranked[0]["take"], ranked[len(ranked) // 2]["take"]]
    for t in examples:
        one(t, data[t])
        print(f"wrote decision_basis_{t}.png")
    # panel: all patients, region heatmap grid
    takes = [r["take"] for r in ranked if r["take"] != "20260305_MySlate_14"]
    regs = ["brow", "eye", "mouth"]
    M = np.array([[data[t]["regions"].get(r, np.nan) for r in regs] for t in takes])
    fig, ax = plt.subplots(figsize=(6, 7))
    im = ax.imshow(M, cmap="RdYlGn_r", vmin=0, vmax=0.4, aspect="auto")
    ax.set_xticks(range(3)); ax.set_xticklabels(["额/眉", "眼", "嘴"], fontsize=10)
    ax.set_yticks(range(len(takes))); ax.set_yticklabels([t.replace("2026", "") for t in takes], fontsize=7)
    for i in range(len(takes)):
        for j in range(3):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title("各患者 分区左右不对称(模型判断依据)", fontsize=11)
    fig.colorbar(im, label="不对称 (绿=健康 → 红=重)", fraction=0.046, pad=0.04)
    fig.savefig(OUT / "facogram_panel.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    print("wrote facogram_panel.png")


if __name__ == "__main__":
    main()
