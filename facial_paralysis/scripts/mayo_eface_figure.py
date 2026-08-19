"""Figure for the Mayo FACES per-action symmetry analysis.
Left: cohort heatmap of label-free regional + dynamic measures.
Right: two eye-closure L-vs-R trajectories where the peak looks symmetric but the
       hold/forcing reveals the deficit (the signal a still cannot capture).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
BS = ROOT / "outputs" / "mayo_blendshapes"
R = json.loads((ROOT / "outputs" / "mayo_eface" / "eface_scores.json").read_text())
OUT = ROOT / "outputs" / "mayo_eface" / "eface_panel.png"

takes = [k for k in R if k != "20260305_MySlate_14"]          # drop duplicate
short = {k: k.split("_", 1)[1] for k in takes}

# ---- heatmap matrix ----
rows = ["brow_asym", "eye_asym", "oral_asym"]
mat = []
for k in takes:
    mat.append([R[k].get(r) if R[k].get(r) is not None else np.nan for r in rows]
               + [abs(R[k]["eye_dynamics"].get("forced_recruitment", np.nan)),
                  R[k]["synkinesis"].get("oral_ocular_synk", np.nan)])
mat = np.array(mat, float).T
labels = ["brow asym", "eye asym\n(static peak)", "oral asym",
          "|forced recruit|\n(DYNAMIC)", "oral-ocular\nsynkinesis (DYNAMIC)"]

fig = plt.figure(figsize=(15, 6))
ax = fig.add_axes([0.13, 0.12, 0.44, 0.78])
im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=0.6)
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=9)
ax.set_xticks(range(len(takes)))
ax.set_xticklabels([short[k] for k in takes], rotation=60, ha="right", fontsize=8)
ax.axhline(2.5, color="k", lw=2)                              # divide static vs dynamic
ax.set_title("Mayo FACES cohort (n=13): static vs DYNAMIC symmetry measures\n"
             "top 3 = obtainable from a still · bottom 2 = only from video", fontsize=10)
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        if not np.isnan(mat[i, j]):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=6.5)
fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="asymmetry / index")

# ---- example trajectories (eyeSquint L vs R during TightEyeSqueeze) ----
examples = ["20260313_FACES021", "20260305_FACES018"]         # flaccid vs recruiting
seg = json.loads((BS / "segments.json").read_text())
LI, RI = 19, 20                                               # eyeSquint L/R
for n, tk in enumerate(examples):
    axr = fig.add_axes([0.66, 0.58 - 0.46 * n, 0.31, 0.34])
    d = np.load(BS / f"{tk}.npz", allow_pickle=True)
    bs, t = d["bs"], d["t"]
    sq = next((s for s in seg[tk] if s["action"] == "TightEyeSqueeze"), None)
    w = (t >= sq["t_start"] - 0.5) & (t <= sq["t_end"] + 0.5)
    axr.plot(t[w], bs[w, LI], "-o", ms=3, label="left eye", color="#1f77b4")
    axr.plot(t[w], bs[w, RI], "-o", ms=3, label="right eye", color="#d62728")
    axr.axvspan(sq["t_start"], sq["t_end"], color="k", alpha=0.06)
    fr = R[tk]["eye_dynamics"].get("forced_recruitment")
    axr.set_title(f"{short[tk]} — Tight Eye Squeeze  (forced_recruit={fr:+.2f})", fontsize=9)
    axr.set_xlabel("time (s)", fontsize=8); axr.set_ylabel("eyeSquint activation", fontsize=8)
    axr.legend(fontsize=8); axr.tick_params(labelsize=7)

fig.text(0.66, 0.02, "Both eyes can peak similarly, yet the weak side's TRAJECTORY over the 3-s hold\n"
                     "and its response to forcing differ — the signal absent from any single still.",
         fontsize=8, style="italic")
fig.savefig(OUT, dpi=130)
print("wrote", OUT)
