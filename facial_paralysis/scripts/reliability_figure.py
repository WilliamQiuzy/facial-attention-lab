"""#5 reliability figure (de-identified: aggregate stats only). arm64 or anaconda python."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import reliability_suite as R

ROOT = Path(__file__).resolve().parent.parent
rep = json.loads((ROOT / "outputs/mayo_eface/reliability.json").read_text())
sh = rep["split_half"]
regions = ["eye_asym", "smile_asym", "brow_asym"]

fig, ax = plt.subplots(1, 3, figsize=(13, 4))

# A. split-half reliability (pure measurement) vs the good/moderate lines
vals = [sh[r]["spearman_brown"] for r in regions]
depth = [rep.get("depth_3d", {}).get("icc", 0), rep.get("depth_3d", {}).get("pooled_full_reliability", 0)]
labels = [r.replace("_asym", "") for r in regions] + ["3D/frame", "3D/pooled"]
allv = vals + depth
colors = ["#4a4"] * 3 + ["#c44", "#4a8"]
ax[0].bar(labels, allv, color=colors)
ax[0].axhline(0.75, ls="--", c="gray", lw=1); ax[0].axhline(0.5, ls=":", c="gray", lw=1)
ax[0].set_ylim(0, 1.05); ax[0].set_ylabel("reliability")
ax[0].set_title("A. Measurement reliability\n2D blendshape asym excellent; 3D needs pooling", fontsize=9)
for i, v in enumerate(allv):
    ax[0].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)

# B. MDC95 = smallest change we can trust (per measure, own units)
mdc = [sh[r]["mdc95"] for r in regions]
ax[1].bar([r.replace("_asym", "") for r in regions], mdc, color="#68a")
ax[1].set_ylabel("MDC95 (asymmetry-index units)")
ax[1].set_title("B. Minimal detectable change (MDC95)\nchange must exceed this to be real", fontsize=9)
for i, v in enumerate(mdc):
    ax[1].text(i, v + 0.003, f"{v:.3f}", ha="center", fontsize=8)

# C. Bland-Altman: eye gentle vs forced (why cross-provocation is only moderate)
data = {t: np.load(R.BS / f"{t}.npz", allow_pickle=True) for t in R.SEG if t != R.DUP and (R.BS / f"{t}.npz").exists()}
g, f = [], []
for t, d in data.items():
    ai = {}
    for s in R.SEG[t]:
        if s["action"] in ("GentleEyeClosure", "TightEyeSqueeze"):
            ai[s["action"]] = R.action_AI(d["bs"], d["t"], s, 9, 10)
    if "GentleEyeClosure" in ai and "TightEyeSqueeze" in ai and np.isfinite(ai["GentleEyeClosure"]) and np.isfinite(ai["TightEyeSqueeze"]):
        g.append(ai["GentleEyeClosure"]); f.append(ai["TightEyeSqueeze"])
g, f = np.array(g), np.array(f)
mean, diff = (g + f) / 2, g - f
ax[2].scatter(mean, diff, c="steelblue")
ax[2].axhline(diff.mean(), color="k", lw=1)
ax[2].axhline(diff.mean() + 1.96 * diff.std(), color="r", ls="--", lw=1)
ax[2].axhline(diff.mean() - 1.96 * diff.std(), color="r", ls="--", lw=1)
ax[2].set_xlabel("mean eye-closure AI"); ax[2].set_ylabel("gentle - forced")
ax[2].set_title("C. Gentle vs forced eye closure (Bland-Altman)\nphysiologically different -> fix probe when tracking", fontsize=9)

fig.suptitle("Direction #5: label-free measures are RELIABLE as measurements (no labels/patients needed). "
             "MDC95 sets the change threshold for #6 home monitoring.", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(ROOT / "outputs/mayo_eface/reliability_suite.png", dpi=130)
print("wrote outputs/mayo_eface/reliability_suite.png")
