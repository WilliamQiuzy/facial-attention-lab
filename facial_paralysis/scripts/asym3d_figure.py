"""Summary figure for #4 landmark-anchored 3D asymmetry. De-identified: uses depth
surfaces + landmark mesh only (no RGB faces). RUN WITH arm64 /usr/bin/python3."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import median_filter
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import asym3d_landmark as A

ROOT = Path(__file__).resolve().parent.parent
D3 = ROOT / "outputs" / "depth3d"
res = json.load(open(D3 / "asym3d_landmark.json"))
res = {k: v for k, v in res.items() if k != "MySlate_14"}          # drop duplicate for cohort
regions = ["eye", "brow", "mouth", "cheek", "jaw"]

fig = plt.figure(figsize=(14, 5))

# Panel A: registration validation on a depth surface (de-identified)
ax = fig.add_subplot(1, 3, 1)
name = "20260219_FACES014"
d = median_filter(np.load(D3 / f"{name}_depth.npy"), 3)
up = np.rot90(d, 1); H, W = up.shape
xy = np.array(json.load(open(D3 / "landmarks.json"))[name]["xy"])
ax.imshow(np.where(up > 0.05, up, np.nan), cmap="turbo_r")
ax.scatter(xy[A.ALL_IDX, 0] * W, xy[A.ALL_IDX, 1] * H, s=4, c="k")
ax.scatter(xy[1, 0] * W, xy[1, 1] * H, s=30, c="lime")
ax.set_title("A. RGB landmarks -> depth\n(exact time-match, rot+2x, validated)", fontsize=9)
ax.axis("off")

# Panel B: per-region 3D asymmetry across the cohort
ax = fig.add_subplot(1, 3, 2)
pats = sorted(res, key=lambda p: -res[p]["overall_mm"])
M = np.array([[res[p]["regions"].get(r, 0) for r in regions] for p in pats])
im = ax.imshow(M, cmap="magma", aspect="auto")
ax.set_xticks(range(len(regions))); ax.set_xticklabels(regions, fontsize=8)
ax.set_yticks(range(len(pats))); ax.set_yticklabels(pats, fontsize=7)
ax.set_title("B. Pose-corrected 3D L-R asymmetry (mm)\nper region, per patient", fontsize=9)
fig.colorbar(im, ax=ax, fraction=0.046, label="mm")

# Panel C: honest 3D-vs-2D relationship + determinism note
ax = fig.add_subplot(1, 3, 3)
ef = json.load(open(ROOT / "outputs/mayo_eface/eface_scores.json"))
ef = {k.split("_", 1)[1]: v for k, v in ef.items()}
def d2(p):
    r = ef.get(p)
    if not r:
        return None
    vs = [r[k] for k in ("brow_asym", "eye_asym", "oral_asym") if r.get(k) is not None]
    return np.mean(vs) if vs else None
P = [p for p in res if d2(p) is not None]
x = [res[p]["overall_mm"] for p in P]; y = [d2(p) for p in P]
ax.scatter(x, y, c="steelblue")
for p, xi, yi in zip(P, x, y):
    ax.annotate(p.replace("MySlate_", "M").replace("FACES", "F"), (xi, yi), fontsize=6)
from scipy.stats import spearmanr
rho, pv = spearmanr(x, y)
ax.set_xlabel("3D depth asymmetry (mm)"); ax.set_ylabel("2D blendshape asymmetry")
ax.set_title(f"C. 3D (Z) vs 2D (XY) asymmetry\nSpearman {rho:+.2f} (p={pv:.2f}, n={len(P)}) - orthogonal axes",
             fontsize=9)

fig.suptitle("Direction #4: landmark-anchored 3D facial asymmetry from decoded iPhone depth "
             "(appearance-invariant). Duplicate take FACES018=MySlate_14 identical (deterministic).",
             fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(D3 / "asym3d_summary.png", dpi=130)
print(f"wrote {D3/'asym3d_summary.png'}  ({len(pats)} patients)")
