"""Fix + visualize depth using the ALIGNED clean RGB as a guide.
(1) RGB-guided (joint guided) filter: edge-preserving denoise that kills per-column streaks on
    smooth skin but keeps nose/eye/mouth edges (uses the RGB we already matched to each frame).
(2) RGB-textured 3D surface: drape the real RGB onto the depth heightmap -> recognizable 3D face.

Run: /Users/williamqiu/opt/anaconda3/bin/python3  (needs PIL/scipy/matplotlib; no decode/dylib)
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from scipy.ndimage import uniform_filter, median_filter, distance_transform_edt
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
D3 = ROOT / "outputs" / "depth3d"


def fill(a):
    m = ~np.isfinite(a)
    if not m.any():
        return a
    idx = distance_transform_edt(m, return_distances=False, return_indices=True)
    return a[tuple(idx)]


def guided_filter(I, p, r=6, eps=1e-4):
    """Edge-preserving filter of p guided by I (both same shape, [0,1] guide)."""
    box = lambda x: uniform_filter(x, size=2 * r + 1, mode="nearest")
    mI, mp = box(I), box(p)
    cov = box(I * p) - mI * mp
    var = box(I * I) - mI * mI
    a = cov / (var + eps)
    b = mp - a * mI
    return box(a) * I + box(b)


def load_pair(name):
    up = np.rot90(np.load(D3 / f"{name}_depth.npy"), 1)          # depth-upright (640,360)
    H, W = up.shape
    xy = np.array(json.load(open(D3 / "landmarks.json"))[name]["xy"])
    rgb = np.array(Image.open(D3 / f"{name}_rgb.jpg").resize((W, H)))  # RGB -> depth-upright grid
    # face crop from landmarks
    lu, lv = xy[:, 0] * W, xy[:, 1] * H
    u0, u1 = max(int(lu.min()) - 6, 0), min(int(lu.max()) + 6, W)
    v0, v1 = max(int(lv.min()) - 6, 0), min(int(lv.max()) + 14, H)
    # nose-anchored face band
    zc = np.median([up[int(xy[i, 1] * H), int(xy[i, 0] * W)] for i in [1, 4, 5, 6]
                    if up[int(xy[i, 1] * H), int(xy[i, 0] * W)] > 0.05])
    band = (up > zc - 0.055) & (up < zc + 0.065)
    z = np.where(band, up, np.nan)[v0:v1, u0:u1]
    rgbc = rgb[v0:v1, u0:u1]
    gray = rgbc[..., :3].mean(2) / 255.0
    return z, rgbc, gray


def main():
    names = [a for a in ["20260219_FACES014", "20260313_FACES021", "20260313_MySlate_23"]
             if (D3 / f"{a}_depth.npy").exists()]
    out = D3 / "mesh"; out.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(4.6 * len(names), 11))
    for j, name in enumerate(names):
        z, rgbc, gray = load_pair(name)
        mask = np.isfinite(z)
        zf = fill(z)
        zf = median_filter(zf, size=(1, 9))                      # horizontal median: kill vertical streaks
        zf = median_filter(zf, 3)
        # RGB-guided edge-preserving denoise
        zg = guided_filter(gray, zf, r=7, eps=6e-4)
        zg[~mask] = np.nan
        elev = -zg                                               # near (nose) -> high

        # row A: RGB reference
        ax = fig.add_subplot(3, len(names), j + 1)
        ax.imshow(rgbc); ax.set_title(f"{name.split('_',1)[1]}  — RGB (guide)", fontsize=9); ax.axis("off")
        # row B: guided-filtered depth relief
        ax = fig.add_subplot(3, len(names), len(names) + j + 1)
        ax.imshow(np.where(mask, zg, np.nan), cmap="turbo_r"); ax.axis("off")
        ax.set_title("RGB-guided denoised depth\n(streaks gone on skin, edges kept)", fontsize=9)
        # row C: RGB-textured 3D surface, tilted so the real relief (nose) shows
        ax = fig.add_subplot(3, len(names), 2 * len(names) + j + 1, projection="3d")
        hh, ww = elev.shape
        Y, X = np.mgrid[0:hh, 0:ww]
        fc = rgbc[..., :3] / 255.0
        from scipy.ndimage import gaussian_filter
        Zs = gaussian_filter(fill(elev), 1.5)                    # smooth for a clean surface
        ax.plot_surface(X, -Y, Zs, facecolors=fc, rcount=hh, ccount=ww,
                        linewidth=0, antialiased=False, shade=False)
        ax.view_init(elev=32, azim=-72)                         # tilted 3/4: shows nose protrusion
        ax.set_box_aspect((ww, hh, max(np.ptp(Zs), 1e-3) * 9))
        ax.set_axis_off()
        ax.set_title("RGB-textured 3D (tilted — real relief)", fontsize=9)
    fig.suptitle("Depth fix via the aligned RGB: guided-filter denoise + RGB-textured 3D surface", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / "depth_guided.png", dpi=135)
    print(f"wrote {out/'depth_guided.png'}")


if __name__ == "__main__":
    main()
