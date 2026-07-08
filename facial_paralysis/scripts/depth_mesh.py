"""Automatable clean 3D face from OUR decoded iPhone depth (no official UE5 needed).
Denoise -> fill holes -> smooth, then (a) relief-shade render (clean face look) and
(b) organized grid mesh .ply (rotatable in any 3D viewer). Uses the real measured depth.

Run with anaconda python (has open3d): /Users/williamqiu/opt/anaconda3/bin/python3
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from scipy.ndimage import median_filter, gaussian_filter, distance_transform_edt
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

ROOT = Path(__file__).resolve().parent.parent
D3 = ROOT / "outputs" / "depth3d"
F = 432.0


def fill_holes(a):
    m = ~np.isfinite(a)
    if not m.any():
        return a, ~m
    idx = distance_transform_edt(m, return_distances=False, return_indices=True)
    return a[tuple(idx)], ~m


def clean_face_depth(name, smooth=2.2):
    up = np.rot90(np.load(D3 / f"{name}_depth.npy"), 1)
    xy = np.array(json.load(open(D3 / "landmarks.json"))[name]["xy"])
    H, W = up.shape
    zc = np.median([up[int(xy[i, 1] * H), int(xy[i, 0] * W)] for i in [1, 4, 5, 6]
                    if up[int(xy[i, 1] * H), int(xy[i, 0] * W)] > 0.05])
    band = (up > zc - 0.055) & (up < zc + 0.065)
    lu, lv = xy[:, 0] * W, xy[:, 1] * H
    u0, u1 = int(lu.min()) - 6, int(lu.max()) + 6
    v0, v1 = int(lv.min()) - 6, int(lv.max()) + 12
    z = np.where(band, up, np.nan)[max(v0, 0):v1, max(u0, 0):u1]
    z = median_filter(z, 3, mode="nearest")
    filled, mask = fill_holes(z)
    sm = gaussian_filter(filled, smooth)
    sm[~mask] = np.nan
    return sm, mask, zc


def write_grid_mesh(z, path):
    """Organized 2.5D grid mesh -> .ply (no Poisson bulges)."""
    hh, ww = z.shape
    vid = -np.ones((hh, ww), int)
    verts = []
    for v in range(hh):
        for u in range(ww):
            if np.isfinite(z[v, u]):
                vid[v, u] = len(verts)
                Z = z[v, u]
                verts.append([(u - ww / 2) * Z / F, -(v - hh / 2) * Z / F, -Z])
    tris = []
    for v in range(hh - 1):
        for u in range(ww - 1):
            a, b, c, d = vid[v, u], vid[v, u + 1], vid[v + 1, u], vid[v + 1, u + 1]
            if a >= 0 and b >= 0 and c >= 0:
                tris.append([a, c, b])
            if b >= 0 and c >= 0 and d >= 0:
                tris.append([b, c, d])
    verts, tris = np.array(verts), np.array(tris)
    with open(path, "w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(verts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {len(tris)}\nproperty list uchar int vertex_indices\nend_header\n")
        for p in verts:
            f.write(f"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f}\n")
        for t in tris:
            f.write(f"3 {t[0]} {t[1]} {t[2]}\n")
    return len(verts), len(tris)


def main():
    names = [a for a in ["20260219_FACES014", "20260313_FACES021", "20260305_FACES018",
                         "20260313_MySlate_23"] if (D3 / f"{a}_depth.npy").exists()]
    out = D3 / "mesh"; out.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, len(names), figsize=(3.6 * len(names), 4.6))
    ls = LightSource(azdeg=315, altdeg=45)
    for ax, name in zip(axes, names):
        z, mask, zc = clean_face_depth(name)
        elev = -z                                          # nose (near) -> high
        elev_f, _ = fill_holes(elev)
        rng = np.nanmax(elev) - np.nanmin(elev)
        shaded = ls.shade(elev_f, cmap=plt.cm.copper, blend_mode="soft",
                          vert_exag=1.0 / max(rng, 1e-3) * 0.8, dx=1, dy=1)
        shaded[~mask] = 1.0                                # background white
        ax.imshow(shaded); ax.axis("off")
        nv, nt = write_grid_mesh(z, out / f"{name}_face.ply")
        ax.set_title(f"{name.split('_',1)[1]}\n{nv} verts (.ply saved)", fontsize=9)
    fig.suptitle("Clean 3D face relief from OUR decoded iPhone depth "
                 "(temporal-median + smoothed, hillshaded; .ply meshes rotatable in MeshLab/Blender)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / "face_relief.png", dpi=140)
    print(f"wrote {out/'face_relief.png'} + {len(names)} .ply meshes in {out}")


if __name__ == "__main__":
    main()
