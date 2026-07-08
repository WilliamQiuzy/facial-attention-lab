"""Clean 3D face = MediaPipe 468-landmark topology (guaranteed face-shaped) displaced by OUR
measured LiDAR depth (robust per-landmark sample + hole-fill). The coarse mesh averages out the
per-column streak noise; the relief is our real measurement, not an RGB hallucination.

Run with anaconda python (numpy/scipy/matplotlib): /Users/williamqiu/opt/anaconda3/bin/python3
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from scipy.spatial import Delaunay
from scipy.interpolate import griddata
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import asym3d_landmark as A                                 # sample_Z / face_anchor (scipy-only)

ROOT = Path(__file__).resolve().parent.parent
D3 = ROOT / "outputs" / "depth3d"
F = 432.0
NL = 468                                                    # face-oval landmarks (skip irises)


def face_verts(name):
    up = np.rot90(__import__("numpy").load(D3 / f"{name}_depth.npy"), 1)
    from scipy.ndimage import median_filter
    up = median_filter(up, 3)
    xy = np.array(json.load(open(D3 / "landmarks.json"))[name]["xy"])[:NL]
    H, W = up.shape
    u, v = xy[:, 0] * W, xy[:, 1] * H
    fm = A.face_anchor(up, xy, W, H)
    lo, hi = fm - 0.07, fm + 0.07
    Z = np.array([A.sample_Z(up, u[i], v[i], lo, hi) for i in range(NL)])
    ok = np.isfinite(Z)
    # fill missing landmark depths by interpolating from valid ones
    Zf = griddata((u[ok], v[ok]), Z[ok], (u, v), method="linear")
    nan = ~np.isfinite(Zf)
    Zf[nan] = griddata((u[ok], v[ok]), Z[ok], (u[nan], v[nan]), method="nearest")
    # 2D Delaunay for connectivity; drop stretched triangles (webbing over eyes/mouth)
    tri = Delaunay(np.stack([u, v], 1))
    T = tri.simplices
    def edge(a, b):
        return np.hypot(u[a] - u[b], v[a] - v[b])
    keep = [t for t in T if max(edge(t[0], t[1]), edge(t[1], t[2]), edge(t[0], t[2])) < 22]
    T = np.array(keep)
    # robust outlier clip (our Z is noisy) then heavy laplacian smoothing over the mesh
    med = np.median(Zf); mad = np.median(np.abs(Zf - med)) + 1e-6
    Zf = np.clip(Zf, med - 2.5 * mad, med + 2.5 * mad)
    nbr = {i: set() for i in range(NL)}
    for a, b, c in T:
        for x, y in [(a, b), (b, c), (a, c)]:
            nbr[x].add(y); nbr[y].add(x)
    for _ in range(18):
        Zf = np.array([np.mean([Zf[i]] + [Zf[j] for j in nbr[i]]) if nbr[i] else Zf[i]
                       for i in range(NL)])
    X = (u - W / 2) * Zf / F
    Y = -(v - H / 2) * Zf / F
    return X, Y, -Zf, T


def main():
    names = [a for a in ["20260219_FACES014", "20260313_FACES021", "20260305_FACES018",
                         "20260313_MySlate_23"] if (D3 / f"{a}_depth.npy").exists()]
    out = D3 / "mesh"; out.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(3.4 * len(names), 7))
    for j, name in enumerate(names):
        X, Y, Z, T = face_verts(name)
        wspan = max(np.ptp(X), np.ptp(Y))
        for r, (elev, azim) in enumerate([(89, -90), (35, -75)]):     # frontal, then 3/4
            ax = fig.add_subplot(2, len(names), r * len(names) + j + 1, projection="3d")
            ax.plot_trisurf(X, Y, Z, triangles=T, cmap="copper", edgecolor="none",
                            linewidth=0, antialiased=True, shade=True)
            ax.view_init(elev=elev, azim=azim)
            ax.set_box_aspect((np.ptp(X), np.ptp(Y), wspan * 0.45))   # proportional depth
            ax.set_axis_off()
            if r == 0:
                ax.set_title(name.split("_", 1)[1], fontsize=9)
    fig.suptitle("Clean 3D face: MediaPipe landmark topology + OUR measured LiDAR depth  "
                 "(top = frontal, bottom = profile — nose protrusion is our real measurement)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / "face_landmark_mesh.png", dpi=140)
    print(f"wrote {out/'face_landmark_mesh.png'}")


if __name__ == "__main__":
    main()
