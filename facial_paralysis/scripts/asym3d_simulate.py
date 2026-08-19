"""Track 4: measurement-error characterization for the 3D asymmetry pipeline, using
SYNTHETIC ground truth (zero patients, zero labels).

Idea: take a real decoded face depth, make it perfectly L-R symmetric by construction
(known asymmetry = 0), then inject a KNOWN depth deformation delta on one side of a
region. Re-run the full landmark sampler + pose-plane + pair asymmetry and check we
recover delta. Sweeping delta gives: the NOISE FLOOR (symmetric baseline), the
SENSITIVITY/linearity (measured vs true slope), and the MINIMAL DETECTABLE asymmetry.

RUN WITH arm64 /usr/bin/python3.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from scipy.ndimage import median_filter
sys.path.insert(0, str(Path(__file__).resolve().parent))
import asym3d_landmark as A

ROOT = Path(__file__).resolve().parent.parent
D3 = ROOT / "outputs" / "depth3d"


def region_asym(up, xy):
    """Per-region pose-corrected 3D asymmetry (mm) from an upright depth + landmarks."""
    H, W = up.shape
    fm = A.face_anchor(up, xy, W, H)
    if np.isnan(fm):
        return None
    lo, hi = fm - 0.07, fm + 0.07
    lm = {}
    for i in A.ALL_IDX:
        Z = A.sample_Z(up, xy[i, 0] * W, xy[i, 1] * H, lo, hi)
        if not np.isnan(Z):
            lm[i] = (xy[i, 0] * W, xy[i, 1] * H, Z)
    if len(lm) < 12:
        return None
    a, b, c = A.fit_pose_plane(lm)
    resid = {i: Z - (a * u + b * v + c) for i, (u, v, Z) in lm.items()}
    out = {}
    for region, ps in A.PAIRS.items():
        diffs = [abs(resid[l] - resid[r]) for l, r in ps if l in resid and r in resid]
        if diffs:
            out[region] = float(np.median(diffs)) * 1000
    return out


def symmetrize(up, axis):
    """Mirror-average about column `axis` -> depth that is L-R symmetric by construction."""
    H, W = up.shape
    sym = up.copy()
    for u in range(W):
        um = int(round(2 * axis - u))
        if 0 <= um < W:
            a, b = up[:, u], up[:, um]
            both = (a > 0.05) & (b > 0.05)
            m = (a + b) / 2
            sym[both, u] = m[both]
    return sym


def main():
    # use a few good takes; average their floor/slope for robustness
    takes = ["20260219_FACES014", "20260313_FACES021", "20260305_FACES018"]
    deltas_mm = [0, 2, 4, 6, 8, 10]
    rows = []
    for name in takes:
        p = D3 / f"{name}_depth.npy"
        if not p.exists():
            continue
        up = np.rot90(median_filter(np.load(p), 3), 1)
        H, W = up.shape
        xy = np.array(json.load(open(D3 / "landmarks.json"))[name]["xy"])
        axis = xy[1, 0] * W                                   # nose-tip column = midline
        sym = symmetrize(up, axis)
        # mouth region bbox (to inject a localized deformation on the LEFT half)
        mi = [i for pr in A.PAIRS["mouth"] for i in pr]
        mu, mv = xy[mi, 0] * W, xy[mi, 1] * H
        u0, u1, v0, v1 = int(mu.min()) - 6, int(mu.max()) + 6, int(mv.min()) - 6, int(mv.max()) + 6
        for dmm in deltas_mm:
            dep = sym.copy()
            reg = np.zeros_like(dep, bool)
            reg[max(v0, 0):v1, max(u0, 0):int(axis)] = True   # left half of the mouth box
            reg &= dep > 0.05
            dep[reg] += dmm / 1000.0                           # known protrusion delta (m)
            r = region_asym(dep, xy)
            if r:
                rows.append((name, dmm, round(r.get("mouth", np.nan), 2)))
    print(f"{'take':20s} {'true_delta_mm':>13s} {'measured_mouth_mm':>18s}")
    for n, d, m in rows:
        print(f"{n:20s} {d:13d} {m:18.2f}")
    # aggregate: floor (delta=0) and slope
    import numpy as np2
    base = [m for n, d, m in rows if d == 0 and not np.isnan(m)]
    td = np.array([d for n, d, m in rows if not np.isnan(m)])
    md = np.array([m for n, d, m in rows if not np.isnan(m)])
    slope, intercept = np.polyfit(td, md, 1)
    floor = float(np.mean(base)) if base else np.nan
    print(f"\nNOISE FLOOR (true delta=0): {floor:.2f} mm  (measured asymmetry on a symmetric face)")
    print(f"SENSITIVITY: measured = {slope:.2f} * true + {intercept:.2f} mm  (slope 1.0 = unbiased)")
    print(f"MINIMAL DETECTABLE (floor + 2*floor-sd): ~{floor + 2*np.std(base):.1f} mm" if len(base) > 1 else "")
    (D3 / "sim_measurement_error.json").write_text(json.dumps(
        {"rows": rows, "noise_floor_mm": round(floor, 2), "slope": round(float(slope), 2),
         "intercept_mm": round(float(intercept), 2)}, indent=1))
    print(f"wrote {D3/'sim_measurement_error.json'}")


if __name__ == "__main__":
    main()
