"""Direction #4 payoff: landmark-anchored 3D facial asymmetry from decoded iPhone depth.

Appearance/camera-INVARIANT (sidesteps the MARLIN domain confound, domain-AUC 1.0). Unlike
the landmark-free version (which measured head POSE, rho~0 with clinical asymmetry), this
anchors on MediaPipe landmarks so we measure the FACE, per region.

Method (needs NO camera intrinsics -- works in metric depth Z directly):
  1. upright depth = rot90(native, 1); landmarks (from matched RGB) map by (x*W, y*H).
  2. sample metric depth Z (median in a window) at each landmark.
  3. remove rigid head pose: least-squares plane Z ~ a*u + b*v + c over all reliable
     landmarks. Residual r = Z - plane cancels yaw (linear in u) and pitch (linear in v).
  4. for each anatomically symmetric pair (L,R): asymmetry = |r_L - r_R| (a mirror-symmetric
     face has r_L=r_R). RMS over pairs = 3D asymmetry (meters); also per region.

Validated: exact temporal depth<->RGB match (frame_log ns clock), rot90 k=1 confirmed by
landmark overlay. RUN WITH arm64 /usr/bin/python3.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from scipy.ndimage import median_filter

ROOT = Path(__file__).resolve().parent.parent
D3 = ROOT / "outputs" / "depth3d"

# Anatomically symmetric MediaPipe FaceMesh pairs (L index, R index) by region.
PAIRS = {
    "eye":   [(33, 263), (133, 362), (159, 386), (145, 374), (160, 385), (144, 380), (158, 387), (153, 373)],
    "brow":  [(46, 276), (52, 282), (53, 283), (65, 295), (70, 300), (63, 293), (105, 334), (66, 296), (107, 336)],
    "mouth": [(61, 291), (40, 270), (37, 267), (84, 314), (91, 321), (146, 375), (78, 308), (88, 318), (185, 409)],
    "cheek": [(50, 280), (205, 425), (101, 330), (36, 266), (207, 427), (147, 376), (123, 352), (187, 411)],
    "jaw":   [(58, 288), (172, 397), (136, 365), (150, 379), (132, 361), (234, 454), (93, 323), (215, 435)],
}
ALL_IDX = sorted({i for ps in PAIRS.values() for p in ps for i in p})
NOSE = [1, 4, 5, 6, 195, 197, 168, 2]                    # central nose: nearest, reliable anchor


def sample_Z(up, u, v, lo, hi):
    """Median depth in the face band [lo,hi], growing the window until >=8 in-band px."""
    H, W = up.shape
    u, v = int(round(u)), int(round(v))
    if not (0 <= u < W and 0 <= v < H):
        return np.nan
    for win in (5, 9, 13):
        patch = up[max(v - win, 0):v + win + 1, max(u - win, 0):u + win + 1]
        vals = patch[(patch >= lo) & (patch <= hi)]      # only face-band pixels
        if vals.size >= 8:
            return float(np.median(vals))
    return np.nan


def face_anchor(up, xy, W, H):
    """Robust face depth from nose landmarks (nearest, central, hole-free)."""
    zs = []
    for i in NOSE:
        u, v = int(round(xy[i, 0] * W)), int(round(xy[i, 1] * H))
        if 0 <= u < W and 0 <= v < H:
            p = up[max(v - 6, 0):v + 7, max(u - 6, 0):u + 7]
            p = p[p > 0.05]
            if p.size >= 8:
                zs.append(np.median(p))
    return float(np.median(zs)) if zs else np.nan


def landmark_depths(name):
    """idx -> (u,v,Z) for landmarks in the nose-anchored face band (rejects hair/bg/streak)."""
    d = median_filter(np.load(D3 / f"{name}_depth.npy"), size=(3, 3))
    up = np.rot90(d, 1)                                   # k=1 upright (validated)
    H, W = up.shape
    xy = np.array(json.load(open(D3 / "landmarks.json"))[name]["xy"])
    fm = face_anchor(up, xy, W, H)
    if np.isnan(fm):
        return {}, W, H
    lo, hi = fm - 0.07, fm + 0.07                         # +/-7cm face band about nose anchor
    out = {}
    for i in ALL_IDX:
        Z = sample_Z(up, xy[i, 0] * W, xy[i, 1] * H, lo, hi)
        if not np.isnan(Z):
            out[i] = (xy[i, 0] * W, xy[i, 1] * H, Z)
    return out, W, H


def fit_pose_plane(lm):
    """Robust LS plane Z ~ a*u+b*v+c: fit, drop the worst residual decile, refit (2x)."""
    idx = list(lm)
    for _ in range(2):
        A = np.array([[lm[i][0], lm[i][1], 1.0] for i in idx])
        z = np.array([lm[i][2] for i in idx])
        coef, *_ = np.linalg.lstsq(A, z, rcond=None)
        r = np.abs(z - A @ coef)
        keep = r <= np.percentile(r, 90)
        if keep.sum() < 12:
            break
        idx = [i for i, k in zip(idx, keep) if k]
    return coef


def asym3d(name):
    lm, W, H = landmark_depths(name)
    if len(lm) < 12:
        return None
    a, b, c = fit_pose_plane(lm)
    resid = {i: Z - (a * u + b * v + c) for i, (u, v, Z) in lm.items()}
    per_region, all_diffs = {}, []
    for region, ps in PAIRS.items():
        diffs = [abs(resid[l] - resid[r]) for l, r in ps if l in resid and r in resid]
        if diffs:
            per_region[region] = round(float(np.median(diffs)) * 1000, 2)   # mm (robust)
            all_diffs += diffs
    if len(all_diffs) < 6:
        return None
    overall = round(float(np.median(all_diffs)) * 1000, 2)                   # mm (robust)
    return {"overall_mm": overall, "regions": per_region, "n_pairs": len(all_diffs),
            "n_landmarks": len(lm)}


def main():
    names = sorted(p.stem[:-6] for p in D3.glob("*_depth.npy"))
    res = {}
    for name in names:
        r = asym3d(name)
        if r:
            res[name.split("_", 1)[1]] = r
    (D3 / "asym3d_landmark.json").write_text(json.dumps(res, indent=1))

    print(f"{'patient':13s} {'overall':>8s}  {'eye':>6s} {'brow':>6s} {'mouth':>6s} {'cheek':>6s} {'jaw':>6s}  (mm, pose-corrected)")
    print("-" * 78)
    for p, r in sorted(res.items(), key=lambda x: -x[1]["overall_mm"]):
        g = r["regions"]
        print(f"{p:13s} {r['overall_mm']:8.2f}  " + " ".join(f"{g.get(k, 0):6.1f}" for k in ("eye", "brow", "mouth", "cheek", "jaw")))

    dup = [res[k]["overall_mm"] for k in ("FACES018", "MySlate_14") if k in res]
    print(f"\nVALIDATION")
    print(f"  determinism (FACES018 vs MySlate_14 duplicate): {dup} -> {'MATCH' if len(dup) == 2 and dup[0] == dup[1] else 'DIFFER'}")
    vv = [r["overall_mm"] for r in res.values()]
    print(f"  spread: {min(vv):.2f}-{max(vv):.2f} mm ({max(vv)/max(min(vv),1e-6):.1f}x), n={len(res)}")
    print(f"  wrote {D3/'asym3d_landmark.json'}")


if __name__ == "__main__":
    main()
