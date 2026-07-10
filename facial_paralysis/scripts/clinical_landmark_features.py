"""Clinical geometric features from raw MediaPipe 468 landmarks (the SOTA recipe:
Emotrics/Auto-eFACE). Pose-normalized (level eyes, scale by interocular distance), per-side
then L/R asymmetry. The static eye signal (palpebral fissure HEIGHT) is what ARKit eyeBlink
misses on open-eye stills -- the hypothesized fix for the eyes-QWK bottleneck.

clinical_feats(xy, w, h) -> (feature_vector, names). xy = 468x(3) normalized landmarks.
"""
from __future__ import annotations
import numpy as np

# MediaPipe FaceMesh indices (subject's RIGHT = image left).
R_EYE_RING = [33, 133, 159, 145, 160, 144, 158, 153]
L_EYE_RING = [263, 362, 386, 374, 387, 373, 385, 380]
R_UP, R_LO = [159, 158, 160], [145, 144, 153]        # right upper / lower lid
L_UP, L_LO = [386, 385, 387], [374, 380, 373]        # left upper / lower lid
R_IN, R_OUT = 133, 33                                # right eye inner/outer corner
L_IN, L_OUT = 362, 263
R_BROW = [70, 63, 105, 66, 107]
L_BROW = [300, 293, 334, 296, 336]
R_CORNER, L_CORNER = 61, 291                         # mouth corners
MIDLINE = [168, 6, 197, 195, 5, 4, 1, 19, 2, 164, 0, 13, 14, 17, 152, 10]
NOSE_TIP = 1
MOUTH_TOP, MOUTH_BOT = 13, 14

NAMES = [
    "fissure_h_R", "fissure_h_L", "fissure_h_asym", "fissure_h_sd",
    "fissure_w_R", "fissure_w_L", "fissure_w_asym",
    "eye_area_R", "eye_area_L", "eye_area_asym",
    "brow_h_R", "brow_h_L", "brow_h_asym", "brow_h_sd",
    "corner_y_R", "corner_y_L", "corner_y_asym", "corner_y_sd",
    "corner_x_R", "corner_x_L", "commissure_asym",
    "mouth_width", "mouth_open",
]


def _pt(xy, idx, s=1.0):
    return xy[idx, :2] * s


def clinical_feats(xy, w, h):
    """xy: (468,>=2) normalized. Returns (vec[len(NAMES)], NAMES). NaN-safe -> zeros."""
    P = xy[:, :2] * np.array([w, h], np.float32)         # to pixels
    rc = P[R_EYE_RING].mean(0)
    lc = P[L_EYE_RING].mean(0)
    iod = np.linalg.norm(lc - rc) + 1e-6                 # interocular distance = scale
    theta = np.arctan2(lc[1] - rc[1], lc[0] - rc[0])     # roll: level the eyes
    c, s = np.cos(-theta), np.sin(-theta)
    Rm = np.array([[c, -s], [s, c]], np.float32)
    ctr = P[MIDLINE].mean(0)
    Q = (P - ctr) @ Rm.T / iod                           # leveled, IOD-scaled, centered
    y = Q[:, 1]; x = Q[:, 0]
    mid_x = Q[MIDLINE, 0].mean()
    eye_line_y = 0.5 * (Q[R_EYE_RING, 1].mean() + Q[L_EYE_RING, 1].mean())

    fh_R = y[R_LO].mean() - y[R_UP].mean()               # fissure height (eye openness)
    fh_L = y[L_LO].mean() - y[L_UP].mean()
    fw_R = abs(x[R_OUT] - x[R_IN]); fw_L = abs(x[L_OUT] - x[L_IN])
    ar_R, ar_L = fh_R * fw_R, fh_L * fw_L
    bh_R = Q[R_EYE_RING, 1].mean() - y[R_BROW].mean()    # brow height above eye
    bh_L = Q[L_EYE_RING, 1].mean() - y[L_BROW].mean()
    cy_R = y[R_CORNER] - eye_line_y                      # mouth-corner height vs eye line
    cy_L = y[L_CORNER] - eye_line_y
    cx_R = abs(x[R_CORNER] - mid_x); cx_L = abs(x[L_CORNER] - mid_x)  # commissure excursion
    mw = abs(x[L_CORNER] - x[R_CORNER])
    mo = y[MOUTH_BOT] - y[MOUTH_TOP]

    def asym(a, b):
        return abs(a - b)

    vec = np.array([
        fh_R, fh_L, asym(fh_R, fh_L), (fh_R - fh_L),
        fw_R, fw_L, asym(fw_R, fw_L),
        ar_R, ar_L, asym(ar_R, ar_L),
        bh_R, bh_L, asym(bh_R, bh_L), (bh_R - bh_L),
        cy_R, cy_L, asym(cy_R, cy_L), (cy_R - cy_L),
        cx_R, cx_L, asym(cx_R, cx_L),
        mw, mo,
    ], np.float32)
    vec[~np.isfinite(vec)] = 0.0
    return vec, NAMES


if __name__ == "__main__":
    import json, sys
    d = json.load(open(sys.argv[1]))
    k = next(iter(d)); e = d[k]
    v, names = clinical_feats(np.array(e["xy"]), e["w"], e["h"])
    for n, val in zip(names, v):
        print(f"  {n:16s} {val:+.3f}")
