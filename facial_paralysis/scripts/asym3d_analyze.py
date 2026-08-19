"""#4 tracks 1-3, label-free, no new patients. From the harvested multi-frame landmarks+depth.

  Track 3 (reliability): CV within-patient + one-way ICC of the resting 3D asymmetry across
                         independent resting frames -> is the measure stable?
  Track 1 (dynamic, within-subject): per region, is 3D asymmetry LARGER at the action that
                         targets it than at rest? (affected side fails to move -> asymmetry
                         opens up on demand). Paired Wilcoxon across patients. No labels/side.
  Track 2 (depth-unique): nasolabial-fold relief asymmetry (rest) and lagophthalmos eye-asym
                         at eye-closure vs rest -- both things 2D blendshapes cannot measure.

RUN WITH arm64 /usr/bin/python3.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from scipy.ndimage import median_filter
from scipy.stats import wilcoxon
sys.path.insert(0, str(Path(__file__).resolve().parent))
import asym3d_landmark as A

ROOT = Path(__file__).resolve().parent.parent
H = ROOT / "outputs" / "depth3d" / "harvest"
MAN = json.load(open(H / "manifest.json"))
LM = json.load(open(H / "landmarks.json"))
ACTION_FOR = {"mouth": ["RelaxedSmile", "ReanimatedSmile", "LipPucker"],
              "eye": ["TightEyeSqueeze", "GentleEyeClosure"], "brow": ["EyebrowRise"]}
NASO = [(205, 425), (206, 426), (216, 436), (92, 322)]      # cheek-fold region pairs


def resid_of(fid):
    """Pose-corrected residual depth per landmark for a harvested frame; None if unusable."""
    p = H / f"{fid}_depth.npy"
    if not p.exists() or fid not in LM:
        return None
    up = np.rot90(median_filter(np.load(p), 3), 1); Hh, Ww = up.shape
    xy = np.array(LM[fid]["xy"])
    fm = A.face_anchor(up, xy, Ww, Hh)
    if np.isnan(fm):
        return None
    lo, hi = fm - 0.07, fm + 0.07
    lm = {}
    idx = A.ALL_IDX + [i for pr in NASO for i in pr]
    for i in set(idx):
        Z = A.sample_Z(up, xy[i, 0] * Ww, xy[i, 1] * Hh, lo, hi)
        if not np.isnan(Z):
            lm[i] = (xy[i, 0] * Ww, xy[i, 1] * Hh, Z)
    if len(lm) < 12:
        return None
    a, b, c = A.fit_pose_plane({i: lm[i] for i in lm if i in A.ALL_IDX} or lm)
    return {i: Z - (a * u + b * v + c) for i, (u, v, Z) in lm.items()}


def region_asym(resid, region):
    diffs = [abs(resid[l] - resid[r]) for l, r in A.PAIRS[region] if l in resid and r in resid]
    return float(np.median(diffs)) * 1000 if diffs else np.nan


def main():
    frames = {fid: resid_of(fid) for fid in MAN}
    frames = {k: v for k, v in frames.items() if v is not None}
    takes = sorted({MAN[f]["take"] for f in frames})

    # ---- Track 3: reliability of resting overall asymmetry ----
    def overall(resid):
        vs = [region_asym(resid, r) for r in A.PAIRS]
        vs = [x for x in vs if not np.isnan(x)]
        return float(np.median(vs)) if vs else np.nan
    per_take_rest = {}
    for t in takes:
        vals = [overall(frames[f]) for f in frames if MAN[f]["take"] == t and MAN[f]["kind"] == "rest"]
        vals = [v for v in vals if not np.isnan(v)]
        if len(vals) >= 3:
            per_take_rest[t] = vals
    cvs = [np.std(v) / np.mean(v) for v in per_take_rest.values() if np.mean(v) > 0]
    # one-way ICC across takes (rest frames as repeats)
    grand = np.mean([x for v in per_take_rest.values() for x in v])
    k = np.mean([len(v) for v in per_take_rest.values()])
    means = [np.mean(v) for v in per_take_rest.values()]
    MSB = k * np.var(means, ddof=1)
    MSW = np.mean([np.var(v, ddof=1) for v in per_take_rest.values()])
    icc = (MSB - MSW) / (MSB + (k - 1) * MSW) if (MSB + (k - 1) * MSW) > 0 else np.nan
    print("TRACK 3 -- reliability of resting 3D asymmetry")
    print(f"  within-patient CV: median {np.median(cvs)*100:.0f}% (n={len(cvs)} patients, ~{k:.0f} frames each)")
    print(f"  one-way ICC (between-patient / total): {icc:.2f}   [>0.75 good, >0.5 moderate]")

    # ---- Track 1: dynamic within-subject (action opens asymmetry vs rest) ----
    print("\nTRACK 1 -- dynamic: is 3D asymmetry larger at the targeting action than at rest?")
    dyn = {}
    for region, acts in ACTION_FOR.items():
        rest_v, act_v = [], []
        for t in takes:
            rest = [region_asym(frames[f], region) for f in frames if MAN[f]["take"] == t and MAN[f]["kind"] == "rest"]
            rest = [x for x in rest if not np.isnan(x)]
            af = [f for f in frames if MAN[f]["take"] == t and MAN[f]["action"] in acts]
            av = [region_asym(frames[f], region) for f in af]
            av = [x for x in av if not np.isnan(x)]
            if rest and av:
                rest_v.append(np.median(rest)); act_v.append(np.max(av))
        if len(rest_v) >= 5:
            d = np.array(act_v) - np.array(rest_v)
            try:
                W, pv = wilcoxon(act_v, rest_v)
            except ValueError:
                pv = np.nan
            dyn[region] = {"n": len(d), "rest_mm": round(float(np.median(rest_v)), 2),
                           "action_mm": round(float(np.median(act_v)), 2),
                           "delta_mm": round(float(np.median(d)), 2), "p": round(float(pv), 3)}
            print(f"  {region:6s}: rest {np.median(rest_v):5.1f} -> action {np.median(act_v):5.1f} mm  "
                  f"(delta {np.median(d):+5.1f}, Wilcoxon p={pv:.3f}, n={len(d)})")

    # ---- Track 2: depth-unique ----
    print("\nTRACK 2 -- depth-only measures (2D cannot compute these)")
    naso = {}
    for t in takes:
        rf = [frames[f] for f in frames if MAN[f]["take"] == t and MAN[f]["kind"] == "rest"]
        rel = []
        for resid in rf:
            ds = [abs(resid[l] - resid[r]) for l, r in NASO if l in resid and r in resid]
            if ds:
                rel.append(np.median(ds) * 1000)
        if rel:
            naso[t.split("_", 1)[1]] = round(float(np.median(rel)), 2)
    print(f"  nasolabial-fold relief asymmetry (rest): {len(naso)} patients, "
          f"range {min(naso.values()):.1f}-{max(naso.values()):.1f} mm")
    # lagophthalmos: eye asym at closure vs rest
    lag = []
    for t in takes:
        rest = [region_asym(frames[f], "eye") for f in frames if MAN[f]["take"] == t and MAN[f]["kind"] == "rest"]
        rest = [x for x in rest if not np.isnan(x)]
        clos = [region_asym(frames[f], "eye") for f in frames if MAN[f]["take"] == t and MAN[f]["action"] in ("TightEyeSqueeze", "GentleEyeClosure")]
        clos = [x for x in clos if not np.isnan(x)]
        if rest and clos:
            lag.append((t.split("_", 1)[1], np.median(rest), np.max(clos)))
    if lag:
        d = [c - r for _, r, c in lag]
        try:
            _, pv = wilcoxon([c for _, _, c in lag], [r for _, r, _ in lag])
        except ValueError:
            pv = np.nan
        print(f"  lagophthalmos (eye 3D asym at closure vs rest): "
              f"{np.median([r for _,r,_ in lag]):.1f} -> {np.median([c for _,_,c in lag]):.1f} mm "
              f"(delta {np.median(d):+.1f}, p={pv:.3f}, n={len(lag)})")

    out = {"reliability": {"cv_median_pct": round(float(np.median(cvs)) * 100, 1), "icc": round(float(icc), 3)},
           "dynamic": dyn, "nasolabial_mm": naso,
           "lagophthalmos": {"delta_mm": round(float(np.median(d)), 2), "p": round(float(pv), 3), "n": len(lag)} if lag else {}}
    (H.parent / "asym3d_tracks.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote {H.parent/'asym3d_tracks.json'}  (usable frames {len(frames)}/{len(MAN)}, takes {len(takes)})")


if __name__ == "__main__":
    main()
