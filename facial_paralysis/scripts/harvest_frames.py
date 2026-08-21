"""Multi-frame harvest for #4 tracks 1/2/3: for each take, several RESTING depth frames
(reliability) + each ACTION-peak depth frame (dynamic 3D, lagophthalmos), each matched to
its RGB frame via the frame_log ns clock. Saves depth medians + RGB jpgs + manifest for the
pod MediaPipe pass. LOCAL, arm64 /usr/bin/python3.
"""
from __future__ import annotations
import re, sys, json, subprocess
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mayo_depth3d as M
from depth_rgb_prep import parse_frame_log, depth_records, near_area

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "livelinkface_data"
BS = ROOT / "outputs" / "mayo_blendshapes"
SEG = json.loads((BS / "segments.json").read_text())
OUT = ROOT / "outputs" / "depth3d" / "harvest"

# blendshape indices whose peak marks each action
PEAK_SIG = {"EyebrowRise": [4, 5, 3], "GentleEyeClosure": [9, 10], "TightEyeSqueeze": [9, 10, 19, 20],
            "RelaxedSmile": [44, 45], "ReanimatedSmile": [44, 45], "LipPucker": [38],
            "LowerTeethShow": [34, 35]}


def depth_time_index(take):
    """List of (time_s, ordinal, key, blk) for each depth record, time via frame_log."""
    dkey2ns, vns, vidx = parse_frame_log(take)
    recs = list(depth_records(take))
    idx = []
    for o, (key, blk) in enumerate(recs):
        ns = dkey2ns.get(key)
        if ns is None:
            continue
        j = int(np.argmin(np.abs(vns - ns)))
        idx.append((int(vidx[j]) / 60.0, o, key, blk))
    return idx, recs


def median_depth(recs, ordi, half=5):
    stack = []
    for o in range(max(ordi - half, 0), min(ordi + half + 1, len(recs))):
        dd = M.decode(recs[o][1])
        if dd is not None and near_area(dd) > 4000:
            stack.append(np.where(dd > 0.05, dd, np.nan))
    if not stack:
        return None
    med = np.nanmedian(np.stack(stack), axis=0)
    return np.where(np.isfinite(med), med, 0.0).astype(np.float32)


def nearest_record(tidx, t_target):
    cand = [(abs(tt - t_target), o) for (tt, o, k, b) in tidx]
    return min(cand)[1] if cand else None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for take in sorted(p.parent for p in DATA.rglob("depth_data.bin")):
        movs = list(take.glob("*_iPhone.mov")) + list(take.glob("*_iPhone.mp4"))
        name = take.name
        if not movs or name not in SEG:
            continue
        tidx, recs = depth_time_index(take)
        if not tidx:
            continue
        mov = movs[0]
        segs = SEG[name]
        first_action = min((s["t_start"] for s in segs), default=8.0)
        targets = []                                          # (frame_id, kind, action, t_target)
        # resting: evenly spaced before the first action (repose hold)
        rest_end = max(min(first_action - 0.5, 7.0), 2.0)
        for k, tr in enumerate(np.linspace(1.0, rest_end, 6)):
            targets.append((f"{name}__rest{k}", "rest", "repose", float(tr)))
        # action peaks: max relevant blendshape within each window
        d = np.load(BS / f"{name}.npz", allow_pickle=True)
        bs, tb = d["bs"], d["t"]
        for s in segs:
            act = s["action"]
            if act not in PEAK_SIG:
                continue
            w = (tb >= s["t_start"]) & (tb <= s["t_end"])
            if w.sum() < 2:
                continue
            trace = bs[w][:, PEAK_SIG[act]].max(1)
            t_peak = float(tb[w][int(np.argmax(trace))])
            targets.append((f"{name}__{act}", "action", act, t_peak))
        # realize each target
        for fid, kind, act, tt in targets:
            ordi = nearest_record(tidx, tt)
            if ordi is None:
                continue
            med = median_depth(recs, ordi)
            if med is None:
                continue
            t_real = [x for x in tidx if x[1] == ordi][0][0]
            np.save(OUT / f"{fid}_depth.npy", med)
            subprocess.run(["ffmpeg", "-y", "-ss", f"{t_real:.4f}", "-i", str(mov),
                            "-frames:v", "1", str(OUT / f"{fid}_rgb.jpg")], capture_output=True)
            manifest[fid] = {"take": name, "kind": kind, "action": act,
                             "t_target": round(tt, 3), "t_depth": round(t_real, 3), "ordi": ordi}
        print(f"  {name}: {sum(1 for m in manifest.values() if m['take']==name)} frames")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"\n{len(manifest)} frames harvested -> {OUT}")


if __name__ == "__main__":
    main()
