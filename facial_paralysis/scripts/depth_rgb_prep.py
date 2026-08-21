"""Prep for landmark-anchored 3D asymmetry (#4). LOCAL, arm64 /usr/bin/python3.

For each take that has BOTH depth_data.bin and the *_iPhone.mov:
  - parse frame_log.csv -> depth-frame timecode-key -> ns clock; video-frame ns -> index
  - pick a resting depth record (early, largest near-face area)
  - decode it (native 360x640 fp16 meters) -> save <take>_depth.npy
  - find the video frame at the SAME ns timestamp -> ffmpeg-extract that upright RGB -> <take>_rgb.jpg
  - record the match (exact dt, indices) in manifest.json

Depth<->RGB geometry: upright RGB is 1280x720 portrait = exactly 2x the upright depth
(rot90 of native 360x640 -> 640x360). MediaPipe runs on the RGB (pod); landmarks map to
depth by rot + /2. Temporal match is exact (shared ns clock, ~0 ms).
"""
from __future__ import annotations
import re, sys, json, subprocess
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mayo_depth3d as M

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "livelinkface_data"
OUT = ROOT / "outputs" / "depth3d"


def parse_frame_log(take: Path):
    """Return dkey2ns {timecode-key -> ns} for D frames and V arrays (ns, idx)."""
    dkey2ns, vns, vidx = {}, [], []
    for ln in (take / "frame_log.csv").read_text().splitlines():
        c = ln.split(",")
        if len(c) < 5:
            continue
        key = int(re.sub(r"\D", "", c[4]))            # timecode digits -> key
        if c[0] == "V":
            vns.append(int(c[2])); vidx.append(int(c[1]))
        elif c[0] == "D":
            dkey2ns[key] = int(c[2])
    order = np.argsort(vns)
    return dkey2ns, np.array(vns)[order], np.array(vidx)[order]


def depth_records(take: Path):
    """Yield (timecode-key, kraken-block) for each 0x05 depth record."""
    data = (take / "depth_data.bin").read_bytes()
    marks = [m.start() for m in re.finditer(rb'[\x00-\x1f]\d{15}(?!\d)', data)] + [len(data)]
    for i in range(len(marks) - 1):
        p = marks[i]
        if data[p] == 0x05:
            yield int(data[p + 1:p + 16]), data[p + 32:marks[i + 1]]


def near_area(d):
    if d is None:
        return 0
    fg = d[d > 0.05]
    if fg.size < 500:
        return 0
    near = np.percentile(fg, 3)
    return int(((d > near - 0.02) & (d < near + 0.25)).sum())


def pick_resting(recs):
    """Among early records, the one with the largest near-face area (a held repose)."""
    best = None
    for ordi in range(30, 200, 10):
        if ordi >= len(recs):
            break
        key, blk = recs[ordi]
        d = M.decode(blk)
        a = near_area(d)
        if a > 6000 and (best is None or a > best[2]):
            best = (ordi, key, a, d)
    return best


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {}
    takes = sorted(p.parent for p in DATA.rglob("depth_data.bin"))
    for take in takes:
        movs = list(take.glob("*_iPhone.mov")) + list(take.glob("*_iPhone.mp4"))
        if not movs:
            continue
        recs = list(depth_records(take))
        if not recs:
            continue
        dkey2ns, vns, vidx = parse_frame_log(take)
        pick = pick_resting(recs)
        if pick is None:
            print(f"  {take.name}: no resting frame"); continue
        ordi, key, area, d = pick
        ns = dkey2ns.get(key)
        if ns is None:
            print(f"  {take.name}: rec key {key} not in frame_log"); continue
        j = int(np.argmin(np.abs(vns - ns)))
        vframe = int(vidx[j]); dt_ms = float(vns[j] - ns) / 1e6
        t = vframe / 60.0
        name = take.name
        # temporal median over +/-7 resting frames: fills holes + crushes ~10mm/px streak noise
        stack = []
        for o in range(max(ordi - 7, 0), min(ordi + 8, len(recs))):
            dd = M.decode(recs[o][1])
            if dd is not None and near_area(dd) > 4000:
                stack.append(np.where(dd > 0.05, dd, np.nan))
        med = np.nanmedian(np.stack(stack), axis=0) if stack else d
        med = np.where(np.isfinite(med), med, 0.0)
        np.save(OUT / f"{name}_depth.npy", med.astype(np.float32))
        subprocess.run(["ffmpeg", "-y", "-ss", f"{t:.4f}", "-i", str(movs[0]),
                        "-frames:v", "1", str(OUT / f"{name}_rgb.jpg")],
                       capture_output=True)
        manifest[name] = {"rec": ordi, "key": key, "vframe": vframe,
                          "dt_ms": round(dt_ms, 2), "time_s": round(t, 4),
                          "near_area": area}
        print(f"  {name}: rec {ordi} -> Vframe {vframe} (dt {dt_ms:+.1f}ms) area {area}")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"\n{len(manifest)} takes prepped -> {OUT}")


if __name__ == "__main__":
    main()
