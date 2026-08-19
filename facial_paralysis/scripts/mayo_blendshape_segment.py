"""Dense per-frame blendshape time-series + per-action segmentation for the Mayo
recordings, using the KNOWN action order (memory: mayo-action-protocol).

The recordings have no timestamps, but the action ORDER is fixed (7-8 actions),
and each action has a distinct MediaPipe blendshape signature. So we:
  1. Decode frames at FPS via ffmpeg (fast) and run MediaPipe FaceLandmarker to get
     a dense (T, 52) blendshape time-series.
  2. Build per-action SIGNATURE activation curves (brow / eye-close / eye-squint /
     smile / pucker / teeth) over time.
  3. Detect movement BOUTS (above-rest activity separated by rest) and label them
     IN ORDER (ordered matching), disambiguating gentle-vs-tight eye closure and
     relaxed-vs-reanimated smile by occurrence order.

This is the proof-of-concept + segmenter. Run on the RunPod A100 (mediapipe there):
    .venv/bin/python scripts/mayo_blendshape_segment.py <take>          # one take + plot
    .venv/bin/python scripts/mayo_blendshape_segment.py --timeseries-all  # extract all
"""
from __future__ import annotations

import glob
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LLF = ROOT / "data" / "livelinkface_data"
MODEL = ROOT / "data" / "mediapipe_out" / "_models" / "face_landmarker.task"
OUT = ROOT / "outputs" / "mayo_blendshapes"
FPS = 6

# Action order (memory: mayo-action-protocol) -> signature blendshapes.
ACTION_ORDER = ["EyebrowRise", "GentleEyeClosure", "TightEyeSqueeze",
                "RelaxedSmile", "LipPucker", "LowerTeethShow", "ReanimatedSmile"]
SIGS = {
    "brow":      ["browInnerUp", "browOuterUpLeft", "browOuterUpRight"],
    "eye_close": ["eyeBlinkLeft", "eyeBlinkRight"],
    "eye_squint":["eyeSquintLeft", "eyeSquintRight", "cheekSquintLeft", "cheekSquintRight"],
    "smile":     ["mouthSmileLeft", "mouthSmileRight"],
    "pucker":    ["mouthPucker"],
    "teeth":     ["mouthLowerDownLeft", "mouthLowerDownRight", "jawOpen"],
}
# which signature group each ordered action keys on
ACTION_SIG = {"EyebrowRise": "brow", "GentleEyeClosure": "eye_close",
              "TightEyeSqueeze": "eye_squint", "RelaxedSmile": "smile",
              "LipPucker": "pucker", "LowerTeethShow": "teeth", "ReanimatedSmile": "smile"}


def extract_timeseries(take: str):
    mov = glob.glob(str(LLF / take / "*.mov"))
    if not mov:
        return None
    import os
    fr = Path(f"/dev/shm/_bsf_{os.getpid()}")   # PID-unique so parallel jobs don't collide
    fr.mkdir(exist_ok=True)
    for f in fr.glob("*.jpg"):
        f.unlink()
    subprocess.run(["ffmpeg", "-y", "-i", mov[0], "-vf", f"fps={FPS}", "-q:v", "3",
                    str(fr / "%05d.jpg")], capture_output=True)
    jpgs = sorted(fr.glob("*.jpg"))
    if not jpgs:
        return None
    import mediapipe as mp
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision as mpv
    lm = mpv.FaceLandmarker.create_from_options(mpv.FaceLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=str(MODEL)),
        running_mode=mpv.RunningMode.IMAGE, output_face_blendshapes=True, num_faces=1))
    names, rows, mask = None, [], []
    for j in jpgs:
        img = cv2.imread(str(j))
        mpimg = mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        res = lm.detect(mpimg)
        if not res.face_blendshapes:
            rows.append(None); mask.append(False); continue
        bs = res.face_blendshapes[0]
        if names is None:
            names = [c.category_name for c in bs]
        rows.append(np.array([c.score for c in bs], dtype=np.float32)); mask.append(True)
    for f in jpgs:
        f.unlink()
    if names is None:
        return None
    F = len(names)
    arr = np.zeros((len(rows), F), dtype=np.float32)
    for k, r in enumerate(rows):
        if r is not None:
            arr[k] = r
    t = np.arange(len(rows)) / FPS
    return arr, np.array(names), t, np.array(mask)


def sig_curves(arr, names):
    idx = {n: i for i, n in enumerate(names)}
    curves = {}
    for grp, bsn in SIGS.items():
        cols = [idx[n] for n in bsn if n in idx]
        curves[grp] = arr[:, cols].mean(axis=1) if cols else np.zeros(len(arr))
    return curves


def _baseline_sub(c, q=25):
    return np.clip(c - np.percentile(c, q), 0.0, None)


def _find_peaks(c, n_max, min_h=0.22, min_dist_s=4.0):
    """Top-n_max prominent peaks of a baseline-subtracted curve, returned by time."""
    from scipy.signal import find_peaks
    cc = _baseline_sub(c)
    pk, _ = find_peaks(cc, height=min_h, distance=max(1, int(min_dist_s * FPS)))
    if len(pk) == 0:
        return []
    pk = sorted(pk, key=lambda i: cc[i], reverse=True)[:n_max]
    return sorted(int(i) for i in pk)


def segment(curves, t):
    """Signature-driven: each action is detected by the PEAK of its signature curve
    (unambiguous for brow/pucker/teeth). Duplicates (2 eye closures, 2 smiles) are
    split by occurrence order. Labels come from the signature, NOT the timeline
    position — home recordings may deviate from the canonical order."""
    segs = []
    n = len(t)

    def add(action, sig, i):
        c = curves[sig]
        pkv = float(c[i])
        thr = max(0.12, 0.4 * pkv)
        lo = i
        while lo > 0 and c[lo] > thr:
            lo -= 1
        hi = i
        while hi < n - 1 and c[hi] > thr:
            hi += 1
        segs.append({"action": action, "sig": sig,
                     "t_start": round(float(t[lo]), 1), "t_peak": round(float(t[i]), 1),
                     "t_end": round(float(t[hi]), 1), "peak_val": round(pkv, 3)})

    # single-occurrence actions: one peak each
    for sig, action in (("brow", "EyebrowRise"), ("pucker", "LipPucker"),
                        ("teeth", "LowerTeethShow")):
        pk = _find_peaks(curves[sig], 1)
        if pk:
            add(action, sig, pk[0])
    # eye closures: up to 2; earlier = Gentle, later = Tight (tight also has squint)
    for k, i in enumerate(_find_peaks(curves["eye_close"], 2)):
        add(["GentleEyeClosure", "TightEyeSqueeze"][min(k, 1)], "eye_close", i)
    # smiles: up to 2; first = Relaxed, second = Reanimated
    for k, i in enumerate(_find_peaks(curves["smile"], 2)):
        add(["RelaxedSmile", "ReanimatedSmile"][min(k, 1)], "smile", i)

    segs.sort(key=lambda s: s["t_peak"])
    return segs


def plot(take, curves, t, segs):
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, 5))
    for grp, c in curves.items():
        ax.plot(t, c, lw=1.1, label=grp)
    for s in segs:
        ax.axvspan(s["t_start"], s["t_end"], color="grey", alpha=0.15)
        ax.text(s["t_peak"], 1.02, s["action"][:6], rotation=90, fontsize=7, va="bottom")
    ax.set_ylim(0, 1.1); ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("seconds"); ax.set_title(f"{take} — blendshape signatures + ordered segmentation ({len(segs)} actions)")
    fig.tight_layout(); fig.savefig(OUT / f"{take}.png", dpi=80); plt.close(fig)


def main():
    args = sys.argv[1:]
    takes = ([p.parent.name for p in sorted(LLF.glob("*/*.mov"))]
             if args == ["--timeseries-all"] else args)
    OUT.mkdir(parents=True, exist_ok=True)
    import json
    allsegs = {}
    for take in takes:
        cached = OUT / f"{take}.npz"
        if cached.exists():                       # reuse blendshape timeseries (fast iteration)
            d = np.load(cached, allow_pickle=True)
            arr, names, t, mask = d["bs"], d["names"], d["t"], d["mask"]
        else:
            r = extract_timeseries(take)
            if r is None:
                print(f"[skip] {take}", flush=True); continue
            arr, names, t, mask = r
        curves = sig_curves(arr, names)
        segs = segment(curves, t)
        plot(take, curves, t, segs)
        np.savez(OUT / f"{take}.npz", bs=arr, names=names, t=t, mask=mask, fps=FPS)
        allsegs[take] = segs
        print(f"{take}: {len(segs)} actions  {[s['action'][:5]+'@'+str(s['t_peak']) for s in segs]}", flush=True)
    (OUT / "segments.json").write_text(json.dumps(allsegs, indent=1))
    print(f"\nwrote {OUT}/segments.json ({len(allsegs)} takes)", flush=True)


if __name__ == "__main__":
    main()
