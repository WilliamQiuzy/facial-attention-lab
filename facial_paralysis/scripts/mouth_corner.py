"""60fps mouth-corner excursion asymmetry — a cleaner geometric mouth-asymmetry target
(the mouth analog of EAR for eyes). Tests whether mouth failed to transfer because the
blendshape oral_asym target was noisy.

corner_{left,right} = mouth-corner vertical position / IOD (from ear_clips.py). A smile
pulls the corner UP (smaller y). Per side excursion = rest_y - peak_y; asymmetry =
|exc_L - exc_R| / (exc_L + exc_R). Aggregated over the smile actions per patient.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
E = json.loads((ROOT / "outputs/mayo_ear/ear.json").read_text())
MAN = json.loads((ROOT / "outputs/mayo_ear/clip_manifest.json").read_text())
SMILE = ("RelaxedSmile", "ReanimatedSmile")


def excursion_asym(tr):
    t = np.array(tr["t"])
    if len(t) < 5:
        return None
    out = {}
    for side in ("left", "right"):
        y = np.array(tr[f"corner_{side}"])
        rest = np.median(y[:max(3, len(y) // 6)])      # rest = early frames
        peak = np.percentile(y, 5)                      # highest pull (smallest y)
        out[side] = max(rest - peak, 0.0)
    ex = out
    tot = ex["left"] + ex["right"]
    if tot < 1e-4:
        return None
    return {"exc_left": round(ex["left"], 4), "exc_right": round(ex["right"], 4),
            "mouth_corner_asym": round(abs(ex["left"] - ex["right"]) / (tot + 1e-6), 3),
            "weaker": "left" if ex["left"] < ex["right"] else "right"}


def main():
    by_pat = {}
    for clip, tr in E.items():
        m = MAN.get(clip)
        if not m or m["action"] not in SMILE:
            continue
        a = excursion_asym(tr)
        if a:
            by_pat.setdefault(m["take"].split("_", 1)[1], []).append(a["mouth_corner_asym"])
    out = {p: round(float(np.mean(v)), 3) for p, v in by_pat.items()}
    (ROOT / "outputs/mayo_ear/mouth_corner_asym.json").write_text(json.dumps(out, indent=1))
    print(f"mouth-corner asymmetry (60fps), {len(out)} patients:")
    for p, v in sorted(out.items(), key=lambda x: -x[1]):
        print(f"  {p:12s} {v}")


if __name__ == "__main__":
    main()
