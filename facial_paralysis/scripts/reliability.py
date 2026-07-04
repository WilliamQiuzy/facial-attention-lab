"""Reliability of the label-free measurements (the trustworthy deliverable).

Without HB labels we can't measure accuracy, but we CAN measure internal reliability:
1. Cross-action side-consistency: within a patient, does the weaker side agree across
   the different actions? Consistent weaker side = a real anatomical deficit, not noise.
2. Cross-measure agreement: do the independent asymmetry measures (per-action blendshape
   vs 60fps EAR closure vs 60fps mouth-corner) point the same way?
These tell us whether the scorecard's numbers are stable enough to trust for triage.
"""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ef = json.loads((ROOT / "outputs/mayo_eface/eface_scores.json").read_text())
ear = json.loads((ROOT / "outputs/mayo_ear/ear_dynamics.json").read_text())

print("1) CROSS-ACTION SIDE-CONSISTENCY (is the weaker side stable within a patient?)")
consist = []
for take, r in ef.items():
    pid = take.split("_", 1)[1]
    sides = [a["weaker"] for a in r["actions"].values() if a.get("weaker")]
    if len(sides) < 2:
        continue
    c = Counter(sides); maj, n = c.most_common(1)[0]
    frac = n / len(sides)
    consist.append(frac)
    print(f"  {pid:12s} {len(sides)} actions, weaker={maj} on {n}/{len(sides)} ({frac:.0%})")
consist = np.array(consist)
print(f"  -> mean side-consistency {consist.mean():.2f}; fully-consistent (100%) in {int((consist==1).sum())}/{len(consist)} patients\n")

print("2) CROSS-MEASURE AGREEMENT on the eye (independent methods):")
# blendshape eye weaker (from TightEyeSqueeze action) vs EAR-closure weaker
agree = tot = 0
for take, r in ef.items():
    pid = take.split("_", 1)[1]
    bs_eye = r["actions"].get("TightEyeSqueeze", {}).get("weaker")
    ear_eye = ear.get(pid, {}).get("TightEyeSqueeze", {}).get("weaker")
    if bs_eye and ear_eye:
        tot += 1; agree += (bs_eye == ear_eye)
        flag = "OK" if bs_eye == ear_eye else "DISAGREE"
        print(f"  {pid:12s} blendshape={bs_eye:5s} EAR={ear_eye:5s}  {flag}")
print(f"  -> blendshape vs EAR weaker-eye agreement: {agree}/{tot}")
