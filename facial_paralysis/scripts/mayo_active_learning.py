"""B-3: active-learning priority list for HB labeling.

We have two independent severity signals on Mayo that DISAGREE (Run #14):
  - model `s` (learned warm-start; appearance-driven, domain-confounded on Mayo)
  - L/R asymmetry severity (label-free, clinically grounded, domain-invariant)
The most INFORMATIVE takes to HB-label first are the ones where these two disagree
most — labeling them maximally resolves which signal to trust and best calibrates
the future HB head. Pure local analysis of the two existing JSONs.

Run:  python3 scripts/mayo_active_learning.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ASYM = ROOT / "outputs" / "mayo_asymmetry" / "asymmetry_severity.json"
SCORES = ROOT / "outputs" / "mayo_action_bundles" / "per_action_scores.json"
OUT = ROOT / "outputs" / "mayo_active_learning.json"
DUP_DROP = "20260305_MySlate_14"


def main():
    asym = {r["take"]: r for r in json.loads(ASYM.read_text())["ranked"]}
    model = {r["take"]: r for r in json.loads(SCORES.read_text())["ranked"]}
    takes = [t for t in asym if t in model and t != DUP_DROP]

    a = np.array([asym[t]["asym_overall"] for t in takes])
    s = np.array([model[t]["s"] for t in takes])
    # rank (0 = least severe) for each signal, then disagreement = |rank diff| (normalized)
    ra = np.argsort(np.argsort(a))
    rs = np.argsort(np.argsort(s))
    n = len(takes)
    disagree = np.abs(ra - rs) / (n - 1)

    rows = []
    for i, t in enumerate(takes):
        rows.append({"take": t, "asym": round(float(a[i]), 3), "asym_rank": int(ra[i]),
                     "model_s": round(float(s[i]), 3), "model_s_rank": int(rs[i]),
                     "disagreement": round(float(disagree[i]), 2),
                     "weak_side": asym[t]["weak_side"],
                     "side_consistency": asym[t]["side_consistency"]})
    rows.sort(key=lambda r: r["disagreement"], reverse=True)

    print("=========== HB-labeling priority (label highest-disagreement first) ===========")
    print(f"{'take':<26s} {'disagree':>8s} {'asym':>5s}(rk) {'model_s':>7s}(rk) {'weak':>5s} {'cons':>4s}")
    for r in rows:
        print(f"{r['take']:<26s} {r['disagreement']:8.2f} {r['asym']:5.2f}({r['asym_rank']:2d}) "
              f"{r['model_s']:7.2f}({r['model_s_rank']:2d}) {str(r['weak_side'])[:5]:>5s} {r['side_consistency']:4.2f}")
    print("\nInterpretation: high disagreement = the appearance-learned model and the "
          "clinical-asymmetry signal conflict most → these takes are the most informative "
          "to HB-label first (resolve the conflict + calibrate the HB head).")
    OUT.write_text(json.dumps({"priority": rows}, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
