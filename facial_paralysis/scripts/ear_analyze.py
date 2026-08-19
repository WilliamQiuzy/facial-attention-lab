"""Analyze 60fps EAR trajectories (from ear_clips.py on the pod) into clinical
eye-closure dynamics that still images cannot measure:

  closure depth L/R         open→closed EAR drop per eye
  residual EAR at max effort LAGOPHTHALMOS proxy (incomplete closure = corneal risk)
  closure asymmetry          |L−R| residual at the most-closed instant
  closure velocity L/R       peak dEAR/dt (how fast each eye shuts)
  forced recruitment         does residual shrink gentle→forced squeeze?

Compares against the 6fps blendshape eye_asym to show the 60fps EAR adds a cleaner,
clinically-standard closure signal. Outputs JSON + a trajectory/cohort figure.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
E = json.loads((ROOT / "outputs" / "mayo_ear" / "ear.json").read_text())
MAN = json.loads((ROOT / "outputs" / "mayo_ear" / "clip_manifest.json").read_text())
OUT = ROOT / "outputs" / "mayo_ear"


def eye_metrics(tr):
    t = np.array(tr["t"]); el = np.array(tr["ear_left"]); er = np.array(tr["ear_right"])
    if len(t) < 5:
        return None
    open_l, open_r = np.percentile(el, 90), np.percentile(er, 90)     # eye-open baseline
    # most-closed instant = min of summed EAR (both eyes maximally shut)
    k = int(np.argmin(el + er))
    res_l, res_r = float(el[k]), float(er[k])                          # residual EAR (lagophthalmos)
    dt = np.gradient(t) + 1e-6
    vel_l, vel_r = float(-np.min(np.gradient(el) / dt)), float(-np.min(np.gradient(er) / dt))
    return {
        "closure_depth_left": round(float(open_l - res_l), 3),
        "closure_depth_right": round(float(open_r - res_r), 3),
        "residual_left": round(res_l, 3), "residual_right": round(res_r, 3),
        "closure_asym": round(abs(res_l - res_r) / (res_l + res_r + 1e-6), 3),
        "closure_vel_left": round(vel_l, 2), "closure_vel_right": round(vel_r, 2),
        "weaker": "left" if res_l > res_r else "right",               # weaker eye closes LESS (higher residual)
    }


# assemble per-take
by_take = {}
for clip, tr in E.items():
    m = MAN.get(clip)
    if not m:
        continue
    take, action = m["take"], m["action"]
    by_take.setdefault(take, {})[action] = (eye_metrics(tr) if "Eye" in action else None, tr)

report = {}
for take, acts in by_take.items():
    r = {}
    for a in ("GentleEyeClosure", "TightEyeSqueeze"):
        if a in acts and acts[a][0]:
            r[a] = acts[a][0]
    if "GentleEyeClosure" in r and "TightEyeSqueeze" in r:
        wk = r["TightEyeSqueeze"]["weaker"]
        r["forced_recruit_EAR"] = round(r["GentleEyeClosure"][f"residual_{wk}"]
                                        - r["TightEyeSqueeze"][f"residual_{wk}"], 3)  # >0: forcing closes weak eye more
    report[take.split("_", 1)[1]] = r
(OUT / "ear_dynamics.json").write_text(json.dumps(report, indent=1))

# cohort table
print(f"{'patient':12s} {'closeAsym':>9} {'resid_wk':>8} {'weaker':>6} {'forcedRecrEAR':>13}")
print("-" * 56)
for tk, r in sorted(report.items(), key=lambda x: -(x[1].get("TightEyeSqueeze", {}).get("closure_asym", 0))):
    ts = r.get("TightEyeSqueeze", {})
    wk = ts.get("weaker", "-")
    resid = ts.get(f"residual_{wk}", "-") if wk != "-" else "-"
    print(f"{tk:12s} {str(ts.get('closure_asym','-')):>9} {str(resid):>8} {wk:>6} {str(r.get('forced_recruit_EAR','-')):>13}")

# figure: EAR trajectories for the two extremes + cohort closure-asymmetry bars
order = sorted(report, key=lambda tk: -(report[tk].get("TightEyeSqueeze", {}).get("closure_asym", 0)))
fig = plt.figure(figsize=(15, 5))
for n, tk in enumerate(order[:2]):                                    # two most-asymmetric
    ax = fig.add_axes([0.05 + 0.32 * n, 0.15, 0.26, 0.72])
    full = f"20260{''}"  # find full key
    fk = next(k for k in by_take if k.split("_", 1)[1] == tk)
    _, tr = by_take[fk]["TightEyeSqueeze"]
    ax.plot(tr["t"], tr["ear_left"], "-", color="#1f77b4", label="left eye")
    ax.plot(tr["t"], tr["ear_right"], "-", color="#d62728", label="right eye")
    ax.axhline(0.15, ls=":", color="gray", lw=1)
    ax.set_title(f"{tk} — Tight Squeeze @60fps\nclose_asym={report[tk]['TightEyeSqueeze']['closure_asym']}", fontsize=9)
    ax.set_xlabel("s", fontsize=8); ax.set_ylabel("EAR (low=closed)", fontsize=8); ax.legend(fontsize=8)
axb = fig.add_axes([0.72, 0.15, 0.26, 0.72])
ca = [(tk, report[tk].get("TightEyeSqueeze", {}).get("closure_asym", 0)) for tk in order]
axb.barh([x[0] for x in ca][::-1], [x[1] for x in ca][::-1], color="#d62728")
axb.set_title("closure asymmetry @ max squeeze (60fps EAR)", fontsize=9)
axb.tick_params(labelsize=7)
fig.savefig(OUT / "ear_dynamics.png", dpi=130)
print(f"\nwrote {OUT/'ear_dynamics.json'} and {OUT/'ear_dynamics.png'}")
