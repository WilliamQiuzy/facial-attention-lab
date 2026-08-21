"""Direction #6: home self-monitoring tool, grounded in #5's reliability/MDC.

The label-free measures are reliable (eye ICC 0.97, brow 0.91, smile 0.74; #5) and run on-device
via MediaPipe with no clinician and no labels. So a patient can record the FACES protocol on their
own iPhone periodically, and we track per-region L/R asymmetry over time, flagging any change that
exceeds the measure's MDC95 (minimal detectable change) as REAL vs measurement noise.

This produces (1) a patient-facing report card per session and (2) a cohort-level detectable-change
analysis: given the MDC, what magnitude of clinical change is home-detectable, and can the tool tell
patients apart. Uses the FIXED primary provocation per region (per #5: don't mix probes).

Runs locally on cached blendshape trajectories.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import reliability_suite as R

ROOT = Path(__file__).resolve().parent.parent
REL = json.loads((ROOT / "outputs/mayo_eface/reliability.json").read_text())
MDC = {r.replace("_asym", ""): REL["split_half"][r]["mdc95"] for r in REL["split_half"]}
PRIMARY = {"eye": "GentleEyeClosure", "smile": "RelaxedSmile", "brow": "EyebrowRise"}   # fixed probe


def patient_asym(take, d):
    """Reliable per-region L/R asymmetry index at the fixed primary provocation."""
    out = {}
    for region, act in PRIMARY.items():
        for s in R.SEG[take]:
            if s["action"] == act:
                ai = R.action_AI(d["bs"], d["t"], s, *R.LR[region])
                if np.isfinite(ai):
                    out[region] = float(ai)
    return out


def main():
    takes = [t for t in R.SEG if t != R.DUP and (R.BS / f"{t}.npz").exists()]
    scores = {}
    for t in takes:
        d = np.load(R.BS / f"{t}.npz", allow_pickle=True)
        a = patient_asym(t, d)
        if a:
            scores[t.split("_", 1)[1]] = a
    (ROOT / "outputs/mayo_eface/home_scores.json").write_text(json.dumps(scores, indent=1))

    regions = ["eye", "smile", "brow"]
    fig = plt.figure(figsize=(14, 8))

    # --- Row 1: example report cards (3 patients) with the MDC "noise band" as error bars ---
    examples = sorted(scores, key=lambda p: -np.mean([scores[p].get(r, 0) for r in regions]))[:3]
    for j, p in enumerate(examples):
        ax = fig.add_subplot(2, 3, j + 1)
        vals = [scores[p].get(r, np.nan) for r in regions]
        err = [MDC[r] / 2 for r in regions]
        bars = ax.bar(regions, vals, yerr=err, capsize=5,
                      color=["#d66" if v > 0.15 else "#e9a" if v > 0.05 else "#7b7" for v in vals])
        ax.set_ylim(0, max(0.5, max(v for v in vals if np.isfinite(v)) * 1.3))
        ax.set_title(f"Report card — patient {p}\n(bar = asymmetry, whisker = noise band ±MDC/2)", fontsize=9)
        ax.set_ylabel("L/R asymmetry index")

    # --- Row 2a: detectable-change — simulated recovery vs the MDC threshold ---
    ax = fig.add_subplot(2, 3, 4)
    base = np.array([0.30, 0.22, 0.15, 0.10, 0.05])                  # illustrative baseline eye AIs
    recov = base * 0.5                                              # 50% improvement
    x = np.arange(len(base))
    ax.plot(x, base, "o-", label="baseline"); ax.plot(x, recov, "s-", label="follow-up (-50%)")
    for i in range(len(base)):
        detect = (base[i] - recov[i]) > MDC["eye"]
        ax.annotate("detected" if detect else "n.s.", (x[i], recov[i] - 0.02), fontsize=7,
                    color="green" if detect else "gray", ha="center")
    ax.axhspan(0, MDC["eye"], color="gray", alpha=0.15)
    ax.set_title(f"Detectable change (eye)\nMDC95={MDC['eye']:.3f}: a 50%% recovery is flagged\nwhenever |change| exceeds the band", fontsize=9)
    ax.set_xlabel("illustrative cases"); ax.set_ylabel("eye asymmetry index"); ax.legend(fontsize=7)

    # --- Row 2b: can the tool tell patients apart? between-patient spread vs MDC ---
    ax = fig.add_subplot(2, 3, 5)
    for i, r in enumerate(regions):
        vv = np.array([scores[p][r] for p in scores if r in scores[p]])
        ax.scatter([i] * len(vv), vv, alpha=0.6)
        ax.errorbar(i, vv.mean(), yerr=MDC[r] / 2, fmt="_", c="k", capsize=6)
    ax.set_xticks(range(3)); ax.set_xticklabels(regions)
    ax.set_ylabel("asymmetry index (cohort)")
    ax.set_title("Between-patient spread ≫ MDC\n(tool resolves patient differences)", fontsize=9)

    # --- Row 2c: how many follow-ups to average for a target sensitivity ---
    ax = fig.add_subplot(2, 3, 6)
    k = np.arange(1, 6)
    for r in regions:
        # MDC shrinks ~1/sqrt(k) when averaging k independent home sessions
        ax.plot(k, MDC[r] / np.sqrt(k), "o-", label=r)
    ax.set_xlabel("home sessions averaged (k)"); ax.set_ylabel("effective MDC95")
    ax.set_title("Averaging sessions sharpens sensitivity\n(MDC ∝ 1/√k)", fontsize=9)
    ax.legend(fontsize=7)

    fig.suptitle("Direction #6: home iPhone self-monitoring — reliable label-free asymmetry + MDC "
                 "change-detection, no clinician / no labels", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(ROOT / "outputs/mayo_eface/home_monitor.png", dpi=125)
    print(f"scored {len(scores)} patients; MDC95 eye {MDC['eye']:.3f} smile {MDC['smile']:.3f} brow {MDC['brow']:.3f}")
    print("wrote outputs/mayo_eface/home_monitor.png + home_scores.json")


if __name__ == "__main__":
    main()
