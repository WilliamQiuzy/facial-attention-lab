"""Direction #5: measurement reliability of the label-free facial-palsy measures (no labels,
no new patients). For each measure we report:

  - split-half reliability (odd vs even frames within an action hold) = PURE measurement error,
    Spearman-Brown-corrected to full length;
  - SEM and MDC95 (minimal detectable change) in the measure's own units;
  - cross-provocation agreement (gentle vs forced eye-closure; relaxed vs reanimated smile) =
    a robustness lower bound (these probes differ physiologically, so this under-estimates
    reliability -- reported as convergence, not pure reliability).

A measure must be reliable before it can be valid. This says which of our label-free signals are
trustworthy enough for clinical / home use (feeds #6).

Runs locally on cached blendshape trajectories. arm64 or anaconda python both fine.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BS = ROOT / "outputs" / "mayo_blendshapes"
SEG = json.loads((BS / "segments.json").read_text())
DUP = "20260305_MySlate_14"                                   # == FACES018, exclude

# region L/R blendshape indices
LR = {"eye": (9, 10), "smile": (44, 45), "brow": (4, 5)}      # blink, mouthSmile, browOuterUp
# which actions probe which region (parallel provocations)
PROBES = {"eye": ["GentleEyeClosure", "TightEyeSqueeze"],
          "smile": ["RelaxedSmile", "ReanimatedSmile"], "brow": ["EyebrowRise"]}


def asym_index(L, R):
    """|L-R|/(L+R) on peak activation (90th pct); 0 = symmetric, 1 = fully one-sided."""
    lp, rp = np.percentile(L, 90), np.percentile(R, 90)
    s = lp + rp
    return abs(lp - rp) / s if s > 1e-4 else np.nan


def action_AI(bs, t, seg, li, ri, frames="all"):
    w = (t >= seg["t_start"]) & (t <= seg["t_end"])
    idx = np.where(w)[0]
    if len(idx) < 4:
        return np.nan
    if frames == "odd":
        idx = idx[1::2]
    elif frames == "even":
        idx = idx[0::2]
    return asym_index(bs[idx, li], bs[idx, ri])


def icc_agreement(x, y):
    """Two-way ICC(A,1) agreement for k=2 repeats; plus SEM and Spearman-Brown full reliability."""
    x, y = np.asarray(x), np.asarray(y)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = len(x)
    if n < 4:
        return None
    rows = np.stack([x, y], 1)
    grand = rows.mean()
    MSR = 2 * ((rows.mean(1) - grand) ** 2).sum() / (n - 1)          # between-subject
    MSC = n * ((rows.mean(0) - grand) ** 2).sum() / 1               # between-form (k-1=1)
    MSE = (((rows - rows.mean(1, keepdims=True) - rows.mean(0) + grand) ** 2).sum()) / (n - 1)
    icc = (MSR - MSE) / (MSR + MSE + 2 * (MSC - MSE) / n) if (MSR + MSE + 2 * (MSC - MSE) / n) > 0 else np.nan
    sem = np.sqrt(np.mean((x - y) ** 2) / 2)                         # within-subject SD
    mdc95 = 1.96 * np.sqrt(2) * sem
    r = float(np.corrcoef(x, y)[0, 1]) if n > 2 else np.nan
    sb = 2 * r / (1 + r) if r > -1 else np.nan                       # split-half -> full length
    return {"n": n, "icc": round(float(icc), 2), "pearson": round(r, 2),
            "spearman_brown": round(float(sb), 2), "sem": round(float(sem), 3),
            "mdc95": round(float(mdc95), 3), "mean_AI": round(float(rows.mean()), 3)}


def main():
    takes = [t for t in SEG if t != DUP]
    data = {t: np.load(BS / f"{t}.npz", allow_pickle=True) for t in takes if (BS / f"{t}.npz").exists()}

    report = {"split_half": {}, "cross_provocation": {}}
    print("=" * 74)
    print("DIRECTION #5 — reliability of label-free measures")
    print("=" * 74)

    # ---- split-half within-action (pure measurement error) ----
    print("\n[A] SPLIT-HALF reliability within an action hold (pure measurement error)")
    print(f"{'measure':22s} {'n':>3s} {'ICC':>5s} {'SplitHalf->full':>15s} {'SEM':>7s} {'MDC95':>7s}")
    for region, (li, ri) in LR.items():
        odd, even = [], []
        for t, d in data.items():
            bs, tb = d["bs"], d["t"]
            for s in SEG[t]:
                if s["action"] in PROBES[region]:
                    o = action_AI(bs, tb, s, li, ri, "odd")
                    e = action_AI(bs, tb, s, li, ri, "even")
                    if np.isfinite(o) and np.isfinite(e):
                        odd.append(o); even.append(e)
        res = icc_agreement(odd, even)
        if res:
            report["split_half"][region + "_asym"] = res
            print(f"{region+'_asym (AI)':22s} {res['n']:>3d} {res['icc']:>5.2f} "
                  f"{res['spearman_brown']:>15.2f} {res['sem']:>7.3f} {res['mdc95']:>7.3f}")

    # ---- cross-provocation agreement (robustness lower bound) ----
    print("\n[B] CROSS-PROVOCATION agreement (two different probes of the same construct)")
    print(f"{'measure':28s} {'n':>3s} {'ICC':>5s} {'Pearson':>8s} {'MDC95':>7s}")
    for region, probes in PROBES.items():
        if len(probes) < 2:
            continue
        a, b = [], []
        for t, d in data.items():
            bs, tb = d["bs"], d["t"]
            ai = {}
            for s in SEG[t]:
                if s["action"] in probes:
                    ai[s["action"]] = action_AI(bs, tb, s, *LR[region])
            if all(p in ai and np.isfinite(ai[p]) for p in probes):
                a.append(ai[probes[0]]); b.append(ai[probes[1]])
        res = icc_agreement(a, b)
        if res:
            report["cross_provocation"][f"{region}: {probes[0]} vs {probes[1]}"] = res
            print(f"{region+' '+probes[0][:4]+'/'+probes[1][:4]:28s} {res['n']:>3d} "
                  f"{res['icc']:>5.2f} {res['pearson']:>8.2f} {res['mdc95']:>7.3f}")

    # ---- pull in the independent 60fps EAR closure asym (gentle vs forced) ----
    ear = json.loads((ROOT / "outputs/mayo_ear/ear_dynamics.json").read_text())
    a, b = [], []
    for p, d in ear.items():
        if "GentleEyeClosure" in d and "TightEyeSqueeze" in d:
            a.append(d["GentleEyeClosure"]["closure_asym"]); b.append(d["TightEyeSqueeze"]["closure_asym"])
    res = icc_agreement(a, b)
    if res:
        report["cross_provocation"]["EAR closure_asym: gentle vs forced (60fps, independent)"] = res
        print(f"{'EAR closure gentle/forced':28s} {res['n']:>3d} {res['icc']:>5.2f} {res['pearson']:>8.2f} {res['mdc95']:>7.3f}")

    # ---- 3D depth asymmetry reliability (from #4 track 3) ----
    tr = json.loads((ROOT / "outputs/depth3d/asym3d_tracks.json").read_text()).get("reliability", {})
    if tr:
        report["depth_3d"] = tr
        print(f"\n[C] 3D depth asymmetry (from #4): per-frame ICC {tr.get('icc')}, "
              f"pooled full reliability {tr.get('pooled_full_reliability')}")

    (ROOT / "outputs/mayo_eface/reliability.json").write_text(json.dumps(report, indent=1))
    print("\nVERDICT (ICC>0.75 good, 0.5-0.75 moderate, <0.5 poor):")
    for m, r in report["split_half"].items():
        v = "GOOD" if r["spearman_brown"] >= 0.75 else ("MODERATE" if r["spearman_brown"] >= 0.5 else "POOR")
        print(f"  {m:18s} split-half->full {r['spearman_brown']:.2f}  MDC95={r['mdc95']:.3f} AI  -> {v}")
    print("wrote outputs/mayo_eface/reliability.json")


if __name__ == "__main__":
    main()
