"""Script-driven per-action facial-symmetry analysis of the Mayo FACES videos.

Uses the FACES protocol (IRB 24-004956, Dr. Menapace) — 8 held actions per patient —
to turn the cached dense blendshape trajectories (`outputs/mayo_blendshapes/*.npz`,
52 ARKit blendshapes @6fps) + the per-action segments (`segments.json`) into
automated, LABEL-FREE regional symmetry + synkinesis indices aligned to the
eFACE / Sunnybrook voluntary-movement battery.

Why this needs the script: the protocol names each action (gentle vs FORCED eye
closure, smile, pucker, lower-teeth) so each clip's clinical meaning + the correct
L/R signature blendshape pair is known. The key payoff vs. web stills: we measure
asymmetry ACROSS the 3-second hold (a trajectory), the gentle-vs-forced contrast,
and synkinesis — none of which exist in a single still image.

No labels are used; outputs are physical-symmetry measures (objective), not HB grades.
Runs fully locally on cached blendshapes (no mediapipe / no GPU).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BS_DIR = ROOT / "outputs" / "mayo_blendshapes"
SEG = BS_DIR / "segments.json"
OUT = ROOT / "outputs" / "mayo_eface"
OUT.mkdir(parents=True, exist_ok=True)
EPS = 1e-6

# blendshape index map (from the 52 ARKit names, verified)
IDX = {
    "browOuterUpLeft": 4, "browOuterUpRight": 5,
    "eyeBlinkLeft": 9, "eyeBlinkRight": 10, "eyeSquintLeft": 19, "eyeSquintRight": 20,
    "mouthSmileLeft": 44, "mouthSmileRight": 45,
    "mouthDimpleLeft": 28, "mouthDimpleRight": 29,
    "mouthLowerDownLeft": 34, "mouthLowerDownRight": 35,
    "mouthPucker": 38, "mouthFrownLeft": 30, "mouthFrownRight": 31,
}
# action -> (region, left_idx, right_idx) signature pair
ACTION_SIG = {
    "EyebrowRise":      ("brow", IDX["browOuterUpLeft"], IDX["browOuterUpRight"]),
    "GentleEyeClosure": ("eye",  IDX["eyeBlinkLeft"],    IDX["eyeBlinkRight"]),
    "TightEyeSqueeze":  ("eye",  IDX["eyeSquintLeft"],   IDX["eyeSquintRight"]),
    "RelaxedSmile":     ("oral", IDX["mouthSmileLeft"],  IDX["mouthSmileRight"]),
    "ReanimatedSmile":  ("oral", IDX["mouthSmileLeft"],  IDX["mouthSmileRight"]),
    "LipPucker":        ("oral", IDX["mouthDimpleLeft"], IDX["mouthDimpleRight"]),
    "LowerTeethShow":   ("oral", IDX["mouthLowerDownLeft"], IDX["mouthLowerDownRight"]),
}
MOUTH_SET = [44, 45, 28, 29, 38, 30, 31]     # smile/dimple/pucker/frown → oral synkinesis probe
EYE_SET = [9, 10, 19, 20]                     # blink/squint → ocular synkinesis probe


def asym(l: float, r: float) -> float:
    """|L-R|/(L+R): 0 = symmetric, 1 = one side fully absent."""
    return abs(l - r) / (l + r + EPS)


def action_metrics(bs: np.ndarray, t: np.ndarray, seg: dict) -> dict:
    """Per-action peak + trajectory asymmetry for one segmented action."""
    li, ri = ACTION_SIG[seg["action"]][1], ACTION_SIG[seg["action"]][2]
    w = (t >= seg["t_start"]) & (t <= seg["t_end"])
    if w.sum() < 2:
        return {}
    L, R = bs[w, li], bs[w, ri]
    peakL, peakR = float(L.max()), float(R.max())
    active = (L + R) > 0.2 * (peakL + peakR + EPS)          # frames where the action is engaged
    ai_traj = float(np.mean([asym(a, b) for a, b in zip(L[active], R[active])])) if active.any() else np.nan
    return {
        "region": ACTION_SIG[seg["action"]][0],
        "ai_peak": round(asym(peakL, peakR), 3),            # what a single best still gives
        "ai_traj": round(ai_traj, 3),                       # asymmetry sustained over the hold (dynamic)
        "peakL": round(peakL, 3), "peakR": round(peakR, 3),
        "excursion": round(0.5 * (peakL + peakR), 3),       # movement magnitude
        "weaker": "left" if peakL < peakR else "right",
        "n_frames": int(w.sum()),
    }


def synkinesis(bs: np.ndarray, t: np.ndarray, segs: list[dict]) -> dict:
    """Cross-region involuntary co-activation (post-paralytic synkinesis)."""
    rest_mouth = float(np.median(bs[:, MOUTH_SET].sum(1)))
    rest_eye = float(np.median(bs[:, EYE_SET].sum(1)))
    out = {}
    for s in segs:
        w = (t >= s["t_start"]) & (t <= s["t_end"])
        if w.sum() < 2:
            continue
        if s["action"] == "TightEyeSqueeze":                # mouth moving while squeezing eyes
            out["oral_ocular_synk"] = round(float(bs[w][:, MOUTH_SET].sum(1).mean()) - rest_mouth, 3)
        if s["action"] == "RelaxedSmile":                   # eye narrowing while smiling
            out["ocular_oral_synk"] = round(float(bs[w][:, EYE_SET].sum(1).mean()) - rest_eye, 3)
    return out


def eye_dynamics(actions: dict) -> dict:
    """Gentle-vs-forced eye-closure contrast — a purely dynamic/effort measure.
    Does forcing recruit residual closure in the weak eye? (prognostic)."""
    g, f = actions.get("GentleEyeClosure"), actions.get("TightEyeSqueeze")
    if not g or not f:
        return {}
    weak_g = min(g["peakL"], g["peakR"])
    weak_f = min(f["peakL"], f["peakR"])
    return {
        "weak_eye_gentle": round(weak_g, 3),
        "weak_eye_forced": round(weak_f, 3),
        "forced_recruitment": round(weak_f - weak_g, 3),    # >0: effort recruits residual function
        "ai_peak_vs_traj_gap": round(abs(f["ai_peak"] - f["ai_traj"]), 3),  # static-vs-dynamic divergence
    }


def main():
    segments = json.loads(SEG.read_text())
    report = {}
    for npz in sorted(BS_DIR.glob("*.npz")):
        take = npz.stem
        if take not in segments or not segments[take]:
            continue
        d = np.load(npz, allow_pickle=True)
        bs, t = d["bs"], d["t"]
        acts = {}
        for s in segments[take]:
            if s["action"] in ACTION_SIG:
                m = action_metrics(bs, t, s)
                if m:
                    acts[s["action"]] = m
        # regional aggregates (peak asymmetry, label-free)
        def region_ai(names):
            vs = [acts[a]["ai_peak"] for a in names if a in acts]
            return round(float(np.mean(vs)), 3) if vs else None
        report[take] = {
            "brow_asym": region_ai(["EyebrowRise"]),
            "eye_asym": region_ai(["GentleEyeClosure", "TightEyeSqueeze"]),
            "oral_asym": region_ai(["RelaxedSmile", "LipPucker", "LowerTeethShow", "ReanimatedSmile"]),
            "synkinesis": synkinesis(bs, t, segments[take]),
            "eye_dynamics": eye_dynamics(acts),
            "actions": acts,
        }
    (OUT / "eface_scores.json").write_text(json.dumps(report, indent=1))

    # cohort table
    print(f"{'take':24s} {'brow':>5} {'eye':>5} {'oral':>5} {'weakEye':>8} "
          f"{'forcedRecr':>10} {'oralOcSynk':>10}")
    print("-" * 78)
    for tk, r in report.items():
        ed = r["eye_dynamics"]
        sk = r["synkinesis"]
        eye_act = r["actions"].get("TightEyeSqueeze", {})
        print(f"{tk:24s} {str(r['brow_asym']):>5} {str(r['eye_asym']):>5} {str(r['oral_asym']):>5} "
              f"{eye_act.get('weaker','-'):>8} {str(ed.get('forced_recruitment','-')):>10} "
              f"{str(sk.get('oral_ocular_synk','-')):>10}")
    print(f"\nwrote {OUT/'eface_scores.json'}  ({len(report)} patients)")


if __name__ == "__main__":
    main()
