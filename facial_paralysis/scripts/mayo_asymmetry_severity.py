"""Unsupervised, label-free per-action LEFT-RIGHT asymmetry severity for Mayo.

Facial palsy IS movement asymmetry, so the left-vs-right imbalance of each action's
signature blendshapes is a clinically-grounded severity signal that needs NO labels
and (per Run #4) is the one signal that is DOMAIN-INVARIANT — so unlike the learned
`s`, it should transfer to the Mayo domain without calibration.

Uses what we already have locally: the dense per-frame blendshape time-series
(outputs/mayo_blendshapes/<take>.npz) + the per-action segmentation (segments.json).
No mediapipe, no GPU, no labels.

Per action, at the held-expression peak:
    AI(pair) = |L - R| / (L + R + eps)         (0 = symmetric/healthy, ->1 = one side dead)
    action_asym = mean over the action's signature L/R pairs
Aggregated into eFACE-like regions (brow / eye / mouth) and an overall score, plus
the affected (weaker) side and its consistency across actions.

Run:  python3 scripts/mayo_asymmetry_severity.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BS = ROOT / "outputs" / "mayo_blendshapes"
SCORES = ROOT / "outputs" / "mayo_action_bundles" / "per_action_scores.json"
OUT = ROOT / "outputs" / "mayo_asymmetry"
EPS = 1e-3
DUP_DROP = "20260305_MySlate_14"   # identical to FACES018; drop one for stats

# action -> signature L/R pairs (LipPucker has no clean L/R pair -> excluded)
ACTION_PAIRS = {
    "EyebrowRise":      [("browOuterUpLeft", "browOuterUpRight")],
    "GentleEyeClosure": [("eyeBlinkLeft", "eyeBlinkRight")],
    "TightEyeSqueeze":  [("eyeBlinkLeft", "eyeBlinkRight"), ("eyeSquintLeft", "eyeSquintRight"),
                         ("cheekSquintLeft", "cheekSquintRight")],
    "RelaxedSmile":     [("mouthSmileLeft", "mouthSmileRight"), ("mouthDimpleLeft", "mouthDimpleRight")],
    "LowerTeethShow":   [("mouthLowerDownLeft", "mouthLowerDownRight")],
    "ReanimatedSmile":  [("mouthSmileLeft", "mouthSmileRight")],
}
REGION = {"EyebrowRise": "brow", "GentleEyeClosure": "eye", "TightEyeSqueeze": "eye",
          "RelaxedSmile": "mouth", "LowerTeethShow": "mouth", "ReanimatedSmile": "mouth"}


def action_asymmetry(bs, names_idx, t, seg, pairs):
    """AI averaged over the held-expression frames of one action window. Returns
    (asym, weaker_side, n_frames) or (None, None, 0) if the action is too weak."""
    fps = 6.0
    i0 = max(0, int(seg["t_start"] * fps))
    i1 = min(len(t) - 1, int(seg["t_end"] * fps))
    if i1 <= i0:
        return None, None, 0
    pcols = [(names_idx[l], names_idx[r]) for l, r in pairs if l in names_idx and r in names_idx]
    if not pcols:
        return None, None, 0
    # activation magnitude per frame = mean of (L+R)/2 over pairs; held frames = near peak
    act = np.mean([(bs[i0:i1 + 1, lc] + bs[i0:i1 + 1, rc]) / 2 for lc, rc in pcols], axis=0)
    if act.max() < 0.12:                       # action barely performed
        return None, None, 0
    held = act >= 0.6 * act.max()
    held &= act > 0.08
    if held.sum() == 0:
        return None, None, 0
    ais, signed = [], []
    for lc, rc in pcols:
        L = bs[i0:i1 + 1, lc][held]; R = bs[i0:i1 + 1, rc][held]
        ais.append(np.mean(np.abs(L - R) / (L + R + EPS)))
        signed.append(np.mean(R - L))          # >0 => left stronger => right weaker
    asym = float(np.mean(ais))
    weaker = "right" if np.mean(signed) > 0 else "left"
    return asym, weaker, int(held.sum())


def main():
    segments = json.loads((BS / "segments.json").read_text())
    model_s = {}
    if SCORES.exists():
        for r in json.loads(SCORES.read_text())["ranked"]:
            model_s[r["take"]] = r
    OUT.mkdir(parents=True, exist_ok=True)

    rows = []
    for take, segs in sorted(segments.items()):
        npz = BS / f"{take}.npz"
        if not npz.exists() or not segs:
            continue
        d = np.load(npz, allow_pickle=True)
        bs, names, t = d["bs"], list(d["names"]), d["t"]
        names_idx = {n: i for i, n in enumerate(names)}
        per_action, region_vals, sides = {}, {"brow": [], "eye": [], "mouth": []}, []
        for seg in segs:
            a = seg["action"]
            if a not in ACTION_PAIRS:
                continue
            asym, weaker, nf = action_asymmetry(bs, names_idx, t, seg, ACTION_PAIRS[a])
            if asym is None:
                continue
            per_action[a] = round(asym, 3)
            region_vals[REGION[a]].append(asym)
            sides.append(weaker)
        if not per_action:
            continue
        regions = {k: round(float(np.mean(v)), 3) for k, v in region_vals.items() if v}
        overall = round(float(np.mean(list(regions.values()))), 3)   # region-balanced
        weak_side = max(set(sides), key=sides.count) if sides else None
        consistency = round(sides.count(weak_side) / len(sides), 2) if sides else 0.0
        rows.append({"take": take, "asym_overall": overall, "regions": regions,
                     "per_action": per_action, "n_actions": len(per_action),
                     "weak_side": weak_side, "side_consistency": consistency,
                     "model_s": model_s.get(take, {}).get("s"),
                     "model_eyes": model_s.get(take, {}).get("eyes_sev"),
                     "model_mouth": model_s.get(take, {}).get("mouth_sev")})

    rows.sort(key=lambda r: r["asym_overall"], reverse=True)

    # determinism check on the duplicate pair
    dup = {r["take"]: r["asym_overall"] for r in rows if r["take"] in
           ("20260305_FACES018", "20260305_MySlate_14")}
    # correlation (Spearman) of asymmetry vs model s, on unique takes
    uniq = [r for r in rows if r["take"] != DUP_DROP]
    pairs_sc = [(r["asym_overall"], r["model_s"]) for r in uniq if r["model_s"] is not None]
    spear = _spearman([p[0] for p in pairs_sc], [p[1] for p in pairs_sc]) if len(pairs_sc) > 2 else None
    spear_eye = _spearman([r["regions"].get("eye", np.nan) for r in uniq],
                          [r["model_eyes"] for r in uniq])
    spear_mouth = _spearman([r["regions"].get("mouth", np.nan) for r in uniq],
                            [r["model_mouth"] for r in uniq])

    summary = {
        "n_takes": len(rows), "n_unique": len(uniq),
        "asym_overall": _spread([r["asym_overall"] for r in uniq]),
        "duplicate_check": dup,
        "spearman_asym_vs_model_s": spear,
        "spearman_eyeAsym_vs_model_eyes": spear_eye,
        "spearman_mouthAsym_vs_model_mouth": spear_mouth,
        "note": ("AI in [0,1]; 0=symmetric/healthy, higher=more asymmetric=more severe. "
                 "Label-free, domain-invariant. Caveat: head pose can inflate AI."),
    }
    print("============ Mayo unsupervised L/R asymmetry severity ============")
    print(json.dumps(summary, indent=1))
    print(f"\n{'take':<26s} {'asym':>5s} {'brow':>5s} {'eye':>5s} {'mouth':>5s} {'weak':>5s} {'cons':>4s} {'mdl_s':>6s}")
    for r in rows:
        rg = r["regions"]
        print(f"{r['take']:<26s} {r['asym_overall']:5.2f} "
              f"{rg.get('brow', float('nan')):5.2f} {rg.get('eye', float('nan')):5.2f} "
              f"{rg.get('mouth', float('nan')):5.2f} {str(r['weak_side'])[:5]:>5s} "
              f"{r['side_consistency']:4.2f} {str(r['model_s']):>6s}")
    _plot(rows)
    (OUT / "asymmetry_severity.json").write_text(json.dumps({"summary": summary, "ranked": rows}, indent=1))
    print(f"\nwrote {OUT}/asymmetry_severity.json and regions.png")


def _spread(x):
    x = np.asarray(x, float)
    return {"mean": round(float(x.mean()), 3), "std": round(float(x.std()), 3),
            "min": round(float(x.min()), 3), "max": round(float(x.max()), 3)}


def _spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 3:
        return None
    ra = np.argsort(np.argsort(a[m])); rb = np.argsort(np.argsort(b[m]))
    ra = ra - ra.mean(); rb = rb - rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    return round(float((ra * rb).sum() / denom), 3) if denom > 0 else None


def _plot(rows):
    takes = [r["take"].replace("2026", "")[:12] for r in rows]
    brow = [r["regions"].get("brow", 0) for r in rows]
    eye = [r["regions"].get("eye", 0) for r in rows]
    mouth = [r["regions"].get("mouth", 0) for r in rows]
    x = np.arange(len(rows)); w = 0.27
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x - w, brow, w, label="brow", color="tab:blue")
    ax.bar(x, eye, w, label="eye", color="tab:orange")
    ax.bar(x + w, mouth, w, label="mouth", color="tab:green")
    ax.set_xticks(x); ax.set_xticklabels(takes, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("L/R asymmetry index (0=symmetric)")
    ax.set_title("Mayo per-region movement asymmetry (label-free, sorted by overall)")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "regions.png", dpi=80); plt.close(fig)


if __name__ == "__main__":
    main()
