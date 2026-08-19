"""Rigorous, label-free synkinesis quantification for the Mayo FACES cohort.

Synkinesis = involuntary co-contraction (e.g. mouth moves when the patient closes the
eye). It is a core facial-palsy problem, hard for clinicians to grade consistently, and
requires NO severity label — it is a physical co-activation measurement.

Key idea vs a naive "how much did the other region move": we require the involuntary
movement to be TIME-LOCKED to the voluntary action (Pearson corr of the two per-frame
traces over the action window). True synkinesis co-moves with the intended action;
coincidental activation does not. Synkinesis score = involuntary excursion × max(corr,0).

Builds a per-patient provoking-action × responding-region matrix, aggregates the two
clinically canonical directions (ocular→oral, oral→ocular), and internally validates
(determinism on the duplicate take; consistency across the two eye-closure provocations).
Runs locally on cached blendshape trajectories.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BS = ROOT / "outputs" / "mayo_blendshapes"
SEG = json.loads((BS / "segments.json").read_text())

EYE = [9, 10, 19, 20]                     # eyeBlink L/R, eyeSquint L/R
ORAL = [44, 45, 28, 29, 38, 7, 8]         # mouthSmile/Dimple L/R, mouthPucker, cheekSquint L/R
BROW = [1, 2, 3, 4, 5]                     # browDown L/R, browInnerUp, browOuterUp L/R
TARGET_SIG = {                            # the blendshapes the action is SUPPOSED to fire
    "GentleEyeClosure": EYE, "TightEyeSqueeze": EYE,
    "RelaxedSmile": [44, 45], "ReanimatedSmile": [44, 45],
    "LipPucker": [38, 28, 29], "LowerTeethShow": [34, 35], "EyebrowRise": [4, 5],
}
# which region is the action in, so we probe the OTHER regions for involuntary movement
ACTION_REGION = {"GentleEyeClosure": "eye", "TightEyeSqueeze": "eye", "RelaxedSmile": "oral",
                 "ReanimatedSmile": "oral", "LipPucker": "oral", "LowerTeethShow": "oral",
                 "EyebrowRise": "brow"}
REGIONS = {"eye": EYE, "oral": ORAL, "brow": BROW}


def synk_for_action(bs, t, seg, rest):
    """For one action: involuntary co-activation in each non-target region, time-locked."""
    w = (t >= seg["t_start"]) & (t <= seg["t_end"])
    if w.sum() < 4:
        return {}
    target = bs[w][:, TARGET_SIG[seg["action"]]].max(1)     # the voluntary action trace
    tgt_region = ACTION_REGION[seg["action"]]
    out = {}
    for rname, idx in REGIONS.items():
        if rname == tgt_region:
            continue
        resp = bs[w][:, idx].sum(1)                          # candidate involuntary trace
        exc = float(resp.mean() - rest[rname])               # excursion above resting
        if target.std() > 1e-4 and resp.std() > 1e-4:
            corr = float(np.corrcoef(target, resp)[0, 1])    # time-locked to the action?
        else:
            corr = 0.0
        out[rname] = round(max(exc, 0.0) * max(corr, 0.0), 4)  # synkinesis score
    return out


def main():
    report = {}
    for npz in sorted(BS.glob("*.npz")):
        take = npz.stem
        if take not in SEG or not SEG[take]:
            continue
        d = np.load(npz, allow_pickle=True)
        bs, t = d["bs"], d["t"]
        rest = {r: float(np.median(bs[:, idx].sum(1))) for r, idx in REGIONS.items()}
        matrix = {}
        for s in SEG[take]:
            if s["action"] in TARGET_SIG:
                m = synk_for_action(bs, t, s, rest)
                if m:
                    matrix[s["action"]] = m
        # aggregate the two canonical directions
        ocular_oral = [matrix[a]["oral"] for a in ("GentleEyeClosure", "TightEyeSqueeze") if a in matrix and "oral" in matrix[a]]
        oral_ocular = [matrix[a]["eye"] for a in ("RelaxedSmile", "LipPucker", "LowerTeethShow", "ReanimatedSmile") if a in matrix and "eye" in matrix[a]]
        report[take.split("_", 1)[1]] = {
            "ocular_oral_synk": round(float(np.mean(ocular_oral)), 4) if ocular_oral else None,
            "oral_ocular_synk": round(float(np.mean(oral_ocular)), 4) if oral_ocular else None,
            "eye_provoke_consistency": round(float(np.std(ocular_oral)), 4) if len(ocular_oral) > 1 else None,
            "matrix": matrix,
        }
    (ROOT / "outputs/mayo_eface/synkinesis.json").write_text(json.dumps(report, indent=1))

    print(f"{'patient':12s} {'ocular->oral':>12s} {'oral->ocular':>12s}  dominant")
    print("-" * 52)
    rows = [(p, r) for p, r in report.items() if p != "MySlate_14"]
    for p, r in sorted(rows, key=lambda x: -(x[1]["ocular_oral_synk"] or 0)):
        oo, orl = r["ocular_oral_synk"], r["oral_ocular_synk"]
        dom = "ocular->oral" if (oo or 0) > (orl or 0) else ("oral->ocular" if (orl or 0) > 0 else "-")
        flag = "  <-- strong synkinesis" if (oo or 0) > 0.05 else ""
        print(f"{p:12s} {str(oo):>12s} {str(orl):>12s}  {dom}{flag}")

    # internal validation
    dup = [(report[k]["ocular_oral_synk"], report[k]["oral_ocular_synk"]) for k in ("FACES018", "MySlate_14") if k in report]
    print(f"\nVALIDATION")
    print(f"  determinism (duplicate take): {dup} {'IDENTICAL' if len(dup)==2 and dup[0]==dup[1] else ''}")
    cons = [r["eye_provoke_consistency"] for _, r in rows if r["eye_provoke_consistency"] is not None]
    print(f"  gentle-vs-forced eye consistency (lower=more consistent): mean std {np.mean(cons):.4f}")
    npos = sum(1 for _, r in rows if (r["ocular_oral_synk"] or 0) > 0.02)
    print(f"  patients with detectable ocular->oral synkinesis: {npos}/{len(rows)}")
    print(f"\nwrote outputs/mayo_eface/synkinesis.json")


if __name__ == "__main__":
    main()
