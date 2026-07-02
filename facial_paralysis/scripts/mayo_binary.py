"""Does the binary palsy DETECTOR transfer to Mayo? (uses the free label: all Mayo = 1)

Severity didn't transfer, but palsy-vs-healthy is a grosser distinction where MARLIN
appearance might survive. Score every Mayo patient's clips through the trained binary
head -> P(palsy). All 14 are positive, so we measure RECALL (flag rate) + whether the
detector SATURATES (calls everyone palsy — the no-in-domain-negatives trap, Run #4) by
comparing to web-healthy P(palsy), and whether P(palsy) tracks clinical asymmetry.
"""
from __future__ import annotations
import os, sys, json, glob
from pathlib import Path
import numpy as np, torch
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../autoresearch_fp")
import prepare_fp as P, runner as R
from mayo_generalization import train_model
ROOT = Path(__file__).resolve().parent.parent
DEVICE = "cpu"


def load_all_mayo():
    out = {}
    for d in sorted(glob.glob(str(ROOT / "outputs/mayo_action_bundles/*/"))):
        pid = os.path.basename(d.rstrip("/")).split("_", 1)[-1]
        recs = []
        for npz in glob.glob(os.path.join(d, "*.npz")):
            z = np.load(npz)
            recs.append({"marlin": z["marlin"].astype(np.float32).mean(0), "mp_seq": z["mp_seq"].astype(np.float32),
                         "mp_mask": z["mp_mask"].astype(bool), "label": 1, "task": "binary"})
        if recs:
            out[pid] = recs
    return out


def p_palsy(models, recs):
    b = P.make_batch(recs, DEVICE)
    ps = []
    for m in models:
        with torch.no_grad():
            s = m.severity(b["marlin"], b["mp_seq"], b["mp_mask"], "binary")
            th = m.thr["binary"]()[0]
            ps.append(torch.sigmoid(s - th).cpu().numpy())
    return float(np.mean(ps))


def main():
    ef = json.loads((ROOT / "outputs/mayo_eface/eface_scores.json").read_text())
    ef = {k.split("_", 1)[1]: v for k, v in ef.items()}
    mayo = load_all_mayo()
    pids = [p for p in mayo if p != "MySlate_14" and p in ef]

    CH = dict(R.DEFAULT); CH.update(json.loads((ROOT / "autoresearch_fp/best_config.json").read_text()))
    GEO = {**CH, "drop_marlin": True}
    # web-healthy reference: PalsyNet binary negatives in val
    web_neg = [r for r in P.load_data() if r["task"] == "binary" and r["label"] == 0]
    web_pos = [r for r in P.load_data() if r["task"] == "binary" and r["label"] == 1]

    for name, cfg in (("champion(MARLIN)", CH), ("geometry_only", GEO)):
        models = [train_model(cfg, s) for s in (0, 1, 2)]
        pmayo = {p: p_palsy(models, mayo[p]) for p in pids}
        vals = np.array(list(pmayo.values()))
        flagged = int((vals > 0.5).sum())
        # web references (in-domain calibration point)
        pw_neg = np.mean([p_palsy(models, [r]) for r in web_neg]) if web_neg else float("nan")
        pw_pos = np.mean([p_palsy(models, [r]) for r in web_pos]) if web_pos else float("nan")
        asym = np.array([ef[p]["eye_asym"] or np.nan for p in pids])
        oral = np.array([ef[p]["oral_asym"] or np.nan for p in pids])
        overall = np.nanmean(np.c_[asym, oral], 1)
        rho, pv = spearmanr(vals, overall)
        print(f"=== {name} ===")
        print(f"  Mayo P(palsy): mean={vals.mean():.2f} range=[{vals.min():.2f},{vals.max():.2f}]  flagged={flagged}/{len(pids)}")
        print(f"  web ref P(palsy): healthy={pw_neg:.2f}  palsy={pw_pos:.2f}   (gap tells if detector discriminates at all)")
        print(f"  P(palsy) vs clinical asymmetry on Mayo: rho={rho:+.2f} (p={pv:.2f})")
        print(f"  -> {'SATURATED (calls ~everyone palsy; no in-domain negatives to trust it)' if vals.min()>0.8 else 'not saturated'}\n")


if __name__ == "__main__":
    main()
