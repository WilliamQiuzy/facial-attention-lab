"""Does mouth transfer to Mayo when measured against the CLEANER 60fps mouth-corner
asymmetry target (vs the noisy blendshape oral_asym)? Re-runs the transfer test for
both targets, geometry-only vs champion.
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../autoresearch_fp")
import runner as R
from mayo_generalization import train_model, load_mayo, score_mayo
ROOT = Path(__file__).resolve().parent.parent


def main():
    ef = json.loads((ROOT / "outputs/mayo_eface/eface_scores.json").read_text())
    ef = {k.split("_", 1)[1]: v for k, v in ef.items()}
    mc = json.loads((ROOT / "outputs/mayo_ear/mouth_corner_asym.json").read_text())
    mayo = load_mayo()
    CH = dict(R.DEFAULT); CH.update(json.loads((ROOT / "autoresearch_fp/best_config.json").read_text()))
    GEO = {**CH, "drop_marlin": True}
    print("Mouth transfer vs two targets (Spearman severity vs asymmetry):\n")
    for name, cfg in (("champion", CH), ("geometry_only", GEO)):
        models = [train_model(cfg, s) for s in (0, 1, 2)]
        sev = score_mayo(models, mayo)
        for tname, tgt in (("blendshape oral_asym", ef), ("60fps mouth_corner", mc)):
            ps = [p for p in sev if p != "MySlate_14" and "mouth" in sev[p]
                  and p in (tgt if tname.startswith("60") else ef)
                  and (mc.get(p) is not None if tname.startswith("60") else ef[p].get("oral_asym") is not None)]
            s = np.array([sev[p]["mouth"] for p in ps])
            a = np.array([mc[p] if tname.startswith("60") else ef[p]["oral_asym"] for p in ps])
            rho, pv = spearmanr(s, a)
            print(f"  {name:14s} vs {tname:22s}: rho={rho:+.2f} p={pv:.2f} (n={len(ps)})")
        print()


if __name__ == "__main__":
    main()
