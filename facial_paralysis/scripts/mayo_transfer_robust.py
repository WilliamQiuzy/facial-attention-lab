"""Robust transfer eval: geometry-only vs champion, region severity vs the CLEAN direct
geometric targets (EAR closure asym for eyes, 60fps corner asym for mouth), with
bootstrap 95% CIs over patients. Gives the definitive transfer numbers with uncertainty.
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


def boot_ci(x, y, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    rhos = []
    idx = np.arange(len(x))
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) < 2 or len(np.unique(x[b])) < 2:
            continue
        rhos.append(spearmanr(x[b], y[b])[0])
    return float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5))


def main():
    ear = json.loads((ROOT / "outputs/mayo_ear/ear_dynamics.json").read_text())
    mc = json.loads((ROOT / "outputs/mayo_ear/mouth_corner_asym.json").read_text())
    eyes_tgt = {p: ear[p]["TightEyeSqueeze"]["closure_asym"] for p in ear if "TightEyeSqueeze" in ear[p]}
    mayo = load_mayo()
    CH = dict(R.DEFAULT); CH.update(json.loads((ROOT / "autoresearch_fp/best_config.json").read_text()))
    GEO = {**CH, "drop_marlin": True}
    print("Transfer to Mayo vs CLEAN geometric targets, bootstrap 95% CI:\n")
    for name, cfg in (("champion(MARLIN)", CH), ("geometry_only", GEO)):
        models = [train_model(cfg, s) for s in (0, 1, 2)]
        sev = score_mayo(models, mayo)
        print(f"=== {name} ===")
        combined = []
        for task, tgt in (("eyes", eyes_tgt), ("mouth", mc)):
            ps = [p for p in sev if p != "MySlate_14" and task in sev[p] and p in tgt]
            s = np.array([sev[p][task] for p in ps]); a = np.array([tgt[p] for p in ps])
            rho = spearmanr(s, a)[0]; lo, hi = boot_ci(s, a)
            print(f"  {task:5s}: rho={rho:+.2f}  95%CI[{lo:+.2f},{hi:+.2f}]  n={len(ps)}")
            # z-standardize per region then pool for a combined estimate
            combined += list(zip((s - s.mean()) / (s.std() + 1e-9), (a - a.mean()) / (a.std() + 1e-9)))
        cs = np.array([c[0] for c in combined]); ca = np.array([c[1] for c in combined])
        rho = spearmanr(cs, ca)[0]; lo, hi = boot_ci(cs, ca)
        print(f"  COMBINED (pooled eyes+mouth): rho={rho:+.2f}  95%CI[{lo:+.2f},{hi:+.2f}]  n={len(cs)}\n")


if __name__ == "__main__":
    main()
