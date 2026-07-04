"""5-seed verification of the broad-sweep candidates that edged above champion (0.649).
Confirms whether the +0.012 is real or noise, and tests stacking the two winners."""
from __future__ import annotations
import os, sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../autoresearch_fp")
import prepare_fp as P, runner as R
ROOT = Path(__file__).resolve().parent.parent
CH = dict(R.DEFAULT); CH.update(json.loads((ROOT / "autoresearch_fp/best_config.json").read_text()))

configs = {
    "champion": CH,
    "mproj256_deep": {**CH, "pr_marlin": {"eyes": 256, "mouth": 768, "binary": 256}, "trunk_layers": 2},
    "bs64": {**CH, "batch_size": 64},
    "bs64_mproj256_deep": {**CH, "batch_size": 64, "pr_marlin": {"eyes": 256, "mouth": 768, "binary": 256}, "trunk_layers": 2},
}
truth = {t: [r["label"] for r in P.val_records(t)] for t in P.REGION_TASKS}
print("5-seed verification (mean region QWK ± sd); champion ref 0.649\n")
for name, cfg in configs.items():
    ms = []
    for s in range(5):
        preds = R.train_one_seed(s, cfg)
        m = 0.5 * (P.quadratic_kappa(truth["eyes"], preds["eyes"], 3)
                   + P.quadratic_kappa(truth["mouth"], preds["mouth"], 3))
        ms.append(m)
    print(f"  {name:22s} {np.mean(ms):.4f} ± {np.std(ms):.4f}   (seeds: {'/'.join(f'{m:.3f}' for m in ms)})")
