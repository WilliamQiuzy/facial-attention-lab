"""#3: Does feature-noise augmentation improve cross-dataset generalization?
Trains champion +/- marlin/geo noise on one web source, tests region QWK on the other.
Also reports the within-mix val QWK so we see if augmentation trades in-domain for transfer.
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../autoresearch_fp")
import prepare_fp as P, runner as R
from cross_dataset import train_on, qwk_on   # reuse
ROOT = Path(__file__).resolve().parent.parent


def main():
    data = P.load_data()
    by_src = {s: [r for r in data if r["source"] == s and r["task"] in ("eyes", "mouth")] for s in ("fnp", "yfp")}
    CH = dict(R.DEFAULT); CH.update(json.loads((ROOT / "autoresearch_fp/best_config.json").read_text()))
    variants = {
        "champion":            dict(CH),
        "+marlin_noise0.2":    {**CH, "marlin_noise": 0.2},
        "+marlin_noise0.5":    {**CH, "marlin_noise": 0.5},
        "+mn0.3+geo0.05+wd0.08": {**CH, "marlin_noise": 0.3, "geo_noise": 0.05, "weight_decay": 0.08, "dropout": 0.3},
    }
    print("Cross-dataset generalization with augmentation (region QWK, mean of 2 seeds)\n")
    print(f"{'variant':24s} {'FNP->YFP eyes/mouth':>22s} {'YFP->FNP eyes/mouth':>22s}")
    for name, cfg in variants.items():
        row = []
        for tr, te in (("fnp", "yfp"), ("yfp", "fnp")):
            models = [train_on(by_src[tr], cfg, s) for s in (0, 1)]
            row.append((qwk_on(models, by_src[te], "eyes"), qwk_on(models, by_src[te], "mouth")))
        print(f"{name:24s} {row[0][0]:.3f}/{row[0][1]:.3f}{'':>12s} {row[1][0]:.3f}/{row[1][1]:.3f}")


if __name__ == "__main__":
    main()
