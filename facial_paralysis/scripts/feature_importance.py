"""Permutation feature-importance for the champion: which input groups drive severity?
Groups: MARLIN appearance (768), raw blendshapes (mp_seq[:52]), L/R asymmetry deltas
(mp_seq[52:72], which also feed the engineered features). Permute each across the val
batch and measure the region-QWK drop (bigger drop = more important).
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../autoresearch_fp")
import prepare_fp as P, runner as R
from mayo_generalization import train_model
ROOT = Path(__file__).resolve().parent.parent
CH = dict(R.DEFAULT); CH.update(json.loads((ROOT / "autoresearch_fp/best_config.json").read_text()))


def qwk(models, b, task, truth):
    preds = []
    for m in models:
        with torch.no_grad():
            s = m.severity(b["marlin"], b["mp_seq"], b["mp_mask"], task)
            preds.append(R.predict_grades(s, m.thr[task](), {"decode": "threshold"}).numpy())
    pred = np.round(np.mean(preds, 0)).astype(int)
    return P.quadratic_kappa(truth, pred, 3)


def main():
    models = [train_model(CH, s) for s in range(3)]
    rng = np.random.default_rng(0)
    for task in P.REGION_TASKS:
        recs = P.val_records(task)
        b0 = P.make_batch(recs, "cpu")
        truth = [r["label"] for r in recs]
        base = qwk(models, b0, task, truth)
        print(f"\n== {task} (champion QWK {base:.3f}) — permutation importance (QWK drop) ==")
        groups = {"MARLIN(768)": ("marlin", slice(None)),
                  "blendshapes[:52]": ("mp_seq", slice(0, 52)),
                  "asym_deltas[52:72]": ("mp_seq", slice(52, 72))}
        for gname, (key, sl) in groups.items():
            drops = []
            for _ in range(5):
                b = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b0.items()}
                perm = rng.permutation(b[key].shape[0])
                if key == "marlin":
                    b[key] = b[key][perm]
                else:
                    b[key][:, :, sl] = b[key][perm][:, :, sl]
                drops.append(base - qwk(models, b, task, truth))
            print(f"   {gname:20s} drop = {np.mean(drops):+.3f} ± {np.std(drops):.3f}")


if __name__ == "__main__":
    main()
