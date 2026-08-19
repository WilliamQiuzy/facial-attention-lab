"""Error analysis + calibration of the champion (0.668) on the web val set.
Confusion matrices, per-class recall/precision, and an ordinal-probability reliability
diagram + expected calibration error (ECE). Standard rigor for the paper.
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../autoresearch_fp")
import prepare_fp as P, runner as R
from mayo_generalization import train_model
ROOT = Path(__file__).resolve().parent.parent
CH = dict(R.DEFAULT); CH.update(json.loads((ROOT / "autoresearch_fp/best_config.json").read_text()))


def cum_probs(model, recs, task):
    b = P.make_batch(recs, "cpu")
    with torch.no_grad():
        s = model.severity(b["marlin"], b["mp_seq"], b["mp_mask"], task)
        th = model.thr[task]()
        cum = torch.sigmoid(s.unsqueeze(1) - th.unsqueeze(0))  # P(y>k), (N, K-1)
    return cum.numpy()


def main():
    models = [train_model(CH, s) for s in range(3)]
    for task in P.REGION_TASKS:
        recs = P.val_records(task)
        y = np.array([r["label"] for r in recs])
        # ensemble cumulative probs -> grade
        cps = np.mean([cum_probs(m, recs, task) for m in models], 0)   # (N,2)
        pred = (cps > 0.5).sum(1)
        print(f"\n===== {task.upper()} (n={len(y)}), champion 0.668 =====")
        print("confusion (rows=true 0/1/2):")
        for row in confusion_matrix(y, pred, labels=[0, 1, 2]):
            print("   ", row)
        print(classification_report(y, pred, labels=[0, 1, 2],
                                     target_names=["Normal", "Slight", "Strong"], zero_division=0, digits=2))
        # calibration of P(y>0) = has-any-deficit
        p = cps[:, 0]; t = (y > 0).astype(int)
        bins = np.linspace(0, 1, 6); ece = 0.0
        print("reliability of P(y>0):  bin   n   mean_pred  emp_freq")
        for i in range(5):
            m = (p >= bins[i]) & (p < bins[i + 1] if i < 4 else p <= 1.0)
            if m.sum():
                mp, ef = p[m].mean(), t[m].mean()
                ece += m.sum() / len(p) * abs(mp - ef)
                print(f"   [{bins[i]:.1f},{bins[i+1]:.1f})  {m.sum():3d}   {mp:.2f}      {ef:.2f}")
        print(f"   ECE (P>0): {ece:.3f}")


if __name__ == "__main__":
    main()
