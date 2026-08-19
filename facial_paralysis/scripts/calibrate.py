"""Post-hoc temperature scaling of the champion's ordinal severity logits.
Error analysis found the EYES head badly miscalibrated (ECE 0.29). Temperature scaling
is the standard no-new-data fix: fit a scalar T on a held-out calibration split, test ECE
on the remaining split. Honest: fit and test on DISJOINT val subsets.
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


def sev_and_theta(models, recs, task):
    b = P.make_batch(recs, "cpu")
    ss, th = [], None
    for m in models:
        with torch.no_grad():
            ss.append(m.severity(b["marlin"], b["mp_seq"], b["mp_mask"], task).numpy())
            th = m.thr[task]().detach().numpy()
    return np.mean(ss, 0), th   # ensemble severity, thresholds


def ece_py0(s, th, y, T=1.0):
    p = 1 / (1 + np.exp(-(s - th[0]) / T))     # P(y>0)
    t = (y > 0).astype(int)
    bins = np.linspace(0, 1, 6); e = 0.0
    for i in range(5):
        m = (p >= bins[i]) & (p < bins[i + 1] if i < 4 else p <= 1.0)
        if m.sum():
            e += m.sum() / len(p) * abs(p[m].mean() - t[m].mean())
    return e


def main():
    models = [train_model(CH, s) for s in range(3)]
    rng = np.random.default_rng(0)
    for task in P.REGION_TASKS:
        recs = P.val_records(task)
        s, th = sev_and_theta(models, recs, task)
        y = np.array([r["label"] for r in recs])
        idx = rng.permutation(len(y)); half = len(y) // 2
        cal, test = idx[:half], idx[half:]
        # grid-search T on calibration split (minimize ECE of P(y>0))
        Ts = np.linspace(0.3, 3.0, 28)
        best_T = min(Ts, key=lambda T: ece_py0(s[cal], th, y[cal], T))
        e0 = ece_py0(s[test], th, y[test], 1.0)
        e1 = ece_py0(s[test], th, y[test], best_T)
        print(f"{task:6s}: fitted T={best_T:.2f}  test ECE(P>0) {e0:.3f} -> {e1:.3f}  "
              f"({'improved' if e1 < e0 else 'no gain'})")


if __name__ == "__main__":
    main()
