"""Per-source breakdown of the champion (which web dataset it handles well).
Region QWK on FNP-val vs YFP-val separately, ensemble over 3 seeds."""
from __future__ import annotations
import os, sys, json
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../autoresearch_fp")
import prepare_fp as P, runner as R
from mayo_generalization import train_model
ROOT = Path(__file__).resolve().parent.parent
CH = dict(R.DEFAULT); CH.update(json.loads((ROOT / "autoresearch_fp/best_config.json").read_text()))


def main():
    models = [train_model(CH, s) for s in range(3)]
    print("Champion 0.668 — region QWK by source (n):")
    for task in P.REGION_TASKS:
        recs = P.val_records(task)
        for src in ("fnp", "yfp"):
            sub = [r for r in recs if r["source"] == src]
            if not sub:
                continue
            b = P.make_batch(sub, "cpu")
            preds = []
            for m in models:
                with torch.no_grad():
                    s = m.severity(b["marlin"], b["mp_seq"], b["mp_mask"], task)
                    preds.append(R.predict_grades(s, m.thr[task](), {"decode": "threshold"}).numpy())
            pred = np.round(np.mean(preds, 0)).astype(int)
            q = P.quadratic_kappa([r["label"] for r in sub], pred, 3)
            print(f"   {task:6s} {src}: QWK={q:.3f}  (n={len(sub)})")


if __name__ == "__main__":
    main()
