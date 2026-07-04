"""autoresearch with a GENERALIZATION objective.

Instead of within-split region QWK (which the champion already maxed but which does NOT
transfer), the metric here is CROSS-DATASET generalization: train on one web source,
test on the other, average both directions + both regions. Non-circular, decent n, and
it directly rewards models that generalize (the property we actually need for Mayo).

Winners are then validated on Mayo transfer separately. Usage:
  python xsearch.py <batch.json> <out.tsv>
"""
from __future__ import annotations
import os, sys, json, time
from pathlib import Path
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../scripts")
import prepare_fp as P, runner as R
from cross_dataset import train_on, qwk_on


def xmetric(cfg, seeds=(0, 1)):
    data = P.load_data()
    by = {s: [r for r in data if r["source"] == s and r["task"] in ("eyes", "mouth")] for s in ("fnp", "yfp")}
    cells = []
    for tr, te in (("fnp", "yfp"), ("yfp", "fnp")):
        models = [train_on(by[tr], cfg, s) for s in seeds]
        for task in ("eyes", "mouth"):
            q = qwk_on(models, by[te], task)
            if q is not None:
                cells.append(q)
    return float(np.mean(cells)), cells


def main():
    batch = json.loads(Path(sys.argv[1]).read_text())
    out = Path(sys.argv[2])
    exps = batch["experiments"] if "experiments" in batch else batch
    for e in exps:
        name = e.get("name", "exp")
        cfg = dict(R.DEFAULT); cfg.update({k: v for k, v in e.items() if k != "name"})
        t0 = time.time()
        try:
            m, cells = xmetric(cfg)
            row = f"{name}\t{m:.4f}\t{'/'.join(f'{c:.2f}' for c in cells)}\t{time.time()-t0:.0f}\t{json.dumps({k:v for k,v in e.items() if k!='name'})}"
        except Exception as ex:  # noqa: BLE001
            row = f"{name}\t0.0\tCRASH\t{time.time()-t0:.0f}\t{type(ex).__name__}: {str(ex)[:120]}"
        print("RESULT\t" + row, flush=True)
        with out.open("a") as f:
            f.write(row + "\n")


if __name__ == "__main__":
    main()
