"""Batch driver for the autoresearch-FP loop. Runs a JSON list of experiment
configs sequentially (reusing the cached data across runs) and appends one result
row per experiment to a per-batch results file. Multiple search.py processes can
run in parallel over disjoint config lists (single-writer per file -> no races).

Usage: python search.py <batch.json> <out.tsv>
  batch.json = {"experiments": [{"name":..., <cfg overrides>}, ...]}
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prepare_fp as P  # noqa: E402
import runner as R  # noqa: E402


def main():
    batch = json.loads(Path(sys.argv[1]).read_text())
    out = Path(sys.argv[2])
    exps = batch["experiments"] if "experiments" in batch else batch
    P.load_data()  # warm cache once
    for e in exps:
        name = e.get("name", "exp")
        cfg = dict(R.DEFAULT)
        cfg.update({k: v for k, v in e.items() if k != "name"})
        t0 = time.time()
        try:
            preds = [R.train_one_seed(s, cfg) for s in P.SEEDS]
            truth = {t: [r["label"] for r in P.val_records(t)] for t in P.REGION_TASKS}
            import numpy as np
            eyes = [P.quadratic_kappa(truth["eyes"], p["eyes"], 3) for p in preds]
            mouth = [P.quadratic_kappa(truth["mouth"], p["mouth"], 3) for p in preds]
            means = [0.5 * (a + b) for a, b in zip(eyes, mouth)]
            row = (f"{name}\t{np.mean(means):.4f}\t{np.std(means):.4f}"
                   f"\t{np.mean(eyes):.3f}\t{np.mean(mouth):.3f}\t{time.time()-t0:.0f}"
                   f"\t{json.dumps({k: v for k, v in e.items() if k != 'name'})}")
        except Exception as ex:  # noqa: BLE001
            row = f"{name}\t0.0\t0.0\tNaN\tNaN\t{time.time()-t0:.0f}\tCRASH {type(ex).__name__}: {str(ex)[:140]}"
        print("RESULT\t" + row, flush=True)
        with out.open("a") as f:
            f.write(row + "\n")


if __name__ == "__main__":
    main()
