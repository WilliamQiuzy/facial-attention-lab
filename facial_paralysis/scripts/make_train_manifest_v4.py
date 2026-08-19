"""Stage 3: build the EXPANDED, leak-safe training manifest (v4).

Reuses the clean v2 manifest (outputs/train_manifest.json — PalsyNet binary +
FNP/YFP eyes/mouth with honest per-source/subject holdout) and APPENDS the
expanded-plan images (anisa/kaggle/stroke) as TRAIN-ONLY records pointing to the
pod-extracted bundles in outputs/expanded_bundles/. No new val/test (leakage policy).

Run:  python3 scripts/make_train_manifest_v4.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"


def main():
    base = json.loads((OUT / "train_manifest.json").read_text())
    plan = json.loads((OUT / "expanded_plan.json").read_text())
    man = list(base)
    for e in plan:
        man.append({"npz": f"expanded_bundles/{e['id']}.npz", "label": int(e["label"]),
                    "task": e["task"], "split": "train", "source": e["dataset"], "pid": e["id"]})
    (OUT / "train_manifest_v4.json").write_text(json.dumps(man))
    c = Counter((e["source"], e["task"], e["split"]) for e in man)
    print(f"v4 manifest: {len(man)} records ({len(base)} base + {len(plan)} expanded)")
    for k in sorted(c):
        print(f"  {k}: {c[k]}")
    print(f"wrote {OUT}/train_manifest_v4.json")


if __name__ == "__main__":
    main()
