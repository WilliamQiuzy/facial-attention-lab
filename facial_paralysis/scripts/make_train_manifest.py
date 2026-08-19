"""Generate a training-label MANIFEST locally (where raw FNP/YFP annotations live)
so the pod can train from cached bundles alone — no raw images/annotations needed.

Replicates run6_unified.py's three loaders' (path, label, task, split, source)
logic exactly, but serializes metadata instead of loading arrays. Output:
outputs/train_manifest.json — a list of {npz (relpath under outputs/), label,
task, split, source, pid}. The pod trainer reads this + the uploaded bundles.

Run locally:  python3 scripts/make_train_manifest.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run3_fnp_region as r3   # noqa: E402
import run5_yfp_region as r5   # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

OUT = ROOT / "outputs"
ACTION = "clip"
FNP_TO_3 = {0: 0, 1: 1, 2: 1, 3: 2}


def main():
    entries = []

    # PalsyNet (binary, stratified 20% val) -------------------------------
    PAL = OUT / "palsynet_bundles"
    labels = {}
    with (PAL / "labels.csv").open() as f:
        for row in csv.DictReader(f):
            labels[row["patient_id"].strip()] = int(row["label"])
    subs = sorted(labels)
    y = [labels[s] for s in subs]
    _, va = train_test_split(subs, test_size=0.2, random_state=0, stratify=y)
    vaset = set(va)
    for s in subs:
        p = PAL / s / f"{ACTION}.npz"
        if p.exists():
            entries.append({"npz": str(p.relative_to(OUT)), "label": labels[s],
                            "task": "binary", "split": "val" if s in vaset else "train",
                            "source": "palsy", "pid": f"palsy_{s}"})

    # FNP eyes/mouth (4->3 level; valid split => val) ---------------------
    for sp in r3.SPLITS:
        labs = r3.parse_split_labels(sp)
        split = "val" if sp == "valid" else "train"
        for fn, lab in labs.items():
            stem = Path(fn).stem[:60]
            p = r3.CACHE / f"{sp}/{stem}" / f"{ACTION}.npz"
            if not p.exists():
                continue
            for task in ("eyes", "mouth"):
                if lab[task] is not None:
                    entries.append({"npz": str(p.relative_to(OUT)),
                                    "label": FNP_TO_3[int(lab[task])], "task": task,
                                    "split": split, "source": "fnp",
                                    "pid": f"fnp_{sp}_{stem}_{task}"})

    # YFP eyes/mouth (3-level; 25% subjects val) --------------------------
    index = r5.build_index()
    ysubs = sorted({it["subject"] for it in index}, key=lambda x: int(x))
    rng = np.random.default_rng(0)
    n_val = max(1, int(round(len(ysubs) * 0.25)))
    val_subs = set(rng.choice(ysubs, size=n_val, replace=False).tolist())
    for it in index:
        p = r5.CACHE / it["subject"] / it["frame"] / f"{ACTION}.npz"
        if not p.exists():
            continue
        split = "val" if it["subject"] in val_subs else "train"
        for task in ("eyes", "mouth"):
            if it[task] is not None:
                entries.append({"npz": str(p.relative_to(OUT)),
                                "label": int(it[task]), "task": task, "split": split,
                                "source": "yfp", "pid": f"yfp_{it['subject']}_{it['frame']}_{task}"})

    (OUT / "train_manifest.json").write_text(json.dumps(entries))
    c = Counter((e["source"], e["task"], e["split"]) for e in entries)
    print(f"{len(entries)} records -> outputs/train_manifest.json")
    for k in sorted(c):
        print(f"  {k}: {c[k]}")


if __name__ == "__main__":
    main()
