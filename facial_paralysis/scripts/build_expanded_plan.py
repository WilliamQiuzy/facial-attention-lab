"""Stage 1: leak-safe plan for adding anisa + kaggle + stroke to TRAINING.

Per docs/leakage_policy.md these web sets are TRAIN-ONLY and must be deduped:
  - anisa  -> coarse3 (Normal=0/Medium=1/Strong=2), keep 1 image per dHash group
  - kaggle -> binary positive (=1), DROP any image that dHash-matches an FNP image
  - stroke -> coarse3 (severity 0-9 binned to 0/1/2), dedup within
Outputs outputs/expanded_plan.json: [{src_path, id, dataset, task, label}] for the
kept images. Stage 2 extracts bundles for exactly these on the pod.

Run:  python3 scripts/build_expanded_plan.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EXT = ROOT / "data" / "external"
OUT = ROOT / "outputs" / "expanded_plan.json"
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def dhash64(path, size=8):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (size + 1, size))
    bits = (img[:, 1:] > img[:, :-1]).flatten().astype(np.uint8)
    return int(np.packbits(bits).view(">u8")[0])


def coarse3_from_score(s):
    return 0 if s <= 2.0 else (1 if s <= 5.0 else 2)


def main():
    # --- FNP hashes (to drop kaggle/sumin dups of FNP, which is in eval) ---
    fnp_hashes = set()
    for p in (EXT / "roboflow_fnp").rglob("*"):
        if p.suffix.lower() in IMG_EXT:
            h = dhash64(p)
            if h is not None:
                fnp_hashes.add(h)
    print(f"FNP hashes: {len(fnp_hashes)}", flush=True)

    plan = []
    dropped = defaultdict(int)

    # --- anisa -> coarse3, keep 1 per dHash group (all splits pooled, train-only) ---
    NAME2C3 = {"Normal": 0, "Medium": 1, "Strong": 2}
    seen = {}
    for sp in ("train", "valid", "test"):
        j = json.load(open(EXT / "roboflow_anisa_paralysis" / sp / "_annotations.coco.json"))
        cats = {c["id"]: c["name"] for c in j["categories"]}
        id2file = {im["id"]: im["file_name"] for im in j["images"]}
        img_label = {}
        for a in j["annotations"]:
            n = cats[a["category_id"]]
            if n in NAME2C3:
                img_label.setdefault(a["image_id"], n)   # first labeled annotation
        for iid, name in img_label.items():
            fp = EXT / "roboflow_anisa_paralysis" / sp / id2file[iid]
            if not fp.exists():
                continue
            h = dhash64(fp)
            if h is None:
                continue
            if h in seen:
                dropped["anisa_dup"] += 1
                continue
            seen[h] = True
            plan.append({"src_path": str(fp), "id": f"anisa_{sp}_{fp.stem[:40]}",
                         "dataset": "anisa", "task": "coarse3", "label": NAME2C3[name]})

    # --- kaggle -> binary positive, drop FNP dups + within-dup ---
    kseen = set()
    for p in (EXT / "kaggle_facial_droop").rglob("*"):
        if p.suffix.lower() not in IMG_EXT:
            continue
        h = dhash64(p)
        if h is None:
            continue
        if h in fnp_hashes:
            dropped["kaggle_dup_fnp"] += 1; continue
        if h in kseen:
            dropped["kaggle_dup_within"] += 1; continue
        kseen.add(h)
        plan.append({"src_path": str(p), "id": f"kaggle_{p.stem[:40]}",
                     "dataset": "kaggle", "task": "binary", "label": 1})

    # --- stroke -> coarse3 (binned), dedup within ---
    sseen = set()
    for p in (EXT / "roboflow_stroke_eye").rglob("*"):
        if p.suffix.lower() not in IMG_EXT:
            continue
        try:
            score = float(p.parent.name)
        except ValueError:
            continue
        h = dhash64(p)
        if h is None:
            continue
        if h in fnp_hashes or h in sseen:
            dropped["stroke_dup"] += 1; continue
        sseen.add(h)
        plan.append({"src_path": str(p), "id": f"stroke_{p.stem[:40]}",
                     "dataset": "stroke", "task": "coarse3", "label": coarse3_from_score(score)})

    OUT.write_text(json.dumps(plan, indent=0))
    from collections import Counter
    by = Counter((e["dataset"], e["task"], e["label"]) for e in plan)
    print(f"\nKEPT {len(plan)} images (all TRAIN-only). dropped: {dict(dropped)}")
    for k in sorted(by):
        print(f"  {k}: {by[k]}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
