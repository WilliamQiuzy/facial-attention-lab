"""Cross-dataset & cross-split near-duplicate detector (leakage guard).

Our palsy image sets are web-scraped (FNP, sumin, anisa, Kaggle droop, YFP frames),
so the SAME face/image very likely appears in more than one set, and Roboflow splits
BY IMAGE not by patient — so a face can land in both train and test. Either leaks
the test set. This computes a perceptual hash (dHash, 64-bit) for every image and
flags:
  (1) EXACT-hash groups spanning >1 dataset      -> cross-dataset pooling leak
  (2) EXACT/NEAR groups spanning >1 split in ONE dataset -> train↔val/test leak
Near-dup (Hamming <= THRESH) is computed on the non-YFP sets + per-subject YFP reps
(YFP has 17k video frames — included as exact-hash only + 1 rep/subject for near).

Output: outputs/dedup_report.json (+ console summary). Use the dup groups to force
each group into a single split when building train/val/test.

Run:  python3 scripts/dedup_check.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EXT = ROOT / "data" / "external"
OUT = ROOT / "outputs" / "dedup_report.json"
NEAR_THRESH = 6          # Hamming distance on the 64-bit dHash
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def dhash_bits(path, size=8):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (size + 1, size))
    diff = (img[:, 1:] > img[:, :-1]).flatten()      # 64 bools
    return diff.astype(np.uint8)


def collect():
    """Return list of dicts {path, dataset, split, subject} for all images."""
    items = []
    # Roboflow COCO sets: <dir>/{train,valid,test}/*.jpg
    for ds in ("roboflow_fnp", "roboflow_facial_paralysis", "roboflow_anisa_paralysis"):
        d = EXT / ds
        for split in ("train", "valid", "test"):
            for p in (d / split).rglob("*"):
                if p.suffix.lower() in IMG_EXT:
                    items.append({"path": str(p), "dataset": ds, "split": split, "subject": None})
    # stroke (classification, train/<score>/)
    for p in (EXT / "roboflow_stroke_eye").rglob("*"):
        if p.suffix.lower() in IMG_EXT:
            split = p.parts[len(EXT.parts) + 1] if len(p.parts) > len(EXT.parts) + 1 else "all"
            items.append({"path": str(p), "dataset": "roboflow_stroke_eye", "split": split, "subject": None})
    # kaggle droop (no split)
    for p in (EXT / "kaggle_facial_droop").rglob("*"):
        if p.suffix.lower() in IMG_EXT:
            items.append({"path": str(p), "dataset": "kaggle_facial_droop", "split": "all", "subject": None})
    # YFP frames: Image*/<subject>/<frame>.bmp  (subject = leakage unit)
    for top in ("Image", "Image2", "Image3", "Image4"):
        base = EXT / "YFP" / top
        if not base.exists():
            continue
        for p in base.rglob("*.bmp"):
            items.append({"path": str(p), "dataset": "yfp", "split": "na", "subject": p.parent.name})
    return items


def main():
    items = collect()
    print(f"hashing {len(items)} images...", flush=True)
    H, keep = [], []
    for it in items:
        b = dhash_bits(it["path"])
        if b is not None:
            H.append(b); keep.append(it)
    H = np.stack(H)                                  # (N,64) uint8
    items = keep
    N = len(items)
    # pack to uint64 for exact grouping
    packed = np.packbits(H, axis=1).view(">u8").ravel()
    print(f"hashed {N}; finding duplicates...", flush=True)

    # ---- (1) exact-hash groups ----
    by_hash = defaultdict(list)
    for i, h in enumerate(packed):
        by_hash[int(h)].append(i)
    exact_groups = [idxs for idxs in by_hash.values() if len(idxs) > 1]

    def datasets_of(idxs): return sorted({items[i]["dataset"] for i in idxs})
    def split_keys(idxs): return sorted({(items[i]["dataset"], items[i]["split"]) for i in idxs})

    cross_dataset = [g for g in exact_groups if len(datasets_of(g)) > 1]
    cross_split = [g for g in exact_groups
                   if len(datasets_of(g)) == 1
                   and len({items[i]["split"] for i in g}) > 1
                   and items[g[0]]["dataset"] != "yfp"]

    # ---- (2) near-dup among non-YFP + 1 rep/YFP-subject ----
    near_idx = [i for i, it in enumerate(items) if it["dataset"] != "yfp"]
    seen_subj = set()
    for i, it in enumerate(items):
        if it["dataset"] == "yfp" and it["subject"] not in seen_subj:
            seen_subj.add(it["subject"]); near_idx.append(i)
    sub = np.array(near_idx)
    Hs = H[sub]
    near_pairs = []
    for a in range(len(sub)):
        ham = (Hs[a] != Hs).sum(axis=1)
        for b in np.where((ham > 0) & (ham <= NEAR_THRESH))[0]:
            if b > a:
                i, j = int(sub[a]), int(sub[b])
                if items[i]["dataset"] != items[j]["dataset"] or items[i]["split"] != items[j]["split"]:
                    near_pairs.append((i, j, int(ham[b])))

    def short(i):
        it = items[i]
        return f"{it['dataset']}/{it['split']}" + (f"/{it['subject']}" if it["subject"] else "") + ":" + Path(it["path"]).name

    report = {
        "n_images": N,
        "n_exact_dup_groups": len(exact_groups),
        "cross_dataset_exact_groups": len(cross_dataset),
        "within_dataset_cross_split_exact_groups": len(cross_split),
        "near_dup_cross_pairs": len(near_pairs),
        "examples_cross_dataset": [[short(i) for i in g] for g in cross_dataset[:15]],
        "examples_cross_split": [[short(i) for i in g] for g in cross_split[:15]],
        "examples_near": [[short(i), short(j), f"ham={h}"] for i, j, h in near_pairs[:15]],
    }
    # dataset-pair overlap counts
    pair_counts = defaultdict(int)
    for g in cross_dataset:
        ds = datasets_of(g)
        for a in range(len(ds)):
            for b in range(a + 1, len(ds)):
                pair_counts[f"{ds[a]} <-> {ds[b]}"] += 1
    report["cross_dataset_pair_counts"] = dict(sorted(pair_counts.items(), key=lambda x: -x[1]))

    OUT.write_text(json.dumps(report, indent=1))
    print("\n================ DEDUP / LEAKAGE REPORT ================")
    print(f"images hashed: {N}")
    print(f"exact-dup groups: {len(exact_groups)}")
    print(f"  CROSS-DATASET exact groups (pooling leak): {len(cross_dataset)}")
    print(f"  within-dataset CROSS-SPLIT exact groups (train<->val/test leak): {len(cross_split)}")
    print(f"near-dup cross pairs (Hamming<= {NEAR_THRESH}): {len(near_pairs)}")
    if report["cross_dataset_pair_counts"]:
        print("cross-dataset overlap (exact) by pair:")
        for k, v in report["cross_dataset_pair_counts"].items():
            print(f"   {k}: {v} groups")
    if cross_split:
        print("\n!! within-dataset cross-split duplicates (these leak train->test):")
        for g in cross_split[:8]:
            print("   ", [short(i) for i in g])
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
