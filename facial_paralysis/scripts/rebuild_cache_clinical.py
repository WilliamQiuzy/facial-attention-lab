"""Rebuild the autoresearch cache with clinical landmark features appended to the geometry
stream. Region images (fnp/yfp) get real clinical features (from pod MediaPipe landmarks);
other records get zeros. Output: fp_ar_cache_clinical.pt + the new feature dim.

Run with anaconda python (torch/numpy). Usage: python rebuild_cache_clinical.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from clinical_landmark_features import clinical_feats, NAMES

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
MANIFEST = OUT / "train_manifest_v4.json"
LANDMARKS = OUT / "web_landmarks.json"
KEY2DIR = OUT / "web_key2dir.json"
SOURCES = ("palsy", "fnp", "yfp")
TASKS = ("binary", "eyes", "mouth")
C = len(NAMES)


def main():
    man = json.loads(MANIFEST.read_text())
    lm = json.loads(LANDMARKS.read_text())
    key2dir = json.loads(KEY2DIR.read_text())
    # clinical vector per bundle_dir
    clin_by_dir = {}
    for key, e in lm.items():
        bundle_dir = key2dir.get(key)
        if bundle_dir is None:
            continue
        vec, _ = clinical_feats(np.array(e["xy"], np.float32), e["w"], e["h"])
        clin_by_dir[bundle_dir] = vec
    print(f"clinical features for {len(clin_by_dir)} images (dim {C})")

    recs, n_real, n_zero = [], 0, 0
    for e in man:
        if e["source"] not in SOURCES or e["task"] not in TASKS:
            continue
        p = OUT / e["npz"]
        if not p.exists():
            continue
        d = np.load(p)
        marlin = d["marlin"].astype(np.float32)
        mp_seq = d["mp_seq"].astype(np.float32)               # (T,72)
        bundle_dir = "/".join(e["npz"].split("/")[:-1])
        clin = clin_by_dir.get(bundle_dir)
        if clin is None:
            clin = np.zeros(C, np.float32); n_zero += 1
        else:
            n_real += 1
        T = mp_seq.shape[0]
        mp_seq_new = np.concatenate([mp_seq, np.tile(clin, (T, 1))], axis=1)  # (T,72+C)
        recs.append({
            "marlin": marlin.mean(0), "mp_seq": mp_seq_new,
            "mp_mask": d["mp_mask"].astype(bool), "label": int(e["label"]),
            "task": e["task"], "source": e["source"], "split": e["split"],
        })
    torch.save(recs, ROOT / "autoresearch_fp" / "fp_ar_cache_clinical.pt")
    print(f"rebuilt {len(recs)} records: {n_real} with clinical, {n_zero} zero-padded")
    print(f"new MP_FEAT_DIM = {72 + C}  (72 blendshapes + {C} clinical)")
    print(f"clinical block indices: [72:{72+C}]  names={NAMES}")


if __name__ == "__main__":
    main()
