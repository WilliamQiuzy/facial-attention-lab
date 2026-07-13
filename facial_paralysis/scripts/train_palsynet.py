"""Real (non-synthetic) training run #1: train the full FacialPalsyModel on
PalsyNet binary palsy/healthy, with subject-level stratified CV.

This is the first real exercise of the whole pipeline end to end:
  PalsyNet videos -> action_bundle (frozen MARLIN windows + MediaPipe 72-d seq)
  -> MultiStreamPatientDataset -> FacialPalsyModel (binary head) -> CV eval.

Each video is one subject; we treat the clip as a single pseudo-action (n_actions=1)
since PalsyNet has no per-action structure. Reports per-fold + mean subject-level
ROC AUC and accuracy. Bundles are cached so re-runs skip extraction.

Run:
  KMP_DUPLICATE_LIB_OK=TRUE \
  /opt/homebrew/Caskroom/miniforge/base/envs/dev/bin/python scripts/train_palsynet.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PALSYNET = ROOT / "data" / "external" / "palsynet" / "data"
CACHE = ROOT / "outputs" / "palsynet_bundles"
ACTION = "clip"               # single pseudo-action
N_MARLIN_WINDOWS = 4
MP_FEAT_DIM = 72


def extract_bundles(reextract: bool = False) -> Path:
    """Extract MARLIN + MediaPipe bundles for all PalsyNet videos. Returns the
    labels.csv path. Caches to CACHE/<subject>/clip.npz."""
    from src.models.backbones.marlin_video import MarlinVideoEncoder
    from src.preprocessing.action_bundle import (
        MediaPipeFeatureExtractor,
        _assert_existing_cache_schema,
        _bundle_npz_payload,
        extract_action_bundle,
    )

    CACHE.mkdir(parents=True, exist_ok=True)
    labels_path = CACHE / "labels.csv"
    rows = []
    need = []
    for label, sub in [(1, "affected"), (0, "unaffected")]:
        for vp in sorted((PALSYNET / sub).glob("*.mp4")):
            sid = f"{sub}_{vp.stem}"
            rows.append({"patient_id": sid, "task": "binary", "label": label})
            out = CACHE / sid / f"{ACTION}.npz"
            if reextract or not out.exists():
                need.append((sid, vp))
            else:
                _assert_existing_cache_schema(
                    out, "mediapipe_bs_lr_v1",
                    expected_capture_mirrored="unknown",
                )

    if need:
        enc = MarlinVideoEncoder.from_default_weights().eval()
        mp_ext = MediaPipeFeatureExtractor(capture_mirrored=None)
        for i, (sid, vp) in enumerate(need, 1):
            b = extract_action_bundle(vp, enc, mp_ext, n_marlin_windows=N_MARLIN_WINDOWS)
            if b is None:
                print(f"  [skip] {sid}: unusable"); continue
            d = CACHE / sid; d.mkdir(parents=True, exist_ok=True)
            np.savez(d / f"{ACTION}.npz", **_bundle_npz_payload(b, mp_ext))
            print(f"  [{i}/{len(need)}] {sid}: marlin{b['marlin'].shape} mp{b['mp_seq'].shape}")
    else:
        print("  all bundles cached")

    with labels_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["patient_id", "task", "label"])
        w.writeheader(); w.writerows(rows)
    return labels_path


def run_cv(n_splits: int = 5, seed: int = 0) -> dict:
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, accuracy_score
    from torch.utils.data import Subset, DataLoader

    from src.datasets.patient_multistream import (
        MultiStreamPatientDataset, collate_multistream)
    from src.models.facial_palsy_model import FacialPalsyModel, FacialPalsyConfig
    from src.models.multitask import TaskSpec
    from src.training.train_multitask import MTTrainConfig, train_multitask
    from src.models.ordinal import cum_probs

    labels_path = CACHE / "labels.csv"
    ds = MultiStreamPatientDataset.from_disk(
        CACHE, labels_path, actions=[ACTION], mp_feat_dim=MP_FEAT_DIM,
        mp_feature_schema="mediapipe_bs_lr_v1")
    y = np.array([r.label for r in ds.records])
    print(f"\ndataset: {len(ds)} subjects, {int((y==1).sum())} palsy / {int((y==0).sum())} healthy")

    def make_model():
        torch.manual_seed(seed)
        return FacialPalsyModel(FacialPalsyConfig(
            mp_feat_dim=MP_FEAT_DIM, n_actions=1, temporal_hidden=64, temporal_out=64,
            trunk_hidden=64, dropout=0.1,
            tasks=[TaskSpec("binary", n_classes=2, coupled=True)]))

    cfg = MTTrainConfig(epochs=60, batch_size=8, lr=5e-4, weight_decay=3e-2,
                        device="cpu", monitor_task="binary", monitor_n_classes=2,
                        log_every=999, seed=seed, early_stopping_patience=12)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    subj_prob = np.zeros(len(ds)); subj_true = y.copy()
    fold_auc, fold_acc = [], []
    for fold, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y), 1):
        model = make_model()
        train_multitask(model, Subset(ds, tr.tolist()), Subset(ds, te.tolist()), cfg)
        # score held-out fold
        model.eval()
        loader = DataLoader(Subset(ds, te.tolist()), batch_size=len(te),
                            collate_fn=collate_multistream)
        b = next(iter(loader))
        with torch.no_grad():
            out = model(b["marlin_emb"], b["marlin_mask"], b["mp_seq"], b["mp_mask"],
                        b["action_present"])
            prob = cum_probs(out["binary"])[:, 0].cpu().numpy()   # P(y>0)=P(palsy)
        subj_prob[te] = prob
        auc = roc_auc_score(y[te], prob); acc = accuracy_score(y[te], (prob > 0.5).astype(int))
        fold_auc.append(auc); fold_acc.append(acc)
        print(f"  fold {fold}: n_val={len(te)}  AUC={auc:.3f}  acc={acc:.3f}")

    pooled_auc = roc_auc_score(subj_true, subj_prob)
    pooled_acc = accuracy_score(subj_true, (subj_prob > 0.5).astype(int))
    res = {
        "n_subjects": int(len(ds)),
        "n_palsy": int((y == 1).sum()), "n_healthy": int((y == 0).sum()),
        "fold_auc": [round(a, 3) for a in fold_auc],
        "fold_acc": [round(a, 3) for a in fold_acc],
        "mean_auc": round(float(np.mean(fold_auc)), 3),
        "std_auc": round(float(np.std(fold_auc)), 3),
        "mean_acc": round(float(np.mean(fold_acc)), 3),
        "pooled_auc": round(float(pooled_auc), 3),
        "pooled_acc": round(float(pooled_acc), 3),
        "config": {"n_marlin_windows": N_MARLIN_WINDOWS, "mp_feat_dim": MP_FEAT_DIM,
                   "epochs": cfg.epochs, "lr": cfg.lr, "weight_decay": cfg.weight_decay,
                   "n_splits": n_splits, "seed": seed},
    }
    print("\n================= PALSYNET TRAINING RESULT =================")
    print(f"  full pipeline (frozen MARLIN + trainable GRU + binary head)")
    print(f"  subject-level {n_splits}-fold CV (stratified):")
    print(f"    AUC  {res['mean_auc']:.3f} ± {res['std_auc']:.3f}   (pooled {res['pooled_auc']:.3f})")
    print(f"    acc  {res['mean_acc']:.3f}                  (pooled {res['pooled_acc']:.3f})")
    print(f"  reference: frozen-MARLIN linear probe was AUC 0.872 (feasibility gate)")
    print("===========================================================")
    (CACHE / "results.json").write_text(json.dumps(res, indent=2))
    return res


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--reextract", action="store_true")
    ap.add_argument("--n-splits", type=int, default=5)
    args = ap.parse_args()
    print("Extracting PalsyNet bundles (MARLIN + MediaPipe)...")
    extract_bundles(reextract=args.reextract)
    run_cv(n_splits=args.n_splits)


if __name__ == "__main__":
    main()
