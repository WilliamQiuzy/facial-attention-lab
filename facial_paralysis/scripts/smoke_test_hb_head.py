"""End-to-end smoke test for the HB head framework.

What this exercises:
  - Synthetic dataset with a planted class-direction signal in 768-d
  - HBHead model factory (mean-pool over actions)
  - 5-fold stratified k-fold CV
  - Early stopping on val_kappa with patience=8
  - Aggregate metrics across folds + pooled confusion matrix

When real clinical labels arrive, replace `_make_synthetic_dataset` with
`PatientVideoDataset.from_disk(data_root, embedding_cache_root, labels_csv)`
and re-run — everything else stays the same.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.datasets.patient_videos import (  # noqa: E402
    PatientRecord, PatientVideoDataset, STANDARD_ACTIONS,
)
from src.models.hb_head import HBHead, HBHeadConfig  # noqa: E402
from src.training.train_hb import (  # noqa: E402
    KFoldConfig, TrainConfig, pretty_kfold_summary, train_hb, train_hb_kfold,
)

N_CLASSES = 6
EMBED_DIM = 768
N_ACTIONS = len(STANDARD_ACTIONS)


def _make_synthetic_dataset(
    n_patients: int,
    n_frames_per_action: int = 9,
    signal_strength: float = 4.0,
    missing_action_prob: float = 0.10,
    seed: int = 0,
) -> PatientVideoDataset:
    """Synthetic patient cohort with a planted signal: class shifts a fixed
    direction in 768-d. Noise std = 0.5. With signal_strength=4 + 60 patients
    the head should learn a meaningful k-fold kappa (validates the framework)."""
    rng = np.random.default_rng(seed)
    signal_dir = rng.standard_normal(EMBED_DIM).astype(np.float32)
    signal_dir /= np.linalg.norm(signal_dir)

    records: list[PatientRecord] = []
    for i in range(n_patients):
        label = i % N_CLASSES                  # roughly balanced
        per_action: list[np.ndarray | None] = []
        for _ in range(N_ACTIONS):
            if rng.random() < missing_action_prob:
                per_action.append(None)
                continue
            base = rng.standard_normal((n_frames_per_action, EMBED_DIM)).astype(np.float32) * 0.5
            shift = (label - (N_CLASSES - 1) / 2.0) * signal_strength
            base += shift * signal_dir
            per_action.append(base)
        records.append(PatientRecord(
            patient_id=f"synthetic_{i:03d}",
            label=label,
            frame_emb_per_action=per_action,
        ))
    return PatientVideoDataset(records, actions=STANDARD_ACTIONS, embed_dim=EMBED_DIM)


def _make_model():
    return HBHead(HBHeadConfig(
        embed_dim=EMBED_DIM,
        n_actions=N_ACTIONS,
        n_classes=N_CLASSES,
        action_pool="mean",
        dropout=0.3,
    ))


def main():
    print("=== HB head framework smoke test (k-fold + early stopping) ===\n")
    torch.manual_seed(0)

    full_ds = _make_synthetic_dataset(n_patients=60, seed=0)
    print(f"dataset: {len(full_ds)} patients   embed_dim={EMBED_DIM}   "
          f"n_actions={N_ACTIONS}   n_classes={N_CLASSES}")
    label_hist = np.bincount(
        np.array([int(full_ds[i]["label"]) for i in range(len(full_ds))]),
        minlength=N_CLASSES,
    )
    print(f"label distribution: HB I-VI = {label_hist.tolist()}")

    train_cfg = TrainConfig(
        epochs=50,
        batch_size=8,
        lr=2e-3,
        weight_decay=5e-2,                     # heavier reg for small data
        log_every=10,
        early_stopping_patience=8,
        early_stopping_monitor="val_kappa",
        early_stopping_warmup_epochs=5,
    )
    kfold_cfg = KFoldConfig(k=5, stratified=True, seed=0, verbose=True)

    # --- Baseline: single 50/10 hold-out split, no early stopping ---
    print("\n" + "=" * 60)
    print("Baseline (1 train/val split, no early stopping)")
    print("=" * 60)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(full_ds))
    n_val = 12
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    from torch.utils.data import Subset
    bl_train = Subset(full_ds, train_idx.tolist())
    bl_val = Subset(full_ds, val_idx.tolist())
    bl_cfg = TrainConfig(
        epochs=50, batch_size=8, lr=2e-3, weight_decay=5e-2,
        log_every=10, early_stopping_patience=0,  # disable
    )
    bl_history = train_hb(_make_model(), bl_train, bl_val, bl_cfg)
    print(f"\nbaseline final kappa: {bl_history['final_metrics'].quadratic_kappa:.3f}  "
          f"(best @ epoch {bl_history['best_epoch']}; ran {bl_history['epochs_run']} epochs)")

    # --- Main: 5-fold stratified CV with early stopping ---
    print("\n" + "=" * 60)
    print("5-fold stratified CV with early stopping (patience=8 on val_kappa)")
    print("=" * 60)
    result = train_hb_kfold(_make_model, full_ds, train_cfg, kfold_cfg)

    print("\n" + "=" * 60)
    print("Per-fold summary")
    print("=" * 60)
    print(f"{'fold':>4s}  {'epochs':>6s}  {'best_ep':>8s}  {'stopped_early':>14s}  "
          f"{'kappa':>6s}  {'acc':>6s}  {'mae':>5s}")
    for i, (h, m) in enumerate(zip(result["per_fold"], result["metrics_per_fold"])):
        print(f"{i:>4d}  {h['epochs_run']:>6d}  {h['best_epoch']:>8d}  "
              f"{str(h['stopped_early']):>14s}  "
              f"{m.quadratic_kappa:>6.3f}  {m.accuracy:>6.3f}  {m.mae_grades:>5.2f}")

    print("\n" + "=" * 60)
    print("Aggregate")
    print("=" * 60)
    print(pretty_kfold_summary(result))

    # Sanity-check assertions so the test fails loudly if framework breaks
    assert result["kappa_mean"] > 0.10, (
        f"kappa_mean={result['kappa_mean']:.3f} too low — framework may be broken"
    )
    assert len(result["metrics_per_fold"]) == kfold_cfg.k
    early_stopped_count = sum(1 for h in result["per_fold"] if h["stopped_early"])
    print(f"\nframework smoke test: PASSED  "
          f"(kappa_mean={result['kappa_mean']:.3f}, "
          f"{early_stopped_count}/{kfold_cfg.k} folds stopped early)")


if __name__ == "__main__":
    main()
