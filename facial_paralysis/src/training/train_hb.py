"""Training utilities for the HB head.

Public entry points:
  - `train_hb(model, train_ds, val_ds, cfg)` — single train/val pass with
    optional early stopping. Returns history dict + best state.
  - `train_hb_kfold(model_factory, dataset, train_cfg, kfold_cfg)` — k-fold
    cross-validation. Optionally stratified by label so each fold sees a
    balanced HB grade distribution. Returns per-fold results + aggregate stats.

Loss: 6-way cross-entropy. For ordinal alternatives, swap `_compute_loss`'s
body — the head's output shape is already correct.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from src.datasets.patient_videos import collate_patients
from src.evaluation.hb_metrics import HBMetrics


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-2
    n_classes: int = 6
    device: str = "auto"
    log_every: int = 5
    seed: int = 0

    # Early stopping
    early_stopping_patience: int = 0          # 0 = disabled
    early_stopping_monitor: Literal["val_kappa", "val_loss"] = "val_kappa"
    early_stopping_min_delta: float = 0.001    # minimum improvement to count
    early_stopping_warmup_epochs: int = 5      # don't trigger before this epoch


@dataclass
class KFoldConfig:
    k: int = 5
    stratified: bool = True
    seed: int = 0
    verbose: bool = True


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _compute_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """6-way cross-entropy. Swap for CORN/CORAL/MSE-on-grade here."""
    return F.cross_entropy(logits, target)


@torch.no_grad()
def _eval_epoch(model: nn.Module, loader: DataLoader, device: torch.device, n_classes: int):
    model.eval()
    all_pred: list[np.ndarray] = []
    all_true: list[np.ndarray] = []
    total_loss, n = 0.0, 0
    for batch in loader:
        frame_emb = batch["frame_emb"].to(device)
        frame_mask = batch["frame_mask"].to(device)
        labels = batch["label"].to(device)
        logits = model.forward_with_frames(frame_emb, frame_mask)
        loss = _compute_loss(logits, labels)
        total_loss += loss.item() * labels.size(0)
        n += labels.size(0)
        all_pred.append(logits.argmax(dim=1).cpu().numpy())
        all_true.append(labels.cpu().numpy())
    if not all_pred:
        return float("nan"), HBMetrics(
            float("nan"), float("nan"), float("nan"),
            np.zeros((n_classes, n_classes), dtype=np.int64),
        )
    pred = np.concatenate(all_pred)
    true = np.concatenate(all_true)
    metrics = HBMetrics.from_predictions(true, pred, n_classes=n_classes)
    return total_loss / max(n, 1), metrics


def _better(curr: float, best: float, monitor: str, min_delta: float) -> bool:
    """Is `curr` a meaningful improvement over `best` for the chosen monitor?
    For kappa we want greater; for loss we want less."""
    if monitor == "val_kappa":
        return curr > best + min_delta
    if monitor == "val_loss":
        return curr < best - min_delta
    raise ValueError(f"unknown monitor: {monitor}")


# ----------------------------------------------------------------------
# Single-pass training
# ----------------------------------------------------------------------
def train_hb(
    model: nn.Module,
    train_ds: Dataset,
    val_ds: Dataset | None,
    cfg: TrainConfig | None = None,
) -> dict:
    cfg = cfg or TrainConfig()
    torch.manual_seed(cfg.seed)

    device = _resolve_device(cfg.device)
    model = model.to(device)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=collate_patients, num_workers=0,
    )
    val_loader = (
        DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                   collate_fn=collate_patients, num_workers=0)
        if val_ds is not None else None
    )

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.epochs)

    history = {"train_loss": [], "val_loss": [], "val_kappa": [], "val_acc": []}
    best_score = -float("inf") if cfg.early_stopping_monitor == "val_kappa" else float("inf")
    best_state: dict | None = None
    best_epoch = 0
    epochs_since_best = 0
    final_metrics: HBMetrics | None = None
    stopped_early = False

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total, n = 0.0, 0
        for batch in train_loader:
            frame_emb = batch["frame_emb"].to(device)
            frame_mask = batch["frame_mask"].to(device)
            labels = batch["label"].to(device)
            optim.zero_grad()
            logits = model.forward_with_frames(frame_emb, frame_mask)
            loss = _compute_loss(logits, labels)
            loss.backward()
            optim.step()
            total += loss.item() * labels.size(0)
            n += labels.size(0)
        train_loss = total / max(n, 1)
        history["train_loss"].append(train_loss)

        val_loss, val_metrics = float("nan"), None
        if val_loader is not None:
            val_loss, val_metrics = _eval_epoch(model, val_loader, device, cfg.n_classes)
            history["val_loss"].append(val_loss)
            history["val_kappa"].append(val_metrics.quadratic_kappa)
            history["val_acc"].append(val_metrics.accuracy)
            final_metrics = val_metrics

            curr = (val_metrics.quadratic_kappa
                    if cfg.early_stopping_monitor == "val_kappa" else val_loss)
            if _better(curr, best_score, cfg.early_stopping_monitor,
                       cfg.early_stopping_min_delta):
                best_score = curr
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch
                epochs_since_best = 0
            else:
                epochs_since_best += 1

        sched.step()

        if epoch == 1 or epoch == cfg.epochs or epoch % cfg.log_every == 0:
            if val_loader is not None:
                print(f"  epoch {epoch:>3d}/{cfg.epochs}  "
                      f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                      f"val_acc={val_metrics.accuracy:.3f}  val_kappa={val_metrics.quadratic_kappa:.3f}")
            else:
                print(f"  epoch {epoch:>3d}/{cfg.epochs}  train_loss={train_loss:.4f}")

        # Early stopping
        if (cfg.early_stopping_patience > 0
                and val_loader is not None
                and epoch >= cfg.early_stopping_warmup_epochs
                and epochs_since_best >= cfg.early_stopping_patience):
            print(f"  early stop at epoch {epoch} "
                  f"(no improvement in {cfg.early_stopping_patience} epochs since best @ epoch {best_epoch})")
            stopped_early = True
            break

    # Restore best state so .final_metrics reflects the model we'd keep
    if best_state is not None:
        model.load_state_dict(best_state)
        if val_loader is not None:
            _, final_metrics = _eval_epoch(model, val_loader, device, cfg.n_classes)

    history["best_score"] = best_score if val_loader is not None else None
    history["best_epoch"] = best_epoch
    history["best_state"] = best_state
    history["final_metrics"] = final_metrics
    history["stopped_early"] = stopped_early
    history["epochs_run"] = len(history["train_loss"])
    return history


# ----------------------------------------------------------------------
# K-fold CV
# ----------------------------------------------------------------------
def _stratified_kfold_indices(labels: np.ndarray, k: int, seed: int) -> list[np.ndarray]:
    """Per-class round-robin assignment to k folds. For class c with n_c samples,
    samples are shuffled and dealt out fold 0, 1, ..., k-1, 0, 1, ... so each
    fold gets ~n_c/k samples of class c."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    fold_of_sample = np.empty(n, dtype=np.int64)
    for c in np.unique(labels):
        idx_c = np.where(labels == c)[0]
        idx_c = rng.permutation(idx_c)
        for j, i in enumerate(idx_c):
            fold_of_sample[i] = j % k
    return [np.where(fold_of_sample == f)[0] for f in range(k)]


def _random_kfold_indices(n: int, k: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    return [perm[f::k] for f in range(k)]


def train_hb_kfold(
    model_factory: Callable[[], nn.Module],
    dataset: Dataset,
    train_cfg: TrainConfig | None = None,
    kfold_cfg: KFoldConfig | None = None,
) -> dict:
    """Train k models, one per fold, and aggregate metrics.

    Each fold's training set is k-1 folds' samples; val set is the held-out fold.
    Returns:
      {
        "per_fold": [history_0, ..., history_{k-1}],
        "metrics_per_fold": [HBMetrics, ...],
        "kappa_mean", "kappa_std",
        "accuracy_mean", "accuracy_std",
        "mae_mean", "mae_std",
        "confusion_pooled": (n_classes, n_classes) summed across folds,
      }
    """
    train_cfg = train_cfg or TrainConfig()
    kfold_cfg = kfold_cfg or KFoldConfig()
    n = len(dataset)

    # Gather labels for stratification
    labels = np.array([int(dataset[i]["label"]) for i in range(n)])

    if kfold_cfg.stratified:
        fold_idx = _stratified_kfold_indices(labels, kfold_cfg.k, kfold_cfg.seed)
    else:
        fold_idx = _random_kfold_indices(n, kfold_cfg.k, kfold_cfg.seed)

    histories: list[dict] = []
    metrics_list: list[HBMetrics] = []
    pooled_conf = np.zeros((train_cfg.n_classes, train_cfg.n_classes), dtype=np.int64)

    for fold in range(kfold_cfg.k):
        val_idx = fold_idx[fold]
        train_idx = np.concatenate([fold_idx[f] for f in range(kfold_cfg.k) if f != fold])
        train_ds = Subset(dataset, train_idx.tolist())
        val_ds = Subset(dataset, val_idx.tolist())

        if kfold_cfg.verbose:
            train_label_hist = np.bincount(labels[train_idx], minlength=train_cfg.n_classes)
            val_label_hist = np.bincount(labels[val_idx], minlength=train_cfg.n_classes)
            print(f"\n--- fold {fold + 1}/{kfold_cfg.k} ---")
            print(f"  train n={len(train_idx)}  label hist: {train_label_hist.tolist()}")
            print(f"  val   n={len(val_idx)}  label hist: {val_label_hist.tolist()}")

        model = model_factory()
        history = train_hb(model, train_ds, val_ds, train_cfg)
        histories.append(history)
        m = history["final_metrics"]
        if m is not None:
            metrics_list.append(m)
            pooled_conf += m.confusion

    kappa = np.array([m.quadratic_kappa for m in metrics_list])
    accs = np.array([m.accuracy for m in metrics_list])
    mae = np.array([m.mae_grades for m in metrics_list])

    return {
        "per_fold": histories,
        "metrics_per_fold": metrics_list,
        "kappa_mean": float(np.nanmean(kappa)),
        "kappa_std": float(np.nanstd(kappa)),
        "accuracy_mean": float(np.nanmean(accs)),
        "accuracy_std": float(np.nanstd(accs)),
        "mae_mean": float(np.nanmean(mae)),
        "mae_std": float(np.nanstd(mae)),
        "confusion_pooled": pooled_conf,
        "n_classes": train_cfg.n_classes,
    }


def pretty_kfold_summary(result: dict) -> str:
    """Multi-line human-readable report of a k-fold result."""
    from src.evaluation.hb_metrics import HB_GRADE_NAMES
    lines = [
        f"k-fold results over {len(result['metrics_per_fold'])} folds:",
        f"  quadratic kappa : {result['kappa_mean']:.3f} ± {result['kappa_std']:.3f}",
        f"  accuracy        : {result['accuracy_mean']:.3f} ± {result['accuracy_std']:.3f}",
        f"  MAE (grades)    : {result['mae_mean']:.3f} ± {result['mae_std']:.3f}",
        "pooled confusion (rows=true, cols=pred; HB I..VI):",
        "       " + " ".join(f"{n:>4s}" for n in HB_GRADE_NAMES[: result['n_classes']]),
    ]
    for i in range(result["n_classes"]):
        row = " ".join(f"{int(result['confusion_pooled'][i, j]):>4d}"
                        for j in range(result["n_classes"]))
        lines.append(f"  {HB_GRADE_NAMES[i]:<4s} {row}")
    return "\n".join(lines)
