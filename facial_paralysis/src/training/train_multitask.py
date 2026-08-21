"""Multi-task trainer for the full FacialPalsyModel (docs/model_design.md §6/§7).

A batch may mix datasets/label-scales; `multitask_loss` supervises only each
sample's own head while the shared severity trunk learns from all samples. We
monitor HB quality (quadratic-weighted kappa) on the held-out HB samples — the
primary clinical target.

Public entry point:
    train_multitask(model, train_ds, val_ds, cfg) -> history dict (+ best_state)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.datasets.patient_multistream import collate_multistream
from src.evaluation.hb_metrics import HBMetrics
from src.models.multitask import multitask_loss
from src.models.ordinal import predict_grade


@dataclass
class MTTrainConfig:
    epochs: int = 40
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-2
    device: str = "auto"
    monitor_task: str = "hb"          # which task's quadratic kappa to early-stop on
    monitor_n_classes: int = 6        # that task's class count
    log_every: int = 5
    seed: int = 0
    early_stopping_patience: int = 0      # 0 = disabled
    early_stopping_warmup: int = 5
    grad_clip: float = 5.0


def _resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _to_device(batch: dict, device: torch.device) -> dict:
    out = dict(batch)
    for k in ("marlin_emb", "marlin_mask", "mp_seq", "mp_mask", "action_present", "label"):
        out[k] = batch[k].to(device)
    return out


def _forward(model: nn.Module, b: dict) -> dict:
    return model(b["marlin_emb"], b["marlin_mask"], b["mp_seq"], b["mp_mask"],
                 b["action_present"])


@torch.no_grad()
def _eval_monitored(model, loader, device, cfg) -> tuple[float, HBMetrics | None]:
    """Loss over all tasks; ordinal metrics over the monitored task's samples only."""
    model.eval()
    tot_loss, n = 0.0, 0
    preds, trues = [], []
    for batch in loader:
        b = _to_device(batch, device)
        out = _forward(model, b)
        loss, _ = multitask_loss(out, b["label"], b["task_ids"], model.multitask.tasks)
        tot_loss += float(loss.detach()) * b["label"].size(0); n += b["label"].size(0)
        rows = [i for i, t in enumerate(b["task_ids"]) if t == cfg.monitor_task]
        if rows:
            idx = torch.tensor(rows, device=device)
            p = predict_grade(out[cfg.monitor_task].index_select(0, idx))
            preds.append(p.cpu().numpy())
            trues.append(b["label"].index_select(0, idx).cpu().numpy())
    if not preds:
        return tot_loss / max(n, 1), None
    metrics = HBMetrics.from_predictions(
        np.concatenate(trues), np.concatenate(preds), n_classes=cfg.monitor_n_classes)
    return tot_loss / max(n, 1), metrics


def train_multitask(
    model: nn.Module,
    train_ds: Dataset,
    val_ds: Dataset | None,
    cfg: MTTrainConfig | None = None,
) -> dict:
    cfg = cfg or MTTrainConfig()
    torch.manual_seed(cfg.seed)
    device = _resolve_device(cfg.device)
    model = model.to(device)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              collate_fn=collate_multistream)
    val_loader = (DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                             collate_fn=collate_multistream) if val_ds is not None else None)

    # Only train parameters that require grad (MARLIN is frozen / not in this model).
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.epochs)

    history = {"train_loss": [], "val_loss": [], "val_kappa": [], "val_acc": [],
               "task_losses": []}
    best_kappa, best_state, best_epoch, since_best = -float("inf"), None, 0, 0
    final_metrics = None

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        tot, n = 0.0, 0
        agg_parts: dict[str, list[float]] = {}
        for batch in train_loader:
            b = _to_device(batch, device)
            optim.zero_grad()
            out = _forward(model, b)
            loss, parts = multitask_loss(out, b["label"], b["task_ids"], model.multitask.tasks)
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            optim.step()
            tot += float(loss.detach()) * b["label"].size(0); n += b["label"].size(0)
            for k, v in parts.items():
                agg_parts.setdefault(k, []).append(v)
        train_loss = tot / max(n, 1)
        history["train_loss"].append(train_loss)
        history["task_losses"].append({k: float(np.mean(v)) for k, v in agg_parts.items()})

        if val_loader is not None:
            vloss, vm = _eval_monitored(model, val_loader, device, cfg)
            history["val_loss"].append(vloss)
            kappa = vm.quadratic_kappa if vm else float("nan")
            acc = vm.accuracy if vm else float("nan")
            history["val_kappa"].append(kappa); history["val_acc"].append(acc)
            final_metrics = vm
            if vm is not None and kappa > best_kappa + 1e-4:
                best_kappa, best_epoch, since_best = kappa, epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                since_best += 1

        sched.step()
        if epoch == 1 or epoch == cfg.epochs or epoch % cfg.log_every == 0:
            msg = f"  epoch {epoch:>3d}/{cfg.epochs}  train_loss={train_loss:.4f}"
            if val_loader is not None:
                msg += f"  val_loss={history['val_loss'][-1]:.4f}  val_kappa={history['val_kappa'][-1]:.3f}"
            print(msg)

        if (cfg.early_stopping_patience > 0 and val_loader is not None
                and epoch >= cfg.early_stopping_warmup
                and since_best >= cfg.early_stopping_patience):
            print(f"  early stop @ {epoch} (best kappa {best_kappa:.3f} @ {best_epoch})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        if val_loader is not None:
            _, final_metrics = _eval_monitored(model, val_loader, device, cfg)
    history["best_kappa"] = best_kappa if val_loader is not None else None
    history["best_epoch"] = best_epoch
    history["best_state"] = best_state
    history["final_metrics"] = final_metrics
    history["epochs_run"] = len(history["train_loss"])
    return history
