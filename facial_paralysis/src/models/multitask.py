"""Coupled multi-task severity model for facial-paralysis grading.

Implements §4–5 of `docs/model_design.md`: a shared trunk maps a patient's
per-action embeddings to a scalar *global severity* `s` (plus a representation
`h`), and each label scale (binary palsy / 3-class coarse / HB I–VI / region
intensity) attaches as an **ordered-threshold head**. Global tasks read the
shared `s`; region-specific tasks own a private severity projected from `h`.

This is the mechanism that lets datasets on incompatible label scales train one
model without fabricating labels: every head is just a different cut-point set
on a common severity axis. See `src/models/ordinal.py` for the head math.

Input contract (matches `src/datasets/patient_videos.py`):
    action_emb : (B, n_actions, D)                    already pooled over frames
  or
    frame_emb  : (B, n_actions, n_frames, D)          before frame pooling
    frame_mask : (B, n_actions, n_frames) bool        True = real frame
An image is the degenerate case n_frames == 1. Missing actions are signalled by
`action_mask` / an all-False frame_mask row and excluded from pooling.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.ordinal import (
    OrderedThresholds,
    ordinal_loss,
    predict_grade,
)


# ----------------------------------------------------------------------
# Task specification
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class TaskSpec:
    """One supervised label scale attached to the severity axis.

    name:        unique key, also the key used to route labels at loss time.
    n_classes:   ordinal levels (>=2). binary=2, Parra coarse=3, HB=6.
    coupled:     True  -> head reads the shared global severity `s`.
                 False -> head projects its own severity from `h` (region tasks).
    loss_weight: scalar multiplier in the aggregated multi-task loss.
    """

    name: str
    n_classes: int
    coupled: bool = True
    loss_weight: float = 1.0

    def __post_init__(self):
        if self.n_classes < 2:
            raise ValueError(f"task {self.name}: n_classes must be >= 2, got {self.n_classes}")
        if self.loss_weight < 0:
            raise ValueError(f"task {self.name}: loss_weight must be >= 0")


# Default task set for the project. HB is the primary target; the rest are
# auxiliary supervision from public datasets (see docs/model_design.md §4).
DEFAULT_TASKS: tuple[TaskSpec, ...] = (
    TaskSpec("hb", n_classes=6, coupled=True, loss_weight=1.0),         # MEEI / Mayo
    TaskSpec("binary", n_classes=2, coupled=True, loss_weight=0.5),     # PalsyNet
    TaskSpec("coarse3", n_classes=3, coupled=True, loss_weight=0.5),    # Parra
    TaskSpec("eyes", n_classes=2, coupled=False, loss_weight=0.3),      # YFP region
    TaskSpec("mouth", n_classes=2, coupled=False, loss_weight=0.3),     # YFP region
)


# ----------------------------------------------------------------------
# Shared trunk: per-action embeddings -> (representation h, severity s)
# ----------------------------------------------------------------------
@dataclass
class TrunkConfig:
    embed_dim: int = 768
    hidden_dim: int = 256
    n_actions: int = 5
    dropout: float = 0.2
    action_pool: Literal["mean", "max", "attention"] = "mean"


class SeverityTrunk(nn.Module):
    """Per-action MLP -> action pooling -> representation `h` and scalar `s`.

    Mirrors the pooling logic of the legacy `HBHead` (masked mean/max/attention)
    so behavior is comparable, but exposes the pre-classifier representation and
    a dedicated global-severity projection for the coupled multi-task design.
    """

    def __init__(self, cfg: TrunkConfig | None = None):
        super().__init__()
        self.cfg = cfg or TrunkConfig()
        D, H = self.cfg.embed_dim, self.cfg.hidden_dim

        self.per_action_mlp = nn.Sequential(
            nn.Linear(D, H),
            nn.ReLU(inplace=True),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(H, H),
            nn.ReLU(inplace=True),
            nn.Dropout(self.cfg.dropout),
        )
        if self.cfg.action_pool == "attention":
            self.action_attn = nn.Sequential(
                nn.Linear(H, H // 2),
                nn.Tanh(),
                nn.Linear(H // 2, 1),
            )
        else:
            self.action_attn = None

        # Global severity projection. Initialized small so the untrained model
        # starts near the threshold spread chosen in OrderedThresholds.
        self.severity_proj = nn.Linear(H, 1)

        self.out_dim = H

    # ----- pooling helpers (validated) -----
    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
        m = mask.to(x.dtype)
        s = (x * m.unsqueeze(-1)).sum(dim=dim)
        denom = m.sum(dim=dim).clamp_min(1.0).unsqueeze(-1)
        return s / denom

    def _pool_actions(self, h: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        if self.cfg.action_pool == "mean":
            return self._masked_mean(h, action_mask, dim=1)
        if self.cfg.action_pool == "max":
            neg = torch.finfo(h.dtype).min
            return h.masked_fill(~action_mask.unsqueeze(-1), neg).amax(dim=1)
        if self.cfg.action_pool == "attention":
            scores = self.action_attn(h).squeeze(-1)             # (B, n_actions)
            scores = scores.masked_fill(~action_mask, torch.finfo(scores.dtype).min)
            weights = F.softmax(scores, dim=1).unsqueeze(-1)
            return (h * weights).sum(dim=1)
        raise ValueError(f"unknown action_pool: {self.cfg.action_pool}")

    def represent(
        self,
        action_emb: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, n_actions, D) -> (h: (B, H), s: (B,))."""
        if action_emb.dim() != 3:
            raise ValueError(f"expected (B, n_actions, D), got {tuple(action_emb.shape)}")
        if action_emb.shape[-1] != self.cfg.embed_dim:
            raise ValueError(
                f"embed_dim mismatch: got {action_emb.shape[-1]}, cfg {self.cfg.embed_dim}"
            )
        if action_mask is None:
            action_mask = torch.ones(action_emb.shape[:2], dtype=torch.bool,
                                     device=action_emb.device)
        elif action_mask.shape != action_emb.shape[:2]:
            raise ValueError(
                f"action_mask shape {tuple(action_mask.shape)} != {tuple(action_emb.shape[:2])}"
            )
        if action_mask.dtype != torch.bool:
            raise TypeError(f"action_mask must be bool, got {action_mask.dtype}")
        if not action_mask.any(dim=1).all():
            raise ValueError("every sample must have at least one present action")

        h_act = self.per_action_mlp(action_emb)                  # (B, n_actions, H)
        h = self._pool_actions(h_act, action_mask)               # (B, H)
        s = self.severity_proj(h).squeeze(1)                     # (B,)
        return h, s

    def represent_with_frames(
        self,
        frame_emb: torch.Tensor,
        frame_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, n_actions, n_frames, D) -> (h, s). Mean-pools real frames first."""
        if frame_emb.dim() != 4:
            raise ValueError(f"expected (B, n_actions, n_frames, D), got {tuple(frame_emb.shape)}")
        if frame_mask is None:
            frame_mask = torch.ones(frame_emb.shape[:3], dtype=torch.bool,
                                    device=frame_emb.device)
        elif frame_mask.shape != frame_emb.shape[:3]:
            raise ValueError(
                f"frame_mask shape {tuple(frame_mask.shape)} != {tuple(frame_emb.shape[:3])}"
            )
        action_emb = self._masked_mean(frame_emb, frame_mask, dim=2)  # (B, n_actions, D)
        action_present = frame_mask.any(dim=2)                        # (B, n_actions)
        return self.represent(action_emb, action_present)


# ----------------------------------------------------------------------
# Full model
# ----------------------------------------------------------------------
class MultiTaskSeverityModel(nn.Module):
    """Shared severity trunk + one ordered-threshold head per task.

    forward(...) returns {task_name: cumulative_logits (B, K_task - 1)} for ALL
    tasks. Supervision is selective (see `multitask_loss`): only the head that
    matches each sample's task receives a gradient, but the shared trunk /
    global severity is shaped by every sample.
    """

    def __init__(
        self,
        tasks: Sequence[TaskSpec] = DEFAULT_TASKS,
        trunk_cfg: TrunkConfig | None = None,
    ):
        super().__init__()
        if not tasks:
            raise ValueError("need at least one task")
        names = [t.name for t in tasks]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate task names: {names}")

        self.trunk = SeverityTrunk(trunk_cfg)
        self.tasks = tuple(tasks)
        H = self.trunk.out_dim

        # Coupled tasks share the trunk's global severity -> only thresholds.
        # Uncoupled tasks own a private severity projection from `h`.
        self.thresholds = nn.ModuleDict({
            t.name: OrderedThresholds(t.n_classes) for t in tasks
        })
        self.region_proj = nn.ModuleDict({
            t.name: nn.Linear(H, 1) for t in tasks if not t.coupled
        })

    @property
    def task_map(self) -> dict[str, TaskSpec]:
        return {t.name: t for t in self.tasks}

    def _heads_from_repr(self, h: torch.Tensor, s: torch.Tensor) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for t in self.tasks:
            if t.coupled:
                severity = s
            else:
                severity = self.region_proj[t.name](h).squeeze(1)
            out[t.name] = self.thresholds[t.name](severity)
        return out

    def forward(
        self,
        action_emb: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        h, s = self.trunk.represent(action_emb, action_mask)
        return self._heads_from_repr(h, s)

    def forward_with_frames(
        self,
        frame_emb: torch.Tensor,
        frame_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        h, s = self.trunk.represent_with_frames(frame_emb, frame_mask)
        return self._heads_from_repr(h, s)

    # Convenience for the primary target.
    def predict_hb(
        self,
        action_emb: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        hb_task: str = "hb",
    ) -> torch.Tensor:
        if hb_task not in self.thresholds:
            raise KeyError(f"no task named {hb_task!r}; have {list(self.thresholds)}")
        return predict_grade(self.forward(action_emb, action_mask)[hb_task])


# ----------------------------------------------------------------------
# Multi-task loss with per-sample routing
# ----------------------------------------------------------------------
def multitask_loss(
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    task_ids: Sequence[str],
    tasks: Sequence[TaskSpec],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Aggregate ordinal loss over a (possibly mixed) batch.

    Each sample i belongs to dataset/task `task_ids[i]` with label `labels[i]`.
    Only the head matching a sample's task is supervised by that sample. The
    total loss is the task-weighted mean over the tasks present in the batch.

    Args:
        outputs:  {task_name: cum_logits (B, K-1)} as returned by the model.
        labels:   (B,) long, each in [0, K_task-1] for that sample's task.
        task_ids: length-B sequence of task names.
        tasks:    the TaskSpec list (for weights + validation).

    Returns:
        (total_loss, parts) where parts maps task_name -> its raw (unweighted)
        loss value for logging. Tasks absent from the batch are omitted.
    """
    if labels.dim() != 1:
        raise ValueError(f"labels must be 1-D, got {tuple(labels.shape)}")
    B = labels.shape[0]
    if len(task_ids) != B:
        raise ValueError(f"task_ids length {len(task_ids)} != batch {B}")
    spec_map = {t.name: t for t in tasks}
    for name in set(task_ids):
        if name not in spec_map:
            raise KeyError(f"sample task {name!r} has no TaskSpec; have {list(spec_map)}")
        if name not in outputs:
            raise KeyError(f"model produced no output for task {name!r}")

    device = labels.device
    total = torch.zeros((), device=device)
    weight_sum = 0.0
    parts: dict[str, float] = {}

    for name in spec_map:
        rows = [i for i, t in enumerate(task_ids) if t == name]
        if not rows:
            continue
        idx = torch.tensor(rows, device=device, dtype=torch.long)
        cl = outputs[name].index_select(0, idx)                  # (n, K-1)
        tg = labels.index_select(0, idx)                         # (n,)
        l = ordinal_loss(cl, tg, reduction="mean")
        w = spec_map[name].loss_weight
        total = total + w * l
        weight_sum += w
        parts[name] = float(l.detach())

    if weight_sum == 0.0:
        raise ValueError("no supervised tasks in batch (all weights zero?)")
    return total / weight_sum, parts
