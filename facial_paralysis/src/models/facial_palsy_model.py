"""Full facial-paralysis severity model (docs/model_design.md §5).

Composes the two streams of §3 into the per-action embedding of §5.0, then runs
the coupled multi-task latent-severity head of §4:

    MARLIN clip vec(s)  ──(masked mean over windows)──┐
                                                      concat ─► SeverityTrunk ─► ordinal heads
    MediaPipe seq ──► TemporalLandmarkEncoder (GRU) ──┘

Only the temporal encoder + trunk + ordinal heads are trainable; MARLIN is frozen
and its embeddings arrive pre-computed (cached) in the input. This is the model
that consumes a `MultiStreamPatientDataset` batch.

Input tensors (B = patients, A = actions, W = MARLIN windows, T = frames):
    marlin_emb  (B, A, W, 768) float   — cached frozen MARLIN clip embeddings
    marlin_mask (B, A, W)       bool    — True = real window
    mp_seq      (B, A, T, F)    float   — MediaPipe per-frame feature sequence
    mp_mask     (B, A, T)       bool    — True = real frame
    action_mask (B, A)          bool    — True = action present (optional; derived if None)
Output: {task_name: cumulative_logits (B, K_task-1)} (see MultiTaskSeverityModel).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch
import torch.nn as nn

from src.models.multitask import (
    DEFAULT_TASKS,
    MultiTaskSeverityModel,
    TaskSpec,
    TrunkConfig,
)
from src.models.ordinal import predict_grade
from src.models.temporal import TemporalLandmarkEncoder

MARLIN_DIM = 768


@dataclass
class FacialPalsyConfig:
    mp_feat_dim: int = 72                 # 52 blendshapes + 20 L/R asymmetry pairs (MediaPipe)
    marlin_dim: int = MARLIN_DIM
    temporal_hidden: int = 128
    temporal_out: int = 128
    temporal_layers: int = 1
    trunk_hidden: int = 256
    n_actions: int = 5
    dropout: float = 0.2
    action_pool: str = "mean"
    # Temporal pooling of the GRU's per-frame outputs (geometric-dynamics stream):
    # "mean" dilutes transient events, "max" keeps the peak-asymmetry frame,
    # "attention" learns which frames matter. See temporal.py / Run #9.
    temporal_pool: str = "mean"
    # Pooling over MARLIN windows when a clip yields several 16-frame windows.
    # "mean" (default) or "max" (peak window). No effect when W == 1.
    marlin_window_pool: str = "mean"
    tasks: Sequence[TaskSpec] = field(default_factory=lambda: DEFAULT_TASKS)
    # Stream balancing: by default the 768-d MARLIN vector dwarfs the small
    # geometric/temporal vector in the concat, drowning the local asymmetry signal
    # (the eyes-head failure mode). Project MARLIN down and/or LayerNorm each stream
    # so the geometric evidence is not lost. Off by default (backward compatible).
    marlin_proj_dim: int | None = None
    stream_layernorm: bool = False


class FacialPalsyModel(nn.Module):
    def __init__(self, cfg: FacialPalsyConfig | None = None):
        super().__init__()
        self.cfg = cfg or FacialPalsyConfig()
        c = self.cfg

        if c.marlin_window_pool not in ("mean", "max"):
            raise ValueError(f"marlin_window_pool must be 'mean' or 'max', got {c.marlin_window_pool!r}")
        self.temporal = TemporalLandmarkEncoder(
            feat_dim=c.mp_feat_dim,
            hidden_dim=c.temporal_hidden,
            out_dim=c.temporal_out,
            num_layers=c.temporal_layers,
            dropout=c.dropout,
            pool=c.temporal_pool,
        )
        marlin_out = c.marlin_proj_dim or c.marlin_dim
        self.marlin_proj = nn.Linear(c.marlin_dim, c.marlin_proj_dim) if c.marlin_proj_dim else None
        if c.stream_layernorm:
            self.marlin_norm = nn.LayerNorm(marlin_out)
            self.dyn_norm = nn.LayerNorm(c.temporal_out)
        else:
            self.marlin_norm = self.dyn_norm = None
        embed_dim = marlin_out + c.temporal_out
        self.multitask = MultiTaskSeverityModel(
            tasks=c.tasks,
            trunk_cfg=TrunkConfig(
                embed_dim=embed_dim,
                hidden_dim=c.trunk_hidden,
                n_actions=c.n_actions,
                dropout=c.dropout,
                action_pool=c.action_pool,
            ),
        )
        self.embed_dim = embed_dim

    # ------------------------------------------------------------------
    @staticmethod
    def _masked_mean_windows(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x (B, A, W, D), mask (B, A, W) -> (B, A, D). All-absent -> zeros."""
        m = mask.to(x.dtype).unsqueeze(-1)
        s = (x * m).sum(dim=2)
        denom = m.sum(dim=2).clamp_min(1.0)
        return s / denom

    @staticmethod
    def _masked_max_windows(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x (B, A, W, D), mask (B, A, W) -> (B, A, D) peak over real windows.
        Padding -> -inf so it never wins; all-absent actions -> zeros."""
        neg = torch.finfo(x.dtype).min
        pooled = x.masked_fill(~mask.unsqueeze(-1), neg).amax(dim=2)   # (B, A, D)
        empty = ~mask.any(dim=2)                                       # (B, A)
        return pooled.masked_fill(empty.unsqueeze(-1), 0.0)

    def _pool_windows(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.cfg.marlin_window_pool == "max":
            return self._masked_max_windows(x, mask)
        return self._masked_mean_windows(x, mask)

    def build_action_embeddings(
        self,
        marlin_emb: torch.Tensor,
        marlin_mask: torch.Tensor,
        mp_seq: torch.Tensor,
        mp_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Assemble per-action embeddings (B, A, marlin_dim + temporal_out)."""
        if marlin_emb.dim() != 4 or marlin_emb.shape[-1] != self.cfg.marlin_dim:
            raise ValueError(
                f"marlin_emb must be (B,A,W,{self.cfg.marlin_dim}), got {tuple(marlin_emb.shape)}"
            )
        if mp_seq.dim() != 4 or mp_seq.shape[-1] != self.cfg.mp_feat_dim:
            raise ValueError(
                f"mp_seq must be (B,A,T,{self.cfg.mp_feat_dim}), got {tuple(mp_seq.shape)}"
            )
        B, A, W, _ = marlin_emb.shape
        if marlin_mask.shape != (B, A, W):
            raise ValueError(f"marlin_mask {tuple(marlin_mask.shape)} != {(B, A, W)}")
        Bm, Am, T, _ = mp_seq.shape
        if (Bm, Am) != (B, A):
            raise ValueError(f"mp_seq batch/action {(Bm, Am)} != {(B, A)}")
        if mp_mask.shape != (B, A, T):
            raise ValueError(f"mp_mask {tuple(mp_mask.shape)} != {(B, A, T)}")

        marlin_vec = self._pool_windows(marlin_emb, marlin_mask)          # (B, A, 768)
        if self.marlin_proj is not None:
            marlin_vec = self.marlin_proj(marlin_vec)                     # (B, A, proj)

        # Temporal encoder runs per (patient, action): flatten to (B*A, T, F).
        seq_flat = mp_seq.reshape(B * A, T, mp_seq.shape[-1])
        mask_flat = mp_mask.reshape(B * A, T)
        dyn = self.temporal(seq_flat, mask_flat).reshape(B, A, self.cfg.temporal_out)

        if self.marlin_norm is not None:                                  # balance streams
            marlin_vec = self.marlin_norm(marlin_vec)
            dyn = self.dyn_norm(dyn)
        return torch.cat([marlin_vec, dyn], dim=-1)                        # (B, A, D)

    @staticmethod
    def _derive_action_mask(marlin_mask: torch.Tensor, mp_mask: torch.Tensor) -> torch.Tensor:
        """An action is present if it has any real MARLIN window OR any real frame."""
        return marlin_mask.any(dim=2) | mp_mask.any(dim=2)

    def forward(
        self,
        marlin_emb: torch.Tensor,
        marlin_mask: torch.Tensor,
        mp_seq: torch.Tensor,
        mp_mask: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        action_emb = self.build_action_embeddings(marlin_emb, marlin_mask, mp_seq, mp_mask)
        if action_mask is None:
            action_mask = self._derive_action_mask(marlin_mask, mp_mask)
        return self.multitask(action_emb, action_mask)

    def predict_hb(self, *args, hb_task: str = "hb", **kwargs) -> torch.Tensor:
        return predict_grade(self.forward(*args, **kwargs)[hb_task])
