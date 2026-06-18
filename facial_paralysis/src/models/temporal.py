"""Trainable temporal encoder over MediaPipe per-frame feature sequences.

This is the geometric-dynamics stream of docs/model_design.md §3.2 / §5.0. It
consumes a per-action sequence of MediaPipe-derived features (52 blendshapes +
left/right asymmetry features) and emits one fixed-width `dynamics_vec` per
action, capturing the *trajectory* of facial asymmetry (synkinesis, asymmetric
onset/peak, incomplete eye closure) — the clinically diagnostic motion signal.

Unlike MARLIN (frozen, cached), this module is **trained** end-to-end with the
severity head, so the model learns the asymmetry dynamics directly from our data.

Input  : x (B, T, F) float, mask (B, T) bool (True = real frame)
Output : (B, H_out) — pooled GRU outputs (zeros for all-padding rows)

Temporal pooling (`pool`) over the per-frame GRU outputs is a first-class knob,
not a hard-coded mean. This matters clinically: the diagnostic facial-palsy
signal is often a *transient* event (a brief incomplete eye closure during a
blink). A masked **mean** over a ~3 s clip dilutes exactly that peak; **max**
("peak") pooling keeps the most-asymmetric frame, and **attention** pooling
learns which frames matter. See docs/model_design.md §5.0 / training_runs.md
Run #7 (mean-pooling washed out YFP blink dynamics) and Run #9 (this ablation).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

POOL_MODES = ("mean", "max", "attention")


class TemporalLandmarkEncoder(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 128,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.1,
        pool: str = "mean",
    ):
        super().__init__()
        if feat_dim < 1:
            raise ValueError(f"feat_dim must be >= 1, got {feat_dim}")
        if pool not in POOL_MODES:
            raise ValueError(f"pool must be one of {POOL_MODES}, got {pool!r}")
        self.feat_dim = feat_dim
        self.out_dim = out_dim
        self.pool = pool
        self.gru = nn.GRU(
            input_size=feat_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        gru_out = hidden_dim * (2 if bidirectional else 1)
        # Attention pooling: a small per-frame scorer over the GRU outputs.
        self.attn = nn.Linear(gru_out, 1) if pool == "attention" else None
        self.proj = nn.Sequential(
            nn.LayerNorm(gru_out),
            nn.Linear(gru_out, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x (B, T, C), mask (B, T) bool -> (B, C). All-padding rows -> zeros."""
        m = mask.to(x.dtype).unsqueeze(-1)                 # (B, T, 1)
        s = (x * m).sum(dim=1)                             # (B, C)
        denom = m.sum(dim=1).clamp_min(1.0)                # (B, 1)
        return s / denom

    @staticmethod
    def _masked_max(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x (B, T, C), mask (B, T) bool -> (B, C) elementwise max over real
        frames. Padding is set to -inf so it never wins; all-padding rows are
        forced to 0 (their values are unused — the caller zeroes empties)."""
        neg = torch.finfo(x.dtype).min
        filled = x.masked_fill(~mask.unsqueeze(-1), neg)   # (B, T, C)
        pooled = filled.amax(dim=1)                        # (B, C)
        empty = ~mask.any(dim=1)                           # (B,)
        return pooled.masked_fill(empty.unsqueeze(-1), 0.0)

    def _masked_attention(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x (B, T, C), mask (B, T) bool -> (B, C). Softmax over real frames of a
        learned per-frame score; all-padding rows -> zeros."""
        scores = self.attn(x).squeeze(-1)                  # (B, T)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B, T, 1)
        pooled = (x * weights).sum(dim=1)                  # (B, C)
        empty = ~mask.any(dim=1)
        return pooled.masked_fill(empty.unsqueeze(-1), 0.0)

    def _pool_time(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.pool == "mean":
            return self._masked_mean(x, mask)
        if self.pool == "max":
            return self._masked_max(x, mask)
        return self._masked_attention(x, mask)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"expected (B, T, F), got {tuple(x.shape)}")
        if x.shape[-1] != self.feat_dim:
            raise ValueError(f"feat_dim mismatch: got {x.shape[-1]}, expected {self.feat_dim}")
        B, T, _ = x.shape
        if mask is None:
            mask = torch.ones(B, T, dtype=torch.bool, device=x.device)
        elif mask.shape != (B, T):
            raise ValueError(f"mask shape {tuple(mask.shape)} != {(B, T)}")
        elif mask.dtype != torch.bool:
            raise TypeError(f"mask must be bool, got {mask.dtype}")

        # Use packing so the (bidirectional) GRU genuinely ignores padded frames.
        # Zeroing the input is NOT enough: the backward pass would otherwise carry
        # padding into valid timesteps' hidden states.
        x = x * mask.to(x.dtype).unsqueeze(-1)
        lengths = mask.sum(dim=1)                           # (B,)
        empty = lengths == 0
        lengths_clamped = lengths.clamp(min=1).to("cpu")    # pack requires len >= 1, on CPU
        packed = pack_padded_sequence(x, lengths_clamped, batch_first=True, enforce_sorted=False)
        out_packed, _ = self.gru(packed)
        out, _ = pad_packed_sequence(out_packed, batch_first=True, total_length=T)  # (B, T, gru_out)
        pooled = self._pool_time(out, mask)                 # (B, gru_out); empty rows -> 0
        result = self.proj(pooled)                          # (B, out_dim)
        # Absent actions (no real frames) contribute nothing downstream; force a
        # clean zero so they are unaffected by the proj bias/LayerNorm.
        return result.masked_fill(empty.unsqueeze(-1), 0.0)
