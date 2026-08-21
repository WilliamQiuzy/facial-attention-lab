"""House-Brackmann grade prediction head over per-patient action embeddings.

Input shape (per patient):
    (n_actions, n_frames, embed_dim)        # variable n_frames per action OK if padded
or
    (n_actions, embed_dim)                  # already mean-pooled over frames

Output:
    (n_classes,) logits — default n_classes=6 for HB Grades I..VI.

Design rationale at this stage (small cohort, ~30 patients eventually):
  - Frame-level pooling: mean (parameter-free). When n_frames varies across
    actions, we accept a mask. Attention pool is an easy drop-in later but
    needs more data to train.
  - Per-action MLP: a shared two-layer projector compresses the 768-d backbone
    feature to a 256-d action representation. Shared (not per-action) to avoid
    blowing up parameter count.
  - Action-level pooling: configurable {mean, max, attention}. Default mean.
    Attention adds ~1k params, fine to enable when N grows.
  - Classifier: a single Linear(256 → 6). We treat HB as 6-way classification
    with cross-entropy. The grading is ordinal, but ordinal-regression heads
    (CORN / CORAL) typically do not beat plain CE on tiny cohorts. Switching
    is a one-line change at the loss site, not in the head.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HBHeadConfig:
    embed_dim: int = 768
    hidden_dim: int = 256
    n_actions: int = 5              # HB standard 5 poses: rest, brow_raise, light_eye_closure, forced_eye_closure, smile
    n_classes: int = 6              # HB Grades I..VI
    dropout: float = 0.2
    action_pool: Literal["mean", "max", "attention"] = "mean"


class HBHead(nn.Module):
    """Patient-level House-Brackmann grade classifier.

    Forward signatures (B = batch / patients):
      forward(action_emb, action_mask=None):
          action_emb:  (B, n_actions, embed_dim)            already-pooled per action
          action_mask: (B, n_actions) bool, True = action present (e.g. patient
                       missing one of the 6-7 videos). Optional.
      forward_with_frames(frame_emb, frame_mask=None):
          frame_emb:   (B, n_actions, n_frames, embed_dim)  before pooling
          frame_mask:  (B, n_actions, n_frames) bool, True = real frame.
                       (We treat 0 real frames as a missing action.)
    """

    def __init__(self, cfg: HBHeadConfig | None = None):
        super().__init__()
        self.cfg = cfg or HBHeadConfig()
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
            # Bahdanau-style score per action: small MLP → 1 logit per action
            self.action_attn = nn.Sequential(
                nn.Linear(H, H // 2),
                nn.Tanh(),
                nn.Linear(H // 2, 1),
            )
        else:
            self.action_attn = None

        self.classifier = nn.Linear(H, self.cfg.n_classes)

    # ------------------------------------------------------------------
    # Pooling helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
        """mask: bool of broadcastable shape, True = keep."""
        m = mask.to(x.dtype)
        s = (x * m.unsqueeze(-1)).sum(dim=dim)
        denom = m.sum(dim=dim).clamp_min(1.0).unsqueeze(-1)
        return s / denom

    def _pool_actions(self, h: torch.Tensor, action_mask: torch.Tensor | None) -> torch.Tensor:
        """h: (B, n_actions, H) → (B, H)."""
        if action_mask is None:
            action_mask = torch.ones(h.shape[:2], dtype=torch.bool, device=h.device)
        if self.cfg.action_pool == "mean":
            return self._masked_mean(h, action_mask, dim=1)
        if self.cfg.action_pool == "max":
            very_negative = torch.finfo(h.dtype).min
            h_masked = h.masked_fill(~action_mask.unsqueeze(-1), very_negative)
            return h_masked.amax(dim=1)
        if self.cfg.action_pool == "attention":
            scores = self.action_attn(h).squeeze(-1)  # (B, n_actions)
            scores = scores.masked_fill(~action_mask, torch.finfo(scores.dtype).min)
            weights = F.softmax(scores, dim=1).unsqueeze(-1)  # (B, n_actions, 1)
            return (h * weights).sum(dim=1)
        raise ValueError(f"unknown action_pool: {self.cfg.action_pool}")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        action_emb: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """action_emb: (B, n_actions, embed_dim). Returns (B, n_classes) logits."""
        if action_emb.dim() != 3:
            raise ValueError(f"expected (B, n_actions, D), got {tuple(action_emb.shape)}")
        h = self.per_action_mlp(action_emb)              # (B, n_actions, H)
        pooled = self._pool_actions(h, action_mask)      # (B, H)
        return self.classifier(pooled)                    # (B, n_classes)

    def forward_with_frames(
        self,
        frame_emb: torch.Tensor,
        frame_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """frame_emb: (B, n_actions, n_frames, embed_dim).
        frame_mask: (B, n_actions, n_frames) bool, True = real frame.
        Mean-pools frames per action, then routes to forward()."""
        if frame_emb.dim() != 4:
            raise ValueError(
                f"expected (B, n_actions, n_frames, D), got {tuple(frame_emb.shape)}"
            )
        if frame_mask is None:
            frame_mask = torch.ones(frame_emb.shape[:3], dtype=torch.bool,
                                     device=frame_emb.device)
        # mean over frames per action
        action_emb = self._masked_mean(frame_emb, frame_mask, dim=2)  # (B, n_actions, D)
        action_present = frame_mask.any(dim=2)                         # (B, n_actions)
        return self.forward(action_emb, action_present)
