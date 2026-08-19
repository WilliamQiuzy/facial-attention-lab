"""Oo MLP-Mixer minimal HB classifier — baseline competitor to HBHead.

Design choice: this is the **simplest possible** HB classifier built on top of
Oo et al. 2025's frozen MLP-Mixer encoder. No MLP capacity, no attention, no
cross-action mixing — just `Linear(768, n_classes)` on top of mean-pooled
embeddings.

Why this exists as a separate baseline (instead of just deleting HBHead):
  - If HBHead beats OoMixerForHB by a wide margin → the extra MLP+attention
    capacity is actually doing work for our cohort size.
  - If they tie → simpler is better, drop HBHead's complexity.
  - This is the standard "linear probe" baseline in representation-learning
    papers; reporting it is good hygiene.

# Interface alignment with HBHead

Input is the same `(B, n_actions, n_frames, 768)` + optional `frame_mask` and
`action_mask` that HBHead consumes — see `src/datasets/patient_videos.py`.
This means the same `PatientVideoDataset` + `train_hb_kfold` infrastructure
works unchanged. You can A/B HBHead vs OoMixerForHB by just swapping the
model_factory passed to `train_hb_kfold`.

# Linear-probe training (the canonical use)

The Oo MLP-Mixer backbone is already frozen and pre-computed into per-take
`.npz` files by `scripts/preprocess.py` — it never appears in the training
loop. So "linear probing" here is literally: feed cached 768-d vectors →
mean-pool → `Linear(768, 6)` → cross-entropy. CPU is plenty.

Total trainable parameters: 768 * 6 + 6 = 4,614.

# Full fine-tune (optional, GPU territory)

If you ever want to fine-tune the Oo backbone too, that's a different code
path: re-run preprocessing with the backbone unfrozen, batched on GPU. Not
this module's job — this module assumes embeddings are cached.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class OoMixerForHBConfig:
    embed_dim: int = 768
    n_classes: int = 6


class OoMixerForHB(nn.Module):
    def __init__(self, cfg: OoMixerForHBConfig | None = None):
        super().__init__()
        self.cfg = cfg or OoMixerForHBConfig()
        self.classifier = nn.Linear(self.cfg.embed_dim, self.cfg.n_classes)

    # ------------------------------------------------------------------
    # Pooling helpers (mirrors HBHead._masked_mean to keep semantics aligned)
    # ------------------------------------------------------------------
    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
        m = mask.to(x.dtype)
        s = (x * m.unsqueeze(-1)).sum(dim=dim)
        denom = m.sum(dim=dim).clamp_min(1.0).unsqueeze(-1)
        return s / denom

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, action_emb: torch.Tensor,
                action_mask: torch.Tensor | None = None) -> torch.Tensor:
        """action_emb: (B, n_actions, embed_dim)."""
        if action_emb.dim() != 3:
            raise ValueError(f"expected (B, n_actions, D), got {tuple(action_emb.shape)}")
        if action_mask is None:
            action_mask = torch.ones(action_emb.shape[:2], dtype=torch.bool,
                                      device=action_emb.device)
        pooled = self._masked_mean(action_emb, action_mask, dim=1)  # (B, D)
        return self.classifier(pooled)

    def forward_with_frames(self, frame_emb: torch.Tensor,
                             frame_mask: torch.Tensor | None = None) -> torch.Tensor:
        """frame_emb: (B, n_actions, n_frames, embed_dim).
        Mean-pools frames per action, then mean-pools across present actions."""
        if frame_emb.dim() != 4:
            raise ValueError(
                f"expected (B, n_actions, n_frames, D), got {tuple(frame_emb.shape)}"
            )
        if frame_mask is None:
            frame_mask = torch.ones(frame_emb.shape[:3], dtype=torch.bool,
                                     device=frame_emb.device)
        action_emb = self._masked_mean(frame_emb, frame_mask, dim=2)
        action_present = frame_mask.any(dim=2)
        return self.forward(action_emb, action_present)
