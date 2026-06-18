"""Gaber 2022 ASI → House-Brackmann head.

This is the simplest possible HB classifier built on top of Gaber's Animation
Symmetry Index (ASI) features. ASI is a deterministic feature transform with
NO learned parameters (see `src/baselines/gaber_fau/asi.py`) — there is no
"Gaber backbone" to fine-tune, and the only learnable component is this head.

Input layout matches the rest of `baselines_hb/` so the same
`PatientVideoDataset` + `train_hb_kfold` infrastructure works unchanged.

Inputs accepted:
  - `(B, n_actions, asi_dim)` already mean-pooled per action
  - `(B, n_actions, n_frames, asi_dim)` raw per-frame ASI

`asi_dim` is 3 by default (eyebrow, eye, mouth) or 4 if you include `total`.
Total is a linear combination of the other three (mean), so it adds no rank;
we keep it as a default-on option only because Gaber's paper reports it.

`n_actions` is 5 by default (rest + 4 movements: smile, closing-eyes, raising-
eyebrows, blowing-cheeks). Whistling is the 6th Gaber movement but is not in
the HB-standard pose set; expose it via `n_actions=6` if you have it.

Per-action features are mean-pooled and concatenated -> Linear(n_actions *
asi_dim, n_classes). Trainable parameter count for the default config:
5 * 4 * 6 + 6 = 126.

How it fits the project:

  1. Run `python -m src.baselines.gaber_fau.adapt_to_ours --mediapipe-root
     data/mediapipe_out --out outputs/asi_ours` to cache per-take ASI series.
  2. Segment per-take ASI into actions (Task 3 — pending clinician timestamps).
  3. Stack into `(B, n_actions, n_frames_padded, asi_dim)` and feed here.

Until segmentation lands, you can use the per-take mean ASI as a
single-action fallback (`n_actions=1`).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class GaberASIForHBConfig:
    asi_dim: int = 4            # eyebrow, eye, mouth, total (paper's 4)
    n_actions: int = 5          # rest, smile, eye-closure, brow-raise, cheeks
    n_classes: int = 6          # HB I..VI
    dropout: float = 0.0        # tiny model, default off


class GaberASIForHB(nn.Module):
    """Linear-probe HB head on Gaber ASI features.

    Forward signatures (B = batch / patients):
      forward(action_emb, action_mask=None):
          action_emb:  (B, n_actions, asi_dim)            already-pooled per action
          action_mask: (B, n_actions) bool, True = action present
      forward_with_frames(frame_emb, frame_mask=None):
          frame_emb:   (B, n_actions, n_frames, asi_dim)  pre-pool
          frame_mask:  (B, n_actions, n_frames) bool, True = real frame
    """

    def __init__(self, cfg: GaberASIForHBConfig | None = None):
        super().__init__()
        self.cfg = cfg or GaberASIForHBConfig()
        in_features = self.cfg.n_actions * self.cfg.asi_dim
        layers: list[nn.Module] = []
        if self.cfg.dropout > 0:
            layers.append(nn.Dropout(self.cfg.dropout))
        layers.append(nn.Linear(in_features, self.cfg.n_classes))
        self.classifier = nn.Sequential(*layers)

    def _flatten(self, action_emb: torch.Tensor,
                 action_mask: torch.Tensor | None) -> torch.Tensor:
        """(B, n_actions, asi_dim) → (B, n_actions*asi_dim), zero-masking missing actions."""
        if action_mask is not None:
            action_emb = action_emb * action_mask.unsqueeze(-1).to(action_emb.dtype)
        return action_emb.reshape(action_emb.shape[0], -1)

    def forward(self, action_emb: torch.Tensor,
                action_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self._flatten(action_emb, action_mask)
        return self.classifier(x)

    def forward_with_frames(self, frame_emb: torch.Tensor,
                            frame_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Mean-pool over frames within each action, then standard forward."""
        if frame_mask is None:
            pooled = frame_emb.mean(dim=2)
            action_mask = None
        else:
            mask = frame_mask.unsqueeze(-1).to(frame_emb.dtype)
            num = (frame_emb * mask).sum(dim=2)
            denom = mask.sum(dim=2).clamp_min(1.0)
            pooled = num / denom
            action_mask = frame_mask.any(dim=2)
        return self.forward(pooled, action_mask=action_mask)
