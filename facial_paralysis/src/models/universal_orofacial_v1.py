"""Small source-blind and auxiliary-head models for universal orofacial research."""
from __future__ import annotations

import torch
from torch import nn


FEATURE_DIM = 110
LOW_RANK_DIM = 16
SOURCE_COUNT = 2


class UniversalLowRankModel(nn.Module):
    """A compact shared trunk whose universal forward never receives source ID."""

    def __init__(self, *, auxiliary_heads: bool):
        super().__init__()
        self.auxiliary_heads_enabled = bool(auxiliary_heads)
        self.encoder = nn.Linear(FEATURE_DIM, LOW_RANK_DIM)
        self.universal_head = nn.Linear(LOW_RANK_DIM, 1)
        self.auxiliary_heads = (
            nn.ModuleList([nn.Linear(LOW_RANK_DIM, 1) for _ in range(SOURCE_COUNT)])
            if self.auxiliary_heads_enabled else nn.ModuleList()
        )

    @staticmethod
    def _validate_features(features: torch.Tensor) -> None:
        if (
            not isinstance(features, torch.Tensor)
            or features.ndim != 2
            or features.shape[1] != FEATURE_DIM
            or not features.is_floating_point()
            or not bool(torch.isfinite(features).all())
        ):
            raise ValueError("universal model requires finite floating (n, 110) features")

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        self._validate_features(features)
        return torch.tanh(self.encoder(features))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return the source-blind universal logit."""
        return self.universal_head(self.encode(features)).squeeze(-1)

    def auxiliary_logits(
        self,
        features: torch.Tensor,
        source_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Route training-only auxiliary predictions to the two endpoint heads."""
        if not self.auxiliary_heads_enabled:
            raise ValueError("auxiliary heads are disabled for this candidate")
        latent = self.encode(features)
        if (
            not isinstance(source_indices, torch.Tensor)
            or source_indices.shape != (features.shape[0],)
            or source_indices.dtype != torch.int64
            or source_indices.device != features.device
            or not bool(((source_indices == 0) | (source_indices == 1)).all())
        ):
            raise ValueError("auxiliary routing requires aligned source indices 0 or 1")
        logits = torch.empty(
            features.shape[0], dtype=features.dtype, device=features.device
        )
        for source_index, head in enumerate(self.auxiliary_heads):
            selected = source_indices == source_index
            if bool(selected.any()):
                logits[selected] = head(latent[selected]).squeeze(-1)
        return logits
