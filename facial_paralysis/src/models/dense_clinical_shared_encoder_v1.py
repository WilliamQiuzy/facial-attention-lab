"""One source-blind dense-clinical encoder with a universal binary head."""
from __future__ import annotations

import torch
from torch import nn


ACTION_VOCAB = (
    "FREE_EARLY",
    "FREE_MID_EARLY",
    "FREE_MID_LATE",
    "FREE_LATE",
    "LIP_PUCKER",
    "MOUTH_OPEN",
    "SMILE_SPREAD",
    "BROW_RAISE",
    "EYE_GENTLE",
    "EYE_FORCEFUL",
    "SMILE_GENTLE",
    "SMILE_FULL",
    "SHOW_BOTTOM_TEETH",
)
TASK_HEAD_COUNT = 3


class DenseClinicalSharedEncoder(nn.Module):
    """Fuse optional dense trajectories with required clinical action tokens."""

    def __init__(self, *, use_dense: bool, model_dim: int = 64):
        super().__init__()
        if type(use_dense) is not bool or model_dim != 64:
            raise ValueError("v1 requires bool use_dense and model_dim=64")
        self.use_dense = use_dense
        self.patient_dim = model_dim
        self.clinical_encoder = nn.Sequential(
            nn.Linear(110, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        self.clinical_pair_projection = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
        )
        self.dense_spatial = nn.Sequential(
            nn.Linear(478 * 3, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, model_dim),
        )
        self.dense_temporal = nn.Sequential(
            nn.Conv1d(model_dim, model_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(model_dim, model_dim, 3, padding=1),
            nn.GELU(),
        )
        self.dense_projection = nn.Linear(model_dim * 2, model_dim)
        self.dense_pair_projection = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
        )
        self.fusion_gate = nn.Linear(model_dim * 2, model_dim)
        self.action_embedding = nn.Embedding(len(ACTION_VOCAB), model_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=4,
            dim_feedforward=model_dim * 2,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.action_encoder = nn.TransformerEncoder(
            layer, num_layers=2, enable_nested_tensor=False
        )
        self.patient_projection = nn.Linear(model_dim * 2, model_dim)
        self.patient_norm = nn.LayerNorm(model_dim)
        self.universal_head = nn.Linear(model_dim, 1)
        self.task_heads = nn.ModuleList(
            nn.Linear(model_dim, 1) for _ in range(TASK_HEAD_COUNT)
        )

    def task_logits_from_embedding(
        self,
        patient_embedding: torch.Tensor,
        task_codes: torch.Tensor,
    ) -> torch.Tensor:
        """Apply only the tiny task head after the common patient embedding."""
        if (
            not isinstance(patient_embedding, torch.Tensor)
            or patient_embedding.ndim != 2
            or patient_embedding.shape[1] != self.patient_dim
            or not patient_embedding.is_floating_point()
            or not bool(torch.isfinite(patient_embedding).all())
            or not isinstance(task_codes, torch.Tensor)
            or task_codes.shape != (patient_embedding.shape[0],)
            or task_codes.dtype != torch.long
            or task_codes.device != patient_embedding.device
            or bool((task_codes < 0).any())
            or bool((task_codes >= TASK_HEAD_COUNT).any())
        ):
            raise ValueError("task routing requires one valid code per shared embedding")
        all_logits = torch.stack(
            [head(patient_embedding).squeeze(-1) for head in self.task_heads],
            dim=1,
        )
        return all_logits.gather(1, task_codes[:, None]).squeeze(1)

    @staticmethod
    def _validate_inputs(
        clinical_original: torch.Tensor,
        clinical_mirrored: torch.Tensor,
        dense_original: torch.Tensor,
        dense_mirrored: torch.Tensor,
        dense_valid_mask: torch.Tensor,
        dense_available: torch.Tensor,
        action_mask: torch.Tensor,
        action_codes: torch.Tensor,
    ) -> tuple[int, int]:
        if not isinstance(clinical_original, torch.Tensor) or clinical_original.ndim != 3:
            raise ValueError("clinical tokens must be a floating (batch, actions, 110) tensor")
        batch, actions, dimension = clinical_original.shape
        if (
            batch < 1
            or actions < 1
            or dimension != 110
            or not clinical_original.is_floating_point()
            or not bool(torch.isfinite(clinical_original).all())
            or not isinstance(clinical_mirrored, torch.Tensor)
            or clinical_mirrored.shape != clinical_original.shape
            or not clinical_mirrored.is_floating_point()
            or not bool(torch.isfinite(clinical_mirrored).all())
            or clinical_mirrored.device != clinical_original.device
            or not isinstance(dense_original, torch.Tensor)
            or dense_original.shape != (batch, actions, 32, 478, 3)
            or not dense_original.is_floating_point()
            or not bool(torch.isfinite(dense_original).all())
            or dense_original.device != clinical_original.device
            or not isinstance(dense_mirrored, torch.Tensor)
            or dense_mirrored.shape != dense_original.shape
            or not dense_mirrored.is_floating_point()
            or not bool(torch.isfinite(dense_mirrored).all())
            or dense_mirrored.device != clinical_original.device
            or dense_valid_mask.shape != (batch, actions, 32)
            or dense_valid_mask.dtype != torch.bool
            or dense_valid_mask.device != clinical_original.device
            or dense_available.shape != (batch, actions)
            or dense_available.dtype != torch.bool
            or dense_available.device != clinical_original.device
            or action_mask.shape != (batch, actions)
            or action_mask.dtype != torch.bool
            or action_mask.device != clinical_original.device
            or action_codes.shape != (batch, actions)
            or action_codes.dtype != torch.long
            or action_codes.device != clinical_original.device
            or bool((action_codes < 0).any())
            or bool((action_codes >= len(ACTION_VOCAB)).any())
        ):
            raise ValueError("dense-clinical model inputs differ from the v1 contract")
        if bool((action_mask.sum(dim=1) == 0).any()):
            raise ValueError("every patient requires at least one action")
        if bool((dense_available & ~action_mask).any()):
            raise ValueError("padded actions cannot expose dense evidence")
        if bool((dense_valid_mask & ~dense_available.unsqueeze(-1)).any()):
            raise ValueError("unavailable dense evidence cannot mark frames valid")
        if bool(dense_available.any()) and bool(
            (dense_valid_mask.sum(dim=-1)[dense_available] < 2).any()
        ):
            raise ValueError("available dense actions require at least two frames")
        return batch, actions

    def _dense_tokens(
        self,
        dense: torch.Tensor,
        dense_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, actions = dense.shape[:2]
        frame_mask = dense_valid_mask.reshape(batch * actions, 32)
        flattened = dense.reshape(batch * actions, 32, 478 * 3)
        flattened = flattened * frame_mask.unsqueeze(-1).to(flattened.dtype)
        spatial = self.dense_spatial(flattened)
        temporal = self.dense_temporal(spatial.transpose(1, 2)).transpose(1, 2)
        weights = frame_mask.unsqueeze(-1).to(temporal.dtype)
        mean = (temporal * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        masked = temporal.masked_fill(~frame_mask.unsqueeze(-1), float("-inf"))
        maximum = masked.max(dim=1).values
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        return self.dense_projection(torch.cat((mean, maximum), dim=-1)).reshape(
            batch, actions, self.patient_dim
        )

    def encode(
        self,
        clinical_original: torch.Tensor,
        clinical_mirrored: torch.Tensor,
        dense_original: torch.Tensor,
        dense_mirrored: torch.Tensor,
        dense_valid_mask: torch.Tensor,
        dense_available: torch.Tensor,
        action_mask: torch.Tensor,
        action_codes: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_inputs(
            clinical_original,
            clinical_mirrored,
            dense_original,
            dense_mirrored,
            dense_valid_mask,
            dense_available,
            action_mask,
            action_codes,
        )
        clinical_first = self.clinical_encoder(clinical_original)
        clinical_second = self.clinical_encoder(clinical_mirrored)
        clinical_token = self.clinical_pair_projection(torch.cat((
            0.5 * (clinical_first + clinical_second),
            torch.abs(clinical_first - clinical_second),
        ), dim=-1))
        if self.use_dense:
            dense_first = self._dense_tokens(dense_original, dense_valid_mask)
            dense_second = self._dense_tokens(dense_mirrored, dense_valid_mask)
            dense_token = self.dense_pair_projection(torch.cat((
                0.5 * (dense_first + dense_second),
                torch.abs(dense_first - dense_second),
            ), dim=-1))
            pair = torch.cat((clinical_token, dense_token), dim=-1)
            gate = torch.sigmoid(self.fusion_gate(pair))
            fused = clinical_token + gate * (dense_token - clinical_token)
            tokens = torch.where(
                dense_available.unsqueeze(-1), fused, clinical_token
            )
        else:
            tokens = clinical_token
        tokens = tokens + self.action_embedding(action_codes)
        encoded = self.action_encoder(tokens, src_key_padding_mask=~action_mask)
        weights = action_mask.unsqueeze(-1).to(encoded.dtype)
        mean = (encoded * weights).sum(dim=1) / weights.sum(dim=1)
        maximum = encoded.masked_fill(
            ~action_mask.unsqueeze(-1), float("-inf")
        ).max(dim=1).values
        patient = self.patient_projection(torch.cat((mean, maximum), dim=-1))
        return self.patient_norm(patient)

    def forward(
        self,
        clinical_original: torch.Tensor,
        clinical_mirrored: torch.Tensor,
        dense_original: torch.Tensor,
        dense_mirrored: torch.Tensor,
        dense_valid_mask: torch.Tensor,
        dense_available: torch.Tensor,
        action_mask: torch.Tensor,
        action_codes: torch.Tensor,
    ) -> torch.Tensor:
        return self.universal_head(self.encode(
            clinical_original,
            clinical_mirrored,
            dense_original,
            dense_mirrored,
            dense_valid_mask,
            dense_available,
            action_mask,
            action_codes,
        )).squeeze(-1)


__all__ = ["ACTION_VOCAB", "TASK_HEAD_COUNT", "DenseClinicalSharedEncoder"]
