"""Medically gated shared 478D+110D action encoder."""
from __future__ import annotations

import torch
from torch import nn

from .dense_clinical_shared_encoder_v1 import ACTION_VOCAB, TASK_HEAD_COUNT
from .medical_shared_candidate_registry_v2 import SharedCandidateV2, candidate_registry


BROW_LANDMARKS = tuple(sorted({
    276, 283, 282, 295, 285, 300, 293, 334, 296, 336,
    46, 53, 52, 65, 55, 70, 63, 105, 66, 107,
}))
EYE_LANDMARKS = tuple(sorted({
    263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387,
    386, 385, 384, 398, 33, 7, 163, 144, 145, 153, 154, 155, 133,
    246, 161, 160, 159, 158, 157, 173,
}))
MOUTH_LANDMARKS = tuple(sorted({
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185, 40,
    39, 37, 0, 267, 269, 270, 409, 78, 95, 88, 178, 87, 14, 317,
    402, 318, 324, 308, 191, 80, 81, 82, 13, 312, 311, 310, 415,
}))
_REGIONS = (
    BROW_LANDMARKS,
    EYE_LANDMARKS,
    MOUTH_LANDMARKS,
    tuple(range(478)),
)


class MedicallyGatedSharedEncoderV2(nn.Module):
    """One source-blind trunk; endpoint routing begins after patient embedding."""

    def __init__(self, candidate: SharedCandidateV2, *, model_dim: int = 64):
        super().__init__()
        if (
            type(candidate) is not SharedCandidateV2
            or candidate not in candidate_registry()
            or model_dim != 64
        ):
            raise ValueError("v2 requires one exact frozen candidate and model_dim=64")
        self.candidate = candidate
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
        self.regional_encoder = nn.Sequential(
            nn.Linear(32, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        fusion_input = model_dim * 3 + 2
        self.masked_concat = nn.Sequential(
            nn.Linear(fusion_input, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
        )
        self.reliability_gate = nn.Linear(fusion_input, 2)
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

    @staticmethod
    def _validate_inputs(
        clinical_original: torch.Tensor,
        clinical_mirrored: torch.Tensor,
        dense_original: torch.Tensor,
        dense_mirrored: torch.Tensor,
        dense_valid_mask: torch.Tensor,
        dense_available: torch.Tensor,
        dense_timestamps: torch.Tensor,
        action_mask: torch.Tensor,
        action_codes: torch.Tensor,
    ) -> tuple[int, int]:
        if not isinstance(clinical_original, torch.Tensor) or clinical_original.ndim != 3:
            raise ValueError("clinical tokens require batch, action and 110D axes")
        batch, actions, dimension = clinical_original.shape
        device = clinical_original.device
        if (
            batch < 1
            or actions < 1
            or dimension != 110
            or not clinical_original.is_floating_point()
            or not bool(torch.isfinite(clinical_original).all())
            or clinical_mirrored.shape != clinical_original.shape
            or not clinical_mirrored.is_floating_point()
            or clinical_mirrored.device != device
            or not bool(torch.isfinite(clinical_mirrored).all())
            or dense_original.shape != (batch, actions, 32, 478, 3)
            or not dense_original.is_floating_point()
            or dense_original.device != device
            or not bool(torch.isfinite(dense_original).all())
            or dense_mirrored.shape != dense_original.shape
            or not dense_mirrored.is_floating_point()
            or dense_mirrored.device != device
            or not bool(torch.isfinite(dense_mirrored).all())
            or dense_valid_mask.shape != (batch, actions, 32)
            or dense_valid_mask.dtype != torch.bool
            or dense_valid_mask.device != device
            or dense_available.shape != (batch, actions)
            or dense_available.dtype != torch.bool
            or dense_available.device != device
            or dense_timestamps.shape != (batch, actions, 32)
            or not dense_timestamps.is_floating_point()
            or dense_timestamps.device != device
            or not bool(torch.isfinite(dense_timestamps).all())
            or action_mask.shape != (batch, actions)
            or action_mask.dtype != torch.bool
            or action_mask.device != device
            or action_codes.shape != (batch, actions)
            or action_codes.dtype != torch.long
            or action_codes.device != device
            or bool((action_codes < 0).any())
            or bool((action_codes >= len(ACTION_VOCAB)).any())
        ):
            raise ValueError("shared v2 inputs differ from the closed tensor contract")
        if bool((action_mask.sum(dim=1) == 0).any()):
            raise ValueError("every participant requires at least one action")
        if bool((dense_available & ~action_mask).any()):
            raise ValueError("padded actions cannot expose dense evidence")
        if bool((dense_valid_mask & ~dense_available.unsqueeze(-1)).any()):
            raise ValueError("unavailable dense evidence cannot mark frames valid")
        if bool(dense_available.any()):
            if bool((dense_valid_mask.sum(dim=-1)[dense_available] != 32).any()):
                raise ValueError("dense actions require the full interpolated grid")
            intervals = torch.diff(dense_timestamps, dim=-1)
            if bool((intervals[dense_available] <= 0).any()):
                raise ValueError("dense velocity requires strictly increasing seconds")
        return batch, actions

    def _clinical_tokens(
        self,
        original: torch.Tensor,
        mirrored: torch.Tensor,
    ) -> torch.Tensor:
        first = self.clinical_encoder(original)
        if self.candidate.view_mode == "original_only":
            return first
        second = self.clinical_encoder(mirrored)
        return self.clinical_pair_projection(torch.cat((
            0.5 * (first + second), torch.abs(first - second)
        ), dim=-1))

    def _dense_tokens(self, dense: torch.Tensor) -> torch.Tensor:
        batch, actions = dense.shape[:2]
        spatial = self.dense_spatial(dense.reshape(batch * actions, 32, 478 * 3))
        temporal = self.dense_temporal(spatial.transpose(1, 2)).transpose(1, 2)
        pooled = torch.cat((temporal.mean(dim=1), temporal.max(dim=1).values), dim=-1)
        return self.dense_projection(pooled).reshape(batch, actions, self.patient_dim)

    @staticmethod
    def _stream_region_statistics(
        dense: torch.Tensor,
        timestamps: torch.Tensor,
        indices: tuple[int, ...],
    ) -> torch.Tensor:
        selected = dense[:, :, :, list(indices), :]
        excursion = torch.linalg.vector_norm(selected, dim=-1)
        mean_excursion = excursion.mean(dim=(2, 3))
        peak_excursion = excursion.amax(dim=(2, 3))
        intervals = torch.diff(timestamps, dim=-1).clamp_min(1e-6)
        displacement = torch.linalg.vector_norm(
            selected[:, :, 1:] - selected[:, :, :-1], dim=-1
        )
        velocity = displacement / intervals.unsqueeze(-1)
        mean_velocity = velocity.mean(dim=(2, 3))
        peak_velocity = velocity.amax(dim=(2, 3))
        return torch.stack(
            (mean_excursion, peak_excursion, mean_velocity, peak_velocity),
            dim=-1,
        )

    def regional_evidence(
        self,
        dense_original: torch.Tensor,
        dense_mirrored: torch.Tensor,
        dense_timestamps: torch.Tensor,
        action_codes: torch.Tensor,
    ) -> torch.Tensor:
        if (
            dense_original.ndim != 5
            or dense_original.shape[-3:] != (32, 478, 3)
            or dense_mirrored.shape != dense_original.shape
            or dense_timestamps.shape != dense_original.shape[:2] + (32,)
            or action_codes.shape != dense_original.shape[:2]
        ):
            raise ValueError("regional evidence received malformed action tensors")
        blocks = []
        for indices in _REGIONS:
            first = self._stream_region_statistics(
                dense_original, dense_timestamps, indices
            )
            if self.candidate.view_mode == "bilateral_invariant":
                second = self._stream_region_statistics(
                    dense_mirrored, dense_timestamps, indices
                )
                block = torch.cat((
                    0.5 * (first + second), torch.abs(first - second)
                ), dim=-1)
            else:
                block = torch.cat((first, torch.zeros_like(first)), dim=-1)
            blocks.append(block)
        evidence = torch.cat(blocks, dim=-1)
        mode = self.candidate.regional_mode
        if mode == "none":
            return torch.zeros_like(evidence)
        if mode != "matched_excursion_velocity":
            velocity_positions = torch.tensor(
                [2, 3, 6, 7], device=evidence.device, dtype=torch.long
            )
            shaped = evidence.reshape(*evidence.shape[:2], 4, 8).clone()
            shaped.index_fill_(-1, velocity_positions, 0.0)
            evidence = shaped.reshape_as(evidence)
        if mode.startswith("matched_"):
            region_mask = torch.zeros(
                *action_codes.shape, 4, dtype=evidence.dtype, device=evidence.device
            )
            region_mask[..., 3] = 1.0
            brow = action_codes == ACTION_VOCAB.index("BROW_RAISE")
            eye = (action_codes == ACTION_VOCAB.index("EYE_GENTLE")) | (
                action_codes == ACTION_VOCAB.index("EYE_FORCEFUL")
            )
            oral_codes = tuple(ACTION_VOCAB.index(name) for name in (
                "LIP_PUCKER", "MOUTH_OPEN", "SMILE_SPREAD", "SMILE_GENTLE",
                "SMILE_FULL", "SHOW_BOTTOM_TEETH",
            ))
            oral = torch.zeros_like(action_codes, dtype=torch.bool)
            for code in oral_codes:
                oral |= action_codes == code
            region_mask[..., 0] = brow.to(evidence.dtype)
            region_mask[..., 1] = eye.to(evidence.dtype)
            region_mask[..., 2] = oral.to(evidence.dtype)
            evidence = (
                evidence.reshape(*evidence.shape[:2], 4, 8)
                * region_mask.unsqueeze(-1)
            ).reshape_as(evidence)
        return evidence

    def _fused_tokens(
        self,
        clinical: torch.Tensor,
        dense: torch.Tensor,
        regional: torch.Tensor,
        dense_available: torch.Tensor,
    ) -> torch.Tensor:
        dense_mask = dense_available.unsqueeze(-1).to(clinical.dtype)
        region_enabled = self.candidate.regional_mode != "none"
        region_mask = dense_mask * float(region_enabled)
        dense = dense * dense_mask
        regional = regional * region_mask
        fusion = torch.cat((clinical, dense, regional, dense_mask, region_mask), dim=-1)
        if self.candidate.fusion_mode == "masked_concat":
            return self.masked_concat(fusion)
        gates = torch.sigmoid(self.reliability_gate(fusion))
        fused = clinical + dense_mask * gates[..., :1] * (dense - clinical)
        return fused + region_mask * gates[..., 1:] * (regional - fused)

    def encode(
        self,
        clinical_original: torch.Tensor,
        clinical_mirrored: torch.Tensor,
        dense_original: torch.Tensor,
        dense_mirrored: torch.Tensor,
        dense_valid_mask: torch.Tensor,
        dense_available: torch.Tensor,
        dense_timestamps: torch.Tensor,
        action_mask: torch.Tensor,
        action_codes: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_inputs(
            clinical_original, clinical_mirrored, dense_original, dense_mirrored,
            dense_valid_mask, dense_available, dense_timestamps, action_mask,
            action_codes,
        )
        clinical = self._clinical_tokens(clinical_original, clinical_mirrored)
        dense_first = self._dense_tokens(dense_original)
        if self.candidate.view_mode == "bilateral_invariant":
            dense_second = self._dense_tokens(dense_mirrored)
            dense = self.dense_pair_projection(torch.cat((
                0.5 * (dense_first + dense_second),
                torch.abs(dense_first - dense_second),
            ), dim=-1))
        else:
            dense = dense_first
        regional_values = self.regional_evidence(
            dense_original, dense_mirrored, dense_timestamps, action_codes
        )
        regional = self.regional_encoder(regional_values)
        tokens = self._fused_tokens(clinical, dense, regional, dense_available)
        tokens = tokens + self.action_embedding(action_codes)
        if self.candidate.pooling_mode == "cross_action_transformer":
            tokens = self.action_encoder(tokens, src_key_padding_mask=~action_mask)
        weights = action_mask.unsqueeze(-1).to(tokens.dtype)
        mean = (tokens * weights).sum(dim=1) / weights.sum(dim=1)
        maximum = tokens.masked_fill(
            ~action_mask.unsqueeze(-1), float("-inf")
        ).max(dim=1).values
        patient = self.patient_projection(torch.cat((mean, maximum), dim=-1))
        return self.patient_norm(patient)

    def task_logits_from_embedding(
        self,
        patient_embedding: torch.Tensor,
        task_codes: torch.Tensor,
    ) -> torch.Tensor:
        if (
            patient_embedding.ndim != 2
            or patient_embedding.shape[1] != self.patient_dim
            or task_codes.shape != (patient_embedding.shape[0],)
            or task_codes.dtype != torch.long
            or task_codes.device != patient_embedding.device
            or bool((task_codes < 0).any())
            or bool((task_codes >= TASK_HEAD_COUNT).any())
        ):
            raise ValueError("task heads require one valid code per shared embedding")
        logits = torch.stack(
            [head(patient_embedding).squeeze(-1) for head in self.task_heads], dim=1
        )
        return logits.gather(1, task_codes[:, None]).squeeze(1)

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        return self.universal_head(self.encode(*inputs)).squeeze(-1)


__all__ = [
    "BROW_LANDMARKS",
    "EYE_LANDMARKS",
    "MOUTH_LANDMARKS",
    "MedicallyGatedSharedEncoderV2",
]
