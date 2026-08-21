"""Low-sample shared action-mechanism router for V9."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn


@dataclass(frozen=True)
class SharedMechanismCandidateV9:
    candidate_id: str
    include_dense: bool
    model_dim: int
    pooling: str
    depth: int
    medical_rationale: str


def candidate_registry_v9() -> tuple[SharedMechanismCandidateV9, ...]:
    rows = []
    index = 0
    for include_dense in (False, True):
        for model_dim in (32, 64):
            for pooling in ("meanmax", "task_attention"):
                for depth in (1, 2):
                    evidence = (
                        "110D action geometry plus translation-referenced regional "
                        "excursion and velocity"
                        if include_dense else "the frozen 110D action geometry"
                    )
                    pool = (
                        "source-specific output queries weight the shared action tokens"
                        if pooling == "task_attention"
                        else "source-blind mean and maximum pool the shared action tokens"
                    )
                    rows.append(SharedMechanismCandidateV9(
                        candidate_id=f"SMR9-{index:03d}",
                        include_dense=include_dense,
                        model_dim=model_dim,
                        pooling=pooling,
                        depth=depth,
                        medical_rationale=(
                            f"Each action is encoded by one shared {model_dim}D network "
                            f"from {evidence}; {pool}."
                        ),
                    ))
                    index += 1
    if len(rows) != 16:
        raise AssertionError("shared mechanism V9 registry drifted")
    return tuple(rows)


class SharedMechanismRouterV9(nn.Module):
    """Apply the same action encoder to every script before tiny endpoint heads."""

    def __init__(self, candidate: SharedMechanismCandidateV9):
        super().__init__()
        if (
            type(candidate) is not SharedMechanismCandidateV9
            or candidate not in candidate_registry_v9()
        ):
            raise ValueError("shared mechanism V9 requires one frozen candidate")
        self.candidate = candidate
        input_dim = 237 if candidate.include_dense else 220
        dimension = candidate.model_dim
        self.action_encoder = nn.Sequential(
            nn.Linear(input_dim, dimension),
            nn.LayerNorm(dimension),
            nn.GELU(),
        )
        self.shared_blocks = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(dimension),
                nn.Linear(dimension, dimension * 2),
                nn.GELU(),
                nn.Linear(dimension * 2, dimension),
            )
            for _ in range(candidate.depth)
        )
        self.action_embedding = nn.Embedding(13, dimension)
        if candidate.pooling == "task_attention":
            self.task_queries = nn.Parameter(torch.empty(3, dimension))
            nn.init.normal_(self.task_queries, mean=0.0, std=0.02)
        else:
            self.register_parameter("task_queries", None)
        self.patient_projection = nn.Linear(dimension * 2, dimension)
        self.patient_norm = nn.LayerNorm(dimension)
        self.universal_head = nn.Linear(dimension, 1)
        self.task_heads = nn.ModuleList(nn.Linear(dimension, 1) for _ in range(3))

    def task_specific_parameter_fraction(self) -> float:
        task_specific = sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if name.startswith("task_heads") or name == "task_queries"
        )
        return task_specific / sum(parameter.numel() for parameter in self.parameters())

    @staticmethod
    def _validate(
        values: torch.Tensor,
        action_mask: torch.Tensor,
        task_codes: torch.Tensor,
    ) -> None:
        if (
            not isinstance(values, torch.Tensor)
            or values.ndim != 3
            or values.shape[1:] != (13, 237)
            or values.dtype != torch.float32
            or not bool(torch.isfinite(values).all())
            or action_mask.shape != values.shape[:2]
            or action_mask.dtype != torch.bool
            or action_mask.device != values.device
            or bool((action_mask.sum(dim=1) == 0).any())
            or task_codes.shape != (values.shape[0],)
            or task_codes.dtype != torch.long
            or task_codes.device != values.device
            or bool((task_codes < 0).any())
            or bool((task_codes >= 3).any())
        ):
            raise ValueError("shared mechanism tensors violate the closed contract")

    def forward(
        self,
        values: torch.Tensor,
        action_mask: torch.Tensor,
        task_codes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate(values, action_mask, task_codes)
        selected = values if self.candidate.include_dense else values[..., :220]
        tokens = self.action_encoder(selected)
        codes = torch.arange(13, device=values.device, dtype=torch.long)
        tokens = tokens + self.action_embedding(codes)[None, :, :]
        for block in self.shared_blocks:
            tokens = tokens + block(tokens)
        weights = action_mask.unsqueeze(-1).to(tokens.dtype)
        if self.candidate.pooling == "task_attention":
            query = self.task_queries.index_select(0, task_codes)
            scores = torch.sum(tokens * query[:, None, :], dim=-1) / math.sqrt(
                self.candidate.model_dim
            )
            scores = scores.masked_fill(~action_mask, float("-inf"))
            attended = torch.sum(
                tokens * torch.softmax(scores, dim=1).unsqueeze(-1), dim=1
            )
        else:
            attended = torch.sum(tokens * weights, dim=1) / torch.sum(weights, dim=1)
        maximum = tokens.masked_fill(
            ~action_mask.unsqueeze(-1), float("-inf")
        ).max(dim=1).values
        patient = self.patient_norm(
            self.patient_projection(torch.cat((attended, maximum), dim=-1))
        )
        universal = self.universal_head(patient).squeeze(-1)
        all_task = torch.stack(
            [head(patient).squeeze(-1) for head in self.task_heads], dim=1
        )
        task = all_task.gather(1, task_codes[:, None]).squeeze(1)
        return 0.75 * task + 0.25 * universal, universal


__all__ = [
    "SharedMechanismCandidateV9",
    "SharedMechanismRouterV9",
    "candidate_registry_v9",
]
