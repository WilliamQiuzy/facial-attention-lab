"""Full-mesh shared action phenotypes with tiny script-specific heads."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .residual_shared_router_v8 import ResidualSharedRouterV8, candidate_registry_v8


@dataclass(frozen=True)
class ScriptPhenotypeCandidateV9:
    candidate_id: str
    phenotype_dim: int
    head_mode: str
    universal_blend: float
    script_blend: float
    medical_rationale: str


def candidate_registry_v9() -> tuple[ScriptPhenotypeCandidateV9, ...]:
    rows = [ScriptPhenotypeCandidateV9(
        candidate_id="SAP9-000", phenotype_dim=4, head_mode="linear",
        universal_blend=0.25, script_blend=0.0,
        medical_rationale="Exact deterministic RSR8-001 comparator.",
    )]
    index = 1
    for dimension in (4, 8, 16):
        for head_mode in ("linear", "small_mlp"):
            for universal_blend in (0.25, 0.50):
                for script_blend in (0.50, 1.00):
                    rows.append(ScriptPhenotypeCandidateV9(
                        candidate_id=f"SAP9-{index:03d}",
                        phenotype_dim=dimension,
                        head_mode=head_mode,
                        universal_blend=universal_blend,
                        script_blend=script_blend,
                        medical_rationale=(
                            f"One shared full-mesh encoder maps every prompted action "
                            f"to a {dimension}D motor phenotype; a tiny {head_mode} "
                            "head weights the registered script actions without changing "
                            "landmark interpretation or assuming a normal side."
                        ),
                    ))
                    index += 1
    if len(rows) != 25:
        raise AssertionError("script phenotype V9 registry drifted")
    return tuple(rows)


def _locked_v8_candidate():
    return next(row for row in candidate_registry_v8() if row.candidate_id == "RSR8-001")


class ScriptPhenotypeRouterV9(nn.Module):
    """Preserve V8 sharing and expose shared per-action motor phenotypes."""

    def __init__(self, candidate: ScriptPhenotypeCandidateV9):
        super().__init__()
        if (
            type(candidate) is not ScriptPhenotypeCandidateV9
            or candidate not in candidate_registry_v9()
        ):
            raise ValueError("script phenotype V9 requires one frozen candidate")
        self.candidate = candidate
        self.base = ResidualSharedRouterV8(_locked_v8_candidate())
        if candidate.script_blend == 0.0:
            self.phenotype_projection = nn.Identity()
            self.script_heads = nn.ModuleList()
            return
        self.phenotype_projection = nn.Sequential(
            nn.LayerNorm(64), nn.Linear(64, candidate.phenotype_dim), nn.GELU()
        )
        head_input = 13 * candidate.phenotype_dim + 13
        if candidate.head_mode == "linear":
            self.script_heads = nn.ModuleList(nn.Linear(head_input, 1) for _ in range(3))
        else:
            self.script_heads = nn.ModuleList(
                nn.Sequential(nn.Linear(head_input, 16), nn.GELU(), nn.Linear(16, 1))
                for _ in range(3)
            )

    def shared_action_tokens(self, *inputs: torch.Tensor) -> torch.Tensor:
        return self.base.shared_action_tokens(*inputs)

    def task_specific_parameter_fraction(self) -> float:
        task_specific = sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if name.startswith("script_heads")
            or name.startswith("base.adapters")
            or name.startswith("base.base.task_queries")
            or name.startswith("base.base.backbone.task_heads")
        )
        return task_specific / sum(parameter.numel() for parameter in self.parameters())

    def shared_action_phenotype_bank(
        self,
        tokens: torch.Tensor,
        action_mask: torch.Tensor,
        action_codes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.candidate.script_blend == 0.0:
            raise ValueError("the exact comparator has no unused phenotype bank")
        if (
            not isinstance(tokens, torch.Tensor) or tokens.ndim != 3
            or tokens.shape[2] != 64 or not tokens.is_floating_point()
            or not bool(torch.isfinite(tokens).all())
            or action_mask.shape != tokens.shape[:2] or action_mask.dtype != torch.bool
            or action_mask.device != tokens.device
            or action_codes.shape != tokens.shape[:2] or action_codes.dtype != torch.long
            or action_codes.device != tokens.device
            or bool((action_codes < 0).any()) or bool((action_codes >= 13).any())
            or bool((action_mask.sum(dim=1) == 0).any())
        ):
            raise ValueError("action phenotype bank received malformed shared tokens")
        phenotypes = self.phenotype_projection(tokens)
        assignments = F.one_hot(action_codes, num_classes=13).to(phenotypes.dtype)
        assignments = assignments * action_mask.unsqueeze(-1).to(phenotypes.dtype)
        summed = torch.einsum("bad,bak->bkd", phenotypes, assignments)
        counts = assignments.sum(dim=1)
        bank = summed / counts.clamp_min(1.0).unsqueeze(-1)
        return bank, counts > 0

    def _script_logits(
        self,
        tokens: torch.Tensor,
        action_mask: torch.Tensor,
        action_codes: torch.Tensor,
        task_codes: torch.Tensor,
    ) -> torch.Tensor:
        bank, present = self.shared_action_phenotype_bank(
            tokens, action_mask, action_codes
        )
        features = torch.cat((
            bank.reshape(bank.shape[0], -1), present.to(bank.dtype)
        ), dim=1)
        all_logits = torch.stack(
            [head(features).squeeze(-1) for head in self.script_heads], dim=1
        )
        return all_logits.gather(1, task_codes[:, None]).squeeze(1)

    def routed_and_universal_logits(
        self,
        tokens: torch.Tensor,
        action_mask: torch.Tensor,
        action_codes: torch.Tensor,
        task_codes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        universal_embedding = self.base.base.universal_embedding(tokens, action_mask)
        universal = self.base.base.universal_head(universal_embedding).squeeze(-1)
        if self.candidate.script_blend == 0.0:
            common = self.base.base.endpoint_embedding(tokens, action_mask, task_codes)
            endpoint = self.base.adapt_endpoint(common, task_codes)
            task = self.base.base.task_logits_from_embedding(endpoint, task_codes)
            return 0.75 * task + 0.25 * universal, universal
        common = self.base.base.endpoint_embedding(tokens, action_mask, task_codes)
        endpoint = self.base.adapt_endpoint(common, task_codes)
        prior_task = self.base.base.task_logits_from_embedding(endpoint, task_codes)
        script = self._script_logits(tokens, action_mask, action_codes, task_codes)
        task = (
            (1.0 - self.candidate.script_blend) * prior_task
            + self.candidate.script_blend * script
        )
        blend = self.candidate.universal_blend
        return (1.0 - blend) * task + blend * universal, universal

    def routed_logits(
        self,
        tokens: torch.Tensor,
        action_mask: torch.Tensor,
        action_codes: torch.Tensor,
        task_codes: torch.Tensor,
    ) -> torch.Tensor:
        return self.routed_and_universal_logits(
            tokens, action_mask, action_codes, task_codes
        )[0]


__all__ = [
    "ScriptPhenotypeCandidateV9", "ScriptPhenotypeRouterV9",
    "candidate_registry_v9",
]
