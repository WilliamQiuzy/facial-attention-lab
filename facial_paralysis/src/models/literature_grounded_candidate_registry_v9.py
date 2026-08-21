"""Frozen paper-grounded candidate registry for the final shared V9 screen."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LiteratureGroundedCandidateV9:
    candidate_id: str
    relation_enabled: bool
    auxiliary_weight: float
    paper_basis: str
    medical_rationale: str


def candidate_registry_v9() -> tuple[LiteratureGroundedCandidateV9, ...]:
    return (
        LiteratureGroundedCandidateV9(
            candidate_id="LGS9-000",
            relation_enabled=False,
            auxiliary_weight=0.0,
            paper_basis="deterministic RSR8-001 comparator",
            medical_rationale="Exact V8 shared-encoder control.",
        ),
        LiteratureGroundedCandidateV9(
            candidate_id="LGS9-001",
            relation_enabled=True,
            auxiliary_weight=0.0,
            paper_basis="ALGRNet, TMI 2023",
            medical_rationale=(
                "Shared bilateral eye, brow, and oral relation modeling preserves "
                "muscle-local and global facial context."
            ),
        ),
        LiteratureGroundedCandidateV9(
            candidate_id="LGS9-002",
            relation_enabled=False,
            auxiliary_weight=0.25,
            paper_basis=(
                "Knowledge-driven AU self-supervision, CVPR 2022; "
                "MLST-Net, JBHI 2025"
            ),
            medical_rationale=(
                "A shared auxiliary head preserves label-free regional excursion, "
                "velocity, and bilateral synchrony in the motor representation."
            ),
        ),
        LiteratureGroundedCandidateV9(
            candidate_id="LGS9-003",
            relation_enabled=True,
            auxiliary_weight=0.25,
            paper_basis="Combination of the two independently authorized mechanisms",
            medical_rationale=(
                "Combine anatomical relational encoding and kinematic preservation "
                "only when neither single mechanism degrades the comparator."
            ),
        ),
    )


__all__ = ["LiteratureGroundedCandidateV9", "candidate_registry_v9"]
