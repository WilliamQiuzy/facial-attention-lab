"""Closed medically justified candidate registry for shared Router V9 distillation."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DistilledSharedCandidateV9:
    candidate_id: str
    teacher_mode: str
    distillation_weight: float
    medical_rationale: str


_RATIONALES = {
    "clinical_logistic_32": (
        "A sparse linear teacher emphasizes a small set of action-conditioned "
        "clinical geometry responses while the deployed encoder remains shared."
    ),
    "clinical_logistic_64": (
        "A wider linear teacher retains distributed brow, eye, and oral geometry "
        "when impairment is not captured by a single unilateral asymmetry axis."
    ),
    "mechanism_logistic_64": (
        "A linear privileged teacher adds translation-referenced excursion and "
        "velocity summaries for bilateral weakness and temporal hypokinesia."
    ),
    "mechanism_rbf_64": (
        "A nonlinear privileged teacher permits clinically plausible interactions "
        "between regional excursion and velocity without entering deployment."
    ),
}


def candidate_registry_v9() -> tuple[DistilledSharedCandidateV9, ...]:
    rows = [DistilledSharedCandidateV9(
        candidate_id="DSR9-000",
        teacher_mode="off",
        distillation_weight=0.0,
        medical_rationale=(
            "Exact deterministic RSR8-001 comparator with no privileged teacher."
        ),
    )]
    index = 1
    for teacher_mode in tuple(_RATIONALES):
        for weight in (0.25, 0.50, 0.75):
            rows.append(DistilledSharedCandidateV9(
                candidate_id=f"DSR9-{index:03d}",
                teacher_mode=teacher_mode,
                distillation_weight=weight,
                medical_rationale=_RATIONALES[teacher_mode],
            ))
            index += 1
    if len(rows) != 13:
        raise AssertionError("the distilled V9 registry drifted")
    return tuple(rows)


__all__ = ["DistilledSharedCandidateV9", "candidate_registry_v9"]
