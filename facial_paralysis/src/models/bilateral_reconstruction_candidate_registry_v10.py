"""Frozen laterality-safe reconstruction candidates based on BLV9-009."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BilateralReconstructionCandidateV10:
    candidate_id: str
    reconstruction_mode: str
    optimizer_mode: str
    medical_rationale: str
    contraindication: str


_RATIONALES = {
    "v9_average": (
        "Exact BLV9-009 masked clinical reconstruction baseline.",
        "Development research only; not Mayo or clinical validation.",
    ),
    "bilateral_decomposition": (
        "Reconstruct symmetric capacity and asymmetry magnitude without assigning a healthy side.",
        "Absolute asymmetry is not a clinical laterality or severity label.",
    ),
    "unordered_twin": (
        "Preserve both unilateral response patterns while treating left and right as an unordered pair.",
        "View reconstruction cannot identify the affected facial nerve or House-Brackmann grade.",
    ),
}


def candidate_registry_v10() -> tuple[BilateralReconstructionCandidateV10, ...]:
    rows = []
    index = 0
    for reconstruction_mode in (
        "v9_average", "bilateral_decomposition", "unordered_twin",
    ):
        for optimizer_mode in ("adamw", "sam"):
            rationale, contraindication = _RATIONALES[reconstruction_mode]
            rows.append(BilateralReconstructionCandidateV10(
                candidate_id=f"BRV10-{index:03d}",
                reconstruction_mode=reconstruction_mode,
                optimizer_mode=optimizer_mode,
                medical_rationale=rationale,
                contraindication=contraindication,
            ))
            index += 1
    if len(rows) != 6:
        raise RuntimeError("the bilateral reconstruction registry drifted")
    return tuple(rows)


__all__ = ["BilateralReconstructionCandidateV10", "candidate_registry_v10"]
