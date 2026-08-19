"""Frozen optimizer-only candidates triggered by measured shared-gradient conflict."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConflictAwareCandidateV5:
    candidate_id: str
    base_candidate_id: str
    projection_scope: str
    projection_strength: float


def candidate_registry_v5() -> tuple[ConflictAwareCandidateV5, ...]:
    candidates = []
    index = 0
    for projection_scope in ("patient_block", "all_shared"):
        for projection_strength in (0.5, 1.0):
            candidates.append(ConflictAwareCandidateV5(
                candidate_id=f"CAR5-{index:03d}",
                base_candidate_id="NMR4-001",
                projection_scope=projection_scope,
                projection_strength=projection_strength,
            ))
            index += 1
    if len(candidates) != 4:
        raise AssertionError("the conflict-aware registry drifted")
    return tuple(candidates)


__all__ = ["ConflictAwareCandidateV5", "candidate_registry_v5"]
