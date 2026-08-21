from __future__ import annotations

from dataclasses import FrozenInstanceError

from _testlib import run_all

from src.models.bilateral_reconstruction_candidate_registry_v10 import (
    BilateralReconstructionCandidateV10,
    candidate_registry_v10,
)


def test_registry_is_exact_immutable_and_mechanism_bounded(c):
    rows = candidate_registry_v10()
    c.eq(type(rows), tuple)
    c.eq(len(rows), 6)
    c.eq(tuple(row.candidate_id for row in rows), tuple(
        f"BRV10-{index:03d}" for index in range(6)
    ))
    c.eq(len({(row.reconstruction_mode, row.optimizer_mode) for row in rows}), 6)
    c.eq({row.reconstruction_mode for row in rows}, {
        "v9_average", "bilateral_decomposition", "unordered_twin",
    })
    c.eq({row.optimizer_mode for row in rows}, {"adamw", "sam"})
    c.true(all(row.medical_rationale and row.contraindication for row in rows))
    c.raises(
        lambda: setattr(rows[0], "optimizer_mode", "sam"), FrozenInstanceError,
    )


def test_v9_research_baseline_is_exact_and_not_relabelled_deployment(c):
    baseline = candidate_registry_v10()[0]
    c.eq(baseline, BilateralReconstructionCandidateV10(
        candidate_id="BRV10-000",
        reconstruction_mode="v9_average",
        optimizer_mode="adamw",
        medical_rationale="Exact BLV9-009 masked clinical reconstruction baseline.",
        contraindication="Development research only; not Mayo or clinical validation.",
    ))
    c.true(all("mayo" not in row.candidate_id.lower() for row in candidate_registry_v10()))


if __name__ == "__main__":
    run_all("test_bilateral_reconstruction_candidate_registry_v10", dict(globals()))
