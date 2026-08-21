from __future__ import annotations

from dataclasses import FrozenInstanceError

from _testlib import run_all

from src.models.broad_literature_candidate_registry_v9 import (
    BroadLiteratureCandidateV9,
    candidate_registry_v9,
)


EXPECTED_MECHANISMS = (
    "exact_v8_comparator",
    "sam",
    "asam",
    "swa",
    "r_drop",
    "modality_dropout",
    "action_dropout_consistency",
    "cross_view_vicreg",
    "cross_view_barlow_twins",
    "masked_clinical_reconstruction",
    "masked_action_reconstruction",
    "clinical_to_dense_reconstruction",
    "focal_loss",
    "ldam_loss",
    "pairwise_auc_loss",
    "high_specificity_partial_auc_loss",
    "brier_composite_loss",
    "progressive_layered_extraction",
    "cross_stitch_endpoint_streams",
    "action_conditioned_film",
    "anatomy_action_graph",
)


def test_registry_contains_exactly_twenty_new_mechanism_models(c):
    registry = candidate_registry_v9()
    c.eq(len(registry), 21)
    c.eq(tuple(row.candidate_id for row in registry), tuple(
        f"BLV9-{index:03d}" for index in range(21)
    ))
    c.eq(tuple(row.mechanism for row in registry), EXPECTED_MECHANISMS)
    c.eq(len({row.mechanism for row in registry[1:]}), 20)
    c.eq(registry[0].family, "comparator")
    c.eq({row.family for row in registry[1:]}, {
        "optimization",
        "missing_evidence_robustness",
        "self_supervision",
        "clinical_objective",
        "shared_architecture",
    })
    c.true(all(type(row) is BroadLiteratureCandidateV9 for row in registry))


def test_every_new_model_is_paper_bound_medically_bounded_and_immutable(c):
    forbidden_padding = {"learning_rate", "width", "threshold", "seed", "epochs"}
    forbidden_repeats = {
        "mmoe", "deep_ensemble", "pcgrad", "coral", "groupdro",
        "anatomical_relational_residual", "kinematic_auxiliary",
    }
    for row in candidate_registry_v9()[1:]:
        c.true(row.paper_title != "")
        c.true(row.paper_url.startswith("https://"))
        c.true(row.medical_rationale != "")
        c.true(row.inference_change in {"none", "training_only", "architecture"})
        c.true(not (forbidden_padding & {name for name, _ in row.settings}))
        c.true(row.mechanism not in forbidden_repeats)
        c.eq(len({name for name, _ in row.settings}), len(row.settings))
    row = candidate_registry_v9()[1]
    try:
        row.mechanism = "mutated"
    except FrozenInstanceError:
        pass
    else:
        raise AssertionError("the scientific registry must be immutable")


def test_one_candidate_changes_one_named_mechanism(c):
    registry = candidate_registry_v9()
    c.eq(sum(row.inference_change == "architecture" for row in registry), 4)
    c.eq(sum(row.family == "clinical_objective" for row in registry), 5)
    c.eq(sum(row.family == "self_supervision" for row in registry), 5)
    c.eq(sum(row.family == "missing_evidence_robustness" for row in registry), 2)
    c.eq(sum(row.family == "optimization" for row in registry), 4)
    c.true(all(row.combinable is False for row in registry))


if __name__ == "__main__":
    run_all("test_broad_literature_candidate_registry_v9", dict(globals()))
