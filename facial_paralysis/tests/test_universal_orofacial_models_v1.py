"""Architecture and training contracts for Universal Orofacial Model v1."""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src.evaluation.universal_orofacial_v1 import (  # noqa: E402
    CANDIDATES,
    FROZEN_NEURAL_CONFIG,
    aggregate_participant_recordings,
    apply_scaler,
    evaluate_candidate_oof,
    evaluate_leave_one_source_out,
    fit_locked_candidate,
    group_dro_objective,
    select_universal_candidate,
    locked_candidate_from_dict,
    locked_candidate_to_dict,
    predict_locked_candidate,
    weighted_mirror_scaler,
)
from src.models.universal_orofacial_v1 import (  # noqa: E402
    LOW_RANK_DIM,
    UniversalLowRankModel,
)
from _testlib import Check, run_all  # noqa: E402


def _group(index: int) -> str:
    return f"grp_{index:064x}"


def _separable_dataset():
    labels = np.asarray(
        [0] * 6 + [1] * 6 + [0] * 6 + [1] * 6,
        dtype=np.int64,
    )
    sources = tuple(["palsynet"] * 12 + ["neuroface"] * 12)
    original = np.zeros((24, 110), dtype=np.float64)
    original[:, 0] = labels * 4.0 - 2.0
    original[:, 1] = np.linspace(-0.1, 0.1, 24)
    mirrored = original.copy()
    mirrored[:, 1] *= -1.0
    return aggregate_participant_recordings(
        original, mirrored, labels,
        tuple(_group(index) for index in range(24)), sources,
    )


def test_candidate_space_and_neural_training_are_frozen(c: Check):
    c.eq(CANDIDATES, (
        "source_balanced_logistic_110d",
        "groupdro_lowrank_110d",
        "multitask_lowrank_110d",
    ), "candidate space is fixed before real evaluation")
    c.eq(FROZEN_NEURAL_CONFIG, {
        "epochs": 120,
        "learning_rate": 0.01,
        "weight_decay": 0.05,
        "group_step": 0.1,
        "gradient_clip": 1.0,
        "seeds": (0, 1, 2),
        "auxiliary_weight": 0.5,
    }, "small-sample regularization protocol is exact")


def test_universal_forward_cannot_receive_source_identity(c: Check):
    model = UniversalLowRankModel(auxiliary_heads=True)
    c.eq(LOW_RANK_DIM, 16, "shared representation is deliberately small")
    c.eq(tuple(inspect.signature(model.forward).parameters), ("features",),
         "universal forward has no source argument")
    features = torch.randn(4, 110)
    universal = model(features)
    c.eq(tuple(universal.shape), (4,), "universal head emits one logit per person")
    source_indices = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
    auxiliary = model.auxiliary_logits(features, source_indices)
    c.eq(tuple(auxiliary.shape), (4,), "auxiliary routing remains row aligned")
    c.true(not torch.allclose(universal, auxiliary),
           "auxiliary heads are distinct from the source-blind output")
    c.raises(lambda: model.auxiliary_logits(features, torch.tensor([0, 2, 0, 1])),
             ValueError, "only the two frozen auxiliary heads are routable")


def test_weighted_scaler_uses_train_mirror_pairs_and_is_immutable(c: Check):
    original = np.vstack((np.zeros(110), np.full(110, 2.0)))
    mirrored = np.vstack((np.zeros(110), np.full(110, 4.0)))
    weights = np.asarray([0.75, 0.25], dtype=np.float64)
    scaler = weighted_mirror_scaler(original, mirrored, weights)
    c.true(np.allclose(scaler.mean, 0.75),
           "each participant weight is divided equally across mirror views")
    scaled = apply_scaler(np.full((1, 110), 0.75), scaler)
    c.true(np.allclose(scaled, 0.0), "training mean maps to zero")
    c.raises(lambda: scaler.mean.setflags(write=True), ValueError,
             "fold scaler statistics are immutable")
    c.raises(lambda: apply_scaler(np.full((1, 110), np.nan), scaler), ValueError,
             "nonfinite held features fail closed")


def test_groupdro_objective_upweights_the_worst_source_class_group(c: Check):
    losses = torch.tensor([0.1, 0.2, 1.2, 0.3], dtype=torch.float32)
    group_indices = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    q = torch.full((4,), 0.25, dtype=torch.float32)
    objective, updated = group_dro_objective(
        losses, group_indices, q, group_step=0.1
    )
    c.true(float(updated[2]) > 0.25,
           "the highest-loss source-class group gains optimization mass")
    c.true(torch.isclose(updated.sum(), torch.tensor(1.0)),
           "DRO group weights remain normalized")
    c.true(float(objective) > 0.0, "DRO objective is finite and positive")


def test_logistic_oof_is_participant_disjoint_and_mirror_mean(c: Check):
    result = evaluate_candidate_oof(
        _separable_dataset(), "source_balanced_logistic_110d", device="cpu"
    )
    c.eq(result.protocol, "six_fold_source_class_stratified_participant_oof",
         "development validation unit is participant")
    c.eq(result.probabilities.shape, (24,),
         "one universal OOF probability is emitted per participant")
    c.true(result.metrics["overall"]["auroc"] >= 0.99,
           "the universal head learns a source-common synthetic signal")
    c.true(all(result.metrics[source]["auroc"] >= 0.99
               for source in ("palsynet", "neuroface")),
           "per-source metrics are recomputed separately")
    c.eq(result.model_fits, 6, "logistic fits once per frozen outer fold")


def test_leave_one_source_out_uses_one_training_source_and_no_refit_on_held(c: Check):
    result = evaluate_leave_one_source_out(
        _separable_dataset(), "source_balanced_logistic_110d", device="cpu"
    )
    c.eq(set(result), {"palsynet_to_neuroface", "neuroface_to_palsynet"},
         "both source-transfer directions are mandatory")
    c.true(all(value["metrics"]["auroc"] >= 0.99 for value in result.values()),
           "a source-common synthetic signal transfers in both directions")
    c.true(all(value["training_source"] != value["held_source"]
               for value in result.values()),
           "held source is absent from model fitting")
    c.true(all(value["model_fits"] == 1 for value in result.values()),
           "logistic is fit exactly once per transfer direction")


def test_selection_uses_only_worst_source_universal_metrics(c: Check):
    summaries = {
        "source_balanced_logistic_110d": {
            "worst_source_auroc": 0.82, "worst_source_balanced_accuracy": 0.80,
            "overall_brier": 0.10, "auxiliary_auroc": 0.99,
        },
        "groupdro_lowrank_110d": {
            "worst_source_auroc": 0.84, "worst_source_balanced_accuracy": 0.72,
            "overall_brier": 0.20, "auxiliary_auroc": 0.10,
        },
        "multitask_lowrank_110d": {
            "worst_source_auroc": 0.84, "worst_source_balanced_accuracy": 0.79,
            "overall_brier": 0.25, "auxiliary_auroc": 1.00,
        },
    }
    winner = select_universal_candidate(summaries)
    c.eq(winner, "multitask_lowrank_110d",
         "selection is worst-source AUROC then balanced accuracy then Brier")
    altered = {name: dict(values) for name, values in summaries.items()}
    altered["source_balanced_logistic_110d"]["auxiliary_auroc"] = 100.0
    c.eq(select_universal_candidate(altered), winner,
         "auxiliary heads cannot promote a universal candidate")


def test_locked_candidate_roundtrip_predicts_without_refit_or_source_input(c: Check):
    dataset = _separable_dataset()
    locked = fit_locked_candidate(
        dataset, "source_balanced_logistic_110d", device="cpu"
    )
    before = predict_locked_candidate(
        locked, dataset.original, dataset.mirrored, device="cpu"
    )
    payload = locked_candidate_to_dict(locked)
    c.eq(set(payload), {"schema_version", "candidate", "scaler", "models"},
         "private artifact schema is closed")
    encoded = str(payload)
    c.true(all(token not in encoded for token in ("grp_", "palsynet", "neuroface")),
           "locked model contains no participant or source identity")
    restored = locked_candidate_from_dict(payload)
    after = predict_locked_candidate(
        restored, dataset.original, dataset.mirrored, device="cpu"
    )
    c.true(np.array_equal(before, after),
           "safe private-artifact roundtrip is prediction exact")
    c.eq(locked.model_fits, 1, "final Logistic is fit once")
    broken = dict(payload)
    broken["models"] = []
    c.raises(lambda: locked_candidate_from_dict(broken), ValueError,
             "missing model state fails closed")


if __name__ == "__main__":
    run_all("test_universal_orofacial_models_v1", dict(globals()))
