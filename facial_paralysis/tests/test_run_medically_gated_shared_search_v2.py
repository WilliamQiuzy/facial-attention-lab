from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _testlib import run_all

from scripts import run_medically_gated_shared_search_v2 as runner
from src.models.medical_shared_candidate_registry_v2 import candidate_registry
from src.preprocessing.shared_clinical_tokens_v1 import ClinicalActionBag


def _bag(actions: int, *, dense: bool) -> ClinicalActionBag:
    rng = np.random.default_rng(actions + int(dense))
    dense_shape = (actions, 32, 478, 3)
    return ClinicalActionBag(
        clinical_original=rng.normal(size=(actions, 110)).astype(np.float64),
        clinical_mirrored=rng.normal(size=(actions, 110)).astype(np.float64),
        dense_original=(
            rng.normal(size=dense_shape).astype(np.float32)
            if dense else np.zeros(dense_shape, dtype=np.float32)
        ),
        dense_mirrored=(
            rng.normal(size=dense_shape).astype(np.float32)
            if dense else np.zeros(dense_shape, dtype=np.float32)
        ),
        dense_valid_mask=np.full(dense_shape[:2], dense, dtype=bool),
        dense_available=np.full(actions, dense, dtype=bool),
        dense_timestamps=(
            np.tile(np.linspace(0.0, 1.0, 32, dtype=np.float64), (actions, 1))
            if dense else np.zeros(dense_shape[:2], dtype=np.float64)
        ),
        action_names=tuple(runner.ACTION_VOCAB[index] for index in range(actions)),
    )


def test_v2_packer_binds_real_timestamps_to_dense_actions(c):
    rows = tuple(
        runner.ParticipantBag(
            bag=_bag(2 + index, dense=source != "palsynet"),
            label=index % 2,
            group_id=f"grp_{index:064x}",
            source=source,
        )
        for index, source in enumerate(runner.SOURCES)
    )
    dataset = runner.pack_participant_bags_v2(rows)
    c.eq(dataset.dense_timestamps.shape, (3, 4, 32))
    c.true(np.all(dataset.dense_timestamps[0] == 0.0))
    c.true(np.all(np.diff(dataset.dense_timestamps[1, :3], axis=-1) > 0.0))
    c.true(not dataset.dense_timestamps.flags.writeable)


def test_cli_contains_three_sources_but_no_mayo_or_protected_test(c):
    source = inspect.getsource(runner._parser).lower()
    for name in (
        "palsynet-cache-root", "reviewed-identity-manifest", "review-ledger",
        "split-registry", "neuroface-cache", "neuroface-manifest",
        "meei-cache", "meei-manifest", "output", "candidate-ids",
    ):
        c.true(name in source)
    c.true("mayo" not in source)
    c.true("outer-test" not in source and "protected-test" not in source)


def test_search_report_is_aggregate_closed_and_medically_bound(c):
    evaluations = {
        candidate.candidate_id: {
            source: {
                "accuracy": 0.91,
                "balanced_accuracy": 0.90,
                "auroc": 0.93,
                "sensitivity": 0.89,
                "specificity": 0.92,
                "brier": 0.12,
            }
            for source in runner.SOURCES
        }
        for candidate in candidate_registry()
    }
    report = runner.build_search_report(
        phase="screen",
        evaluations=evaluations,
        ranking=tuple(evaluations),
        counts={"palsynet": 38, "neuroface": 36, "meei": 56},
        runtime={"gpu": "NVIDIA H200", "epochs": 20, "seed": 0, "folds": 6},
        commitments={"implementation_sha256": "a" * 64},
    )
    c.eq(report["model"]["shared_patient_embedding_dim"], 64)
    c.true(report["model"]["source_identifier_input"] is False)
    c.eq(len(report["candidate_registry"]), 32)
    c.eq(report["selection"]["primary_metric"], "minimum_source_accuracy")
    c.eq(report["audit"]["palsynet_protected_reads"], 0)
    c.eq(report["audit"]["mayo_reads"], 0)
    emitted = str(report).lower()
    c.true("group_id" not in emitted and "probabilities" not in emitted)
    c.true("v6" not in emitted)


def test_screen_requires_all_32_and_confirm_requires_locked_subset(c):
    metric = {
        source: {
            "accuracy": 0.5, "balanced_accuracy": 0.5, "auroc": 0.5,
            "sensitivity": 0.5, "specificity": 0.5, "brier": 0.25,
        }
        for source in runner.SOURCES
    }
    all_ids = tuple(candidate.candidate_id for candidate in candidate_registry())
    c.raises(
        lambda: runner.validate_candidate_phase("screen", all_ids[:-1]),
        ValueError,
    )
    runner.validate_candidate_phase("screen", all_ids)
    runner.validate_candidate_phase("confirm", all_ids[:4])
    c.raises(
        lambda: runner.validate_candidate_phase("confirm", all_ids[:5]),
        ValueError,
    )
    report = runner.build_search_report(
        phase="confirm",
        evaluations={candidate_id: metric for candidate_id in all_ids[:4]},
        ranking=all_ids[:4],
        counts={"palsynet": 38, "neuroface": 36, "meei": 56},
        runtime={"gpu": "NVIDIA H200", "epochs": 20, "seed": 1, "folds": 6},
        commitments={"implementation_sha256": "b" * 64},
    )
    c.eq(report["candidate_ids"], list(all_ids[:4]))


if __name__ == "__main__":
    run_all("test_run_medically_gated_shared_search_v2", dict(globals()))
