from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _testlib import run_all

from scripts import run_dense_clinical_shared_encoder_v1 as runner
from src.preprocessing.shared_clinical_tokens_v1 import ClinicalActionBag


def _bag(actions: int, *, dense: bool) -> ClinicalActionBag:
    rng = np.random.default_rng(actions + int(dense))
    dense_shape = (actions, 32, 478, 3)
    return ClinicalActionBag(
        clinical_original=rng.normal(size=(actions, 110)).astype(np.float64),
        clinical_mirrored=rng.normal(size=(actions, 110)).astype(np.float64),
        dense_original=rng.normal(size=dense_shape).astype(np.float32) if dense
        else np.zeros(dense_shape, dtype=np.float32),
        dense_mirrored=rng.normal(size=dense_shape).astype(np.float32) if dense
        else np.zeros(dense_shape, dtype=np.float32),
        dense_valid_mask=np.full(dense_shape[:2], dense, dtype=bool),
        dense_available=np.full(actions, dense, dtype=bool),
        dense_timestamps=np.tile(
            np.linspace(0.0, 1.0, 32, dtype=np.float64), (actions, 1)
        ) if dense else np.zeros(dense_shape[:2], dtype=np.float64),
        action_names=tuple(runner.ACTION_VOCAB[index] for index in range(actions)),
    )


def test_participant_packer_pads_actions_and_preserves_dense_availability(c):
    rows = []
    for source_index, source in enumerate(runner.SOURCES):
        rows.append(runner.ParticipantBag(
            bag=_bag(2 + source_index, dense=source != "palsynet"),
            label=source_index % 2,
            group_id=f"grp_{source_index:064x}",
            source=source,
        ))
    dataset = runner.pack_participant_bags(tuple(rows))
    c.eq(dataset.clinical_original.shape, (3, 4, 110))
    c.eq(dataset.dense_original.shape, (3, 4, 32, 478, 3))
    c.eq(dataset.action_mask.sum(axis=1).tolist(), [2, 3, 4])
    c.true(not dataset.dense_available[0].any())
    c.true(dataset.dense_available[1, :3].all())
    c.true(not dataset.clinical_original.flags.writeable)


def test_runner_cli_has_three_private_sources_and_no_mayo_input(c):
    source = inspect.getsource(runner._parser)
    for name in (
        "palsynet-cache-root", "reviewed-identity-manifest", "review-ledger",
        "split-registry", "neuroface-cache", "neuroface-manifest",
        "meei-cache", "meei-manifest", "output",
    ):
        c.true(name in source)
    c.true("mayo" not in source.lower())


def test_smoke_report_is_aggregate_closed_and_keeps_v6_nonshared(c):
    evaluations = {
        "110d_only": {
            source: {"accuracy": 0.5, "balanced_accuracy": 0.5, "auroc": 0.5,
                     "sensitivity": 0.5, "specificity": 0.5, "brier": 0.25}
            for source in runner.SOURCES
        },
        "dense_clinical": {
            source: {"accuracy": 0.6, "balanced_accuracy": 0.6, "auroc": 0.7,
                     "sensitivity": 0.6, "specificity": 0.6, "brier": 0.2}
            for source in runner.SOURCES
        },
    }
    report = runner.build_smoke_report(
        evaluations=evaluations,
        counts={"palsynet": 38, "neuroface": 36, "meei": 56},
        runtime={"gpu": "NVIDIA H200", "epochs": 20, "seed": 0},
        commitments={"implementation_sha256": "a" * 64},
    )
    c.eq(
        report["model"]["primary_head"],
        "source_specific_heads_after_one_shared_patient_embedding",
    )
    c.eq(report["model"]["universal_auxiliary_head_weight"], 0.25)
    c.eq(report["comparators"]["v6"]["shared_encoder"], False)
    c.eq(report["audit"]["palsynet_protected_reads"], 0)
    c.eq(report["audit"]["mayo_reads"], 0)
    emitted = str(report).lower()
    c.true("group_id" not in emitted and "probabilities" not in emitted)
    c.eq(report["decision"]["promotion_authorized"], False)


def test_output_is_no_overwrite(c):
    import tempfile
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "report.json"
        runner.write_report_no_overwrite(path, {"schema_version": "fixture"})
        c.raises(
            lambda: runner.write_report_no_overwrite(path, {"schema_version": "again"}),
            FileExistsError,
        )


def test_dense_membership_allows_only_the_frozen_meei_retained_subset(c):
    neuroface = {f"grp_{index:064x}" for index in range(36)}
    meei_manifest = {f"grp_{index:064x}" for index in range(60)}
    meei_retained = {f"grp_{index:064x}" for index in range(56)}
    runner.validate_dense_membership("neuroface", neuroface, neuroface)
    runner.validate_dense_membership("meei", meei_retained, meei_manifest)
    c.raises(
        lambda: runner.validate_dense_membership(
            "meei", meei_retained | {f"grp_{99:064x}"}, meei_manifest
        ),
        ValueError,
    )
    c.raises(
        lambda: runner.validate_dense_membership(
            "neuroface", set(list(neuroface)[:-1]), neuroface
        ),
        ValueError,
    )


if __name__ == "__main__":
    run_all("test_run_dense_clinical_shared_encoder_v1", dict(globals()))
