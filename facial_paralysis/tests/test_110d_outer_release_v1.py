"""Fail-closed contracts for the one-shot 110D protected release."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from scripts.run_110d_generalization_v1 import (  # noqa: E402
    GateAudit,
    canonical_json_sha256,
    prepare_development_candidates,
    run_development_comparison,
    validate_development_gate,
)
from scripts.run_110d_outer_release_v1 import _parser  # noqa: E402
from src.evaluation.outer_release_110d_v1 import (  # noqa: E402
    AUTHORIZATION_BASIS,
    DEFAULT_OUTER_REPORT_RELATIVE,
    LOCKED_CANDIDATE,
    OUTER_BOOTSTRAP_REPEATS,
    OUTER_BOOTSTRAP_SEED,
    OuterReleaseAudit,
    build_expected_authorization,
    load_authorized_cache_records,
    prepare_locked_views,
    release_implementation_fingerprints,
    run_protected_outer,
    validate_outer_authorization,
    validate_outer_report_against_predictions,
    write_private_no_overwrite_json,
)
from test_110d_generalization_v1 import _artifacts, _dataset  # noqa: E402
from _testlib import Check, run_all  # noqa: E402


def _fixture(bootstrap_repeats: int = 16):
    dataset = _dataset()
    manifest, ledger, registry, manifest_sha, ledger_sha = _artifacts(dataset)
    gate_audit = GateAudit()
    gate = validate_development_gate(
        dataset,
        manifest,
        ledger,
        registry,
        reviewed_manifest_sha256=manifest_sha,
        review_ledger_sha256=ledger_sha,
        audit=gate_audit,
    )
    development_audit = GateAudit(
        gate_attempts=1,
        gate_passes=1,
        development_cache_records_loaded=gate.development_indices.size,
    )
    prepared = prepare_development_candidates(
        dataset, gate, audit=development_audit
    )
    development = run_development_comparison(
        dataset,
        gate,
        prepared,
        audit=development_audit,
        bootstrap_repeats=bootstrap_repeats,
    )
    development_bytes = (
        json.dumps(
            development.report,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    development_sha = hashlib.sha256(development_bytes).hexdigest()
    components, implementation_sha = release_implementation_fingerprints()
    authorization = build_expected_authorization(
        gate=gate,
        development_report_sha256=development_sha,
        release_implementation_sha256=implementation_sha,
    )
    authorization_sha = canonical_json_sha256(authorization)
    return (
        dataset,
        manifest,
        ledger,
        registry,
        gate,
        development.report,
        development_sha,
        components,
        authorization,
        authorization_sha,
    )


def test_authorization_is_exact_pinned_and_precedes_work(c: Check):
    (
        _, _, _, _, gate, development_report, development_sha, _,
        authorization, authorization_sha,
    ) = _fixture()
    c.eq(authorization["candidate"], LOCKED_CANDIDATE,
         "authorization binds the locked 110D candidate")
    c.eq(authorization["authorization_basis"], AUTHORIZATION_BASIS,
         "authorization cites the pre-result researcher instruction")
    c.eq(authorization["output_relative_path"], DEFAULT_OUTER_REPORT_RELATIVE,
         "authorization cannot redirect protected output")

    accepted = validate_outer_authorization(
        authorization,
        authorization_sha256=authorization_sha,
        pinned_authorization_sha256=authorization_sha,
        gate=gate,
        development_report=development_report,
        development_report_sha256=development_sha,
        audit=OuterReleaseAudit(),
        expected_development_bootstrap_repeats=16,
    )
    c.eq(accepted.authorization_sha256, authorization_sha,
         "the exact out-of-band pin authorizes once")

    mutations = []
    for path, value in (
        (("candidate",), "landmark_mi_110d_action_proxy_168d"),
        (("claim_unit",), "video_held_out"),
        (("source_collection_sha256",), "f" * 64),
        (("person_split_registry_sha256",), "e" * 64),
        (("release_implementation_sha256",), "d" * 64),
        (("output_relative_path",), "outputs/alternate.json"),
        (("authorized_once",), False),
    ):
        mutated = copy.deepcopy(authorization)
        mutated[path[0]] = value
        mutations.append(mutated)
    unexpected = copy.deepcopy(authorization)
    unexpected["prediction"] = 0.9
    mutations.append(unexpected)
    for mutated in mutations:
        audit = OuterReleaseAudit()
        forged_sha = canonical_json_sha256(mutated)
        c.raises(
            lambda m=mutated, s=forged_sha, a=audit: validate_outer_authorization(
                m,
                authorization_sha256=s,
                pinned_authorization_sha256=authorization_sha,
                gate=gate,
                development_report=development_report,
                development_report_sha256=development_sha,
                audit=a,
                expected_development_bootstrap_repeats=16,
            ),
            ValueError,
            "stale or self-consistent-forged authorization fails closed",
        )
        c.eq(audit.as_dict(), OuterReleaseAudit(authorization_attempts=1).as_dict(),
             "authorization failure occurs before cache/extract/fit/predict")


def test_outer_fit_is_fixed_grouped_mirrored_and_closed(c: Check):
    (
        dataset, _, _, _, gate, development_report, development_sha, _,
        authorization, authorization_sha,
    ) = _fixture()
    audit = OuterReleaseAudit()
    state = validate_outer_authorization(
        authorization,
        authorization_sha256=authorization_sha,
        pinned_authorization_sha256=authorization_sha,
        gate=gate,
        development_report=development_report,
        development_report_sha256=development_sha,
        audit=audit,
        expected_development_bootstrap_repeats=16,
    )
    c.raises(lambda: prepare_locked_views(
        dataset, gate, state=state, audit=audit
    ), ValueError, "feature extraction cannot precede authenticated cache loading")
    audit.development_cache_records_loaded = gate.development_indices.size
    audit.protected_cache_records_loaded = gate.protected_indices.size
    views = prepare_locked_views(dataset, gate, state=state, audit=audit)
    result = run_protected_outer(
        dataset,
        gate,
        views,
        state=state,
        audit=audit,
        bootstrap_repeats=32,
    )
    validate_outer_report_against_predictions(
        result.report,
        dataset=dataset,
        gate=gate,
        probabilities=result.group_probabilities,
        group_labels=result.group_labels,
        state=state,
        expected_bootstrap_repeats=32,
    )
    c.eq(result.report["protocol"]["model"]["c"], 0.01,
         "outer fit keeps fixed regularization")
    c.eq(result.report["protocol"]["validation_inference"],
         "mean_original_and_horizontal_mirror_probability",
         "outer inference is symmetrized")
    c.eq(result.report["counts"]["protected_groups"],
         len(result.group_labels), "one result exists per protected group")
    c.eq(audit.model_fits, 1, "outer evaluator performs exactly one model fit")
    c.eq(audit.scaler_fits, 1, "outer evaluator performs exactly one scaler fit")
    c.eq(audit.protected_predictions, gate.protected_indices.size,
         "every protected recording is predicted exactly once per fused view")
    encoded = json.dumps(result.report, allow_nan=False)
    c.true("rec_" not in encoded and "grp_" not in encoded,
           "protected report contains no record/group identifiers")
    c.true('"probabilities"' not in encoded and "/Users/" not in encoded,
           "protected report contains no row outcomes or local paths")

    tampered = copy.deepcopy(result.report)
    tampered["metrics"]["auroc"]["point"] = (
        0.0 if result.report["metrics"]["auroc"]["point"] != 0.0 else 1.0
    )
    c.raises(lambda: validate_outer_report_against_predictions(
        tampered,
        dataset=dataset,
        gate=gate,
        probabilities=result.group_probabilities,
        group_labels=result.group_labels,
        state=state,
        expected_bootstrap_repeats=32,
    ), ValueError, "metric tamper is caught by independent recomputation")


def test_authorized_loader_and_cli_have_no_tuning_surface(c: Check):
    (
        dataset, manifest, _, _, gate, development_report, development_sha, _,
        authorization, authorization_sha,
    ) = _fixture()
    audit = OuterReleaseAudit()
    state = validate_outer_authorization(
        authorization,
        authorization_sha256=authorization_sha,
        pinned_authorization_sha256=authorization_sha,
        gate=gate,
        development_report=development_report,
        development_report_sha256=development_sha,
        audit=audit,
        expected_development_bootstrap_repeats=16,
    )
    by_record = {row["recording_id"]: row for row in manifest["recordings"]}

    def loader(path: Path):
        recording_id = path.stem
        index = dataset.recording_ids.index(recording_id)
        row = by_record[recording_id]
        return SimpleNamespace(
            recording_id=recording_id,
            group_id=f"grp_{index:064x}",
            source_sha256=row["source_sha256"],
            label=int(dataset.labels[index]),
            features=dataset.features[index].copy(),
            valid_mask=dataset.valid_masks[index].copy(),
            timestamps=dataset.timestamps[index].copy(),
            source_frame_indices=dataset.source_frame_indices[index].copy(),
        )

    collection_rows = {
        recording_id: {
            "recording_id": recording_id,
            "group_id": f"grp_{index:064x}",
            "source_sha256": by_record[recording_id]["source_sha256"],
            "label": "affected" if dataset.labels[index] else "unaffected",
        }
        for index, recording_id in enumerate(dataset.recording_ids)
    }
    load_authorized_cache_records(
        Path("/sealed-cache"), dataset, gate, collection_rows,
        state=state, audit=audit, record_loader=loader,
    )
    c.eq(audit.development_cache_records_loaded, gate.development_indices.size,
         "authorized loader opens every development cache once")
    c.eq(audit.protected_cache_records_loaded, gate.protected_indices.size,
         "authorized loader opens every protected cache once")
    actions = {action.dest for action in _parser()._actions}
    c.eq(actions, {
        "help", "palsynet_cache_root", "reviewed_identity_manifest",
        "review_ledger", "split_registry", "locked_development_report",
        "authorization",
    }, "outer CLI exposes only authenticated input locations")


def test_release_constants_and_no_overwrite_writer(c: Check):
    c.eq((OUTER_BOOTSTRAP_REPEATS, OUTER_BOOTSTRAP_SEED),
         (5000, 20260805), "outer bootstrap is frozen")
    components, aggregate = release_implementation_fingerprints()
    c.true({"outer_release", "outer_runner", "artifact_freezer"}.issubset(components),
           "release implementation binds its core and both executable CLIs")
    c.eq(aggregate, canonical_json_sha256(components),
         "release aggregate is canonical")
    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory) / "nested" / "report.json"
        write_private_no_overwrite_json(target, {"schema_version": "fixture"})
        c.true(target.is_file(), "writer publishes the first artifact")
        c.raises(lambda: write_private_no_overwrite_json(
            target, {"schema_version": "changed"}
        ), FileExistsError, "writer refuses overwrite")


if __name__ == "__main__":
    run_all("test_110d_outer_release_v1", dict(globals()))
