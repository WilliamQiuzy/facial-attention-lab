"""Fail-closed contracts for one-shot frozen 110D MEEI validation."""
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

from scripts.run_meei_external_v1 import _parser  # noqa: E402
from src.evaluation.meei_external_v1 import (  # noqa: E402
    DEFAULT_REPORT_RELATIVE,
    ExternalAudit,
    build_expected_authorization,
    build_external_report,
    cache_artifact_inventory,
    canonical_json_sha256,
    score_authenticated_cache,
    validate_external_authorization,
    validate_external_report,
    write_private_no_overwrite_json,
)
from src.datasets.dynamic_landmark import DYNAMIC_FEATURE_NAMES  # noqa: E402
from _testlib import Check, run_all  # noqa: E402


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def _authorization_inputs() -> dict[str, object]:
    return {
        "final_artifact_sha256": _sha("artifact"),
        "participant_manifest_sha256": _sha("participants"),
        "cache_manifest_sha256": _sha("cache-manifest"),
        "cache_artifact_collection_sha256": _sha("cache-artifacts"),
        "implementation_sha256": _sha("implementation"),
        "expected_participants": 4,
        "expected_affected": 2,
        "expected_unaffected": 2,
        "expected_eligible_videos": 4,
    }


def _authorized_state():
    values = _authorization_inputs()
    authorization = build_expected_authorization(**values)
    authorization_sha = canonical_json_sha256(authorization)
    audit = ExternalAudit()
    state = validate_external_authorization(
        authorization,
        authorization_sha256=authorization_sha,
        pinned_authorization_sha256=authorization_sha,
        audit=audit,
        **values,
    )
    return state, audit, authorization, authorization_sha


def test_authorization_is_exact_pinned_and_cli_has_no_tuning(c: Check):
    state, audit, authorization, authorization_sha = _authorized_state()
    c.eq(state.authorization_sha256, authorization_sha,
         "out-of-band authorization pin is retained")
    c.eq(audit.authorization_passes, 1,
         "authorization passes exactly once")
    c.eq(authorization["output_relative_path"], DEFAULT_REPORT_RELATIVE,
         "authorization freezes the output path")
    c.eq(authorization["protocol"]["threshold"], 0.5,
         "threshold is frozen before outcomes")

    for field, changed in (
        ("final_artifact_sha256", "f" * 64),
        ("cache_artifact_collection_sha256", "e" * 64),
        ("implementation_sha256", "d" * 64),
        ("expected_participants", 3),
        ("output_relative_path", "outputs/alternate.json"),
    ):
        forged = copy.deepcopy(authorization)
        forged[field] = changed
        failed = ExternalAudit()
        c.raises(lambda f=forged, a=failed: validate_external_authorization(
            f,
            authorization_sha256=canonical_json_sha256(f),
            pinned_authorization_sha256=authorization_sha,
            audit=a,
            **_authorization_inputs(),
        ), ValueError, "self-consistent authorization forgery is rejected")
        c.eq(failed.cache_records_loaded, 0,
             "failed authorization cannot reach cache decoding")

    actions = {action.dest for action in _parser()._actions}
    c.eq(actions, {
        "help", "final_artifact", "participant_manifest",
        "feature_cache_root", "authorization",
    }, "CLI exposes inputs only, with no fit, calibration, threshold, seed, or output")


def test_cache_inventory_binds_bytes_and_rejects_substitution(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        rows = []
        for index in range(2):
            recording_id = f"rec_{index:064x}"
            blob = b"npz-fixture-" + bytes([index])
            (root / f"{recording_id}.npz").write_bytes(blob)
            rows.append({"recording_id": recording_id})
        audit = ExternalAudit()
        inventory = cache_artifact_inventory(root, rows, audit=audit)
        c.eq(set(inventory.blobs), {row["recording_id"] for row in rows},
             "every and only declared cache artifact is retained in memory")
        c.eq(audit.cache_artifacts_hashed, 2,
             "every cache byte artifact is hashed before decoding")
        first = inventory.collection_sha256
        (root / f"{rows[0]['recording_id']}.npz").write_bytes(b"substituted")
        second = cache_artifact_inventory(root, rows, audit=ExternalAudit())
        c.true(first != second.collection_sha256,
               "one-byte-bundle substitution changes the collection commitment")
        (root / "unexpected.npz").write_bytes(b"extra")
        c.raises(lambda: cache_artifact_inventory(
            root, rows, audit=ExternalAudit()
        ), ValueError, "undeclared cache artifacts fail closed")


def test_report_is_participant_level_closed_and_independently_recomputed(c: Check):
    state, audit, _, _ = _authorized_state()
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    probabilities = np.asarray([0.08, 0.35, 0.71, 0.94], dtype=np.float64)
    audit.cache_artifacts_hashed = 4
    audit.cache_records_loaded = 4
    audit.feature_extractions = 4
    audit.mirror_transforms = 8
    audit.artifact_predictions = 4
    audit.participant_aggregations = 4
    components = {"core": _sha("core")}
    implementation_sha = canonical_json_sha256(components)
    state = type(state)(
        **{
            **state.__dict__,
            "implementation_sha256": implementation_sha,
        }
    )
    report = build_external_report(
        labels,
        probabilities,
        state=state,
        audit=audit,
        final_artifact_sha256=_sha("artifact"),
        participant_manifest_sha256=_sha("participants"),
        cache_manifest_sha256=_sha("cache-manifest"),
        cache_artifact_collection_sha256=_sha("cache-artifacts"),
        implementation_components_sha256=components,
        implementation_sha256=implementation_sha,
        participants_total=4,
        eligible_videos=4,
        bootstrap_repeats=32,
    )
    validate_external_report(
        report,
        labels=labels,
        probabilities=probabilities,
        state=state,
        audit=audit,
        expected_bootstrap_repeats=32,
    )
    c.eq(report["counts"]["participants_scored"], 4,
         "metrics use one row per participant")
    c.eq(report["counts"]["photos_scored"], 0,
         "static photos never enter dynamic scoring")
    c.eq(report["audit"]["model_fits"], 0,
         "external validation performs no model fit")
    c.eq(report["audit"]["calibration_fits"], 0,
         "external validation performs no calibration")
    encoded = json.dumps(report, allow_nan=False)
    c.true(all(token not in encoded for token in (
        "rec_", "grp_", "/Users/", "probabilities", "participant_id"
    )), "aggregate report leaks no IDs, rows, paths, or scores")

    tampered = copy.deepcopy(report)
    tampered["metrics"]["auroc"]["point"] = 0.0
    c.raises(lambda: validate_external_report(
        tampered,
        labels=labels,
        probabilities=probabilities,
        state=state,
        audit=audit,
        expected_bootstrap_repeats=32,
    ), ValueError, "metric tamper is caught by independent recomputation")


def test_authenticated_same_bytes_reach_frozen_artifact_without_fit(c: Check):
    artifact = json.loads((
        ROOT / "outputs/dynamic_landmark/artifacts/110d-generalization-v1/"
        "final_palsynet_artifact.json"
    ).read_text())
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        rows = []
        for index, label_name in enumerate(("unaffected", "affected"), start=1):
            recording_id = f"rec_{index:064x}"
            group_id = f"grp_{index:064x}"
            source_sha = f"{index + 10:064x}"
            features = np.zeros((4, 32, 95), dtype=np.float32)
            features[..., 52:] = np.float32(index * 0.01)
            starts = np.asarray([0, 32, 64, 96], dtype=np.int64)
            source_indices = starts[:, None] + np.arange(32)[None, :]
            timestamps = source_indices.astype(np.float64) / 30.0
            np.savez_compressed(
                root / f"{recording_id}.npz",
                features=features,
                valid_mask=np.ones((4, 32), dtype=bool),
                timestamps=timestamps,
                timestamp_unit=np.asarray("seconds"),
                source_frame_indices=source_indices,
                source_frame_count=np.asarray(128, dtype=np.int64),
                feature_schema=np.asarray("mediapipe_bs_lr_v1+clinical23_v2"),
                feature_names=np.asarray(DYNAMIC_FEATURE_NAMES),
                recording_id=np.asarray(recording_id),
                group_id=np.asarray(group_id),
                label=np.asarray(1 if label_name == "affected" else 0, dtype=np.int64),
                source_sha256=np.asarray(source_sha),
            )
            rows.append({
                "recording_id": recording_id,
                "group_id": group_id,
                "source_sha256": source_sha,
                "label": label_name,
            })
        audit = ExternalAudit()
        inventory = cache_artifact_inventory(root, rows, audit=audit)
        values = {
            **_authorization_inputs(),
            "cache_artifact_collection_sha256": inventory.collection_sha256,
            "expected_participants": 2,
            "expected_affected": 1,
            "expected_unaffected": 1,
            "expected_eligible_videos": 2,
        }
        authorization = build_expected_authorization(**values)
        authorization_sha = canonical_json_sha256(authorization)
        state = validate_external_authorization(
            authorization,
            authorization_sha256=authorization_sha,
            pinned_authorization_sha256=authorization_sha,
            audit=audit,
            **values,
        )
        labels, scores = score_authenticated_cache(
            inventory, rows, artifact, state=state, audit=audit
        )
        c.true(bool(np.array_equal(labels, np.asarray([0, 1]))),
               "participant label ordering is deterministic")
        c.true(scores.shape == (2,) and bool(np.isfinite(scores).all()),
               "frozen artifact emits one finite score per participant")
        c.eq((audit.scaler_fits, audit.model_fits, audit.calibration_fits),
             (0, 0, 0), "external inference exposes no fitting path")
        c.eq((audit.cache_records_loaded, audit.artifact_predictions), (2, 2),
             "each authenticated byte bundle is decoded and predicted once")


def test_private_report_refuses_overwrite(c: Check):
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "nested" / "report.json"
        write_private_no_overwrite_json(path, {"sealed": True})
        c.eq(path.stat().st_mode & 0o777, 0o600,
             "one-shot report is owner-private")
        c.raises(lambda: write_private_no_overwrite_json(
            path, {"sealed": False}
        ), FileExistsError, "one-shot report cannot be overwritten")


if __name__ == "__main__":
    run_all("test_meei_external_v1", dict(globals()))
