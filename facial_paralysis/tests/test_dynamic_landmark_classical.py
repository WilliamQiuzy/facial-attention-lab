"""Leakage-proof tests for the dynamic-landmark classical runner."""
from __future__ import annotations

import hashlib
import io
import json
import sys
import tempfile
from contextlib import redirect_stderr
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_dynamic_landmark_classical import (  # noqa: E402
    BOOTSTRAP_REPEATS,
    CANDIDATE_REGISTRY,
    C_GRID,
    NUISANCE_FEATURE_NAMES,
    PINNED_TASK7_REGISTRY_SHA256,
    ClassicalDataset,
    binary_group_metrics,
    choose_regularization_c,
    frozen_classical_protocol,
    group_mean_predictions,
    group_sample_weights,
    load_classical_dataset,
    paired_stratified_group_bootstrap,
    run_inner_candidate_selection,
    _run_synthetic_outer_candidate,
    _run_outer_candidate_unlocked,
    _parser,
    validate_outer_registry,
)
from src.datasets.dynamic_landmark import (  # noqa: E402
    DYNAMIC_FEATURE_NAMES,
    DYNAMIC_FEATURE_SCHEMA,
)
from _testlib import Check, run_all  # noqa: E402


def _opaque(prefix: str, value: int) -> str:
    return f"{prefix}_{value:064x}"


def _synthetic_dataset(records_per_group: int = 1) -> ClassicalDataset:
    rng = np.random.default_rng(2026)
    feature_rows = []
    masks = []
    times = []
    source_rows = []
    nuisance = []
    labels = []
    groups = []
    recordings = []
    recording_number = 1
    for label in (0, 1):
        for group_number in range(5):
            group = _opaque("grp", label * 5 + group_number + 1)
            for repeat in range(records_per_group):
                features = rng.normal(0.0, 0.2, size=(4, 32, 95)).astype(np.float32)
                # Add both blendshape and clinical-geometry class signal.
                features[..., :72] += label * 0.35
                features[..., 72:] += label * 0.25
                mask = np.ones((4, 32), dtype=bool)
                timestamps = np.stack([
                    window * 20.0 + np.arange(32) / 30.0
                    for window in range(4)
                ]).astype(np.float64)
                source = np.stack([
                    window * 1000 + np.arange(32)
                    for window in range(4)
                ]).astype(np.int64)
                feature_rows.append(features)
                masks.append(mask)
                times.append(timestamps)
                source_rows.append(source)
                nuisance.append(np.asarray([
                    10.0 + group_number,
                    1000.0 + repeat,
                    1.0,
                    100.0 + label,
                    2.0,
                    0.3,
                    0.01,
                    0.0,
                    1.0,
                ]))
                labels.append(label)
                groups.append(group)
                recordings.append(_opaque("rec", recording_number))
                recording_number += 1
    return ClassicalDataset(
        features=np.stack(feature_rows),
        valid_masks=np.stack(masks),
        timestamps=np.stack(times),
        source_frame_indices=np.stack(source_rows),
        nuisance=np.stack(nuisance),
        labels=np.asarray(labels, dtype=np.int64),
        group_ids=np.asarray(groups),
        recording_ids=tuple(recordings),
    )


def test_registry_is_complete_before_any_outer_evaluation(c: Check):
    c.eq(tuple(CANDIDATE_REGISTRY), (
        "nuisance", "blendshape", "landmark", "fusion", "rao_fusion"
    ), "all five paper-style candidates are frozen")
    c.eq(C_GRID, (0.01, 0.1, 1.0, 10.0), "regularization grid is frozen")
    c.eq(BOOTSTRAP_REPEATS, 5000, "paired bootstrap count is frozen")
    c.eq(NUISANCE_FEATURE_NAMES, (
        "duration_seconds", "bitrate_proxy_bytes_per_second", "detection_rate",
        "luminance_mean", "frame_difference_mean", "face_scale_mean",
        "face_scale_std", "eye_line_roll_degrees_mean",
        "eye_line_roll_degrees_std",
    ), "same-detection nuisance audit uses the final Task2 field contract")
    protocol = frozen_classical_protocol()
    c.eq(protocol["candidates"], list(CANDIDATE_REGISTRY),
         "registry serializes candidates in frozen order")
    c.eq(protocol["primary_metric"], "pooled_group_auroc",
         "pooled group AUROC is primary")
    c.eq(protocol["probability_threshold"], 0.5,
         "secondary threshold metrics stay fixed at 0.5")


def test_group_weights_and_group_means_treat_each_identity_once(c: Check):
    labels = np.asarray([0, 0, 1])
    groups = np.asarray([_opaque("grp", 1), _opaque("grp", 1), _opaque("grp", 2)])
    probabilities = np.asarray([0.2, 0.4, 0.8])
    weights = group_sample_weights(groups)
    c.true(np.allclose(weights, (0.5, 0.5, 1.0)),
           "duplicate recordings split one group's total training weight")
    group_labels, group_ids, group_probabilities = group_mean_predictions(
        labels, groups, probabilities
    )
    c.eq(tuple(group_labels.tolist()), (0, 1), "one label per group")
    c.eq(tuple(group_ids.tolist()), (_opaque("grp", 1), _opaque("grp", 2)),
         "groups are deterministically ordered")
    c.true(np.allclose(group_probabilities, (0.3, 0.8)),
           "recording probabilities are averaged within group")
    metrics = binary_group_metrics(labels, groups, probabilities)
    c.eq(metrics["auroc"], 1.0, "group AUROC")
    c.eq(metrics["average_precision"], 1.0, "group AP")
    c.true(abs(metrics["brier"] - 0.065) < 1e-12, "group Brier score")
    c.eq(metrics["balanced_accuracy"], 1.0, "fixed-threshold balanced accuracy")
    c.eq(metrics["sensitivity"], 1.0, "fixed-threshold sensitivity")
    c.eq(metrics["specificity"], 1.0, "fixed-threshold specificity")


def test_c_selection_uses_pooled_inner_oof_auc_and_ties_choose_smaller(c: Check):
    scores = {0.01: 0.7, 0.1: 0.8, 1.0: 0.8, 10.0: 0.6}
    c.eq(choose_regularization_c(scores), 0.1,
         "numerically tied best scores choose the smaller C")
    near_tie = {0.01: 0.8, 0.1: 0.8 + 5e-13, 1.0: 0.7, 10.0: 0.6}
    c.eq(choose_regularization_c(near_tie), 0.01,
         "floating noise below the frozen tolerance cannot change C")


def test_inner_rao_selection_never_fits_on_outer_or_noncontrol_rows(c: Check):
    dataset = _synthetic_dataset()
    result = run_inner_candidate_selection(dataset, "rao_fusion", outer_fold_number=0)
    outer_test = set(result.outer_test_indices)
    c.eq(set(result.c_scores), set(C_GRID), "every fixed C is evaluated")
    c.true(result.selected_c in C_GRID, "selected C belongs to the frozen grid")
    c.eq(len(result.oof_probabilities), len(result.outer_train_indices),
         "inner OOF predictions cover outer train exactly once")
    c.true(np.isfinite(result.oof_probabilities).all(), "inner probabilities are finite")
    prototype_events = [event for event in result.audit_events
                        if event.fit_kind == "healthy_reference"]
    scaler_events = [event for event in result.audit_events
                     if event.fit_kind == "standard_scaler"]
    c.eq(len(prototype_events), 4, "one train-control prototype per inner fold")
    c.eq(len(scaler_events), 4, "one train-only scaler per inner fold")
    for event in result.audit_events:
        c.true(outer_test.isdisjoint(event.fit_indices),
               "outer test never enters inner fitting state")
        c.true(set(event.fit_indices).issubset(set(result.outer_train_indices)),
               "all inner fit rows come from outer train")
        if event.fit_kind == "healthy_reference":
            c.true(all(dataset.labels[index] == 0 for index in event.fit_indices),
                   "healthy prototype receives controls only")


def test_outer_registry_gate_requires_exact_supplied_sha_and_protocol(c: Check):
    def parse_removed_outer_mode():
        with redirect_stderr(io.StringIO()):
            _parser().parse_args([
                "--cache-root", "/synthetic", "--mode", "outer"
            ])

    c.raises(parse_removed_outer_mode, SystemExit,
             "CLI exposes no real outer-scoring mode before Task7")
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "experiment_registry.json"
        path.write_text(json.dumps({
            "schema_version": "dynamic_landmark_experiment_registry_v1",
            "classical_protocol": frozen_classical_protocol(),
        }, sort_keys=True), encoding="utf-8")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        c.raises(lambda: validate_outer_registry(path, None), ValueError,
                 "outer scoring needs an explicitly supplied frozen hash")
        c.raises(lambda: validate_outer_registry(path, "0" * 64), ValueError,
                 "wrong registry bytes fail closed")
        c.eq(PINNED_TASK7_REGISTRY_SHA256, None,
             "outer scoring stays disabled until Task 7 pins a repository hash")
        c.raises(lambda: validate_outer_registry(path, digest), ValueError,
                 "a caller-created matching file cannot authorize outer scoring")

        bad = Path(temporary) / "bad_registry.json"
        bad.write_text(json.dumps({
            "schema_version": "dynamic_landmark_experiment_registry_v1",
            "classical_protocol": {
                **frozen_classical_protocol(),
                "c_grid": [0.1, 1.0],
            },
        }, sort_keys=True), encoding="utf-8")
        bad_digest = hashlib.sha256(bad.read_bytes()).hexdigest()
        c.raises(lambda: validate_outer_registry(bad, bad_digest), ValueError,
                 "a matching hash cannot authorize a drifted protocol")


def test_authorized_synthetic_outer_run_reports_every_fold_without_leakage(c: Check):
    dataset = _synthetic_dataset()
    with tempfile.TemporaryDirectory() as temporary:
        registry = Path(temporary) / "experiment_registry.json"
        registry.write_text(json.dumps({
            "schema_version": "dynamic_landmark_experiment_registry_v1",
            "classical_protocol": frozen_classical_protocol(),
        }, sort_keys=True), encoding="utf-8")
        digest = hashlib.sha256(registry.read_bytes()).hexdigest()
        result = _run_synthetic_outer_candidate(dataset, "nuisance")
    c.eq(len(result.fold_metrics), 5, "all five outer folds report metrics")
    c.eq(tuple(sorted(set(result.outer_fold_by_record.tolist()))), (0, 1, 2, 3, 4),
         "every record is assigned to exactly one outer test fold")
    c.true(np.isfinite(result.probabilities).all(),
           "every synthetic outer record receives one probability")
    expected_metrics = {
        "auroc", "average_precision", "brier", "balanced_accuracy",
        "sensitivity", "specificity",
    }
    c.true(all(set(metrics) == expected_metrics for metrics in result.fold_metrics),
           "each fold reports all required metrics")
    for outer_fold in range(5):
        outer_test = set(np.flatnonzero(result.outer_fold_by_record == outer_fold))
        for event in result.audit_events:
            if event.outer_fold == outer_fold:
                c.true(outer_test.isdisjoint(event.fit_indices),
                       "the current outer test never enters any fit event")
    real_like = replace(
        dataset,
        claim_unit="video_held_out",
        identity_status="unreviewed",
        collection_manifest_sha256="a" * 64,
    )
    c.raises(lambda: _run_synthetic_outer_candidate(
        real_like,
        "nuisance",
    ), ValueError, "synthetic outer helper cannot expose a real cohort")
    c.raises(lambda: _run_outer_candidate_unlocked(
        real_like, "nuisance"
    ), ValueError, "private outer core also requires an issued authorization token")


def test_paired_bootstrap_resamples_groups_within_class(c: Check):
    labels = np.asarray([0] * 6 + [1] * 6)
    groups = np.asarray([_opaque("grp", index + 1) for index in range(12)])
    baseline = np.asarray([0.1, 0.2, 0.8, 0.4, 0.3, 0.7,
                           0.9, 0.8, 0.2, 0.6, 0.7, 0.3])
    candidate = np.asarray([0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
                            0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
    result = paired_stratified_group_bootstrap(
        labels, groups, baseline, candidate, seed=17
    )
    c.eq(result["repeats"], 5000, "runner uses the required repeat count")
    c.true(result["delta_auroc"] > 0.0, "candidate improves pooled group AUROC")
    c.true(result["probability_delta_gt_zero"] > 0.9,
           "paired resampling preserves a strong positive improvement")
    c.true(result["ci95"][0] <= result["delta_auroc"] <= result["ci95"][1],
           "point delta lies inside its descriptive interval")


def _write_cache(path: Path, recording_number: int, label: int, nuisance: dict):
    features = np.zeros((4, 32, 95), dtype=np.float32)
    features[..., :72] = label * 0.1
    features[..., 72:] = np.linspace(0.0, 1.0, 23, dtype=np.float32)
    features[..., 72] += np.arange(32, dtype=np.float32)[None, :] / 31.0
    mask = np.ones((4, 32), dtype=bool)
    source_indices = np.arange(128, dtype=np.int64).reshape(4, 32)
    timestamps = source_indices.astype(np.float64) / 30.0
    recording_id = _opaque("rec", recording_number)
    group_id = _opaque("grp", recording_number)
    source_sha = f"{recording_number:064x}"
    np.savez(
        path / f"{recording_id}.npz",
        features=features,
        valid_mask=mask,
        timestamps=timestamps,
        timestamp_unit=np.asarray("seconds"),
        source_frame_indices=source_indices,
        source_frame_count=np.asarray(128, dtype=np.int64),
        feature_schema=np.asarray(DYNAMIC_FEATURE_SCHEMA),
        feature_names=np.asarray(DYNAMIC_FEATURE_NAMES),
        recording_id=np.asarray(recording_id),
        group_id=np.asarray(group_id),
        label=np.asarray(label, dtype=np.int64),
        source_sha256=np.asarray(source_sha),
    )
    return {
        "recording_id": recording_id,
        "group_id": group_id,
        "source_sha256": source_sha,
        "label": "affected" if label == 1 else "unaffected",
        "source_frame_count": 128,
        "fps": 30.0,
        "window_starts": [0, 32, 64, 96],
        "frames_per_window": 32,
        "timestamp_unit": "seconds",
        "frame_width": 640,
        "frame_height": 480,
        "file_size_bytes": 1000 + recording_number,
        "coverage": 1.0,
        "landmark_varied": True,
        "landmark_variation_stat": 1.0,
        "nuisance": nuisance,
    }


def _excluded_record(recording_number: int, label: str) -> dict[str, object]:
    return {
        "recording_id": _opaque("rec", recording_number),
        "group_id": _opaque("grp", recording_number),
        "source_sha256": f"{recording_number:064x}",
        "label": label,
        "reason": "synthetic_extraction_exclusion",
    }


def _collection_manifest(records: list[dict], excluded: list[dict]) -> dict:
    discovered = records + excluded
    source_fingerprint = hashlib.sha256()
    for row in sorted(discovered, key=lambda item: (item["label"], item["source_sha256"])):
        source_fingerprint.update(
            f"{row['label']}:{row['source_sha256']}\n".encode("ascii")
        )
    components = {
        "action_bundle": "1" * 64,
        "builder": "2" * 64,
        "clinical_landmarks": "3" * 64,
        "dynamic_landmark_loader": "4" * 64,
        "feature_registry": "5" * 64,
    }
    producer_aggregate = hashlib.sha256()
    for name, digest in sorted(components.items()):
        producer_aggregate.update(f"{name}:{digest}\n".encode("ascii"))
    retained_affected = sum(row["label"] == "affected" for row in records)
    retained_unaffected = sum(row["label"] == "unaffected" for row in records)
    return {
        "schema_version": "palsynet_clinical23_v2_windows_v1",
        "dataset": "PalsyNet",
        "feature_schema": DYNAMIC_FEATURE_SCHEMA,
        "feature_shape": [4, 32, 95],
        "capture_mirrored": None,
        "claim_unit": "video_held_out",
        "identity_status": "unreviewed",
        "protocol": {
            "windows_per_recording": 4,
            "frames_per_window": 32,
            "minimum_coverage": 0.9,
            "minimum_retained": 47,
            "minimum_landmark_variation_fraction": 0.95,
        },
        "provenance": {
            "model_sha256": "6" * 64,
            "identity_manifest_sha256": "7" * 64,
            "identity_fingerprints": {
                "bundle_provenance_sha256": "8" * 64,
                "embedding_collection_sha256": "9" * 64,
                "source_collection_sha256": source_fingerprint.hexdigest(),
            },
            "source_collection_sha256": source_fingerprint.hexdigest(),
            "corpus_inventory": {
                "recordings": 49,
                "fps": 30.0,
                "total_frames": 177_511,
                "minimum_frames": 172,
                "duration_minutes": 98.61722222222221,
            },
            "dependency_versions": {
                "python": "python==3.10.2",
                "numpy": "numpy==1.26.4",
                "mediapipe": "mediapipe==0.10.35",
                "opencv": "opencv-contrib-python==4.11.0.86",
                "torch": "torch==2.2.1",
            },
            "producer_sources": {
                "components": components,
                "aggregate_sha256": producer_aggregate.hexdigest(),
            },
        },
        "counts": {
            "discovered": 49,
            "retained": len(records),
            "excluded": len(excluded),
            "retained_affected": retained_affected,
            "retained_unaffected": retained_unaffected,
            "retained_groups": len({row["group_id"] for row in records}),
        },
        "records": records,
        "excluded": excluded,
    }


def test_cache_adapter_uses_manifest_and_public_validating_loader(c: Check):
    nuisance = {
        "duration_seconds": 128.0 / 30.0,
        "bitrate_proxy_bytes_per_second": 250.0,
        "detection_rate": 1.0,
        "luminance_mean": 100.0,
        "frame_difference_mean": 2.0,
        "face_scale_mean": 0.3,
        "face_scale_std": 0.01,
        "eye_line_roll_degrees_mean": 0.0,
        "eye_line_roll_degrees_std": 1.0,
    }
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        records = [
            _write_cache(root, index, 1 if index <= 26 else 0, nuisance)
            for index in range(1, 48)
        ]
        excluded = [
            _excluded_record(48, "affected"),
            _excluded_record(49, "unaffected"),
        ]
        manifest = _collection_manifest(records, excluded)
        serialized_manifest = json.dumps(manifest)
        (root / "collection_manifest.json").write_text(
            serialized_manifest, encoding="utf-8"
        )
        dataset = load_classical_dataset(root)
        c.eq(dataset.features.shape, (47, 4, 32, 95),
             "adapter loads validated public cache objects")
        c.eq((int(dataset.labels.sum()), int((dataset.labels == 0).sum())), (26, 21),
             "retained manifest/cache labels agree")
        c.eq(dataset.claim_unit, "video_held_out",
             "adapter preserves the audit's conservative claim unit")
        c.eq(dataset.identity_status, "unreviewed",
             "adapter never upgrades an unreviewed identity claim")
        c.eq(dataset.collection_manifest_sha256,
             hashlib.sha256(serialized_manifest.encode("utf-8")).hexdigest(),
             "validated collection bytes are frozen for the future Task7 registry")
        c.raises(lambda: replace(
            dataset, claim_unit="patient_held_out", identity_status="unreviewed"
        ), ValueError, "unknown or misspelled claim units fail closed")
        c.true(np.allclose(dataset.nuisance[0], tuple(nuisance.values())),
               "nuisance fields follow an explicit order")

        drifted = json.loads(json.dumps(manifest))
        drifted["protocol"]["minimum_retained"] = 1
        (root / "collection_manifest.json").write_text(
            json.dumps(drifted), encoding="utf-8"
        )
        c.raises(lambda: load_classical_dataset(root), ValueError,
                 "a drifted Task2 collection-level gate fails closed")

        for mutate, message in (
            (lambda value: value["counts"].__setitem__("retained_affected", 25),
             "retained label counts are cross-checked"),
            (lambda value: value["provenance"]["producer_sources"].__setitem__(
                "aggregate_sha256", "f" * 64),
             "producer provenance aggregate is cross-checked"),
            (lambda value: value["provenance"]["dependency_versions"].__setitem__(
                "torch", "numpy==999.0"),
             "dependency field is bound to its exact distribution name"),
            (lambda value: value["records"][0].__setitem__(
                "source_frame_count", 129),
             "record temporal metadata is cross-checked against NPZ"),
            (lambda value: value.pop("protocol"),
             "complete Task2 top-level schema is required"),
        ):
            tampered = json.loads(json.dumps(manifest))
            mutate(tampered)
            (root / "collection_manifest.json").write_text(
                json.dumps(tampered), encoding="utf-8"
            )
            c.raises(lambda: load_classical_dataset(root), ValueError, message)

        records[0]["source_sha256"] = "f" * 64
        (root / "collection_manifest.json").write_text(
            json.dumps({**manifest, "records": records}), encoding="utf-8"
        )
        c.raises(lambda: load_classical_dataset(root), ValueError,
                 "manifest/cache provenance disagreement fails closed")


if __name__ == "__main__":
    run_all("test_dynamic_landmark_classical", dict(globals()))
