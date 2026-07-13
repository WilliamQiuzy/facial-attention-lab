"""Fail-closed tests for on-disk MediaPipe feature schemas."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.patient_multistream import (  # noqa: E402
    MP_FEATURE_NAMES_BY_SCHEMA,
    MP_SIDE_CONVENTION_BY_SCHEMA,
    MultiStreamPatientDataset,
)
from _testlib import Check, run_all  # noqa: E402


SCHEMA = "mediapipe_bs_lr_v1+clinical23_v2"


def _fixture(
    dim: int = 95,
    schema: str = SCHEMA,
    include_schema: bool = True,
    names: list[str] | tuple[str, ...] | None = None,
):
    td = tempfile.TemporaryDirectory()
    root = Path(td.name)
    (root / "p1").mkdir()
    (root / "labels.csv").write_text("patient_id,hb_grade\np1,3\n")
    fields = {
        "marlin": np.zeros((1, 768), np.float32),
        "mp_seq": np.zeros((4, dim), np.float32),
        "mp_mask": np.ones(4, bool),
        "mp_feat_dim": np.asarray(dim, np.int32),
    }
    if include_schema:
        fields["mp_feature_schema"] = np.asarray(schema)
        expected_names = MP_FEATURE_NAMES_BY_SCHEMA.get(
            schema, tuple(f"f{i}" for i in range(dim)))
        fields["mp_feature_names"] = np.asarray(
            names if names is not None else expected_names)
        fields["mp_side_convention"] = np.asarray(
            MP_SIDE_CONVENTION_BY_SCHEMA[schema])
        fields["mp_capture_mirrored"] = np.asarray("false")
    np.savez(root / "p1" / "rest.npz", **fields)
    return td, root


def test_matching_schema_loads(c: Check):
    td, root = _fixture()
    try:
        ds = MultiStreamPatientDataset.from_disk(
            root, root / "labels.csv", actions=["rest"], mp_feat_dim=95,
            mp_feature_schema=SCHEMA)
        c.eq(len(ds), 1, "matching cache accepted")
        c.eq(ds.mp_feature_schema, SCHEMA, "schema retained")
    finally:
        td.cleanup()


def test_dimension_mismatch_rejected(c: Check):
    td, root = _fixture(dim=95)
    try:
        c.raises(lambda: MultiStreamPatientDataset.from_disk(
            root, root / "labels.csv", actions=["rest"], mp_feat_dim=72,
            mp_feature_schema=SCHEMA), ValueError, "wrong requested dimension")
    finally:
        td.cleanup()


def test_schema_mismatch_rejected(c: Check):
    td, root = _fixture(schema="mediapipe_bs_lr_v1")
    try:
        c.raises(lambda: MultiStreamPatientDataset.from_disk(
            root, root / "labels.csv", actions=["rest"], mp_feat_dim=95,
            mp_feature_schema=SCHEMA), ValueError, "wrong requested schema")
    finally:
        td.cleanup()


def test_legacy_dimension_only_requires_opt_in(c: Check):
    td, root = _fixture(dim=72, include_schema=False)
    try:
        c.raises(lambda: MultiStreamPatientDataset.from_disk(
            root, root / "labels.csv", actions=["rest"], mp_feat_dim=72,
            mp_feature_schema="mediapipe_bs_lr_v1"), ValueError,
            "metadata-free cache rejected by default")
        ds = MultiStreamPatientDataset.from_disk(
            root, root / "labels.csv", actions=["rest"], mp_feat_dim=72,
            mp_feature_schema="mediapipe_bs_lr_v1", allow_legacy_schema=True)
        c.eq(len(ds), 1, "explicit legacy opt-in accepted")
    finally:
        td.cleanup()


def test_mask_length_mismatch_rejected(c: Check):
    td, root = _fixture()
    try:
        path = root / "p1" / "rest.npz"
        with np.load(path) as saved:
            fields = {name: saved[name] for name in saved.files}
        fields["mp_mask"] = np.ones(2, bool)
        np.savez(path, **fields)
        c.raises(lambda: MultiStreamPatientDataset.from_disk(
            root, root / "labels.csv", actions=["rest"], mp_feat_dim=95,
            mp_feature_schema=SCHEMA), ValueError,
            "mask length must match the original mp_seq time axis")
    finally:
        td.cleanup()


def test_nonboolean_disk_mask_is_rejected_before_cast(c: Check):
    td, root = _fixture()
    try:
        path = root / "p1" / "rest.npz"
        with np.load(path) as saved:
            fields = {name: saved[name] for name in saved.files}
        fields["mp_mask"] = np.asarray((1, 2, 0, 1), dtype=np.int64)
        np.savez(path, **fields)
        c.raises(lambda: MultiStreamPatientDataset.from_disk(
            root, root / "labels.csv", actions=["rest"], mp_feat_dim=95,
            mp_feature_schema=SCHEMA), ValueError,
            "integer masks cannot be silently coerced to detector validity")
    finally:
        td.cleanup()


def test_missing_mask_rejected_when_mp_sequence_exists(c: Check):
    td, root = _fixture()
    try:
        path = root / "p1" / "rest.npz"
        with np.load(path) as saved:
            fields = {name: saved[name] for name in saved.files if name != "mp_mask"}
        np.savez(path, **fields)
        c.raises(lambda: MultiStreamPatientDataset.from_disk(
            root, root / "labels.csv", actions=["rest"], mp_feat_dim=95,
            mp_feature_schema=SCHEMA), ValueError,
            "mp_seq requires an explicit frame-validity mask")
    finally:
        td.cleanup()


def test_schemaful_and_metadata_free_actions_cannot_mix(c: Check):
    td, root = _fixture()
    try:
        np.savez(
            root / "p1" / "smile.npz",
            marlin=np.zeros((1, 768), np.float32),
            mp_seq=np.zeros((4, 95), np.float32),
            mp_mask=np.ones(4, bool),
            mp_feat_dim=np.asarray(95, np.int32),
        )
        c.raises(lambda: MultiStreamPatientDataset.from_disk(
            root, root / "labels.csv", actions=["rest", "smile"],
            mp_feat_dim=95), ValueError,
            "schema metadata must be complete across all MediaPipe actions")
    finally:
        td.cleanup()


def test_legacy_opt_in_does_not_promote_a_mixed_schema_dataset(c: Check):
    td, root = _fixture()
    try:
        np.savez(
            root / "p1" / "smile.npz",
            marlin=np.zeros((1, 768), np.float32),
            mp_seq=np.zeros((4, 95), np.float32),
            mp_mask=np.ones(4, bool),
            mp_feat_dim=np.asarray(95, np.int32),
        )
        c.raises(lambda: MultiStreamPatientDataset.from_disk(
            root, root / "labels.csv", actions=["rest", "smile"],
            mp_feat_dim=95, mp_feature_schema=SCHEMA,
            allow_legacy_schema=True), ValueError,
            "legacy opt-in applies only when every MediaPipe bundle is legacy")
    finally:
        td.cleanup()


def test_schema_rejects_unbound_feature_names(c: Check):
    td, root = _fixture(names=["garbage"] * 95)
    try:
        c.raises(lambda: MultiStreamPatientDataset.from_disk(
            root, root / "labels.csv", actions=["rest"], mp_feat_dim=95,
            mp_feature_schema=SCHEMA), ValueError,
            "schema must bind the exact ordered feature names")
    finally:
        td.cleanup()


def test_embeddings_only_bundle_ignores_mp_schema_requirement(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "p1").mkdir()
        (root / "labels.csv").write_text("patient_id,hb_grade\np1,3\n")
        np.savez(root / "p1" / "rest.npz",
                 embeddings=np.zeros((1, 768), np.float32))
        ds = MultiStreamPatientDataset.from_disk(
            root, root / "labels.csv", actions=["rest"], mp_feat_dim=95,
            mp_feature_schema=SCHEMA)
        c.eq(len(ds), 1, "MARLIN-only legacy bundle remains loadable")
        c.true(not bool(ds[0]["mp_mask"].any()),
               "absent MediaPipe stream remains fully masked")


def test_metadata_free_mp_cache_requires_explicit_opt_in_without_requested_schema(c: Check):
    td, root = _fixture(dim=72, include_schema=False)
    try:
        c.raises(lambda: MultiStreamPatientDataset.from_disk(
            root, root / "labels.csv", actions=["rest"], mp_feat_dim=72),
            ValueError, "dimension-only disk cache must be explicitly trusted")
        ds = MultiStreamPatientDataset.from_disk(
            root, root / "labels.csv", actions=["rest"], mp_feat_dim=72,
            allow_legacy_schema=True)
        c.eq(len(ds), 1, "all-legacy cache loads after explicit opt-in")
        c.eq(ds.mp_feature_schema, None, "dimension-only legacy cache stays unversioned")
    finally:
        td.cleanup()


def test_side_provenance_round_trips(c: Check):
    td, root = _fixture()
    try:
        ds = MultiStreamPatientDataset.from_disk(
            root, root / "labels.csv", actions=["rest"], mp_feat_dim=95,
            mp_feature_schema=SCHEMA)
        c.eq(ds.mp_side_convention, MP_SIDE_CONVENTION_BY_SCHEMA[SCHEMA],
             "dataset preserves the topology-side contract")
        c.eq(ds.mp_capture_mirrored, "false",
             "dataset preserves capture mirror provenance")
        c.eq(ds.records[0].actions[0].mp_capture_mirrored, "false",
             "action bundle preserves capture mirror provenance")
    finally:
        td.cleanup()


def test_mixed_capture_mirror_provenance_rejected(c: Check):
    td, root = _fixture()
    try:
        np.savez(
            root / "p1" / "smile.npz",
            marlin=np.zeros((1, 768), np.float32),
            mp_seq=np.zeros((4, 95), np.float32),
            mp_mask=np.ones(4, bool),
            mp_feat_dim=np.asarray(95, np.int32),
            mp_feature_schema=np.asarray(SCHEMA),
            mp_feature_names=np.asarray(MP_FEATURE_NAMES_BY_SCHEMA[SCHEMA]),
            mp_side_convention=np.asarray(MP_SIDE_CONVENTION_BY_SCHEMA[SCHEMA]),
            mp_capture_mirrored=np.asarray("true"),
        )
        c.raises(lambda: MultiStreamPatientDataset.from_disk(
            root, root / "labels.csv", actions=["rest", "smile"],
            mp_feat_dim=95, mp_feature_schema=SCHEMA), ValueError,
            "opposite capture orientations cannot silently mix")
    finally:
        td.cleanup()


def test_schemaful_cache_requires_side_provenance(c: Check):
    td, root = _fixture()
    try:
        path = root / "p1" / "rest.npz"
        with np.load(path) as saved:
            fields = {
                name: saved[name] for name in saved.files
                if name not in {"mp_side_convention", "mp_capture_mirrored"}
            }
        np.savez(path, **fields)
        c.raises(lambda: MultiStreamPatientDataset.from_disk(
            root, root / "labels.csv", actions=["rest"], mp_feat_dim=95,
            mp_feature_schema=SCHEMA), ValueError,
            "schemaful MediaPipe cache requires side provenance")
    finally:
        td.cleanup()


def test_valid_media_pipe_row_must_be_finite(c: Check):
    td, root = _fixture()
    try:
        path = root / "p1" / "rest.npz"
        with np.load(path) as saved:
            fields = {name: saved[name] for name in saved.files}
        fields["mp_seq"][0, 0] = np.nan
        np.savez(path, **fields)
        c.raises(lambda: MultiStreamPatientDataset.from_disk(
            root, root / "labels.csv", actions=["rest"], mp_feat_dim=95,
            mp_feature_schema=SCHEMA), ValueError,
            "NaN in a detector-valid row is rejected")
    finally:
        td.cleanup()


def test_masked_nonfinite_row_is_zero_before_model(c: Check):
    td, root = _fixture()
    try:
        path = root / "p1" / "rest.npz"
        with np.load(path) as saved:
            fields = {name: saved[name] for name in saved.files}
        fields["mp_mask"][1] = False
        fields["mp_seq"][1] = np.nan
        np.savez(path, **fields)
        ds = MultiStreamPatientDataset.from_disk(
            root, root / "labels.csv", actions=["rest"], mp_feat_dim=95,
            mp_feature_schema=SCHEMA)
        item = ds[0]
        c.true(bool(torch.isfinite(item["mp_seq"]).all()),
               "masked audit-style NaN never reaches the model")
        c.true(bool((item["mp_seq"][0, 1] == 0).all()),
               "masked frame is canonical zero padding")
    finally:
        td.cleanup()


def test_nonfinite_marlin_embedding_is_rejected(c: Check):
    td, root = _fixture()
    try:
        path = root / "p1" / "rest.npz"
        with np.load(path) as saved:
            fields = {name: saved[name] for name in saved.files}
        fields["marlin"][0, 0] = np.inf
        np.savez(path, **fields)
        c.raises(lambda: MultiStreamPatientDataset.from_disk(
            root, root / "labels.csv", actions=["rest"], mp_feat_dim=95,
            mp_feature_schema=SCHEMA), ValueError,
            "nonfinite video embeddings are rejected")
    finally:
        td.cleanup()


if __name__ == "__main__":
    run_all("test_feature_schema", dict(globals()))
