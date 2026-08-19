"""Tests for the provenance-locked PalsyNet MARLIN regeneration command."""
from __future__ import annotations

import importlib
import importlib.util
import hashlib
import io
import json
import os
import sys
import tempfile
import traceback
import types
from contextlib import redirect_stderr
from pathlib import Path
from typing import NamedTuple

import numpy as np

from _testlib import Check, run_all


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "regenerate_palsynet_marlin_identity.py"
EXPECTED_BUNDLE_FIELDS = {
    "marlin",
    "schema_version",
    "source_sha256",
    "n_clips",
    "capture_mirrored",
    "marlin_model_sha256",
    "marlin_config_sha256",
    "face_landmarker_model_sha256",
}
EXPECTED_MANIFEST_FIELDS = {
    "schema_version",
    "dataset",
    "counts",
    "config",
    "collection_fingerprints",
    "dependency_versions",
    "asset_hashes",
    "implementation_source_tree_sha256",
    "records",
}


class Fixture(NamedTuple):
    video_root: Path
    implementation_root: Path
    marlin_dir: Path
    face_model: Path
    output_root: Path


class FakeEncoder:
    def __init__(self, behavior=None):
        self.behavior = behavior
        self.calls = []
        self.to_calls = []
        self.eval_calls = 0

    def to(self, device):
        self.to_calls.append(device)
        return self

    def eval(self):
        self.eval_calls += 1
        return self

    def encode_video_path(self, source, n_clips, landmarker):
        source = Path(source)
        self.calls.append((source, n_clips, landmarker))
        if self.behavior is not None:
            override = self.behavior(source, len(self.calls))
            if override is not NotImplemented:
                return override
        digest = hashlib.sha256(source.read_bytes()).digest()
        offset = int.from_bytes(digest[:4], "big") / float(2**32)
        values = np.arange(4 * 768, dtype=np.float32).reshape(4, 768)
        return values / np.float32(10000.0) + np.float32(offset)


def _write_sources(video_root: Path, reverse_creation: bool, name_token: str) -> None:
    for label, count in (("affected", 27), ("unaffected", 22)):
        directory = video_root / label
        directory.mkdir(parents=True)
        indices = list(range(count))
        if reverse_creation:
            indices.reverse()
        for index in indices:
            name_index = count - index if reverse_creation else index
            (directory / f"{name_token}_{name_index:02d}.mp4").write_bytes(
                f"fixed-source-payload:{label}:{index}".encode("ascii")
            )


def _make_fixture(
    root: Path,
    *,
    reverse_creation: bool = False,
    name_token: str = "Sensitive_Person_Name",
) -> Fixture:
    video_root = root / "palsynet" / "data"
    _write_sources(video_root, reverse_creation, name_token)

    implementation_root = root / "implementation"
    marlin_backbone = implementation_root / "src" / "models" / "backbones" / "marlin_video.py"
    marlin_backbone.parent.mkdir(parents=True)
    marlin_backbone.write_text("# frozen MARLIN implementation\n")
    oo_utils = implementation_root / "src" / "baselines" / "oo_multimodal" / "utils"
    oo_utils.mkdir(parents=True)
    (oo_utils / "__init__.py").write_text("# explicit landmarker implementation\n")
    (oo_utils / "crop.py").write_text("# explicit crop implementation\n")

    marlin_dir = implementation_root / "data" / "external" / "marlin_vit_base_ytf"
    marlin_dir.mkdir(parents=True)
    (marlin_dir / "config.json").write_text('{"model":"fake-marlin"}\n')
    (marlin_dir / "model.safetensors").write_bytes(b"fixed-fake-model-weights")
    (marlin_dir / "marlin.py").write_text("# upstream MARLIN source\n")

    face_model = (
        implementation_root
        / "src"
        / "baselines"
        / "oo_multimodal"
        / "weights"
        / "face_landmarker.task"
    )
    face_model.parent.mkdir(parents=True)
    face_model.write_bytes(b"fixed-face-landmarker-model")
    output_root = video_root.parent / "derived" / "identity_marlin_v1"
    return Fixture(video_root, implementation_root, marlin_dir, face_model, output_root)


def _generate(module, fixture: Fixture, encoder: FakeEncoder | None = None, calls=None):
    encoder = encoder or FakeEncoder()
    calls = calls if calls is not None else []
    landmarker = object()

    def encoder_factory(implementation_root, marlin_dir):
        calls.append(("encoder", Path(implementation_root), Path(marlin_dir)))
        return encoder

    def landmarker_factory(implementation_root, model_path):
        calls.append(("landmarker", Path(implementation_root), Path(model_path)))
        return landmarker

    result = module.generate_identity_cache(
        video_root=fixture.video_root,
        implementation_root=fixture.implementation_root,
        marlin_dir=fixture.marlin_dir,
        face_landmarker_model=fixture.face_model,
        output_root=fixture.output_root,
        encoder_factory=encoder_factory,
        landmarker_factory=landmarker_factory,
        dependency_versions_fn=lambda: {
            "mediapipe": "test-1",
            "numpy": "test-2",
            "opencv-python": "test-3",
            "python": "test-4",
            "safetensors": "test-5",
            "torch": "test-6",
        },
        deterministic_setup_fn=lambda: calls.append(("deterministic", "cpu")),
    )
    return result, encoder, landmarker, calls


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _load_module():
    spec = importlib.util.spec_from_file_location("regenerate_palsynet_marlin_identity", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load regeneration script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_dependency_versions_accepts_one_explicit_opencv_distribution(c: Check):
    module = _load_module()
    original_version = module.importlib.metadata.version
    common = {
        "numpy": "1.26.4",
        "torch": "2.2.1",
        "mediapipe": "0.10.35",
        "safetensors": "0.5.3",
    }

    def install_set(extra):
        installed = {**common, **extra}

        def fake_version(name):
            if name not in installed:
                raise module.importlib.metadata.PackageNotFoundError(name)
            return installed[name]

        module.importlib.metadata.version = fake_version

    try:
        install_set({"opencv-contrib-python": "4.11.0.86"})
        versions = module._default_dependency_versions()
        c.eq(
            versions["opencv-python"],
            "opencv-contrib-python==4.11.0.86",
            "a contrib-only runtime records its unambiguous distribution and version",
        )

        install_set({"opencv-python": "4.11.0", "opencv-contrib-python": "4.11.0.86"})
        c.raises(
            module._default_dependency_versions,
            ValueError,
            "multiple installed OpenCV distributions fail closed",
        )

        install_set({})
        c.raises(
            module._default_dependency_versions,
            ValueError,
            "a runtime without an OpenCV distribution fails closed",
        )
    finally:
        module.importlib.metadata.version = original_version


def test_implementation_tree_hash_accepts_empty_python_package_markers(c: Check):
    module = _load_module()
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        implementation = root / "implementation"
        marlin = implementation / "data" / "external" / "marlin_vit_base_ytf"
        source = implementation / "src"
        source.mkdir(parents=True)
        marlin.mkdir(parents=True)
        (source / "__init__.py").write_bytes(b"")
        (source / "runtime.py").write_text("VALUE = 1\n")
        (marlin / "model.py").write_text("VALUE = 2\n")
        digest = module._hash_source_tree(implementation, marlin)
        c.eq(len(digest), 64, "empty __init__.py files remain hashable source inputs")
        c.true(all(char in "0123456789abcdef" for char in digest),
               "implementation tree fingerprint is lowercase SHA-256")


def test_count_gate_rejects_incomplete_dataset_before_runtime_or_output(c: Check):
    module = _load_module()
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        video_root = root / "palsynet" / "data"
        (video_root / "affected").mkdir(parents=True)
        (video_root / "unaffected").mkdir()
        (video_root / "affected" / "one.mp4").write_bytes(b"one")

        called = []
        expected_output = video_root.parent / "derived" / "identity_marlin_v1"
        c.raises(
            lambda: module.generate_identity_cache(
                video_root=video_root,
                implementation_root=root / "implementation",
                marlin_dir=root / "marlin",
                face_landmarker_model=root / "face.task",
                output_root=expected_output,
                encoder_factory=lambda *_: called.append("encoder"),
                landmarker_factory=lambda *_: called.append("landmarker"),
            ),
            ValueError,
            "49-video class counts are a fail-closed gate",
        )
        c.eq(called, [], "runtime factories are not called before count validation")
        c.true(not expected_output.exists(), "invalid input creates no generation")


def test_count_gate_rejects_untracked_video_outside_locked_classes(c: Check):
    module = _load_module()
    with tempfile.TemporaryDirectory() as td:
        fixture = _make_fixture(Path(td))
        (fixture.video_root / "untracked_extra.mp4").write_bytes(b"extra-video")
        calls = []
        c.raises(lambda: _generate(module, fixture, calls=calls), ValueError,
                 "the locked collection contains exactly 49 videos total")
        c.eq(calls, [], "unexpected videos fail before runtime construction")
        c.true(not fixture.output_root.exists())


def test_unreadable_input_subtree_fails_closed_instead_of_being_skipped(c: Check):
    module = _load_module()
    with tempfile.TemporaryDirectory() as td:
        fixture = _make_fixture(Path(td))
        hidden = fixture.video_root / "hidden"
        hidden.mkdir()
        (hidden / "untracked_extra.mp4").write_bytes(b"hidden-extra-video")
        hidden.chmod(0)
        try:
            c.raises(lambda: module.collect_sources(fixture.video_root), ValueError,
                     "unreadable input trees cannot silently disappear from inventory")
        finally:
            hidden.chmod(0o700)


def test_success_writes_exact_deidentified_schemas_and_uses_cpu_runtime(c: Check):
    module = _load_module()
    with tempfile.TemporaryDirectory() as td:
        fixture = _make_fixture(Path(td))
        result, encoder, landmarker, calls = _generate(module, fixture)

        c.eq(result["counts"], {"affected": 27, "unaffected": 22, "total": 49})
        c.eq(encoder.to_calls, ["cpu"], "the encoder is pinned to CPU")
        c.eq(encoder.eval_calls, 1, "the frozen encoder is put in eval mode")
        c.eq(len(encoder.calls), 49, "every source is encoded exactly once")
        c.true(all(row[1] == 4 for row in encoder.calls), "n_clips is fixed at four")
        c.true(all(row[2] is landmarker for row in encoder.calls),
               "one explicit landmarker is reused")
        c.eq(calls[0], ("deterministic", "cpu"), "determinism is configured first")
        c.eq(calls[1], ("encoder", fixture.implementation_root, fixture.marlin_dir))
        c.eq(calls[2], ("landmarker", fixture.implementation_root, fixture.face_model))

        provenance = json.loads((fixture.output_root / "provenance.json").read_text())
        manifest = json.loads((fixture.output_root / "extraction_manifest.json").read_text())
        c.eq(set(provenance), {"schema_version", "dataset", "records"})
        c.eq(provenance["schema_version"], "palsynet_bundle_provenance_v1")
        c.eq(provenance["dataset"], "PalsyNet")
        c.eq(len(provenance["records"]), 49)
        c.true(all(set(row) == {"source_sha256", "bundle_key", "bundle_sha256"}
                   for row in provenance["records"]),
               "audit provenance records have the exact consumable schema")
        c.eq(set(manifest), EXPECTED_MANIFEST_FIELDS)
        c.eq(manifest["schema_version"], "palsynet_marlin_extraction_manifest_v1")
        c.eq(manifest["config"], {
            "bundle_schema_version": "palsynet_marlin_identity_v1",
            "capture_mirrored": "unknown",
            "device": "cpu",
            "embedding_dim": 768,
            "n_clips": 4,
        })
        c.eq(len(manifest["records"]), 49)
        c.true(all(set(row) == {
            "source_sha256", "label", "bundle_key", "bundle_sha256", "embedding_sha256"
        } for row in manifest["records"]), "manifest records have no identifier field")
        c.eq(
            [row["source_sha256"] for row in manifest["records"]],
            sorted(row["source_sha256"] for row in manifest["records"]),
            "records are ordered by content, never by filename",
        )
        c.true(len(manifest["implementation_source_tree_sha256"]) == 64)

        row = manifest["records"][0]
        bundle_path = fixture.output_root / row["bundle_key"]
        c.eq(hashlib.sha256(bundle_path.read_bytes()).hexdigest(), row["bundle_sha256"])
        with np.load(bundle_path, allow_pickle=False) as bundle:
            c.eq(set(bundle.files), EXPECTED_BUNDLE_FIELDS)
            c.eq(bundle["marlin"].shape, (4, 768))
            c.eq(bundle["marlin"].dtype, np.dtype(np.float32))
            c.eq(bundle["schema_version"].shape, ())
            c.eq(bundle["schema_version"].item(), "palsynet_marlin_identity_v1")
            c.eq(bundle["source_sha256"].item(), row["source_sha256"])
            c.eq(bundle["n_clips"].shape, ())
            c.eq(bundle["n_clips"].dtype, np.dtype(np.int64))
            c.eq(bundle["n_clips"].item(), 4)
            c.eq(bundle["capture_mirrored"].item(), "unknown")


def test_outputs_never_contain_raw_names_stems_or_paths(c: Check):
    module = _load_module()
    with tempfile.TemporaryDirectory() as td:
        secret = "Alice_Mayo_Secret_Identifier"
        fixture = _make_fixture(Path(td), name_token=secret)
        _generate(module, fixture)
        artifacts = [path for path in fixture.output_root.rglob("*") if path.is_file()]
        c.eq(len(artifacts), 51, "49 bundles plus two manifests are the only files")
        for path in artifacts:
            relative = path.relative_to(fixture.output_root).as_posix()
            c.true(secret not in relative, "raw stem is absent from artifact keys")
            payload = path.read_bytes()
            c.true(secret.encode() not in payload, "raw stem is absent from artifact bytes")
            c.true(str(fixture.video_root).encode() not in payload,
                   "raw source root is absent from artifact bytes")
        bundle_dirs = sorted((fixture.output_root / "bundles").iterdir())
        c.true(all(len(path.name) == 64 and path.name == path.name.lower()
                   for path in bundle_dirs), "bundle directories are SHA-256 only")


def test_content_order_is_independent_of_creation_and_filename_order(c: Check):
    module = _load_module()
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        first = _make_fixture(root / "first", reverse_creation=False, name_token="FirstNames")
        second = _make_fixture(root / "second", reverse_creation=True, name_token="OtherNames")
        _generate(module, first)
        _generate(module, second)
        first_manifest = json.loads((first.output_root / "extraction_manifest.json").read_text())
        second_manifest = json.loads((second.output_root / "extraction_manifest.json").read_text())
        first_rows = [(r["source_sha256"], r["label"], r["embedding_sha256"],
                       r["bundle_sha256"])
                      for r in first_manifest["records"]]
        second_rows = [(r["source_sha256"], r["label"], r["embedding_sha256"],
                        r["bundle_sha256"])
                       for r in second_manifest["records"]]
        c.eq(first_rows, second_rows, "content-identical collections have identical order")
        c.eq(first_manifest["collection_fingerprints"]["source_collection_sha256"],
             second_manifest["collection_fingerprints"]["source_collection_sha256"])
        c.eq(first_manifest["collection_fingerprints"]["bundle_collection_sha256"],
             second_manifest["collection_fingerprints"]["bundle_collection_sha256"],
             "deterministic bundles have an identical collection fingerprint")


def test_missing_wrong_nonfinite_or_wrong_dtype_windows_fail_closed(c: Check):
    module = _load_module()
    cases = {
        "missing": None,
        "wrong_windows": np.zeros((3, 768), dtype=np.float32),
        "wrong_width": np.zeros((4, 767), dtype=np.float32),
        "wrong_dtype": np.zeros((4, 768), dtype=np.float64),
        "nonfinite": np.full((4, 768), np.nan, dtype=np.float32),
    }
    for name, invalid in cases.items():
        with tempfile.TemporaryDirectory() as td:
            fixture = _make_fixture(Path(td))
            encoder = FakeEncoder(lambda _source, _index, invalid=invalid: invalid)
            c.raises(lambda: _generate(module, fixture, encoder), ValueError,
                     f"{name} MARLIN output must fail")
            c.true(not fixture.output_root.exists(), f"{name} leaves no generation")


def test_source_mutation_and_asset_hash_drift_preserve_existing_output(c: Check):
    module = _load_module()
    for drift_kind in ("source", "asset"):
        with tempfile.TemporaryDirectory() as td:
            fixture = _make_fixture(Path(td))
            fixture.output_root.mkdir(parents=True)
            marker = fixture.output_root / "previous_generation.marker"
            marker.write_bytes(b"must-survive")

            def behavior(source, index, kind=drift_kind):
                if index == 1:
                    if kind == "source":
                        source.write_bytes(source.read_bytes() + b"-changed")
                    else:
                        model = fixture.marlin_dir / "model.safetensors"
                        model.write_bytes(model.read_bytes() + b"-changed")
                return NotImplemented

            c.raises(lambda: _generate(module, fixture, FakeEncoder(behavior)), ValueError,
                     f"{drift_kind} drift fails closed")
            c.eq(marker.read_bytes(), b"must-survive", "old output is untouched")
            leftovers = [p.name for p in fixture.output_root.parent.iterdir()
                         if ".staging." in p.name or p.name.endswith(".lock")]
            c.eq(leftovers, [], "transaction debris is removed")


def test_late_source_addition_is_detected_before_promotion(c: Check):
    module = _load_module()
    with tempfile.TemporaryDirectory() as td:
        fixture = _make_fixture(Path(td))
        fixture.output_root.mkdir(parents=True)
        marker = fixture.output_root / "previous_generation.marker"
        marker.write_bytes(b"must-survive")

        def add_source(_source, index):
            if index == 1:
                (fixture.video_root / "unaffected" / "late_extra.mp4").write_bytes(
                    b"late-collection-member"
                )
            return NotImplemented

        c.raises(lambda: _generate(module, fixture, FakeEncoder(add_source)), ValueError,
                 "collection membership drift fails closed")
        c.eq(marker.read_bytes(), b"must-survive", "the previous output is untouched")
        c.true((fixture.video_root / "unaffected" / "late_extra.mp4").exists(),
               "the input mutation is observed but never silently deleted")


def test_post_validation_source_or_asset_drift_blocks_promotion(c: Check):
    module = _load_module()
    for drift_kind in ("source", "asset"):
        with tempfile.TemporaryDirectory() as td:
            fixture = _make_fixture(Path(td))
            fixture.output_root.mkdir(parents=True)
            marker = fixture.output_root / "previous.marker"
            marker.write_bytes(b"old-valid-generation")
            original_validate = module._validate_generation

            def validate_then_mutate(*args, kind=drift_kind, **kwargs):
                original_validate(*args, **kwargs)
                if kind == "source":
                    source = next((fixture.video_root / "affected").glob("*.mp4"))
                    source.write_bytes(source.read_bytes() + b"-late-change")
                else:
                    config = fixture.marlin_dir / "config.json"
                    config.write_text('{"model":"late-changed"}\n')

            module._validate_generation = validate_then_mutate
            try:
                c.raises(lambda: _generate(module, fixture), ValueError,
                         f"post-validation {drift_kind} drift fails closed")
            finally:
                module._validate_generation = original_validate
            c.eq(marker.read_bytes(), b"old-valid-generation",
                 "late drift never replaces the previous target")


def test_failed_49th_preserves_valid_generation_and_rerun_removes_stale(c: Check):
    module = _load_module()
    with tempfile.TemporaryDirectory() as td:
        fixture = _make_fixture(Path(td))
        _generate(module, fixture)
        before = _tree_hashes(fixture.output_root)

        def fail_last(_source, index):
            if index == 49:
                raise RuntimeError("injected final encode failure")
            return NotImplemented

        c.raises(lambda: _generate(module, fixture, FakeEncoder(fail_last)), RuntimeError,
                 "a final-record failure aborts the whole generation")
        c.eq(_tree_hashes(fixture.output_root), before,
             "the previous valid generation is byte-for-byte preserved")

        stale = fixture.output_root / "stale_from_old_generation.txt"
        stale.write_text("stale")
        _generate(module, fixture)
        c.true(not stale.exists(), "whole-directory promotion removes stale files")


def test_promotion_durability_failure_restores_previous_generation(c: Check):
    module = _load_module()
    with tempfile.TemporaryDirectory() as td:
        parent = Path(td)
        target = parent / "identity_marlin_v1"
        staging = parent / ".identity_marlin_v1.staging.test"
        target.mkdir()
        staging.mkdir()
        (target / "old.marker").write_bytes(b"old-valid-generation")
        (staging / "new.marker").write_bytes(b"new-generation")
        original_fsync = module._fsync_directory
        calls = []

        def fail_second_fsync(path):
            calls.append(Path(path))
            if len(calls) == 2:
                raise OSError("injected durability failure")
            return original_fsync(path)

        module._fsync_directory = fail_second_fsync
        try:
            c.raises(lambda: module._promote_generation(staging, target), OSError,
                     "a failed durable promotion is reported")
        finally:
            module._fsync_directory = original_fsync
        c.eq((target / "old.marker").read_bytes(), b"old-valid-generation",
             "the previously valid target is restored")
        c.true(not (target / "new.marker").exists(), "failed new output is not exposed")


def test_post_backup_durability_failure_restores_previous_generation(c: Check):
    module = _load_module()
    with tempfile.TemporaryDirectory() as td:
        parent = Path(td)
        target = parent / "identity_marlin_v1"
        staging = parent / ".identity_marlin_v1.staging.test"
        target.mkdir()
        staging.mkdir()
        (target / "old.marker").write_bytes(b"old-valid-generation")
        (staging / "new.marker").write_bytes(b"new-generation")
        original_fsync = module._fsync_directory
        calls = []

        def fail_first_fsync(path):
            calls.append(Path(path))
            if len(calls) == 1:
                raise OSError("injected post-backup durability failure")
            return original_fsync(path)

        module._fsync_directory = fail_first_fsync
        try:
            c.raises(lambda: module._promote_generation(staging, target), OSError,
                     "a failed backup durability barrier is reported")
        finally:
            module._fsync_directory = original_fsync
        c.eq((target / "old.marker").read_bytes(), b"old-valid-generation",
             "the target is restored even before staging promotion begins")
        c.true(not (target / "new.marker").exists(), "new output was never exposed")


def test_first_backup_rename_failure_never_moves_or_deletes_old_target(c: Check):
    module = _load_module()
    with tempfile.TemporaryDirectory() as td:
        parent = Path(td)
        target = parent / "identity_marlin_v1"
        staging = parent / ".identity_marlin_v1.staging.test"
        target.mkdir()
        staging.mkdir()
        (target / "old.marker").write_bytes(b"old-valid-generation")
        (staging / "new.marker").write_bytes(b"new-generation")
        original_replace = module.os.replace
        calls = []

        def fail_first_replace(source, destination):
            calls.append((Path(source), Path(destination)))
            if len(calls) == 1:
                raise OSError("injected first rename failure")
            return original_replace(source, destination)

        module.os.replace = fail_first_replace
        try:
            c.raises(lambda: module._promote_generation(staging, target), OSError,
                     "the first backup rename failure is reported")
        finally:
            module.os.replace = original_replace
        c.eq((target / "old.marker").read_bytes(), b"old-valid-generation",
             "the old target is untouched when no backup was created")
        c.eq((staging / "new.marker").read_bytes(), b"new-generation",
             "the fresh staging tree remains available to caller cleanup")


def test_staging_barrier_failure_cleans_stage_and_lock_without_touching_old(c: Check):
    module = _load_module()
    with tempfile.TemporaryDirectory() as td:
        fixture = _make_fixture(Path(td))
        fixture.output_root.mkdir(parents=True)
        marker = fixture.output_root / "previous.marker"
        marker.write_bytes(b"old-valid-generation")
        original_fsync = module._fsync_directory
        injected = []

        def fail_when_staging_exists(path):
            parent = Path(path)
            has_stage = parent.is_dir() and any(
                child.name.startswith(".identity_marlin_v1.staging.")
                for child in parent.iterdir()
            )
            if has_stage and not injected:
                injected.append(True)
                raise OSError("injected staging durability failure")
            return original_fsync(path)

        module._fsync_directory = fail_when_staging_exists
        try:
            c.raises(lambda: _generate(module, fixture), OSError,
                     "staging durability failure aborts generation")
        finally:
            module._fsync_directory = original_fsync
        c.eq(marker.read_bytes(), b"old-valid-generation")
        debris = [child.name for child in fixture.output_root.parent.iterdir()
                  if ".staging." in child.name or child.name.endswith(".lock")]
        c.eq(debris, [], "failed staging creation leaves neither stage nor lock")


def test_staging_name_collision_never_deletes_directory_not_owned_by_run(c: Check):
    module = _load_module()
    with tempfile.TemporaryDirectory() as td:
        fixture = _make_fixture(Path(td))
        output_parent = fixture.output_root.parent
        output_parent.mkdir(parents=True)
        collision = output_parent / ".identity_marlin_v1.staging.fixedtoken"
        collision.mkdir()
        marker = collision / "preexisting.marker"
        marker.write_bytes(b"not-owned-by-this-run")
        original_token_hex = module.secrets.token_hex
        module.secrets.token_hex = lambda _n: "fixedtoken"
        try:
            c.raises(lambda: _generate(module, fixture), FileExistsError,
                     "fresh staging names never overwrite existing siblings")
        finally:
            module.secrets.token_hex = original_token_hex
        c.eq(marker.read_bytes(), b"not-owned-by-this-run",
             "a collided staging directory is preserved byte-for-byte")
        c.true(not (output_parent / ".identity_marlin_v1.lock").exists(),
               "lock is still released")


def test_encoder_failure_diagnostic_never_exposes_sensitive_source_path(c: Check):
    module = _load_module()
    with tempfile.TemporaryDirectory() as td:
        secret = "Highly_Sensitive_Raw_Person_Name"
        fixture = _make_fixture(Path(td), name_token=secret)

        def fail_with_path(source, _index):
            raise IOError(f"cannot open video: {source}")

        try:
            _generate(module, fixture, FakeEncoder(fail_with_path))
        except RuntimeError:
            diagnostic = traceback.format_exc()
        else:
            raise AssertionError("injected encoder failure did not propagate")
        c.true(secret not in diagnostic, "raw source stems are suppressed from diagnostics")
        c.true(str(fixture.video_root) not in diagnostic,
               "raw source roots are suppressed from diagnostics")


def test_cli_boundary_suppresses_all_underlying_path_diagnostics(c: Check):
    module = _load_module()
    secret = "Highly_Sensitive_Raw_Person_Name.mp4"
    original_generate = module.generate_identity_cache

    def fail_with_sensitive_path(**_kwargs):
        raise FileNotFoundError(f"missing /private/raw/patients/{secret}")

    module.generate_identity_cache = fail_with_sensitive_path
    stderr = io.StringIO()
    try:
        with redirect_stderr(stderr):
            c.raises(
                lambda: module.main([
                    "--video-root", "/private/raw/patients",
                    "--implementation-root", "/private/code",
                    "--marlin-dir", "/private/model",
                    "--face-landmarker-model", "/private/face.task",
                    "--output-root", "/private/output",
                ]),
                SystemExit,
                "CLI fails with one sanitized diagnostic",
            )
    finally:
        module.generate_identity_cache = original_generate
    diagnostic = stderr.getvalue()
    c.true(secret not in diagnostic and "/private/" not in diagnostic,
           "no underlying filename or path reaches CLI stderr")
    c.true("canonical output state requires verification" in diagnostic,
           "the operator still gets a useful fail-closed status")


def test_path_escape_symlink_and_missing_assets_fail_before_runtime(c: Check):
    module = _load_module()
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        fixture = _make_fixture(root)
        calls = []
        wrong_output = root / "escaped-output"
        c.raises(
            lambda: module.generate_identity_cache(
                video_root=fixture.video_root,
                implementation_root=fixture.implementation_root,
                marlin_dir=fixture.marlin_dir,
                face_landmarker_model=fixture.face_model,
                output_root=wrong_output,
                encoder_factory=lambda *_: calls.append("encoder"),
                landmarker_factory=lambda *_: calls.append("landmarker"),
            ),
            ValueError,
            "output cannot escape the canonical derived location",
        )
        c.eq(calls, [])
        c.true(not wrong_output.exists())

    with tempfile.TemporaryDirectory() as td:
        fixture = _make_fixture(Path(td))
        source = next((fixture.video_root / "affected").glob("*.mp4"))
        target = source.with_name("symlinked_secret.mp4")
        target.symlink_to(source)
        c.raises(lambda: _generate(module, fixture), ValueError,
                 "a source symlink is rejected")
        c.true(not fixture.output_root.exists())

    with tempfile.TemporaryDirectory() as td:
        fixture = _make_fixture(Path(td))
        fixture.face_model.unlink()
        calls = []
        c.raises(lambda: _generate(module, fixture, calls=calls), ValueError,
                 "missing explicit landmarker assets fail before runtime")
        c.eq(calls, [], "no runtime factory can trigger a fallback download")
        c.true(not fixture.output_root.exists())
        source_text = SCRIPT.read_text()
        c.true("urlretrieve" not in source_text and "urllib" not in source_text
               and "requests" not in source_text and "http://" not in source_text
               and "https://" not in source_text,
               "the regeneration command has no network implementation path")


def test_intermediate_asset_symlink_is_rejected_before_staging(c: Check):
    module = _load_module()
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        fixture = _make_fixture(root)
        original_data = fixture.implementation_root / "data"
        relocated_data = root / "relocated_data"
        original_data.rename(relocated_data)
        original_data.symlink_to(relocated_data, target_is_directory=True)
        calls = []
        c.raises(lambda: _generate(module, fixture, calls=calls), ValueError,
                 "an intermediate MARLIN asset symlink is unsafe")
        c.eq(calls, [], "runtime is not loaded through a symlinked asset tree")
        c.true(not fixture.output_root.exists(), "no staging output is created")


def test_default_landmarker_runtime_disables_missing_asset_fallback(c: Check):
    module = _load_module()
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        implementation = root / "implementation"
        utils_root = implementation / "src" / "baselines" / "oo_multimodal" / "utils"
        utils_root.mkdir(parents=True)
        marker = root / "fallback_was_used.marker"
        model = root / "face.task"
        model.write_bytes(b"present-before-construction")
        (utils_root / "__init__.py").write_text(
            "from .mediapipe_wrapper import MediaPipeFaceLandmarker\n"
        )
        (utils_root / "mediapipe_wrapper.py").write_text(
            "from pathlib import Path\n"
            f"MARKER = Path({str(marker)!r})\n"
            "def _ensure_model(path):\n"
            "    path = Path(path)\n"
            "    if not path.exists():\n"
            "        MARKER.write_text('unsafe fallback')\n"
            "        path.write_bytes(b'fallback')\n"
            "    return path\n"
            "class MediaPipeFaceLandmarker:\n"
            "    def __init__(self, model_path):\n"
            "        path = Path(model_path)\n"
            "        path.unlink()\n"
            "        _ensure_model(path)\n"
        )
        previous = {key: sys.modules.pop(key) for key in list(sys.modules)
                    if key == "utils" or key.startswith("utils.")}
        try:
            c.raises(
                lambda: module._default_landmarker_factory(implementation, model),
                ValueError,
                "an asset disappearing at construction must fail closed",
            )
            c.true(not marker.exists(), "the wrapper fallback is never reached")
        finally:
            for key in list(sys.modules):
                if key == "utils" or key.startswith("utils."):
                    sys.modules.pop(key)
            sys.modules.update(previous)


def test_default_runtime_fresh_imports_all_explicit_modules_until_encoding_finishes(c: Check):
    module = _load_module()
    prefixes = ("src", "marlin_vit_base_ytf", "utils")
    saved = {
        key: value for key, value in list(sys.modules.items())
        if any(key == prefix or key.startswith(prefix + ".") for prefix in prefixes)
    }
    for key in saved:
        sys.modules.pop(key, None)
    with tempfile.TemporaryDirectory() as td:
        fixture = _make_fixture(Path(td))
        implementation = fixture.implementation_root
        for init_path in (
            implementation / "src" / "__init__.py",
            implementation / "src" / "models" / "__init__.py",
            implementation / "src" / "models" / "backbones" / "__init__.py",
        ):
            init_path.parent.mkdir(parents=True, exist_ok=True)
            init_path.write_text("# explicit package\n")
        (fixture.marlin_dir / "__init__.py").write_text("# trusted MARLIN package\n")
        (fixture.marlin_dir / "marlin.py").write_text(
            "class Marlin:\n"
            "    ORIGIN = 'trusted'\n"
        )
        (implementation / "src" / "models" / "backbones" / "marlin_video.py").write_text(
            "from pathlib import Path\n"
            "import sys\n"
            "import numpy as np\n"
            "from marlin_vit_base_ytf.marlin import Marlin\n"
            "sys.path.insert(0, '/runtime-added-marlin-path')\n"
            "class MarlinVideoEncoder:\n"
            "    def __init__(self, origin): self.origin = origin\n"
            "    @classmethod\n"
            "    def from_default_weights(cls, marlin_dir):\n"
            "        expected = Path(__file__).parents[3] / 'data' / 'external' / 'marlin_vit_base_ytf'\n"
            "        if Path(marlin_dir) != expected: raise RuntimeError('wrong explicit MARLIN dir')\n"
            "        return cls(Marlin.ORIGIN)\n"
            "    def to(self, device):\n"
            "        if device != 'cpu': raise RuntimeError('not cpu')\n"
            "        return self\n"
            "    def eval(self): return self\n"
            "    def encode_video_path(self, source, n_clips, landmarker):\n"
            "        import utils\n"
            "        if '/runtime-added-lazy-path' not in sys.path:\n"
            "            sys.path.insert(0, '/runtime-added-lazy-path')\n"
            "        if self.origin != 'trusted' or utils.RUNTIME_ORIGIN != 'trusted':\n"
            "            raise RuntimeError('ambient module cache used')\n"
            "        return np.ones((n_clips, 768), dtype=np.float32)\n"
        )
        utils_root = implementation / "src" / "baselines" / "oo_multimodal" / "utils"
        (utils_root / "__init__.py").write_text(
            "from .mediapipe_wrapper import MediaPipeFaceLandmarker\n"
            "RUNTIME_ORIGIN = 'trusted'\n"
        )
        (utils_root / "mediapipe_wrapper.py").write_text(
            "from pathlib import Path\n"
            "def _ensure_model(path):\n"
            "    path = Path(path)\n"
            "    if not path.is_file(): raise ValueError('missing explicit model')\n"
            "    return path\n"
            "class MediaPipeFaceLandmarker:\n"
            "    def __init__(self, model_path):\n"
            "        _ensure_model(model_path)\n"
            "        self.origin = 'trusted'\n"
        )

        sys.path.insert(0, str(fixture.marlin_dir.parent))
        trusted_package = importlib.import_module("marlin_vit_base_ytf")
        sys.path.pop(0)
        evil_marlin = types.ModuleType("marlin_vit_base_ytf.marlin")
        evil_marlin.__file__ = str(Path(td) / "outside" / "marlin.py")
        evil_marlin.Marlin = type("Marlin", (), {"ORIGIN": "evil"})
        sys.modules["marlin_vit_base_ytf.marlin"] = evil_marlin
        trusted_package.marlin = evil_marlin

        oo_root = implementation / "src" / "baselines" / "oo_multimodal"
        sys.path.insert(0, str(oo_root))
        poisoned_utils = importlib.import_module("utils")
        sys.path.pop(0)
        poisoned_utils.RUNTIME_ORIGIN = "evil"
        poisoned_utils.MediaPipeFaceLandmarker = type(
            "EvilLandmarker", (), {"__init__": lambda self, model_path: None}
        )
        path_before = list(sys.path)
        try:
            result = module.generate_identity_cache(
                video_root=fixture.video_root,
                implementation_root=implementation,
                marlin_dir=fixture.marlin_dir,
                face_landmarker_model=fixture.face_model,
                output_root=fixture.output_root,
                dependency_versions_fn=lambda: {
                    "mediapipe": "test-1", "numpy": "test-2",
                    "opencv-python": "test-3", "python": "test-4",
                    "safetensors": "test-5", "torch": "test-6",
                },
                deterministic_setup_fn=lambda: None,
            )
            c.eq(result["counts"]["total"], 49)
            c.true(sys.modules["marlin_vit_base_ytf.marlin"] is evil_marlin,
                   "ambient MARLIN modules are restored after extraction")
            c.true(sys.modules["utils"] is poisoned_utils,
                   "ambient utils package is restored after extraction")
            c.eq(sys.path, path_before,
                 "runtime imports cannot permanently change subsequent import resolution")
        finally:
            for key in list(sys.modules):
                if any(key == prefix or key.startswith(prefix + ".") for prefix in prefixes):
                    sys.modules.pop(key)
            sys.modules.update(saved)


def test_default_deterministic_setup_reseeds_cpu_random_sources(c: Check):
    import torch

    module = _load_module()
    module._default_deterministic_setup()
    first = (
        module.random.random(),
        float(np.random.random()),
        torch.rand(4, device="cpu").tolist(),
    )
    module._default_deterministic_setup()
    second = (
        module.random.random(),
        float(np.random.random()),
        torch.rand(4, device="cpu").tolist(),
    )
    c.eq(first, second, "Python, NumPy, and Torch CPU RNGs are reproducibly seeded")
    c.true(torch.are_deterministic_algorithms_enabled(),
           "Torch rejects nondeterministic algorithms")


def test_runtime_isolation_executes_current_source_not_stale_timestamp_pyc(c: Check):
    module = _load_module()
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        module_name = "stale_pyc_probe"
        source = root / f"{module_name}.py"
        source.write_text("VALUE = 'old'\n")
        sys.path.insert(0, str(root))
        previous_dont_write = sys.dont_write_bytecode
        try:
            sys.dont_write_bytecode = False
            imported = importlib.import_module(module_name)
            c.eq(imported.VALUE, "old")
            sys.modules.pop(module_name, None)
            info = source.stat()
            source.write_text("VALUE = 'new'\n")
            os.utime(source, ns=(info.st_atime_ns, info.st_mtime_ns))
            importlib.invalidate_caches()
            with module._isolated_runtime_namespaces():
                current = importlib.import_module(module_name)
                c.eq(current.VALUE, "new",
                     "isolated runtime bypasses a same-size same-mtime stale pyc")
        finally:
            sys.modules.pop(module_name, None)
            sys.dont_write_bytecode = previous_dont_write
            if sys.path and sys.path[0] == str(root):
                sys.path.pop(0)


if __name__ == "__main__":
    run_all(__name__, globals())
