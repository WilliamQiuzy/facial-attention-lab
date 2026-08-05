"""Pure-contract tests for the local, deidentified PalsyNet identity audit."""
from __future__ import annotations

import hashlib
import hmac
import inspect
import json
import os
import stat
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.audit_palsynet_identity as audit_module  # noqa: E402
from scripts.audit_palsynet_identity import (  # noqa: E402
    CANONICAL_OUTPUT_ROOT,
    IdentityRecord,
    build_manifest,
    compose_contact_sheet,
    default_group_id,
    deterministic_window_starts,
    generate_contact_sheets,
    load_group_overrides,
    load_marlin_embedding,
    load_or_create_salt,
    opaque_recording_id,
    rank_cosine_pairs,
    read_representative_frames,
    run_audit,
    validate_output_root,
    validate_group_overrides,
)
from _testlib import Check, run_all  # noqa: E402


def _record(
    source_sha256: str,
    label: str,
    salt: bytes,
    private_name: str,
    embedding: np.ndarray | None = None,
) -> IdentityRecord:
    return IdentityRecord(
        source_path=Path("/private/raw PalsyNet") / private_name,
        source_stem=Path(private_name).stem,
        label=label,
        source_sha256=source_sha256,
        bundle_sha256=hashlib.sha256(
            ("synthetic-bundle:" + private_name).encode("utf-8")
        ).hexdigest(),
        recording_id=opaque_recording_id(source_sha256, salt),
        group_id=default_group_id(source_sha256, salt),
        embedding=(
            np.asarray(embedding, dtype=np.float64)
            if embedding is not None
            else np.r_[1.0, np.zeros(767, dtype=np.float64)]
        ),
    )


def _write_bundle(path: Path, marker: float) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    marlin = np.zeros((1, 768), dtype=np.float32)
    marlin[0, 0] = marker
    marlin[0, 1] = 1.0
    np.savez(path, marlin=marlin)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_provenance(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_text(json.dumps({
        "schema_version": "palsynet_bundle_provenance_v1",
        "dataset": "PalsyNet",
        "records": rows,
    }))


def _temporary_canonical(root: Path):
    """Patch the module-level canonical path; caller restores in ``finally``."""
    original_project_root = audit_module.PROJECT_ROOT
    original_output_root = audit_module.CANONICAL_OUTPUT_ROOT
    project_root = root / "facial_paralysis"
    project_root.mkdir(mode=0o700)
    canonical = (
        project_root / "outputs" / "palsynet_identity_audit"
        / "generation"
    )
    audit_module.PROJECT_ROOT = project_root
    audit_module.CANONICAL_OUTPUT_ROOT = canonical
    return original_project_root, original_output_root, project_root, canonical


def test_hmac_ids_are_deterministic_and_domain_separated(c: Check):
    salt = bytes(range(32))
    source_sha256 = hashlib.sha256(b"synthetic-video").hexdigest()
    expected_recording = "rec_" + hmac.new(
        salt, source_sha256.encode("ascii"), hashlib.sha256
    ).hexdigest()
    expected_group = "grp_" + hmac.new(
        salt, b"group:" + source_sha256.encode("ascii"), hashlib.sha256
    ).hexdigest()

    c.eq(opaque_recording_id(source_sha256, salt), expected_recording,
         "recording id uses HMAC-SHA256 over the source digest")
    c.eq(default_group_id(source_sha256, salt), expected_group,
         "group id uses an independent HMAC domain")
    c.true(expected_recording[4:] != expected_group[4:],
           "recording and group identifiers cannot coincide")


def test_local_salt_is_stable_and_private(c: Check):
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "local" / "audit_salt.bin"
        first = load_or_create_salt(path)
        second = load_or_create_salt(path)
        mode = stat.S_IMODE(path.stat().st_mode)
    c.eq(first, second, "existing local salt is reused")
    c.eq(len(first), 32, "new salts contain 256 random bits")
    c.eq(mode & 0o077, 0, "salt is not group/world accessible")


def test_insecure_existing_salt_fails_without_mutating_prior_audit(c: Check):
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "audit_salt.bin"
        path.write_bytes(b"i" * 32)
        path.chmod(0o640)
        c.raises(lambda: load_or_create_salt(path), ValueError,
                 "an unexpectedly accessible existing salt fails closed")
        c.eq(stat.S_IMODE(path.stat().st_mode), 0o640,
             "validation does not chmod a previous generation before commit")


def test_output_root_is_constrained_to_the_ignored_audit_directory(c: Check):
    c.eq(validate_output_root(CANONICAL_OUTPUT_ROOT), CANONICAL_OUTPUT_ROOT.resolve(),
         "the repository's precise ignored audit root is accepted")
    with tempfile.TemporaryDirectory() as td:
        tracked_typo = Path(td) / "palsynet_identity_audti"
        c.raises(lambda: validate_output_root(tracked_typo), ValueError,
                 "a typo cannot publish face sheets or salt outside the ignored root")


def test_output_tree_rejects_every_symlink_escape_before_any_write(c: Check):
    scenarios = (
        "root",
        "contact_sheets",
        "recordings",
        "pairs",
        "manifest",
        "image",
    )
    for scenario in scenarios:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            originals = _temporary_canonical(root)
            original_project_root, original_output_root, _, canonical = originals
            canonical.parent.mkdir(parents=True, mode=0o700)
            outside = root / "outside"
            outside.mkdir()
            sentinel = outside / "sentinel.txt"
            sentinel.write_text("untouched")
            try:
                if scenario == "root":
                    canonical.symlink_to(outside, target_is_directory=True)
                else:
                    canonical.mkdir()
                    contact = canonical / "contact_sheets"
                    if scenario == "contact_sheets":
                        contact.symlink_to(outside, target_is_directory=True)
                    else:
                        contact.mkdir()
                        recordings = contact / "recordings"
                        pairs = contact / "pairs"
                        if scenario == "recordings":
                            recordings.symlink_to(outside, target_is_directory=True)
                            pairs.mkdir()
                        elif scenario == "pairs":
                            recordings.mkdir()
                            pairs.symlink_to(outside, target_is_directory=True)
                        else:
                            recordings.mkdir()
                            pairs.mkdir()
                            if scenario == "manifest":
                                (canonical / "identity_manifest.json").symlink_to(
                                    outside / "escaped-manifest.json"
                                )
                            elif scenario == "image":
                                (recordings / ("rec_" + "a" * 64 + ".jpg")).symlink_to(
                                    outside / "escaped-image.jpg"
                                )
                c.raises(
                    lambda: run_audit(
                        video_root=root / "missing-videos",
                        bundle_root=root / "missing-bundles",
                        bundle_provenance=root / "missing-provenance.json",
                        output_root=canonical,
                    ),
                    ValueError,
                    f"{scenario} symlink must fail before audit generation",
                )
                c.eq(sentinel.read_text(), "untouched",
                     f"{scenario} symlink cannot mutate its outside target")
                c.true(not (outside / "escaped-manifest.json").exists(),
                       "manifest symlink destination is never written")
                c.true(not (outside / "escaped-image.jpg").exists(),
                       "image symlink destination is never written")
                c.true(not list(canonical.parent.glob(
                    ".palsynet_identity_audit.staging-*"
                )), "symlink rejection happens before staging is created")
            finally:
                audit_module.PROJECT_ROOT = original_project_root
                audit_module.CANONICAL_OUTPUT_ROOT = original_output_root


def test_generation_rejects_outputs_ancestor_symlink_before_any_io(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        originals = _temporary_canonical(root)
        original_project_root, original_output_root, project_root, canonical = originals
        outside = root / "outside"
        outside.mkdir(mode=0o700)
        sentinel = outside / "sentinel.txt"
        sentinel.write_text("untouched")
        (project_root / "outputs").symlink_to(
            outside, target_is_directory=True
        )
        calls: list[str] = []
        guarded = {
            "_read_salt": audit_module._read_salt,
            "collect_identity_records": audit_module.collect_identity_records,
            "sha256_file": audit_module.sha256_file,
        }

        def forbidden(name: str):
            def invoke(*_args, **_kwargs):
                calls.append(name)
                raise AssertionError(f"{name} ran before ancestor validation")
            return invoke

        for name in guarded:
            setattr(audit_module, name, forbidden(name))
        try:
            c.raises(lambda: run_audit(
                video_root=root / "videos",
                bundle_root=root / "bundles",
                bundle_provenance=root / "provenance.json",
                output_root=canonical,
            ), ValueError, "an outputs ancestor symlink fails before audit IO")
            c.eq(calls, [], "ancestor validation precedes reads and hashes")
            c.eq(sentinel.read_text(), "untouched",
                 "ancestor rejection cannot mutate the outside target")
            c.true(not (outside / "palsynet_identity_audit").exists(),
                   "no lifecycle path is created through the outputs symlink")
        finally:
            for name, value in guarded.items():
                setattr(audit_module, name, value)
            audit_module.PROJECT_ROOT = original_project_root
            audit_module.CANONICAL_OUTPUT_ROOT = original_output_root


def test_cli_rejects_noncanonical_output_and_external_salt_before_writing(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        originals = _temporary_canonical(root)
        original_project_root, original_output_root, project_root, canonical = originals
        wrong_output = project_root / "outputs" / "identity_typo"
        outside_salt = root / "tracked" / "audit_salt.bin"
        original_argv = sys.argv
        try:
            sys.argv = [
                "audit_palsynet_identity.py",
                "--video-root", str(root / "missing-videos"),
                "--bundle-root", str(root / "missing-bundles"),
                "--bundle-provenance", str(root / "missing-provenance.json"),
                "--output-root", str(wrong_output),
            ]
            c.raises(audit_module.main, ValueError,
                     "CLI rejects a noncanonical output root")
            c.true(not wrong_output.exists(), "bad output root is never created")

            sys.argv = [
                "audit_palsynet_identity.py",
                "--video-root", str(root / "missing-videos"),
                "--bundle-root", str(root / "missing-bundles"),
                "--bundle-provenance", str(root / "missing-provenance.json"),
                "--output-root", str(canonical),
                "--salt-file", str(outside_salt),
            ]
            c.raises(audit_module.main, ValueError,
                     "CLI rejects a salt outside the canonical ignored root")
            c.true(not canonical.exists(), "invalid salt fails before output mkdir")
            c.true(not outside_salt.exists(), "invalid salt file is never created")
        finally:
            sys.argv = original_argv
            audit_module.PROJECT_ROOT = original_project_root
            audit_module.CANONICAL_OUTPUT_ROOT = original_output_root


def test_generation_cli_has_no_reviewed_promotion_surface(c: Check):
    destinations = {
        action.dest for action in audit_module._parser()._actions
        if action.dest != "help"
    }
    c.eq(destinations, {
        "video_root",
        "bundle_root",
        "bundle_provenance",
        "output_root",
        "salt_file",
        "top_pairs",
    }, "generation cannot accept groups, evidence, or a reviewed-status flag")
    c.eq(audit_module._parser().get_default("top_pairs"), 1176,
         "generation defaults to every unordered recording pair")


def test_planned_generation_cli_creates_fresh_private_lifecycle_parent(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        originals = _temporary_canonical(root)
        original_project_root, original_output_root, _, canonical = originals
        original_argv = sys.argv
        replacements = {
            "collect_identity_records": lambda *_args, **_kwargs: [],
            "build_manifest": lambda *_args, **_kwargs: {
                "counts": {"total": 0},
                "identity_review": {"status": "unreviewed"},
            },
            "generate_contact_sheets": lambda *_args, **_kwargs: {
                "recordings": 0, "ranked_pairs": 0, "overview": 0,
            },
            "build_contact_sheet_inventory": lambda *_args, **_kwargs: {
                "probe": "inventory",
            },
            "build_review_ledger_template": lambda *_args, **_kwargs: {
                "probe": "ledger",
            },
            "sha256_file": lambda path: hashlib.sha256(
                Path(path).as_posix().encode("utf-8")
            ).hexdigest(),
            "_validate_generation": lambda *_args, **_kwargs: None,
        }
        original_functions = {
            name: getattr(audit_module, name) for name in replacements
        }

        def observe_collection(*_args, **_kwargs):
            c.true(canonical.parent.is_dir(),
                   "canonical lifecycle parent exists before source collection")
            c.eq(stat.S_IMODE(canonical.parent.stat().st_mode), 0o700,
                 "fresh lifecycle parent is owner-only")
            return []

        replacements["collect_identity_records"] = observe_collection
        for name, value in replacements.items():
            setattr(audit_module, name, value)
        try:
            c.true(not canonical.parent.exists(),
                   "fresh-checkout probe does not precreate the lifecycle parent")
            sys.argv = [
                "audit_palsynet_identity.py",
                "--video-root", str(root / "videos"),
                "--bundle-root", str(root / "bundles"),
                "--bundle-provenance", str(root / "provenance.json"),
                "--output-root", str(canonical),
            ]
            audit_module.main()
            c.true(canonical.is_dir(), "the exact planned CLI publishes generation")
            c.eq(stat.S_IMODE(canonical.stat().st_mode), 0o700,
                 "published generation remains owner-only")
        finally:
            sys.argv = original_argv
            for name, value in original_functions.items():
                setattr(audit_module, name, value)
            audit_module.PROJECT_ROOT = original_project_root
            audit_module.CANONICAL_OUTPUT_ROOT = original_output_root


def test_existing_generation_is_immutable_and_fails_before_collection(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        originals = _temporary_canonical(root)
        original_project_root, original_output_root, _, canonical = originals
        canonical.mkdir(parents=True)
        old_manifest = b'{"complete":"old"}\n'
        (canonical / "identity_manifest.json").write_bytes(old_manifest)
        original_collect = audit_module.collect_identity_records
        audit_module.collect_identity_records = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("immutable generation must fail before source collection")
        )
        try:
            c.raises(
                lambda: run_audit(
                    video_root=root / "videos",
                    bundle_root=root / "bundles",
                    bundle_provenance=root / "provenance.json",
                    output_root=canonical,
                ),
                FileExistsError,
                "a generated audit can never be replaced in place",
            )
            c.eq((canonical / "identity_manifest.json").read_bytes(), old_manifest,
                 "failed rerun preserves generated evidence byte-for-byte")
            c.true(not list(canonical.parent.glob(
                ".generation.staging-*"
            )), "failed staging directories are cleaned")
        finally:
            audit_module.collect_identity_records = original_collect
            audit_module.PROJECT_ROOT = original_project_root
            audit_module.CANONICAL_OUTPUT_ROOT = original_output_root


def test_generation_directory_publication_is_no_overwrite(c: Check):
    exclusive_rename = getattr(
        audit_module, "_rename_directory_exclusive", None
    )
    c.true(callable(exclusive_rename),
           "generation publication has an atomic exclusive rename primitive")
    c.true(
        "_rename_directory_exclusive" in inspect.getsource(
            audit_module._promote_generation
        ),
        "promotion uses atomic no-replace rather than check-then-rename",
    )
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        originals = _temporary_canonical(root)
        original_project_root, original_output_root, _, canonical = originals
        canonical.parent.mkdir(parents=True, mode=0o700)
        first_staging = canonical.parent / ".generation.staging-first"
        first_staging.mkdir()
        (first_staging / "sentinel").write_text("first")
        second_staging = canonical.parent / ".generation.staging-second"
        second_staging.mkdir()
        (second_staging / "sentinel").write_text("second")
        try:
            audit_module._promote_generation(first_staging, canonical)
            c.eq((canonical / "sentinel").read_text(), "first",
                 "fresh generation is published exactly once")
            c.raises(
                lambda: audit_module._promote_generation(second_staging, canonical),
                FileExistsError,
                "a second generation cannot replace the first",
            )
            c.eq((canonical / "sentinel").read_text(), "first",
                 "no-overwrite preserves the first generation")
        finally:
            audit_module.PROJECT_ROOT = original_project_root
            audit_module.CANONICAL_OUTPUT_ROOT = original_output_root


def test_marlin_loader_means_windows_and_l2_normalizes(c: Check):
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "clip.npz"
        windows = np.zeros((2, 768), dtype=np.float32)
        windows[0, 0] = 1.0
        windows[1, 1] = 1.0
        np.savez(path, marlin=windows)
        embedding = load_marlin_embedding(path)
    c.eq(embedding.shape, (768,), "window mean has frozen MARLIN width")
    c.true(np.isclose(np.linalg.norm(embedding), 1.0), "mean is L2 normalized")
    c.true(np.allclose(embedding[:2], np.sqrt(0.5)), "windows are averaged first")


def test_marlin_loader_rejects_missing_malformed_or_nonfinite_arrays(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cases = {
            "missing.npz": {"other": np.zeros((2, 768), dtype=np.float32)},
            "one_dimensional.npz": {"marlin": np.zeros(768, dtype=np.float32)},
            "wrong_width.npz": {"marlin": np.zeros((2, 767), dtype=np.float32)},
            "nonfinite.npz": {"marlin": np.full((2, 768), np.nan, dtype=np.float32)},
            "complex.npz": {"marlin": np.ones((2, 768), dtype=np.complex64) * (1 + 1j)},
            "zero_mean.npz": {"marlin": np.vstack([
                np.ones(768, dtype=np.float32), -np.ones(768, dtype=np.float32)
            ])},
        }
        for name, payload in cases.items():
            path = root / name
            np.savez(path, **payload)
            c.raises(lambda path=path: load_marlin_embedding(path), ValueError,
                     f"{name} must fail closed")


def test_bundle_provenance_requires_exact_source_and_bundle_hashes(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        bundle_root = root / "bundles"
        first_key, second_key = "opaque-a/clip.npz", "opaque-b/clip.npz"
        first_bundle = _write_bundle(bundle_root / first_key, 1.0)
        second_bundle = _write_bundle(bundle_root / second_key, 2.0)
        first_source = hashlib.sha256(b"source-a").hexdigest()
        second_source = hashlib.sha256(b"source-b").hexdigest()
        expected = {first_source, second_source}
        valid_rows = [
            {"source_sha256": first_source, "bundle_key": first_key,
             "bundle_sha256": first_bundle},
            {"source_sha256": second_source, "bundle_key": second_key,
             "bundle_sha256": second_bundle},
        ]
        provenance = root / "provenance.json"
        _write_provenance(provenance, valid_rows)

        loader = getattr(audit_module, "load_bundle_provenance", None)
        c.true(callable(loader), "a fail-closed trusted provenance loader is required")
        loaded = loader(provenance, bundle_root, expected)
        c.eq(set(loaded), expected, "trusted provenance covers every source exactly")
        c.eq(loaded[first_source].bundle_sha256, first_bundle,
             "validated provenance retains the observed bundle digest")

        cases = {
            "missing": valid_rows[:1],
            "extra": valid_rows + [{
                "source_sha256": hashlib.sha256(b"source-extra").hexdigest(),
                "bundle_key": first_key,
                "bundle_sha256": first_bundle,
            }],
            "stale": [
                {**valid_rows[0], "bundle_sha256": "f" * 64},
                valid_rows[1],
            ],
            "swapped": [
                {**valid_rows[0], "bundle_sha256": second_bundle},
                {**valid_rows[1], "bundle_sha256": first_bundle},
            ],
        }
        for name, rows in cases.items():
            _write_provenance(provenance, rows)
            c.raises(
                lambda: loader(provenance, bundle_root, expected),
                ValueError,
                f"{name} provenance cannot select a frozen bundle",
            )


def test_bundle_provenance_rejects_path_escape_duplicate_keys_and_symlinks(c: Check):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        bundle_root = root / "bundles"
        source_sha256 = hashlib.sha256(b"one-source").hexdigest()
        valid_path = bundle_root / "opaque" / "clip.npz"
        bundle_sha256 = _write_bundle(valid_path, 1.0)
        provenance = root / "provenance.json"
        loader = getattr(audit_module, "load_bundle_provenance", None)
        c.true(callable(loader), "trusted provenance loader exists")

        _write_provenance(provenance, [{
            "source_sha256": source_sha256,
            "bundle_key": "../escaped/clip.npz",
            "bundle_sha256": bundle_sha256,
        }])
        c.raises(lambda: loader(provenance, bundle_root, {source_sha256}),
                 ValueError, "bundle keys cannot escape the cache root")

        symlink_key = bundle_root / "linked" / "clip.npz"
        symlink_key.parent.mkdir(parents=True)
        symlink_key.symlink_to(valid_path)
        _write_provenance(provenance, [{
            "source_sha256": source_sha256,
            "bundle_key": "linked/clip.npz",
            "bundle_sha256": bundle_sha256,
        }])
        c.raises(lambda: loader(provenance, bundle_root, {source_sha256}),
                 ValueError, "trusted bundle paths cannot traverse symlinks")

        provenance.write_text(
            '{"schema_version":"palsynet_bundle_provenance_v1",'
            '"dataset":"PalsyNet","records":['
            '{"source_sha256":"' + source_sha256 + '",'
            '"source_sha256":"' + source_sha256 + '",'
            '"bundle_key":"opaque/clip.npz",'
            '"bundle_sha256":"' + bundle_sha256 + '"}]}'
        )
        c.raises(lambda: loader(provenance, bundle_root, {source_sha256}),
                 ValueError, "duplicate provenance keys are rejected")


def test_cosine_pair_ranking_is_complete_and_deterministic(c: Check):
    embeddings = {
        "rec_" + "c" * 64: np.array([0.0, 1.0]),
        "rec_" + "a" * 64: np.array([1.0, 0.0]),
        "rec_" + "b" * 64: np.array([1.0, 0.0]),
    }
    first = rank_cosine_pairs(embeddings)
    second = rank_cosine_pairs(dict(reversed(list(embeddings.items()))))
    c.eq(first, second, "input mapping order cannot change ranking")
    c.eq(len(first), 3, "all cross-recording pairs are ranked")
    c.eq([row["rank"] for row in first], [1, 2, 3], "ranks are contiguous")
    c.eq((first[0]["recording_id_a"], first[0]["recording_id_b"]),
         ("rec_" + "a" * 64, "rec_" + "b" * 64),
         "highest cosine pair comes first")
    c.true(np.isclose(first[0]["cosine"], 1.0), "cosine is preserved")


def test_manifest_is_deidentified_and_defaults_to_unreviewed_video_holdout(c: Check):
    salt = b"s" * 32
    records = [
        _record("1" * 64, "affected", salt, "Alice Raw Name.mp4"),
        _record("2" * 64, "unaffected", salt, "Bob Raw Name.mp4",
                np.r_[0.0, 1.0, np.zeros(766)]),
    ]
    pairs = rank_cosine_pairs({r.recording_id: r.embedding for r in records})
    manifest = build_manifest(records, pairs)
    encoded = json.dumps(manifest, allow_nan=False, sort_keys=True)

    c.eq(manifest["claim_unit"], "video_held_out",
         "identity is not inferred from hash uniqueness")
    c.eq(manifest["identity_review"]["status"], "unreviewed",
         "manual review remains explicit")
    c.true(all(r["identity_status"] == "unreviewed"
               and r["claim_unit"] == "video_held_out"
               for r in manifest["recordings"]),
           "recording claims remain conservative by default")
    c.true("Alice Raw Name" not in encoded and "Bob Raw Name" not in encoded,
           "raw filenames never enter the manifest")
    c.true("/private/" not in encoded and "source_path" not in encoded,
           "raw paths and path fields never enter the manifest")
    c.true(all(set(row) == {
        "recording_id", "group_id", "label", "source_sha256",
        "identity_status", "claim_unit"
    } for row in manifest["recordings"]),
           "per-record source hash supports deidentified cache joins")
    c.eq(
        {row["source_sha256"] for row in manifest["recordings"]},
        {"1" * 64, "2" * 64},
        "source hashes join raw videos to opaque recording and group ids",
    )
    c.true(all("bundle_sha256" not in row and "bundle_key" not in row
               for row in manifest["recordings"]),
           "cache-specific bundle provenance remains aggregate-only")
    c.eq(manifest["counts"], {"affected": 1, "unaffected": 1, "total": 2,
                              "ranked_pairs": 1},
         "aggregate counts are explicit")
    c.true(set(manifest["fingerprints"]) == {
        "source_collection_sha256", "bundle_provenance_sha256",
        "embedding_collection_sha256"
    }, "aggregate fingerprints are present")


def test_collection_fingerprints_are_salt_independent_and_use_normalized_embeddings(c: Check):
    def records_for(salt: bytes) -> list[IdentityRecord]:
        return [
            _record("a" * 64, "affected", salt, "one.mp4"),
            _record("b" * 64, "unaffected", salt, "two.mp4",
                    np.r_[0.0, 1.0, np.zeros(766)]),
        ]

    normal = records_for(b"x" * 32)
    swapped = records_for(b"y" * 32)
    scaled = records_for(b"x" * 32)
    for record in scaled:
        record.embedding *= 7.0
    scaled_before = [record.embedding.copy() for record in scaled]
    normal_pairs = rank_cosine_pairs({r.recording_id: r.embedding for r in normal})
    swapped_pairs = rank_cosine_pairs({r.recording_id: r.embedding for r in swapped})
    scaled_pairs = rank_cosine_pairs({r.recording_id: r.embedding for r in scaled})

    c.true(normal[0].recording_id < normal[1].recording_id
           and swapped[0].recording_id > swapped[1].recording_id,
           "the fixed salts exercise opposite opaque-id sort orders")

    normal_fingerprints = build_manifest(normal, normal_pairs)["fingerprints"]
    swapped_fingerprints = build_manifest(swapped, swapped_pairs)["fingerprints"]
    scaled_fingerprints = build_manifest(scaled, scaled_pairs)["fingerprints"]
    c.eq(
        normal_fingerprints["source_collection_sha256"],
        swapped_fingerprints["source_collection_sha256"],
        "source fingerprint depends only on label and source digest",
    )
    c.eq(
        normal_fingerprints["embedding_collection_sha256"],
        swapped_fingerprints["embedding_collection_sha256"],
        "embedding fingerprint excludes salt-derived opaque ids",
    )
    c.eq(
        normal_fingerprints["embedding_collection_sha256"],
        scaled_fingerprints["embedding_collection_sha256"],
        "embedding fingerprint hashes normalized embedding bytes",
    )
    c.true(
        all(np.array_equal(record.embedding, before)
            for record, before in zip(scaled, scaled_before)),
        "fingerprinting does not mutate in-memory embeddings",
    )


def test_manifest_rejects_missing_or_duplicate_source_hashes(c: Check):
    salt = b"t" * 32
    duplicate = [
        _record("3" * 64, "affected", salt, "first.mp4"),
        _record("3" * 64, "affected", salt, "second.mp4"),
    ]
    missing = [_record("4" * 64, "affected", salt, "third.mp4")]
    missing[0].source_sha256 = ""
    c.raises(lambda: build_manifest(duplicate, []), ValueError,
             "duplicate source files are forbidden")
    c.raises(lambda: build_manifest(missing, []), ValueError,
             "every source must have a SHA-256 digest")


def test_group_overrides_require_exact_valid_coverage_but_allow_cross_label(c: Check):
    salt = b"u" * 32
    records = [
        _record("5" * 64, "affected", salt, "one.mp4"),
        _record("6" * 64, "affected", salt, "two.mp4"),
        _record("7" * 64, "unaffected", salt, "three.mp4"),
    ]
    group_a, group_b = "grp_" + "a" * 64, "grp_" + "b" * 64
    valid = {
        records[0].recording_id: group_a,
        records[1].recording_id: group_a,
        records[2].recording_id: group_b,
    }
    c.eq(validate_group_overrides(valid, records), valid,
         "complete opaque grouping is accepted")
    c.raises(lambda: validate_group_overrides({
        records[0].recording_id: group_a,
        records[1].recording_id: group_a,
    }, records), ValueError, "every recording must appear exactly once")
    c.raises(lambda: validate_group_overrides({
        **valid, "rec_" + "f" * 64: group_b
    }, records), ValueError, "unknown recording ids are rejected")
    cross_label = {
        records[0].recording_id: group_a,
        records[1].recording_id: group_a,
        records[2].recording_id: group_a,
    }
    c.eq(validate_group_overrides(cross_label, records), cross_label,
         "identity review never splits a real person because labels differ")
    c.raises(lambda: validate_group_overrides({
        **valid, records[2].recording_id: "group-human-name"
    }, records), ValueError, "group ids stay canonical and opaque")


def test_override_json_rejects_duplicate_keys(c: Check):
    salt = b"v" * 32
    record = _record("8" * 64, "affected", salt, "raw.mp4")
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "groups.json"
        path.write_text(
            "{" + json.dumps(record.recording_id) + ":" + json.dumps("grp_" + "a" * 64)
            + "," + json.dumps(record.recording_id) + ":" + json.dumps("grp_" + "b" * 64) + "}"
        )
        c.raises(lambda: load_group_overrides(path, [record]), ValueError,
                 "duplicate JSON keys are not silently collapsed")


def test_group_overrides_alone_remain_unreviewed_video_holdout(c: Check):
    salt = b"w" * 32
    record = _record("9" * 64, "affected", salt, "Private Person.mp4")
    group_id = "grp_" + "c" * 64
    manifest = build_manifest(
        [record], [], group_overrides={record.recording_id: group_id}
    )
    encoded = json.dumps(manifest, allow_nan=False)
    c.eq(manifest["identity_review"]["status"], "unreviewed",
         "a group mapping is not manual-review attestation")
    c.eq(manifest["claim_unit"], "video_held_out",
         "group mapping alone cannot enable person-held-out claims")
    c.eq(manifest["recordings"][0]["group_id"], group_id,
         "conservative manifests may still apply opaque grouping")
    c.true("Private Person" not in encoded, "review does not deidentify by leakage")


def test_generation_builder_cannot_promote_person_holdout(c: Check):
    salt = b"z" * 32
    record = _record("c" * 64, "affected", salt, "Private Person.mp4")
    parameters = set(inspect.signature(build_manifest).parameters)
    c.true("identity_review_status" not in parameters
           and "reviewer_evidence_sha256" not in parameters,
           "only the separate finalizer can create a reviewed claim")
    generated = build_manifest([record], [])
    c.eq(generated["identity_review"]["status"], "unreviewed")
    c.eq(generated["claim_unit"], "video_held_out")


def test_contact_sheet_sampling_is_deterministic_and_aspect_preserving(c: Check):
    c.eq(deterministic_window_starts(128), (0, 32, 64, 96),
         "minimum valid video gives four exact windows")
    c.eq(deterministic_window_starts(160), deterministic_window_starts(160),
         "sampling has no random state")
    c.raises(lambda: deterministic_window_starts(127), ValueError,
             "short videos are rejected rather than repeated")

    wide = np.full((10, 20, 3), 255, dtype=np.uint8)
    tall = np.full((20, 10, 3), 127, dtype=np.uint8)
    sheet = compose_contact_sheet([[wide, tall]], panel_height=40)
    c.eq(sheet.shape, (40, 160, 3),
         "panels scale by height then pad without geometric distortion")
    c.true((sheet[:, :80] > 0).all(), "wide panel retains its full image")
    c.true((sheet[:, 80:100] > 0).all() and (sheet[:, 100:] == 0).all(),
           "tall panel is pillar-boxed, not stretched")


def test_contact_sheet_generation_adds_label_blind_owner_only_overview(c: Check):
    salt = b"o" * 32
    records = [
        _record("d" * 64, "affected", salt, "Private A.mp4"),
        _record("e" * 64, "unaffected", salt, "Private B.mp4"),
    ]
    pairs = rank_cosine_pairs({
        record.recording_id: record.embedding for record in records
    })
    original_reader = audit_module.read_representative_frames
    audit_module.read_representative_frames = lambda _path: [
        np.full((8, 8, 3), value, dtype=np.uint8)
        for value in (32, 64, 96, 128)
    ]
    try:
        with tempfile.TemporaryDirectory() as td:
            output_root = Path(td) / "generation"
            output_root.mkdir(mode=0o700)
            counts = generate_contact_sheets(
                records, pairs, output_root, top_pairs=1
            )
            overview = output_root / "contact_sheets" / "overview.jpg"
            c.eq(counts, {"recordings": 2, "ranked_pairs": 1, "overview": 1},
                 "generation declares one label-blind overview")
            decoded = audit_module.cv2.imread(str(overview))
            c.true(decoded is not None and decoded.size > 0,
                   "overview is a decodable opaque-record contact sheet")
            for path in output_root.rglob("*"):
                expected_mode = 0o700 if path.is_dir() else 0o600
                c.eq(stat.S_IMODE(path.stat().st_mode), expected_mode,
                     "generated identity artifacts remain owner-only")
    finally:
        audit_module.read_representative_frames = original_reader


def test_representative_frame_read_rejects_post_seek_position_mismatch(c: Check):
    class WrongSeekCapture:
        def __init__(self, path):
            self.requested = 0
            self.released = False

        def isOpened(self):
            return True

        def get(self, prop):
            if prop == audit_module.cv2.CAP_PROP_FRAME_COUNT:
                return 128.0
            if prop == audit_module.cv2.CAP_PROP_POS_FRAMES:
                return float(self.requested + 7)
            return 0.0

        def set(self, prop, value):
            self.requested = int(value)
            return True

        def read(self):
            return True, np.zeros((8, 8, 3), dtype=np.uint8)

        def release(self):
            self.released = True

    original_capture = audit_module.cv2.VideoCapture
    audit_module.cv2.VideoCapture = WrongSeekCapture
    try:
        c.raises(
            lambda: read_representative_frames("ignored.mp4"),
            RuntimeError,
            "a decoder that lands on another frame cannot enter an identity sheet",
        )
    finally:
        audit_module.cv2.VideoCapture = original_capture


if __name__ == "__main__":
    run_all("test_palsynet_identity_audit", dict(globals()))
