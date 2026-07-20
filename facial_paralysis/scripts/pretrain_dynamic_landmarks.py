"""Receipt-bound two-stage dynamic-landmark pretraining runner."""
from __future__ import annotations

import argparse
import ctypes
import errno
import fcntl
import hashlib
import io
import json
import os
import re
import secrets
import stat
import statistics
import sys
from collections.abc import Mapping
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Callable, Iterator

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRETRAINING_ROOT = (
    PROJECT_ROOT / "outputs" / "dynamic_landmark" / "pretraining"
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if __name__ == "__main__":
    from scripts import pretrain_dynamic_landmarks as canonical_runner

    canonical_runner._entrypoint()

from src.pretraining import dynamic_landmark_ssl as ssl_core  # noqa: E402
from src.models.dynamic_landmark import (  # noqa: E402
    ARM_BLENDSHAPE,
    ARM_FUSION,
    ARM_LANDMARK,
)


_RUN_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_MAX_EXACT_RESULT_TREE_ENTRIES = 64
_MAX_EXACT_RESULT_TREE_DEPTH = 4
_MAX_EXACT_RESULT_TREE_REGULAR_BYTES = 128 * 1024 * 1024
_ABLATION_ARMS = (ARM_BLENDSHAPE, ARM_LANDMARK, ARM_FUSION)
_ABLATION_BASELINES = ("trained", "fresh_untrained", "train_mean")
_FORMAL_SEEDS = (0, 1, 2)
_MAYO_ABLATION_NAMESPACE = (
    PRETRAINING_ROOT / "ablation" / "mayo-input-arm-v1"
)


def _authorization_factories(
    args: argparse.Namespace,
):
    from scripts import prepare_dynamic_landmark_ssl_inputs as inputs_cli

    return inputs_cli._authorization_factories(args)


def _transaction_authorizer(authorizer: Callable[[], object]):
    """Run one expensive live authorizer once inside a transaction boundary."""
    missing = object()
    authorization: object = missing

    def authorize():
        nonlocal authorization
        if authorization is missing:
            authorization = authorizer()
        return authorization

    return authorize


def _require_publication_edge_authorization(
    *,
    stage: str,
    evidence,
    authorization: object,
) -> None:
    """Compare one fresh live generation closure to its frozen stage receipt."""
    if stage == "ravdess":
        expected_schema = evidence.source_schema
        commitments = {
            "manifest_sha256": getattr(
                authorization, "manifest_sha256", None,
            ),
        }
    elif stage == "mayo":
        expected_schema = "mayo_mediapipe_clinical23_ssl_v2"
        generation_commitment = getattr(authorization, "commitment", None)
        if not isinstance(generation_commitment, Mapping):
            raise ValueError("live Mayo generation commitment is unavailable")
        commitments = {
            "collection_manifest_sha256": getattr(
                authorization, "collection_manifest_sha256", None,
            ),
            "exposure_manifest_sha256": getattr(
                authorization, "exposure_manifest_sha256", None,
            ),
            "generation_commitment_sha256": ssl_core._canonical_sha256(
                generation_commitment,
            ),
        }
    else:
        raise ValueError("publication-edge SSL stage is unsupported")
    if (
        getattr(authorization, "schema", None) != expected_schema
        or ssl_core._canonical_sha256(commitments)
        != evidence.upstream_manifest_commitments_sha256
        or getattr(authorization, "generation_closure_hmac", None)
        != evidence.upstream_generation_closure_hmac
        or getattr(authorization, "key_file_identity_sha256", None)
        != evidence.canonical_key_identity_sha256
    ):
        raise ValueError("live SSL generation changed before publication")


def _producer_sha256() -> str:
    from scripts import prepare_dynamic_landmark_ssl_inputs as inputs_cli

    return inputs_cli._producer_sha256()


def _privacy_forbidden(
    args: argparse.Namespace,
    ravdess_authorizer,
    mayo_authorizer,
):
    from scripts import prepare_dynamic_landmark_ssl_inputs as inputs_cli

    ravdess_inventory, mayo_inventory = inputs_cli._live_privacy_inventories(args)
    return inputs_cli._build_live_forbidden_tokens(
        mayo_roots=(args.mayo_data_root, args.mayo_existing_export_root),
        ravdess_authorization=ravdess_authorizer(),
        mayo_authorization=mayo_authorizer(),
        ravdess_inventory=ravdess_inventory,
        mayo_inventory=mayo_inventory,
    )


def _scan_private_results(roots: tuple[Path, ...], forbidden) -> None:
    from scripts import prepare_dynamic_landmark_ssl_inputs as inputs_cli

    modes_ok, privacy_ok, non_0600 = inputs_cli._scan_private_trees(
        roots, forbidden=forbidden,
    )
    if not modes_ok or not privacy_ok or non_0600 != 0:
        raise ValueError("private SSL result verification failed")


def _assert_summary_private(payload: bytes, forbidden) -> None:
    from scripts import prepare_dynamic_landmark_ssl_inputs as inputs_cli

    matcher = inputs_cli._ByteMatcher(forbidden.tokens)
    _, leaked = matcher.feed(payload)
    if leaked:
        raise ValueError("private fact entered the SSL summary")


def _quiet_call(function, /, *args, **kwargs):
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        value = function(*args, **kwargs)
    if stdout.getvalue() or stderr.getvalue():
        raise ValueError("pretraining dependency emitted unexpected output")
    return value


class _PathRedactingArgumentParser(argparse.ArgumentParser):
    def error(self, _message: str) -> None:
        self.exit(2, "pretraining command arguments are invalid\n")


def _parser() -> argparse.ArgumentParser:
    parser = _PathRedactingArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    def add_execution_arguments(command: argparse.ArgumentParser) -> None:
        command.add_argument("--run-root", type=Path, required=True)
        command.add_argument("--bridge-root", type=Path, required=True)
        command.add_argument("--ravdess-data-root", type=Path, required=True)
        command.add_argument("--ravdess-key", type=Path, required=True)
        command.add_argument("--mayo-data-root", type=Path, required=True)
        command.add_argument(
            "--mayo-existing-export-root", type=Path, required=True,
        )
        command.add_argument("--mayo-cache-root", type=Path, required=True)
        command.add_argument(
            "--mayo-exposure-manifest", type=Path, required=True,
        )
        command.add_argument("--mayo-key", type=Path, required=True)

    command = commands.add_parser("two-stage")
    command.add_argument("--mode", choices=("smoke", "formal"), required=True)
    add_execution_arguments(command)
    add_execution_arguments(commands.add_parser("mayo-ablation"))
    commands.add_parser("dry-run")
    return parser


def _formal_job_matrix() -> dict[str, object]:
    """Return the preregistered path-free formal job schedule."""
    jobs: list[dict[str, object]] = []
    for seed in _FORMAL_SEEDS:
        jobs.extend((
            {
                "experiment": "two_stage_fusion",
                "stage": "ravdess",
                "input_arm": ssl_core.ARM_SEMANTIC23,
                "target_schema": ssl_core.TARGET_SEMANTIC23,
                "seed": seed,
                "epochs": 30,
                "optimizer": "adamw",
                "initialization": "same_seed_fresh",
            },
            {
                "experiment": "two_stage_fusion",
                "stage": "mayo",
                "input_arm": ARM_FUSION,
                "target_schema": ssl_core.TARGET_FULL95,
                "seed": seed,
                "epochs": 30,
                "optimizer": "adamw",
                "initialization": "seed_matched_ravdess_prior",
            },
        ))
    for arm in _ABLATION_ARMS:
        for seed in _FORMAL_SEEDS:
            jobs.append({
                "experiment": "mayo_input_arm_ablation",
                "stage": "mayo",
                "input_arm": arm,
                "target_schema": ssl_core.TARGET_FULL95,
                "seed": seed,
                "epochs": 30,
                "optimizer": "adamw",
                "initialization": "same_seed_fresh",
            })
    return {
        "schema_version": "dynamic_landmark_ssl_job_matrix_v1",
        "jobs": jobs,
    }


def _expected_ablation_result_files(
    arms: tuple[str, ...] = _ABLATION_ARMS,
    seeds: tuple[int, ...] = _FORMAL_SEEDS,
) -> set[str]:
    if arms != _ABLATION_ARMS or seeds != _FORMAL_SEEDS:
        raise ValueError("Mayo ablation result matrix is not exact")
    expected = {
        "reports/mayo_input_arm_ablation.json",
    }
    for arm in arms:
        for seed in seeds:
            checkpoint = f"checkpoints/{arm}/seed_{seed}.pt"
            expected.add(checkpoint)
            expected.add(f"{checkpoint}.receipt.json")
    return expected


def _finite_nonnegative(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("ablation metric must be finite and nonnegative")
    observed = float(value)
    if observed < 0.0 or observed == float("inf") or observed != observed:
        raise ValueError("ablation metric must be finite and nonnegative")
    return observed


def _normalize_ablation_metric(value: object) -> dict[str, object]:
    if (
        type(value) is not dict
        or set(value) != {
            "raw_mae", "standardized_mae", "standardized_smooth_l1",
        }
        or type(value["raw_mae"]) is not dict
        or set(value["raw_mae"]) != {
            "blendshape72", "clinical23", "equal_block_macro", "full95",
        }
    ):
        raise ValueError("ablation common-target metric schema is not exact")
    return {
        "raw_mae": {
            name: _finite_nonnegative(metric)
            for name, metric in value["raw_mae"].items()
        },
        "standardized_mae": _finite_nonnegative(value["standardized_mae"]),
        "standardized_smooth_l1": _finite_nonnegative(
            value["standardized_smooth_l1"]
        ),
    }


def _ablation_statistics(
    runs: list[dict[str, object]],
) -> tuple[dict[str, object], dict[str, object]]:
    """Aggregate exact common-target metrics and paired seed differences."""
    if type(runs) is not list or len(runs) != 9:
        raise ValueError("ablation statistics require exactly nine jobs")
    indexed: dict[tuple[str, int], dict[str, object]] = {}
    for run in runs:
        if type(run) is not dict or set(run) != {"arm", "seed", "metrics"}:
            raise ValueError("ablation metric row schema is not exact")
        arm, seed, metrics = run["arm"], run["seed"], run["metrics"]
        if (
            arm not in _ABLATION_ARMS
            or seed not in _FORMAL_SEEDS
            or isinstance(seed, bool)
            or type(metrics) is not dict
            or set(metrics) != set(_ABLATION_BASELINES)
            or (arm, seed) in indexed
        ):
            raise ValueError("ablation metric matrix is not exact")
        indexed[(str(arm), int(seed))] = {
            baseline: _normalize_ablation_metric(metrics[baseline])
            for baseline in _ABLATION_BASELINES
        }
    if set(indexed) != {
        (arm, seed) for arm in _ABLATION_ARMS for seed in _FORMAL_SEEDS
    }:
        raise ValueError("ablation metric matrix is incomplete")

    def summarize(values: list[float]) -> dict[str, float]:
        return {
            "mean": statistics.fmean(values),
            "sd": statistics.stdev(values),
        }

    aggregate: dict[str, object] = {}
    for arm in _ABLATION_ARMS:
        aggregate[arm] = {
            baseline: {
                "raw_mae": {
                    name: summarize([
                        indexed[(arm, seed)][baseline]["raw_mae"][name]  # type: ignore[index]
                        for seed in _FORMAL_SEEDS
                    ])
                    for name in (
                        "blendshape72", "clinical23", "equal_block_macro",
                        "full95",
                    )
                },
                "standardized_mae": summarize([
                    indexed[(arm, seed)][baseline]["standardized_mae"]  # type: ignore[index]
                    for seed in _FORMAL_SEEDS
                ]),
                "standardized_smooth_l1": summarize([
                    indexed[(arm, seed)][baseline][
                        "standardized_smooth_l1"
                    ]  # type: ignore[index]
                    for seed in _FORMAL_SEEDS
                ]),
            }
            for baseline in _ABLATION_BASELINES
        }

    paired: dict[str, object] = {}
    for left, right in (
        (ARM_LANDMARK, ARM_BLENDSHAPE),
        (ARM_FUSION, ARM_BLENDSHAPE),
        (ARM_FUSION, ARM_LANDMARK),
    ):
        comparison: dict[str, object] = {"raw_mae": {}}
        for name in (
            "blendshape72", "clinical23", "equal_block_macro", "full95",
        ):
            values = {
                str(seed): (
                    indexed[(left, seed)]["trained"]["raw_mae"][name]  # type: ignore[index]
                    - indexed[(right, seed)]["trained"]["raw_mae"][name]  # type: ignore[index]
                )
                for seed in _FORMAL_SEEDS
            }
            comparison["raw_mae"][name] = {  # type: ignore[index]
                "by_seed": values,
                **summarize(list(values.values())),
            }
        for name in ("standardized_mae", "standardized_smooth_l1"):
            values = {
                str(seed): (
                    indexed[(left, seed)]["trained"][name]  # type: ignore[index]
                    - indexed[(right, seed)]["trained"][name]  # type: ignore[index]
                )
                for seed in _FORMAL_SEEDS
            }
            comparison[name] = {
                "by_seed": values,
                **summarize(list(values.values())),
            }
        paired[f"{left}_minus_{right}"] = comparison
    return aggregate, paired


def _ablation_inputs_root(run_root: Path) -> Path:
    """Return the one common atomically frozen Mayo ablation input tree."""
    return _private_directory(run_root / "inputs", "ablation inputs")


def _common_ablation_evidence(
    evidence_by_arm: Mapping[str, object],
) -> tuple[str, str, str]:
    """Require the arm snapshots to share one split, scaler, and source."""
    if set(evidence_by_arm) != set(_ABLATION_ARMS):
        raise ValueError("ablation evidence arm set is not exact")
    common_fields = (
        "source", "cache_commitment_sha256", "cache_count", "split_unit",
        "claim_unit", "patient_held_out", "train_indices_sha256",
        "heldout_indices_sha256", "group_ids_sha256", "scaler_sha256",
        "train_count", "heldout_count", "development_only",
        "sample_ids_sha256", "source_unit_ids_sha256",
        "cache_integrity_ids_sha256", "original_mapping_sha256",
        "bundle_sha256", "bundle_size_bytes", "bundle_file_count",
        "sample_count", "source_unit_count", "unique_group_count",
        "upstream_cache_count", "exclusion_count", "feature_names_sha256",
        "adapter_sha256", "temporal_policy_sha256",
        "bridge_generation_sha256", "upstream_manifest_commitments_sha256",
        "upstream_generation_closure_hmac", "canonical_key_identity_sha256",
        "source_schema", "mode", "target_schema", "experiment_kind",
        "initialization_policy",
    )
    first = evidence_by_arm[_ABLATION_ARMS[0]]
    baseline = tuple(getattr(first, name) for name in common_fields)
    for arm in _ABLATION_ARMS:
        evidence = evidence_by_arm[arm]
        if (
            tuple(getattr(evidence, name) for name in common_fields) != baseline
            or getattr(evidence, "stage", None) != "mayo"
            or getattr(evidence, "mode", None) != "formal"
            or getattr(evidence, "experiment_kind", None)
            != "mayo_input_arm_ablation"
            or getattr(evidence, "input_arm", None) != arm
            or getattr(evidence, "target_schema", None) != ssl_core.TARGET_FULL95
            or getattr(evidence, "initialization_policy", None)
            != "same_seed_fresh"
            or getattr(evidence, "prior_checkpoint_sha256", None) is not None
        ):
            raise ValueError("ablation arms do not share one frozen input contract")
    return (
        str(getattr(first, "heldout_indices_sha256")),
        str(getattr(first, "scaler_sha256")),
        str(getattr(first, "cache_commitment_sha256")),
    )


def _materialized_heldout_mask(
    evidence: object,
) -> tuple[torch.Tensor, bytes, str]:
    """Load the one persisted common formal held-out mask before any job."""
    mask = ssl_core.load_frozen_mayo_ablation_mask(evidence)
    payload = ssl_core._tensor_fingerprint_bytes(mask)
    digest = hashlib.sha256(payload).hexdigest()
    if digest != ssl_core._mask_sha256(mask):
        raise RuntimeError("materialized held-out mask digest is inconsistent")
    return mask, payload, digest


def _fresh_ablation_initialization(seed: int) -> tuple[str, str]:
    """Fingerprint the same-seed model and empty AdamW state pre-job."""
    if isinstance(seed, bool) or seed not in _FORMAL_SEEDS:
        raise ValueError("ablation initialization seed is unsupported")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = ssl_core.DynamicLandmarkSSLModel().to("cpu")
        model_digest = ssl_core._model_state_sha256(model.state_dict())
        parameter_by_name = dict(model.named_parameters())
        trainable_names = ssl_core._trainable_parameter_names(model, "mayo")
        optimizer = torch.optim.AdamW(
            [parameter_by_name[name] for name in trainable_names],
            lr=0.001,
            weight_decay=0.0001,
        )
        initial_state = optimizer.state_dict()
        if initial_state.get("state") != {}:
            raise RuntimeError("fresh AdamW unexpectedly contains mutable state")
        optimizer_material = {
            "schema_version": "adamw_empty_named_state_v1",
            "trainable_parameters": [
                {
                    "name": name,
                    "shape": list(parameter_by_name[name].shape),
                    "dtype": str(parameter_by_name[name].dtype),
                }
                for name in trainable_names
            ],
            "state_dict": initial_state,
        }
        optimizer_digest = ssl_core._canonical_sha256(optimizer_material)
    return model_digest, optimizer_digest


def _ablation_common_target_metrics(result: object) -> dict[str, object]:
    report = getattr(result, "heldout_report", None)
    if type(report) is not dict:
        raise ValueError("ablation training report is unavailable")
    common = report.get("common_target_metrics")
    if type(common) is not dict or set(common) != set(_ABLATION_BASELINES):
        raise ValueError("ablation common-target metrics are unavailable")
    metrics = {
        baseline: _normalize_ablation_metric(common[baseline])
        for baseline in _ABLATION_BASELINES
    }
    return metrics


def _private_directory(path: Path, name: str) -> Path:
    lexical = path.absolute()
    try:
        resolved = lexical.resolve(strict=True)
        status = lexical.lstat()
    except OSError as exc:
        raise ValueError(f"{name} is unavailable") from exc
    canonical = resolved == lexical
    if not canonical and sys.platform == "darwin":
        for source, destination in (
            (Path("/var"), Path("/private/var")),
            (Path("/tmp"), Path("/private/tmp")),
        ):
            try:
                relative = lexical.relative_to(source)
            except ValueError:
                continue
            if resolved == destination / relative:
                canonical = True
                break
    if (
        not canonical
        or not stat.S_ISDIR(status.st_mode)
        or status.st_uid != os.geteuid()
        or stat.S_IMODE(status.st_mode) != 0o700
    ):
        raise ValueError(f"{name} must be an owner-only canonical directory")
    return resolved


def _require_existing_input(path: Path, *, directory: bool, name: str) -> None:
    """Reject an absent or symlinked live input before any run-side mutation."""
    lexical = path.absolute()
    try:
        status = lexical.lstat()
    except OSError as exc:
        raise ValueError(f"{name} is unavailable") from exc
    expected = stat.S_ISDIR(status.st_mode) if directory else stat.S_ISREG(
        status.st_mode
    )
    if not expected or lexical.is_symlink():
        raise ValueError(f"{name} has the wrong storage type")


def _preflight_live_inputs(args: argparse.Namespace) -> None:
    for path, name in (
        (args.ravdess_data_root, "RAVDESS data root"),
        (args.mayo_data_root, "Mayo data root"),
        (args.mayo_existing_export_root, "Mayo existing-export root"),
        (args.mayo_cache_root, "Mayo cache root"),
    ):
        _require_existing_input(path, directory=True, name=name)
    for path, name in (
        (args.ravdess_key, "RAVDESS key"),
        (args.mayo_exposure_manifest, "Mayo exposure manifest"),
        (args.mayo_key, "Mayo key"),
    ):
        _require_existing_input(path, directory=False, name=name)


def _anchor_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev), int(value.st_ino), int(value.st_mode),
        int(value.st_uid), int(value.st_gid),
    )


def _validate_run_root(
    descriptor: int,
    run_root: Path,
    expected_identity: tuple[int, ...],
) -> None:
    opened = os.fstat(descriptor)
    current = os.stat(run_root, follow_symlinks=False)
    if (
        _anchor_identity(opened) != expected_identity
        or _anchor_identity(current) != expected_identity
        or not stat.S_ISDIR(opened.st_mode)
        or opened.st_uid != os.geteuid()
        or stat.S_IMODE(opened.st_mode) != 0o700
    ):
        raise ValueError("run root changed during execution")


def _validate_results_lock(
    descriptor: int,
    lock_name: str,
    run_descriptor: int,
) -> None:
    opened = os.fstat(descriptor)
    current = os.stat(
        lock_name, dir_fd=run_descriptor, follow_symlinks=False,
    )
    identity = lambda value: (
        value.st_dev, value.st_ino, value.st_mode, value.st_uid,
        value.st_gid, value.st_nlink, value.st_size,
    )
    if (
        identity(opened) != identity(current)
        or not stat.S_ISREG(opened.st_mode)
        or opened.st_uid != os.geteuid()
        or stat.S_IMODE(opened.st_mode) != 0o600
        or opened.st_nlink != 1
        or opened.st_size != 0
    ):
        raise ValueError("results lock storage is unsafe")


def _close_descriptor_sequence(descriptors: tuple[int, ...]) -> None:
    """Attempt every close while preserving active and cleanup exceptions."""
    closer = ExitStack()
    for descriptor in descriptors:
        closer.callback(os.close, descriptor)
    closer.__exit__(*sys.exc_info())


def _generation_lease_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_uid),
        int(value.st_gid),
        int(value.st_nlink),
        int(value.st_size),
    )


def _validate_generation_lease(
    *,
    parent_descriptor: int,
    parent_path: Path,
    parent_identity: tuple[int, ...],
    descriptor: int,
    lock_name: str,
    lock_identity: tuple[int, ...],
    private_parent: bool,
) -> None:
    parent_opened = os.fstat(parent_descriptor)
    parent_current = os.stat(parent_path, follow_symlinks=False)
    if (
        _anchor_identity(parent_opened) != parent_identity
        or _anchor_identity(parent_current) != parent_identity
        or not stat.S_ISDIR(parent_opened.st_mode)
        or parent_opened.st_uid != os.geteuid()
        or (
            stat.S_IMODE(parent_opened.st_mode) != 0o700
            if private_parent
            else stat.S_IMODE(parent_opened.st_mode) & 0o022 != 0
        )
    ):
        raise ValueError("generation lease parent changed")
    opened = os.fstat(descriptor)
    current = os.stat(
        lock_name,
        dir_fd=parent_descriptor,
        follow_symlinks=False,
    )
    if (
        _generation_lease_identity(opened) != lock_identity
        or _generation_lease_identity(current) != lock_identity
        or not stat.S_ISREG(opened.st_mode)
        or opened.st_uid != os.geteuid()
        or stat.S_IMODE(opened.st_mode) != 0o600
        or opened.st_nlink != 1
        or opened.st_size != 0
    ):
        raise ValueError("generation lease storage is unsafe")


@contextmanager
def _shared_generation_leases(
    args: argparse.Namespace,
) -> Iterator[Callable[[], None]]:
    """Freeze both upstream generations through final result publication."""
    leases = sorted(
        (
            (
                args.ravdess_data_root.absolute(),
                ".derived_semantic23.lock",
                False,
            ),
            (
                PRETRAINING_ROOT.absolute(),
                ".mayo_ssl_cache.lock",
                True,
            ),
        ),
        key=lambda item: os.fspath(item[0] / item[1]),
    )
    closer = ExitStack()
    held: list[
        tuple[int, Path, tuple[int, ...], int, str, tuple[int, ...], bool]
    ] = []
    try:
        for parent_path, lock_name, private_parent in leases:
            parent_descriptor = os.open(
                parent_path,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
            )
            closer.callback(os.close, parent_descriptor)
            parent_info = os.fstat(parent_descriptor)
            parent_identity = _anchor_identity(parent_info)
            if (
                not stat.S_ISDIR(parent_info.st_mode)
                or parent_info.st_uid != os.geteuid()
                or (
                    stat.S_IMODE(parent_info.st_mode) != 0o700
                    if private_parent
                    else stat.S_IMODE(parent_info.st_mode) & 0o022 != 0
                )
            ):
                raise ValueError("generation lease parent is unsafe")
            try:
                descriptor = os.open(
                    lock_name,
                    os.O_RDONLY
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_CLOEXEC", 0),
                    dir_fd=parent_descriptor,
                )
            except OSError as exc:
                if exc.errno == errno.ELOOP:
                    raise ValueError(
                        "generation lease must not be a symlink"
                    ) from exc
                raise
            closer.callback(os.close, descriptor)
            lock_identity = _generation_lease_identity(os.fstat(descriptor))
            _validate_generation_lease(
                parent_descriptor=parent_descriptor,
                parent_path=parent_path,
                parent_identity=parent_identity,
                descriptor=descriptor,
                lock_name=lock_name,
                lock_identity=lock_identity,
                private_parent=private_parent,
            )
            fcntl.flock(descriptor, fcntl.LOCK_SH)
            _validate_generation_lease(
                parent_descriptor=parent_descriptor,
                parent_path=parent_path,
                parent_identity=parent_identity,
                descriptor=descriptor,
                lock_name=lock_name,
                lock_identity=lock_identity,
                private_parent=private_parent,
            )
            held.append((
                parent_descriptor,
                parent_path,
                parent_identity,
                descriptor,
                lock_name,
                lock_identity,
                private_parent,
            ))

        def validate() -> None:
            for (
                parent_descriptor,
                parent_path,
                parent_identity,
                descriptor,
                lock_name,
                lock_identity,
                private_parent,
            ) in held:
                _validate_generation_lease(
                    parent_descriptor=parent_descriptor,
                    parent_path=parent_path,
                    parent_identity=parent_identity,
                    descriptor=descriptor,
                    lock_name=lock_name,
                    lock_identity=lock_identity,
                    private_parent=private_parent,
                )

        yield validate
        validate()
    finally:
        closer.__exit__(*sys.exc_info())


@contextmanager
def _exclusive_results_lock(
    run_root: Path,
) -> Iterator[tuple[int, str, int, tuple[int, ...]]]:
    run_descriptor = os.open(
        run_root,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    descriptor: int | None = None
    try:
        run_identity = _anchor_identity(os.fstat(run_descriptor))
        _validate_run_root(run_descriptor, run_root, run_identity)
        lock_name = ".results.lock"
        created = False
        try:
            descriptor = os.open(
                lock_name,
                os.O_RDWR | os.O_CREAT | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=run_descriptor,
            )
            created = True
        except FileExistsError:
            descriptor = os.open(
                lock_name,
                os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=run_descriptor,
            )
        if created:
            os.fchmod(descriptor, 0o600)
            os.fsync(descriptor)
            os.fsync(run_descriptor)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        _validate_run_root(run_descriptor, run_root, run_identity)
        _validate_results_lock(descriptor, lock_name, run_descriptor)
        yield descriptor, lock_name, run_descriptor, run_identity
    finally:
        descriptors = (
            (run_descriptor,)
            + ((descriptor,) if descriptor is not None else ())
        )
        _close_descriptor_sequence(descriptors)


@contextmanager
def _exclusive_results_transaction(
    run_root: Path,
    args: argparse.Namespace,
) -> Iterator[
    tuple[int, str, int, tuple[int, ...], Callable[[], None]]
]:
    with _shared_generation_leases(args) as validate_generation_leases:
        with _exclusive_results_lock(run_root) as locked:
            yield (*locked, validate_generation_leases)


def _rename_directory_no_replace(
    source_name: str,
    destination_name: str,
    *,
    parent_descriptor: int,
) -> None:
    """Atomically publish one directory without replacing any destination."""
    library = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source_name)
    destination_bytes = os.fsencode(destination_name)
    if sys.platform == "darwin":
        rename = library.renameatx_np
        rename.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        rename.restype = ctypes.c_int
        result = rename(
            parent_descriptor,
            source_bytes,
            parent_descriptor,
            destination_bytes,
            0x00000004 | 0x00000010,
        )
    elif sys.platform.startswith("linux"):
        rename = library.renameat2
        rename.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        rename.restype = ctypes.c_int
        result = rename(
            parent_descriptor,
            source_bytes,
            parent_descriptor,
            destination_bytes,
            1,
        )
    else:
        raise OSError(
            errno.ENOTSUP,
            "atomic no-replace directory publication is unavailable",
        )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(
            error_number,
            os.strerror(error_number),
            destination_name,
        )


def _classify_results_publication(
    parent_descriptor: int,
    source_name: str,
    destination_name: str,
    expected_identity: tuple[int, int],
) -> str:
    """Classify an atomic rename outcome against the held staging inode."""
    observed: dict[str, tuple[int, int] | None] = {}
    for field, name in (
        ("source", source_name), ("destination", destination_name),
    ):
        try:
            status = os.stat(
                name, dir_fd=parent_descriptor, follow_symlinks=False,
            )
        except FileNotFoundError:
            observed[field] = None
        else:
            observed[field] = (int(status.st_dev), int(status.st_ino))
    if (
        observed["source"] == expected_identity
        and observed["destination"] is None
    ):
        return "staged"
    if (
        observed["destination"] == expected_identity
        and observed["source"] is None
    ):
        return "published"
    return "indeterminate"


def _write_private_json(path: Path, value: dict[str, object]) -> None:
    encoded = (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    ).encode("ascii")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        os.fchmod(descriptor, 0o600)
        offset = 0
        while offset < len(encoded):
            written = os.write(descriptor, encoded[offset:])
            if written < 1:
                raise OSError("private JSON write made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ledger_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev), int(value.st_ino), int(value.st_mode),
        int(value.st_uid), int(value.st_gid), int(value.st_nlink),
        int(value.st_size), int(value.st_mtime_ns), int(value.st_ctime_ns),
    )


def _directory_ledger_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev), int(value.st_ino), int(value.st_mode),
        int(value.st_uid), int(value.st_gid), int(value.st_nlink),
    )


def _hash_held_file(descriptor: int, expected_size: int) -> str:
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    total = 0
    while chunk := os.read(descriptor, 1024 * 1024):
        total += len(chunk)
        if total > expected_size:
            raise ValueError("result file grew during held verification")
        digest.update(chunk)
    if total != expected_size:
        raise ValueError("result file changed size during held verification")
    return digest.hexdigest()


@contextmanager
def _hold_exact_result_tree(
    run_descriptor: int,
    staging_name: str,
    expected_files: set[str],
) -> Iterator[Callable[..., None]]:
    directory_flags = (
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    root_descriptor = os.open(
        staging_name, directory_flags, dir_fd=run_descriptor,
    )
    directory_descriptors: dict[tuple[str, ...], int] = {
        (): root_descriptor,
    }
    file_descriptors: dict[tuple[str, ...], int] = {}
    identities: dict[tuple[str, ...], tuple[int, ...]] = {}
    digests: dict[tuple[str, ...], str] = {}
    expected_children: dict[tuple[str, ...], set[str]] = {(): set()}
    try:
        file_parts = {tuple(Path(value).parts) for value in expected_files}
        directory_parts: set[tuple[str, ...]] = set()
        for parts in file_parts:
            for size in range(1, len(parts)):
                directory_parts.add(parts[:size])
        if (
            not file_parts
            or 1 + len(directory_parts) + len(file_parts)
            > _MAX_EXACT_RESULT_TREE_ENTRIES
            or max(len(parts) for parts in file_parts)
            > _MAX_EXACT_RESULT_TREE_DEPTH
        ):
            raise ValueError("result tree exceeds its exact structural budget")
        for parts in sorted(directory_parts, key=lambda value: (len(value), value)):
            parent = parts[:-1]
            expected_children.setdefault(parent, set()).add(parts[-1])
            expected_children.setdefault(parts, set())
            descriptor = os.open(
                parts[-1], directory_flags,
                dir_fd=directory_descriptors[parent],
            )
            directory_descriptors[parts] = descriptor
            status = os.fstat(descriptor)
            if (
                not stat.S_ISDIR(status.st_mode)
                or status.st_uid != os.geteuid()
                or stat.S_IMODE(status.st_mode) != 0o700
            ):
                raise ValueError("result directory is not owner-only")
        for parts in sorted(file_parts):
            parent = parts[:-1]
            expected_children.setdefault(parent, set()).add(parts[-1])
            descriptor = os.open(
                parts[-1],
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_descriptors[parent],
            )
            file_descriptors[parts] = descriptor
            status = os.fstat(descriptor)
            if (
                not stat.S_ISREG(status.st_mode)
                or status.st_uid != os.geteuid()
                or stat.S_IMODE(status.st_mode) != 0o600
                or status.st_nlink != 1
            ):
                raise ValueError("result file is not private regular storage")
            identities[parts] = _ledger_identity(status)
        total_regular_bytes = sum(
            int(identity[6]) for identity in identities.values()
        )
        if total_regular_bytes > _MAX_EXACT_RESULT_TREE_REGULAR_BYTES:
            raise ValueError("result tree exceeds its regular-payload budget")
        for parts, descriptor in file_descriptors.items():
            digests[parts] = _hash_held_file(
                descriptor, int(identities[parts][6]),
            )
        for parts, descriptor in directory_descriptors.items():
            identities[parts] = _directory_ledger_identity(
                os.fstat(descriptor)
            )

        def validate(live_root_name: str = staging_name) -> None:
            for parts, descriptor in directory_descriptors.items():
                parent_descriptor = (
                    run_descriptor
                    if not parts
                    else directory_descriptors[parts[:-1]]
                )
                name = live_root_name if not parts else parts[-1]
                live = os.stat(
                    name, dir_fd=parent_descriptor, follow_symlinks=False,
                )
                if (
                    _directory_ledger_identity(os.fstat(descriptor))
                    != identities[parts]
                    or _directory_ledger_identity(live) != identities[parts]
                    or set(os.listdir(descriptor))
                    != expected_children.get(parts, set())
                ):
                    raise ValueError("result directory changed before publication")
            for parts, descriptor in file_descriptors.items():
                live = os.stat(
                    parts[-1],
                    dir_fd=directory_descriptors[parts[:-1]],
                    follow_symlinks=False,
                )
                expected = identities[parts]
                if (
                    _ledger_identity(os.fstat(descriptor)) != expected
                    or _ledger_identity(live) != expected
                    or _hash_held_file(descriptor, expected[6]) != digests[parts]
                ):
                    raise ValueError("result file changed before publication")

        validate()
        yield validate
    finally:
        directory_order = tuple(
            directory_descriptors[parts]
            for parts in sorted(
                directory_descriptors, key=lambda value: (len(value), value),
            )
        )
        _close_descriptor_sequence(
            directory_order + tuple(file_descriptors.values())
        )


def _stage_report(result, checkpoint_fingerprint: str) -> dict[str, object]:
    report = result.heldout_report
    value: dict[str, object] = {
        "checkpoint_fingerprint": checkpoint_fingerprint,
        "optimizer_steps": result.training_receipt.optimizer_steps,
        "seed": result.training_receipt.seed,
    }
    if result.training_receipt.prior_checkpoint_sha256 is not None:
        value["prior_checkpoint_fingerprint"] = (
            result.training_receipt.prior_checkpoint_sha256
        )
    if result.stage_evidence.mode == "smoke":
        value["train_loss"] = report["train_loss"]
    else:
        value["reconstruction"] = dict(report)
    return value


def _formal_aggregates(
    execution: list[dict[str, object]],
) -> dict[str, object]:
    metrics = {
        "ravdess_only": ("trained", "untrained", "train_mean"),
        "ravdess_then_mayo": (
            "trained", "prior_ravdess", "fresh_untrained", "train_mean",
        ),
    }
    aggregate: dict[str, object] = {}
    for stage, names in metrics.items():
        stage_aggregate: dict[str, object] = {}
        for name in names:
            values = [
                float(run[stage]["reconstruction"][name])  # type: ignore[index]
                for run in execution
            ]
            if len(values) != 3 or not all(
                value >= 0.0 and value < float("inf") for value in values
            ):
                raise ValueError("formal reconstruction evidence is incomplete")
            stage_aggregate[name] = {
                "mean": statistics.fmean(values),
                "sd": statistics.stdev(values),
            }
        aggregate[stage] = stage_aggregate
    return aggregate


def _expected_result_files(mode: str, seeds: tuple[int, ...]) -> set[str]:
    if mode not in {"smoke", "formal"}:
        raise ValueError("result file mode is unsupported")
    expected: set[str] = set()
    for seed in seeds:
        prefix = "checkpoints" if mode == "smoke" else (
            f"checkpoints/seed_{seed}"
        )
        for name in ("ravdess_only.pt", "ravdess_then_mayo.pt"):
            expected.add(f"{prefix}/{name}")
            expected.add(f"{prefix}/{name}.receipt.json")
    report_name = (
        "execution_only.json"
        if mode == "smoke"
        else "formal_pretraining_results.json"
    )
    expected.add(f"reports/{report_name}")
    return expected


def _publish_validated_results(
    *,
    run_descriptor: int,
    run_root: Path,
    run_identity: tuple[int, ...],
    results_lock: int,
    lock_name: str,
    staging_name: str,
    staged_identity: tuple[int, int],
    inputs_root: Path,
    privacy_forbidden,
    validate_result_tree: Callable[..., None],
    final_authorization: Callable[[], None],
) -> None:
    """Publish and classify one exact result tree through held descriptors."""
    staging = run_root / staging_name
    validate_result_tree()
    _quiet_call(
        _scan_private_results,
        (inputs_root, staging),
        privacy_forbidden,
    )
    validate_result_tree()
    _validate_run_root(run_descriptor, run_root, run_identity)
    _validate_results_lock(results_lock, lock_name, run_descriptor)
    os.fsync(run_descriptor)
    validate_result_tree()
    _quiet_call(final_authorization)
    rename_error: BaseException | None = None
    try:
        _rename_directory_no_replace(
            staging_name,
            "results",
            parent_descriptor=run_descriptor,
        )
    except BaseException as caught:
        rename_error = caught
    publication_state = _classify_results_publication(
        run_descriptor,
        staging_name,
        "results",
        staged_identity,
    )
    if publication_state != "published":
        if publication_state == "staged" and rename_error is not None:
            raise rename_error
        raise RuntimeError(
            "results publication outcome is indeterminate"
        ) from rename_error
    try:
        # A faulting wrapper may report failure after the syscall committed.
        # Success requires the canonical name, exact held tree, privacy closure,
        # lock identity, and parent durability all to revalidate.
        validate_result_tree("results")
        _quiet_call(
            _scan_private_results,
            (inputs_root, run_root / "results"),
            privacy_forbidden,
        )
        validate_result_tree("results")
        _validate_run_root(run_descriptor, run_root, run_identity)
        _validate_results_lock(results_lock, lock_name, run_descriptor)
        os.fsync(run_descriptor)
        _validate_run_root(run_descriptor, run_root, run_identity)
        _validate_results_lock(results_lock, lock_name, run_descriptor)
        validate_result_tree("results")
        _quiet_call(
            _scan_private_results,
            (inputs_root, run_root / "results"),
            privacy_forbidden,
        )
        validate_result_tree("results")
    except BaseException as caught:
        raise RuntimeError(
            "published results are retained as indeterminate evidence"
        ) from caught
    if rename_error is not None:
        raise RuntimeError(
            "published results are retained after a rename return fault"
        ) from rename_error


def _run_two_stage(
    args: argparse.Namespace,
    *,
    producer_sha256: str,
) -> dict[str, object]:
    run_root = _private_directory(args.run_root.absolute(), "run root")
    if args.mode == "smoke":
        namespace_root = run_root.parent.parent
        namespace_ok = (
            run_root.parent.name == "smoke"
            and _RUN_ID.fullmatch(run_root.name) is not None
        )
    else:
        namespace_root = run_root.parent
        namespace_ok = run_root.name == "formal"
    if (
        not namespace_ok
        or namespace_root != PRETRAINING_ROOT.absolute()
    ):
        raise ValueError("run root is outside the exact mode/run-id namespace")
    inputs_root = _private_directory(run_root / "inputs", "frozen inputs")
    bridge_root = _private_directory(
        args.bridge_root.absolute(), "bridge generation"
    )
    if bridge_root != (PRETRAINING_ROOT / "bridge").absolute():
        raise ValueError("bridge root is outside the canonical namespace")
    _preflight_live_inputs(args)
    raw_ravdess_authorizer, raw_mayo_authorizer = _quiet_call(
        _authorization_factories, args,
    )
    ravdess_authorizer = _transaction_authorizer(raw_ravdess_authorizer)
    mayo_authorizer = _transaction_authorizer(raw_mayo_authorizer)
    privacy_forbidden = _quiet_call(
        _privacy_forbidden,
        args, ravdess_authorizer, mayo_authorizer,
    )
    seeds = (0,) if args.mode == "smoke" else (0, 1, 2)
    summary = {
        "checkpoint_count": len(seeds) * 2,
        "mode": args.mode,
        "seed_count": len(seeds),
        "stage_count": 2,
    }
    _assert_summary_private(
        json.dumps(
            summary, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("ascii"),
        privacy_forbidden,
    )
    with _exclusive_results_transaction(run_root, args) as (
        results_lock,
        lock_name,
        run_descriptor,
        run_identity,
        validate_generation_leases,
    ):
        names = set(os.listdir(run_descriptor))
        if "results" in names:
            raise FileExistsError("results generation already exists")
        if any(name.startswith(".results.staging-") for name in names):
            raise ValueError("unresolved results staging state blocks execution")

        ravdess_evidence = _quiet_call(
            ssl_core.authorize_frozen_ssl_stage,
            stage="ravdess",
            mode=args.mode,
            inputs_root=inputs_root,
            bridge_root=bridge_root,
            ravdess_authorizer=ravdess_authorizer,
            mayo_authorizer=mayo_authorizer,
            producer_sha256=producer_sha256,
        )
        staging = run_root / f".results.staging-{secrets.token_hex(16)}"
        staging.mkdir(mode=0o700)
        os.chmod(staging, 0o700)
        staged_status = staging.stat()
        staged_identity = (
            staged_status.st_dev,
            staged_status.st_ino,
        )
        checkpoints = staging / "checkpoints"
        reports = staging / "reports"
        checkpoints.mkdir(mode=0o700)
        reports.mkdir(mode=0o700)
        os.chmod(checkpoints, 0o700)
        os.chmod(reports, 0o700)

        execution: list[dict[str, object]] = []
        reload_checks: list[tuple[Path, object, object, str]] = []
        publication_lineages: list[
            tuple[Path, object, str, Path, object, str]
        ] = []
        retained_runtime: list[object] = []
        checkpoint_directories = {checkpoints}
        for seed in seeds:
            seed_checkpoints = checkpoints
            if args.mode == "formal":
                seed_checkpoints = checkpoints / f"seed_{seed}"
                seed_checkpoints.mkdir(mode=0o700)
                os.chmod(seed_checkpoints, 0o700)
                checkpoint_directories.add(seed_checkpoints)
            ravdess_result = _quiet_call(
                ssl_core.train_ssl_stage,
                stage_evidence=ravdess_evidence,
                seed=seed,
            )
            ravdess_payload = _quiet_call(
                ssl_core.build_ssl_checkpoint_payload, ravdess_result,
            )
            ravdess_path = seed_checkpoints / "ravdess_only.pt"
            ravdess_receipt = _quiet_call(
                ssl_core.save_ssl_checkpoint,
                ravdess_path,
                ravdess_payload,
                stage_evidence=ravdess_evidence,
            )
            persisted_ravdess = _quiet_call(
                ssl_core.load_ssl_checkpoint,
                ravdess_path,
                receipt=ravdess_receipt,
                stage_evidence=ravdess_evidence,
            )
            retained_runtime.extend((ravdess_receipt, persisted_ravdess))
            ravdess_fingerprint = _quiet_call(
                ssl_core.ssl_checkpoint_fingerprint, persisted_ravdess,
            )
            reload_checks.append((
                ravdess_path, ravdess_receipt, ravdess_evidence,
                ravdess_fingerprint,
            ))

            mayo_evidence = _quiet_call(
                ssl_core.authorize_frozen_ssl_stage,
                stage="mayo",
                mode=args.mode,
                inputs_root=inputs_root,
                bridge_root=bridge_root,
                ravdess_authorizer=ravdess_authorizer,
                mayo_authorizer=mayo_authorizer,
                producer_sha256=producer_sha256,
                prior_ravdess_checkpoint=persisted_ravdess,
                prior_ravdess_evidence=ravdess_evidence,
            )
            retained_runtime.append(mayo_evidence)
            mayo_result = _quiet_call(
                ssl_core.train_ssl_stage,
                stage_evidence=mayo_evidence,
                seed=seed,
                prior_ravdess_checkpoint=persisted_ravdess,
                prior_stage_evidence=ravdess_evidence,
            )
            mayo_payload = _quiet_call(
                ssl_core.build_ssl_checkpoint_payload, mayo_result,
            )
            mayo_path = seed_checkpoints / "ravdess_then_mayo.pt"
            mayo_receipt = _quiet_call(
                ssl_core.save_ssl_checkpoint,
                mayo_path,
                mayo_payload,
                stage_evidence=mayo_evidence,
            )
            persisted_mayo = _quiet_call(
                ssl_core.load_ssl_checkpoint,
                mayo_path,
                receipt=mayo_receipt,
                stage_evidence=mayo_evidence,
            )
            retained_runtime.extend((mayo_receipt, persisted_mayo))
            mayo_fingerprint = _quiet_call(
                ssl_core.ssl_checkpoint_fingerprint, persisted_mayo,
            )
            reload_checks.append((
                mayo_path, mayo_receipt, mayo_evidence, mayo_fingerprint,
            ))
            publication_lineages.append((
                ravdess_path,
                ravdess_receipt,
                ravdess_fingerprint,
                mayo_path,
                mayo_receipt,
                mayo_fingerprint,
            ))
            execution.append({
                "seed": seed,
                "ravdess_only": _stage_report(
                    ravdess_result, ravdess_fingerprint,
                ),
                "ravdess_then_mayo": _stage_report(
                    mayo_result, mayo_fingerprint,
                ),
            })

        report = {
            "schema_version": "dynamic_landmark_ssl_execution_report_v1",
            "mode": args.mode,
            "seed_count": len(seeds),
            "stage_count": 2,
            "runs": execution,
            "medical_generalization": False,
            "outer_mayo_predictions_viewed": False,
        }
        if args.mode == "formal":
            report["aggregate"] = _formal_aggregates(execution)
        report_name = (
            "execution_only.json"
            if args.mode == "smoke"
            else "formal_pretraining_results.json"
        )
        report_path = reports / report_name
        _quiet_call(_write_private_json, report_path, report)
        (
            _report_resolved,
            report_bytes,
            _report_sha256,
            _report_identity,
        ) = _quiet_call(
            ssl_core._private_regular_file_snapshot,
            report_path,
            "SSL execution report",
        )
        report_value = _quiet_call(
            ssl_core._strict_json_mapping,
            report_bytes,
            "SSL execution report",
        )
        if not ssl_core._exact_json_value(report_value, report):
            raise ValueError("SSL execution report changed after writing")
        for path, receipt, evidence, expected in reload_checks:
            reloaded = _quiet_call(
                ssl_core.load_ssl_checkpoint,
                path, receipt=receipt, stage_evidence=evidence,
            )
            if _quiet_call(
                ssl_core.ssl_checkpoint_fingerprint, reloaded,
            ) != expected:
                raise ValueError("checkpoint changed during result finalization")
        expected_files = _expected_result_files(args.mode, seeds)

        def final_authorization() -> None:
            for (
                ravdess_path,
                ravdess_receipt,
                expected_ravdess,
                mayo_path,
                mayo_receipt,
                expected_mayo,
            ) in publication_lineages:
                final_ravdess = ssl_core.load_ssl_checkpoint(
                    ravdess_path,
                    receipt=ravdess_receipt,
                    stage_evidence=ravdess_evidence,
                )
                if (
                    ssl_core.ssl_checkpoint_fingerprint(final_ravdess)
                    != expected_ravdess
                ):
                    raise ValueError(
                        "RAVDESS checkpoint changed before publication"
                    )
                final_mayo_evidence = ssl_core.authorize_frozen_ssl_stage(
                    stage="mayo",
                    mode=args.mode,
                    inputs_root=inputs_root,
                    bridge_root=bridge_root,
                    ravdess_authorizer=ravdess_authorizer,
                    mayo_authorizer=mayo_authorizer,
                    producer_sha256=producer_sha256,
                    prior_ravdess_checkpoint=final_ravdess,
                    prior_ravdess_evidence=ravdess_evidence,
                )
                final_mayo = ssl_core.load_ssl_checkpoint(
                    mayo_path,
                    receipt=mayo_receipt,
                    stage_evidence=final_mayo_evidence,
                )
                if (
                    ssl_core.ssl_checkpoint_fingerprint(final_mayo)
                    != expected_mayo
                ):
                    raise ValueError(
                        "Mayo checkpoint changed before publication"
                    )
            validate_generation_leases()
            edge_mayo = raw_mayo_authorizer()
            validate_generation_leases()
            edge_ravdess = raw_ravdess_authorizer()
            validate_generation_leases()
            _require_publication_edge_authorization(
                stage="mayo",
                evidence=mayo_evidence,
                authorization=edge_mayo,
            )
            _require_publication_edge_authorization(
                stage="ravdess",
                evidence=ravdess_evidence,
                authorization=edge_ravdess,
            )
            validate_generation_leases()

        with _hold_exact_result_tree(
            run_descriptor, staging.name, expected_files,
        ) as validate_result_tree:
            _quiet_call(
                _scan_private_results,
                (inputs_root, staging), privacy_forbidden,
            )
            for checkpoint_directory in sorted(
                checkpoint_directories,
                key=lambda path: len(path.parts),
                reverse=True,
            ):
                _quiet_call(_fsync_directory, checkpoint_directory)
            _quiet_call(_fsync_directory, reports)
            _quiet_call(_fsync_directory, staging)
            if "results" in set(os.listdir(run_descriptor)):
                raise FileExistsError(
                    "results generation appeared during execution"
                )
            before_publish = os.stat(
                staging.name,
                dir_fd=run_descriptor,
                follow_symlinks=False,
            )
            if (before_publish.st_dev, before_publish.st_ino) != staged_identity:
                raise ValueError(
                    "results staging identity changed before publication"
                )
            _quiet_call(
                _scan_private_results,
                (inputs_root, staging), privacy_forbidden,
            )
            validate_result_tree()
            _validate_run_root(run_descriptor, run_root, run_identity)
            _validate_results_lock(
                results_lock, lock_name, run_descriptor,
            )
            os.fsync(run_descriptor)
            _publish_validated_results(
                run_descriptor=run_descriptor,
                run_root=run_root,
                run_identity=run_identity,
                results_lock=results_lock,
                lock_name=lock_name,
                staging_name=staging.name,
                staged_identity=staged_identity,
                inputs_root=inputs_root,
                privacy_forbidden=privacy_forbidden,
                validate_result_tree=validate_result_tree,
                final_authorization=final_authorization,
            )

    return summary


def _run_mayo_ablation(
    args: argparse.Namespace,
    *,
    producer_sha256: str,
) -> dict[str, object]:
    """Run and atomically publish the exact nine-job Mayo-only ablation."""
    run_root = _private_directory(args.run_root.absolute(), "ablation run root")
    if run_root != _MAYO_ABLATION_NAMESPACE.absolute():
        raise ValueError("ablation run root is outside its canonical namespace")
    inputs_root = _ablation_inputs_root(run_root)
    bridge_root = _private_directory(
        args.bridge_root.absolute(), "bridge generation",
    )
    if bridge_root != (PRETRAINING_ROOT / "bridge").absolute():
        raise ValueError("bridge root is outside the canonical namespace")
    _preflight_live_inputs(args)
    raw_ravdess_authorizer, raw_mayo_authorizer = _quiet_call(
        _authorization_factories, args,
    )
    ravdess_authorizer = _transaction_authorizer(raw_ravdess_authorizer)
    mayo_authorizer = _transaction_authorizer(raw_mayo_authorizer)
    privacy_forbidden = _quiet_call(
        _privacy_forbidden,
        args, ravdess_authorizer, mayo_authorizer,
    )
    summary = {
        "arm_count": 3,
        "checkpoint_count": 9,
        "job_count": 9,
        "mode": "formal",
        "seed_count": 3,
        "stage_count": 1,
    }
    _assert_summary_private(
        json.dumps(
            summary, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("ascii"),
        privacy_forbidden,
    )
    with _exclusive_results_transaction(run_root, args) as (
        results_lock,
        lock_name,
        run_descriptor,
        run_identity,
        validate_generation_leases,
    ):
        names = set(os.listdir(run_descriptor))
        if "results" in names:
            raise FileExistsError("results generation already exists")
        if any(name.startswith(".results.staging-") for name in names):
            raise ValueError("unresolved results staging state blocks execution")

        evidence_by_arm = {
            arm: _quiet_call(
                ssl_core.authorize_frozen_ssl_stage,
                stage="mayo",
                mode="formal",
                inputs_root=inputs_root,
                bridge_root=bridge_root,
                ravdess_authorizer=ravdess_authorizer,
                mayo_authorizer=mayo_authorizer,
                producer_sha256=producer_sha256,
                experiment_kind="mayo_input_arm_ablation",
                mayo_input_arm=arm,
            )
            for arm in _ABLATION_ARMS
        }
        (
            common_split_sha256,
            common_scaler_sha256,
            common_cache_sha256,
        ) = _quiet_call(_common_ablation_evidence, evidence_by_arm)
        common_mask, common_mask_bytes, common_mask_sha256 = _quiet_call(
            _materialized_heldout_mask, evidence_by_arm[ARM_FUSION],
        )

        initialization: dict[tuple[str, int], tuple[str, str]] = {}
        for seed in _FORMAL_SEEDS:
            seed_digests: list[tuple[str, str]] = []
            for arm in _ABLATION_ARMS:
                digests = _quiet_call(_fresh_ablation_initialization, seed)
                initialization[(arm, seed)] = digests
                seed_digests.append(digests)
            if len(set(seed_digests)) != 1:
                raise ValueError(
                    "same-seed ablation model or optimizer initialization differs"
                )

        staging = run_root / f".results.staging-{secrets.token_hex(16)}"
        staging.mkdir(mode=0o700)
        os.chmod(staging, 0o700)
        staged_status = staging.stat()
        staged_identity = (staged_status.st_dev, staged_status.st_ino)
        checkpoints = staging / "checkpoints"
        reports = staging / "reports"
        checkpoints.mkdir(mode=0o700)
        reports.mkdir(mode=0o700)
        os.chmod(checkpoints, 0o700)
        os.chmod(reports, 0o700)

        arm_directories: dict[str, Path] = {}
        checkpoint_directories = {checkpoints}
        for arm in _ABLATION_ARMS:
            directory = checkpoints / arm
            directory.mkdir(mode=0o700)
            os.chmod(directory, 0o700)
            arm_directories[arm] = directory
            checkpoint_directories.add(directory)

        metric_rows: list[dict[str, object]] = []
        report_runs: list[dict[str, object]] = []
        reload_checks: list[tuple[Path, object, object, str]] = []
        publication_lineages: list[
            tuple[str, int, Path, object, str]
        ] = []
        retained_runtime: list[object] = list(evidence_by_arm.values())
        for arm in _ABLATION_ARMS:
            evidence = evidence_by_arm[arm]
            for seed in _FORMAL_SEEDS:
                result = _quiet_call(
                    ssl_core.train_ssl_stage,
                    stage_evidence=evidence,
                    seed=seed,
                    heldout_mask=common_mask,
                )
                receipt = result.training_receipt
                model_init, optimizer_init = initialization[(arm, seed)]
                if (
                    receipt.input_arm != arm
                    or receipt.target_schema != ssl_core.TARGET_FULL95
                    or receipt.experiment_kind != "mayo_input_arm_ablation"
                    or receipt.initialization_policy != "same_seed_fresh"
                    or receipt.prior_checkpoint_sha256 is not None
                    or receipt.seed != seed
                    or receipt.epochs != 30
                    or receipt.optimizer != "adamw"
                    or receipt.optimizer_steps != 30
                    or receipt.pre_state_sha256 != model_init
                    or receipt.heldout_mask_schedule_sha256
                    != common_mask_sha256
                    or receipt.heldout_indices_sha256 != common_split_sha256
                    or receipt.scaler_sha256 != common_scaler_sha256
                    or receipt.cache_binding_sha256 != common_cache_sha256
                ):
                    raise ValueError("ablation job violated its frozen contract")
                payload = _quiet_call(
                    ssl_core.build_ssl_checkpoint_payload, result,
                )
                path = arm_directories[arm] / f"seed_{seed}.pt"
                checkpoint_receipt = _quiet_call(
                    ssl_core.save_ssl_checkpoint,
                    path,
                    payload,
                    stage_evidence=evidence,
                )
                persisted = _quiet_call(
                    ssl_core.load_ssl_checkpoint,
                    path,
                    receipt=checkpoint_receipt,
                    stage_evidence=evidence,
                )
                fingerprint = _quiet_call(
                    ssl_core.ssl_checkpoint_fingerprint, persisted,
                )
                retained_runtime.extend((
                    result, checkpoint_receipt, persisted,
                ))
                reload_checks.append((
                    path, checkpoint_receipt, evidence, fingerprint,
                ))
                publication_lineages.append((
                    arm, seed, path, checkpoint_receipt, fingerprint,
                ))
                metrics = _quiet_call(_ablation_common_target_metrics, result)
                metric_rows.append({
                    "arm": arm,
                    "seed": seed,
                    "metrics": metrics,
                })
                per_recording = result.heldout_report.get(
                    "per_recording_metrics",
                )
                if type(per_recording) is not list or not per_recording:
                    raise ValueError(
                        "ablation private per-recording metrics are unavailable"
                    )
                report_runs.append({
                    "arm": arm,
                    "seed": seed,
                    "checkpoint_fingerprint": fingerprint,
                    "optimizer_steps": receipt.optimizer_steps,
                    "pre_state_sha256": model_init,
                    "optimizer_initial_state_sha256": optimizer_init,
                    "heldout_mask_sha256": common_mask_sha256,
                    "common_target_metrics": metrics,
                    "per_recording_metrics": per_recording,
                })

        aggregate, paired = _quiet_call(_ablation_statistics, metric_rows)
        report = {
            "schema_version": "dynamic_landmark_ssl_ablation_report_v1",
            "mode": "formal",
            "experiment_kind": "mayo_input_arm_ablation",
            "arms": list(_ABLATION_ARMS),
            "seeds": list(_FORMAL_SEEDS),
            "job_count": 9,
            "epochs": 30,
            "optimizer": "adamw",
            "target_schema": ssl_core.TARGET_FULL95,
            "shared_split_sha256": common_split_sha256,
            "shared_scaler_sha256": common_scaler_sha256,
            "shared_cache_sha256": common_cache_sha256,
            "shared_heldout_mask_sha256": common_mask_sha256,
            "heldout_mask_materialized_once": True,
            "input_mask_after_common_scaler": True,
            "prior_ravdess": False,
            "early_stopping": False,
            "retry_count": 0,
            "aggregation": "per_recording_then_equal_recording_mean",
            "runs": report_runs,
            "aggregate": aggregate,
            "paired_differences": paired,
            "medical_generalization": False,
            "hb_evaluation": False,
        }
        report_path = reports / "mayo_input_arm_ablation.json"
        _quiet_call(_write_private_json, report_path, report)
        (
            _report_resolved,
            report_bytes,
            _report_sha256,
            _report_identity,
        ) = _quiet_call(
            ssl_core._private_regular_file_snapshot,
            report_path,
            "Mayo ablation report",
        )
        report_value = _quiet_call(
            ssl_core._strict_json_mapping,
            report_bytes,
            "Mayo ablation report",
        )
        if not ssl_core._exact_json_value(report_value, report):
            raise ValueError("Mayo ablation report changed after writing")
        for path, receipt, evidence, expected in reload_checks:
            reloaded = _quiet_call(
                ssl_core.load_ssl_checkpoint,
                path,
                receipt=receipt,
                stage_evidence=evidence,
            )
            if _quiet_call(
                ssl_core.ssl_checkpoint_fingerprint, reloaded,
            ) != expected:
                raise ValueError("ablation checkpoint changed during finalization")

        def final_authorization() -> None:
            edge_ravdess = _transaction_authorizer(raw_ravdess_authorizer)
            edge_mayo = _transaction_authorizer(raw_mayo_authorizer)
            final_evidence = {
                arm: ssl_core.authorize_frozen_ssl_stage(
                    stage="mayo",
                    mode="formal",
                    inputs_root=inputs_root,
                    bridge_root=bridge_root,
                    ravdess_authorizer=edge_ravdess,
                    mayo_authorizer=edge_mayo,
                    producer_sha256=producer_sha256,
                    experiment_kind="mayo_input_arm_ablation",
                    mayo_input_arm=arm,
                )
                for arm in _ABLATION_ARMS
            }
            _common_ablation_evidence(final_evidence)
            final_mask = ssl_core.load_frozen_mayo_ablation_mask(
                final_evidence[ARM_FUSION]
            )
            if (
                ssl_core._tensor_fingerprint_bytes(final_mask)
                != common_mask_bytes
                or ssl_core._mask_sha256(final_mask) != common_mask_sha256
            ):
                raise ValueError(
                    "common Mayo ablation mask changed before publication"
                )
            for arm, seed, path, receipt, expected in publication_lineages:
                loaded = ssl_core.load_ssl_checkpoint(
                    path,
                    receipt=receipt,
                    stage_evidence=final_evidence[arm],
                )
                if ssl_core.ssl_checkpoint_fingerprint(loaded) != expected:
                    raise ValueError(
                        "ablation checkpoint changed before publication"
                    )
                if loaded["metadata"]["seed"] != seed:
                    raise ValueError(
                        "ablation checkpoint seed changed before publication"
                    )
            validate_generation_leases()

        expected_files = _expected_ablation_result_files()
        with _hold_exact_result_tree(
            run_descriptor, staging.name, expected_files,
        ) as validate_result_tree:
            _quiet_call(
                _scan_private_results,
                (inputs_root, staging),
                privacy_forbidden,
            )
            for directory in sorted(
                checkpoint_directories,
                key=lambda path: len(path.parts),
                reverse=True,
            ):
                _quiet_call(_fsync_directory, directory)
            _quiet_call(_fsync_directory, reports)
            _quiet_call(_fsync_directory, staging)
            if "results" in set(os.listdir(run_descriptor)):
                raise FileExistsError(
                    "results generation appeared during execution"
                )
            before_publish = os.stat(
                staging.name,
                dir_fd=run_descriptor,
                follow_symlinks=False,
            )
            if (before_publish.st_dev, before_publish.st_ino) != staged_identity:
                raise ValueError(
                    "results staging identity changed before publication"
                )
            validate_result_tree()
            _validate_run_root(run_descriptor, run_root, run_identity)
            _validate_results_lock(results_lock, lock_name, run_descriptor)
            os.fsync(run_descriptor)
            # Scan the common parent as well as every arm-specific frozen tree.
            _quiet_call(
                _scan_private_results,
                (inputs_root, staging),
                privacy_forbidden,
            )
            _publish_validated_results(
                run_descriptor=run_descriptor,
                run_root=run_root,
                run_identity=run_identity,
                results_lock=results_lock,
                lock_name=lock_name,
                staging_name=staging.name,
                staged_identity=staged_identity,
                inputs_root=run_root / "inputs",
                privacy_forbidden=privacy_forbidden,
                validate_result_tree=validate_result_tree,
                final_authorization=final_authorization,
            )
    return summary


def main(argv: list[str] | None = None) -> dict[str, object]:
    producer_sha256 = _producer_sha256()
    args = _parser().parse_args(argv)
    if args.command == "dry-run":
        result = _formal_job_matrix()
        print(json.dumps(
            result, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ))
        return result
    if args.command not in {"two-stage", "mayo-ablation"}:
        raise ValueError("unsupported pretraining command")
    from scripts import prepare_dynamic_landmark_ssl_inputs as inputs_cli

    captured = inputs_cli._run_mayo_cli_captured(
        args,
        lambda: (
            _run_two_stage(args, producer_sha256=producer_sha256)
            if args.command == "two-stage"
            else _run_mayo_ablation(args, producer_sha256=producer_sha256)
        ),
    )
    print(captured.json_line)
    result = json.loads(captured.json_line)
    if type(result) is not dict:
        raise ValueError("pretraining result is malformed")
    return result


def _entrypoint() -> None:
    try:
        main()
    except (FileExistsError, OSError, RuntimeError, ValueError):
        print("dynamic landmark pretraining failed closed", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(0)
