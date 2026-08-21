from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from _testlib import run_all

from src.deployment.shared_v8_release import (
    DEPLOYMENT_MODEL_ID,
    PROTOCOL_TASK_CODES,
    load_release,
    validate_request_arrays,
    write_release,
)
from src.models.residual_shared_router_v8 import (
    ResidualSharedRouterV8,
    candidate_registry_v8,
)


def _candidate():
    return next(row for row in candidate_registry_v8() if row.candidate_id == "RSR8-001")


def _provenance():
    return {
        "git_commit": "a" * 40,
        "training_seed": 0,
        "training_epochs": 20,
        "training_device": "NVIDIA H200",
        "source_counts": {"palsynet": 38, "neuroface": 36, "meei": 56},
        "source_commitments": {
            "palsynet": "1" * 64,
            "neuroface": "2" * 64,
            "meei": "3" * 64,
        },
    }


def _request(protocol: str) -> dict[str, np.ndarray]:
    actions = {
        "free_motion_four_window": 4,
        "scripted_three_action": 3,
        "cue_aligned_action": 7,
    }[protocol]
    generator = np.random.default_rng(712)
    dense_available = np.zeros(actions, dtype=bool)
    if protocol != "free_motion_four_window":
        dense_available[:] = True
    dense_valid = np.repeat(dense_available[:, None], 32, axis=1)
    dense_original = np.zeros((actions, 32, 478, 3), dtype=np.float32)
    dense_mirrored = np.zeros_like(dense_original)
    dense_timestamps = np.zeros((actions, 32), dtype=np.float32)
    if dense_available.any():
        dense_original[:] = generator.normal(size=dense_original.shape).astype(np.float32)
        dense_mirrored[:] = generator.normal(size=dense_mirrored.shape).astype(np.float32)
        dense_timestamps[:] = np.linspace(0.0, 1.0, 32, dtype=np.float32)
    return {
        "clinical_original": generator.normal(size=(actions, 110)).astype(np.float32),
        "clinical_mirrored": generator.normal(size=(actions, 110)).astype(np.float32),
        "dense_original": dense_original,
        "dense_mirrored": dense_mirrored,
        "dense_valid_mask": dense_valid,
        "dense_available": dense_available,
        "dense_timestamps": dense_timestamps,
        "action_mask": np.ones(actions, dtype=bool),
        "action_codes": np.arange(actions, dtype=np.int64),
    }


def test_release_identity_is_exact_and_protocols_are_clinical(c):
    c.eq(DEPLOYMENT_MODEL_ID, "residual_shared_router_v8_rsr8_001")
    c.eq(PROTOCOL_TASK_CODES, {
        "free_motion_four_window": 0,
        "scripted_three_action": 1,
        "cue_aligned_action": 2,
    })
    c.true(all(token not in " ".join(PROTOCOL_TASK_CODES) for token in (
        "palsynet", "neuroface", "meei", "mayo",
    )))


def test_release_roundtrip_is_checksum_bound_and_deterministic(c):
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "release"
        torch.manual_seed(91)
        model = ResidualSharedRouterV8(_candidate()).eval()
        manifest = write_release(
            root,
            model=model,
            scaler_mean=np.zeros(110, dtype=np.float64),
            scaler_scale=np.ones(110, dtype=np.float64),
            provenance=_provenance(),
        )
        c.eq(manifest["model_id"], DEPLOYMENT_MODEL_ID)
        c.eq(manifest["candidate_id"], "RSR8-001")
        c.eq(
            hashlib.sha256((root / "weights.npz").read_bytes()).hexdigest(),
            manifest["weights_sha256"],
        )
        serialized = json.dumps(manifest, sort_keys=True)
        c.true("participant" not in serialized and "group_id" not in serialized)
        predictor = load_release(root, device="cpu")
        request = _request("scripted_three_action")
        first = predictor.predict("scripted_three_action", request)
        second = predictor.predict("scripted_three_action", request)
        c.eq(first, second)
        c.true(0.0 <= first.probability <= 1.0)
        c.eq(first.predicted_class, int(first.probability >= 0.5))
        c.eq(first.model_id, DEPLOYMENT_MODEL_ID)

        second_root = Path(temporary) / "release-repeat"
        repeated = write_release(
            second_root,
            model=model,
            scaler_mean=np.zeros(110, dtype=np.float64),
            scaler_scale=np.ones(110, dtype=np.float64),
            provenance=_provenance(),
        )
        c.eq(
            (root / "weights.npz").read_bytes(),
            (second_root / "weights.npz").read_bytes(),
        )
        c.eq(manifest["weights_sha256"], repeated["weights_sha256"])

        payload = bytearray((root / "weights.npz").read_bytes())
        payload[-1] ^= 1
        (root / "weights.npz").write_bytes(payload)
        c.raises(lambda: load_release(root, device="cpu"), ValueError)


def test_request_validation_rejects_hidden_or_malformed_signal(c):
    valid = _request("free_motion_four_window")
    normalized = validate_request_arrays("free_motion_four_window", valid)
    c.eq(normalized["clinical_original"].shape, (1, 4, 110))

    nonfinite = {key: value.copy() for key, value in valid.items()}
    nonfinite["clinical_original"][0, 0] = np.nan
    c.raises(
        lambda: validate_request_arrays("free_motion_four_window", nonfinite),
        ValueError,
    )
    hidden = {key: value.copy() for key, value in valid.items()}
    hidden["dense_original"][0, 0, 0, 0] = 1.0
    c.raises(
        lambda: validate_request_arrays("free_motion_four_window", hidden),
        ValueError,
    )
    wrong_dtype = {key: value.copy() for key, value in valid.items()}
    wrong_dtype["action_codes"] = wrong_dtype["action_codes"].astype(np.int32)
    c.raises(
        lambda: validate_request_arrays("free_motion_four_window", wrong_dtype),
        ValueError,
    )
    extra = {**valid, "participant_id": np.asarray([1], dtype=np.int64)}
    c.raises(
        lambda: validate_request_arrays("free_motion_four_window", extra),
        ValueError,
    )


def test_release_publication_is_no_overwrite(c):
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "release"
        model = ResidualSharedRouterV8(_candidate()).eval()
        arguments = dict(
            model=model,
            scaler_mean=np.zeros(110, dtype=np.float64),
            scaler_scale=np.ones(110, dtype=np.float64),
            provenance=_provenance(),
        )
        write_release(root, **arguments)
        c.raises(lambda: write_release(root, **arguments), FileExistsError)


if __name__ == "__main__":
    run_all("test_shared_v8_release", dict(globals()))
