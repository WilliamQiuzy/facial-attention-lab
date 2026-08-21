from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tempfile
import zipfile

import numpy as np
import torch

from _testlib import run_all

from src.deployment.shared_v9_research_release import (
    RESEARCH_CANDIDATE_ID,
    RESEARCH_MODEL_ID,
    RESEARCH_SEEDS,
    load_release,
    write_release,
)
from src.models.broad_literature_candidate_registry_v9 import candidate_registry_v9
from src.models.broad_literature_shared_router_v9 import BroadLiteratureSharedRouterV9


def _candidate():
    return next(row for row in candidate_registry_v9() if row.candidate_id == "BLV9-009")


def _models():
    result = []
    for seed in RESEARCH_SEEDS:
        torch.manual_seed(seed)
        result.append(BroadLiteratureSharedRouterV9(_candidate()).eval())
    return tuple(result)


def _provenance():
    return {
        "git_commit": "a" * 40,
        "training_seeds": list(RESEARCH_SEEDS),
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
    generator = np.random.default_rng(9909)
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


def test_v9_identity_and_roundtrip_are_exact(c):
    c.eq(RESEARCH_CANDIDATE_ID, "BLV9-009")
    c.eq(RESEARCH_SEEDS, (0, 1, 2))
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "release"
        manifest = write_release(
            root,
            models=_models(),
            scaler_mean=np.zeros(110, dtype=np.float64),
            scaler_scale=np.ones(110, dtype=np.float64),
            provenance=_provenance(),
        )
        c.eq(manifest["model_id"], RESEARCH_MODEL_ID)
        c.eq(manifest["candidate_id"], RESEARCH_CANDIDATE_ID)
        c.eq(tuple(manifest["ensemble_seeds"]), RESEARCH_SEEDS)
        for row in manifest["weights"]:
            payload = (root / row["file"]).read_bytes()
            c.eq(hashlib.sha256(payload).hexdigest(), row["sha256"])
        serialized = json.dumps(manifest, sort_keys=True)
        c.true(all(token not in serialized.lower() for token in (
            "participant_id", "group_id", "/users/", "/home/",
        )))

        predictor = load_release(root, device="cpu")
        request = _request("scripted_three_action")
        first = predictor.predict("scripted_three_action", request)
        second = predictor.predict("scripted_three_action", request)
        c.eq(first, second)
        c.eq(first.model_id, RESEARCH_MODEL_ID)
        c.eq(len(first.member_probabilities), 3)
        c.true(np.isclose(
            first.probability, float(np.mean(first.member_probabilities)), atol=1e-8
        ))
        c.eq(first.predicted_class, int(first.probability >= 0.5))


def test_v9_release_rejects_tampering_and_duplicate_npz_members(c):
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "release"
        write_release(
            root,
            models=_models(),
            scaler_mean=np.zeros(110, dtype=np.float64),
            scaler_scale=np.ones(110, dtype=np.float64),
            provenance=_provenance(),
        )
        target = root / "weights-seed0.npz"
        payload = bytearray(target.read_bytes())
        payload[-1] ^= 1
        target.write_bytes(payload)
        c.raises(lambda: load_release(root, device="cpu"), ValueError)

        clean = Path(temporary) / "clean"
        write_release(
            clean,
            models=_models(),
            scaler_mean=np.zeros(110, dtype=np.float64),
            scaler_scale=np.ones(110, dtype=np.float64),
            provenance=_provenance(),
        )
        target = clean / "weights-seed0.npz"
        original = target.read_bytes()
        with np.load(io.BytesIO(original), allow_pickle=False) as saved:
            names = tuple(saved.files)
            first_name = names[0]
            first_payload = io.BytesIO()
            np.lib.format.write_array(first_payload, np.asarray(saved[first_name]))
        duplicate = io.BytesIO()
        with zipfile.ZipFile(duplicate, "w") as archive:
            archive.writestr(first_name + ".npy", first_payload.getvalue())
            archive.writestr(first_name + ".npy", first_payload.getvalue())
        target.write_bytes(duplicate.getvalue())
        manifest = json.loads((clean / "manifest.json").read_text())
        manifest["weights"][0]["sha256"] = hashlib.sha256(duplicate.getvalue()).hexdigest()
        (clean / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
        )
        c.raises(lambda: load_release(clean, device="cpu"), ValueError)


def test_v9_release_is_immutable(c):
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "release"
        arguments = dict(
            models=_models(),
            scaler_mean=np.zeros(110, dtype=np.float64),
            scaler_scale=np.ones(110, dtype=np.float64),
            provenance=_provenance(),
        )
        write_release(root, **arguments)
        c.raises(lambda: write_release(root, **arguments), FileExistsError)


if __name__ == "__main__":
    run_all("test_shared_v9_research_release", dict(globals()))
