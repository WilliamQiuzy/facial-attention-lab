"""Focused, development-only SSL encoder transfer smoke tests."""
from __future__ import annotations

import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _testlib import Check, run_all  # noqa: E402
from src.evaluation.nested_group_cv import InnerGroupFold, NestedGroupFold  # noqa: E402
from src.models.dynamic_landmark import ARM_FUSION, DynamicLandmarkModel  # noqa: E402
from src.pretraining.dynamic_landmark_ssl import DynamicLandmarkSSLModel  # noqa: E402
from src.training.dynamic_landmark_benchmark import BenchmarkConfig  # noqa: E402
from src.training.dynamic_landmark_transfer_smoke import (  # noqa: E402
    DEVELOPMENT_CANDIDATES,
    FUSION_RANDOM,
    FUSION_SSL_WARMSTART,
    LANDMARK_RANDOM,
    run_development_inner_oof,
    transfer_focused_fusion_encoder,
)


TRANSFER_PREFIXES = (
    "proj_bs_x.", "proj_bs_dx.", "proj_lm_x.", "proj_lm_dx.",
    "temporal.", "attention_score.", "pool_projection.",
)


def _source_state() -> OrderedDict[str, torch.Tensor]:
    state = DynamicLandmarkSSLModel().state_dict()
    for number, value in enumerate(state.values(), start=1):
        value.fill_(number / 100.0)
    return state


def _snapshot(model: DynamicLandmarkModel) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _state_is_identical(
    model: DynamicLandmarkModel,
    expected: dict[str, torch.Tensor],
) -> bool:
    return all(
        torch.equal(value, expected[name])
        for name, value in model.state_dict().items()
    )


def _data(n: int = 10):
    generator = torch.Generator().manual_seed(1701)
    features = torch.randn(n, 4, 32, 95, generator=generator)
    valid_mask = torch.ones(n, 4, 32, dtype=torch.bool)
    source_indices = torch.arange(128, dtype=torch.int64).reshape(4, 32)
    source_indices = source_indices.unsqueeze(0).repeat(n, 1, 1)
    timestamps = source_indices.to(torch.float32) / 30.0
    labels = torch.tensor([index % 2 for index in range(n)], dtype=torch.float32)

    # Protected outer rows are intentionally unusable. Development code may
    # inspect tensor metadata, but must never subset, validate, or forward them.
    features[8:] = float("nan")
    valid_mask[8:] = False
    timestamps[8:] = float("nan")
    source_indices[8:] = -1
    labels[8:] = float("nan")
    return features, valid_mask, timestamps, source_indices, labels


def _fold() -> NestedGroupFold:
    outer_train = np.arange(8, dtype=np.int64)
    outer_test = np.arange(8, 10, dtype=np.int64)
    inner = []
    for start in range(0, 8, 2):
        validation = np.arange(start, start + 2, dtype=np.int64)
        train = np.setdiff1d(outer_train, validation)
        inner.append(InnerGroupFold(train, validation))
    return NestedGroupFold(outer_train, outer_test, tuple(inner))


def _config() -> BenchmarkConfig:
    return BenchmarkConfig(
        max_epochs=1,
        learning_rate=1e-3,
        weight_decay=0.0,
        mirror_probability=0.0,
    )


def test_candidate_registry_is_closed_and_ordered(c: Check):
    c.eq(LANDMARK_RANDOM, "landmark_random")
    c.eq(FUSION_RANDOM, "fusion_random")
    c.eq(FUSION_SSL_WARMSTART, "fusion_ssl_warmstart")
    c.eq(DEVELOPMENT_CANDIDATES, (
        LANDMARK_RANDOM, FUSION_RANDOM, FUSION_SSL_WARMSTART,
    ))


def test_transfer_copies_exact_encoder_and_preserves_fresh_head(c: Check):
    source = _source_state()
    c.eq(len(source), 22, "focused SSL schema is exactly 22 tensors")
    downstream = DynamicLandmarkModel(ARM_FUSION)
    before = _snapshot(downstream)

    copied = transfer_focused_fusion_encoder(source, downstream)
    expected = tuple(sorted(
        name for name in downstream.state_dict()
        if name.startswith(TRANSFER_PREFIXES)
    ))
    c.eq(len(expected), 16, "transfer allowlist contains exactly 16 tensors")
    c.eq(copied, expected, "returned audit keys are exact and sorted")
    for name in expected:
        c.true(torch.equal(downstream.state_dict()[name], source[name]), name)
        c.true(downstream.state_dict()[name].data_ptr() != source[name].data_ptr(),
               f"{name} is cloned, not aliased")
    for name in before:
        if name.startswith("binary_head."):
            c.true(torch.equal(downstream.state_dict()[name], before[name]),
                   f"fresh head changed at {name}")


def test_transfer_rejects_invalid_sources_atomically(c: Check):
    source = _source_state()
    invalid_states = []

    missing = OrderedDict(source)
    missing.pop(next(iter(missing)))
    invalid_states.append(missing)
    extra = OrderedDict(source)
    extra["unexpected.weight"] = torch.zeros(1, dtype=torch.float32)
    invalid_states.append(extra)
    wrong_shape = OrderedDict(source)
    wrong_shape["proj_bs_x.weight"] = torch.zeros(1, dtype=torch.float32)
    invalid_states.append(wrong_shape)
    wrong_dtype = OrderedDict(source)
    wrong_dtype["proj_bs_x.weight"] = wrong_dtype["proj_bs_x.weight"].to(torch.float64)
    invalid_states.append(wrong_dtype)
    nonfinite = OrderedDict(source)
    nonfinite["mayo_decoder.bias"] = nonfinite["mayo_decoder.bias"].clone()
    nonfinite["mayo_decoder.bias"][0] = float("nan")
    invalid_states.append(nonfinite)

    for invalid in invalid_states:
        downstream = DynamicLandmarkModel(ARM_FUSION)
        before = _snapshot(downstream)
        c.raises(
            lambda invalid=invalid, downstream=downstream:
                transfer_focused_fusion_encoder(invalid, downstream),
            ValueError,
            "invalid source must fail closed",
        )
        c.true(_state_is_identical(downstream, before),
               "rejected transfer mutated destination")

    nonfusion = DynamicLandmarkModel("landmark_only")
    before = _snapshot(nonfusion)
    c.raises(lambda: transfer_focused_fusion_encoder(source, nonfusion), ValueError,
             "only a Fusion destination is eligible")
    c.true(_state_is_identical(nonfusion, before),
           "non-Fusion rejection mutated destination")


def test_candidate_source_contract_rejects_before_training(c: Check):
    data = _data()
    common = dict(fold=_fold(), seed=0, epochs=1, config=_config())
    c.raises(lambda: run_development_inner_oof(
        *data, candidate="unknown", **common,
    ), ValueError, "candidate registry is closed")
    c.raises(lambda: run_development_inner_oof(
        *data, candidate=FUSION_SSL_WARMSTART, **common,
    ), ValueError, "warm-start requires a source")
    c.raises(lambda: run_development_inner_oof(
        *data, candidate=FUSION_RANDOM, source_state=_source_state(), **common,
    ), ValueError, "random initialization forbids a source")
    c.raises(lambda: run_development_inner_oof(
        *data, candidate=LANDMARK_RANDOM, source_state=_source_state(), **common,
    ), ValueError, "Landmark random initialization forbids a source")


def test_four_fold_oof_is_complete_and_outer_rows_remain_untouched(c: Check):
    features, mask, timestamps, source_indices, labels = _data()
    result = run_development_inner_oof(
        features, mask, timestamps, source_indices, labels,
        fold=_fold(),
        candidate=FUSION_SSL_WARMSTART,
        seed=0,
        epochs=1,
        config=_config(),
        source_state=_source_state(),
    )
    expected_keys = tuple(sorted(
        name for name in DynamicLandmarkModel(ARM_FUSION).state_dict()
        if name.startswith(TRANSFER_PREFIXES)
    ))
    c.eq(result.candidate, FUSION_SSL_WARMSTART)
    c.eq(result.seed, 0)
    c.eq(result.epochs, 1)
    c.true(np.array_equal(result.outer_train_indices, np.arange(8)),
           "result preserves outer-train order")
    c.true(np.array_equal(result.labels, np.asarray([0, 1] * 4)),
           "labels contain outer-train rows only and preserve order")
    c.eq(result.probabilities.shape, (8,), "every outer-train row has one OOF value")
    c.true(bool(np.isfinite(result.probabilities).all()), "OOF values are finite")
    c.true(bool(((result.probabilities >= 0) & (result.probabilities <= 1)).all()),
           "OOF values are probabilities")
    c.eq(result.transferred_keys_by_fold, (expected_keys,) * 4,
         "all four fresh models record the exact warm-start transfer")

    random_result = run_development_inner_oof(
        features, mask, timestamps, source_indices, labels,
        fold=_fold(), candidate=FUSION_RANDOM, seed=0, epochs=1, config=_config(),
    )
    c.eq(random_result.transferred_keys_by_fold, ((),) * 4,
         "all four random-init models record an empty transfer audit")


if __name__ == "__main__":
    run_all("test_dynamic_landmark_transfer_smoke", dict(globals()))
