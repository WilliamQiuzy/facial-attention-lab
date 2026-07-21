"""Focused, development-only SSL encoder transfer smoke tests."""
from __future__ import annotations

import copy
import io
import stat
import sys
from collections import OrderedDict
from contextlib import redirect_stderr
from pathlib import Path
from tempfile import TemporaryDirectory

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
from scripts.run_dynamic_landmark_transfer_smoke import (  # noqa: E402
    COMMITMENT_FIELDS,
    DEFAULT_REPORT_PATH,
    RUN_EPOCHS,
    RUN_OUTER_FOLD,
    RUN_SEED,
    _atomic_write_report,
    _build_report,
    _canonical_json_bytes,
    _extract_authenticated_winner,
    _parser,
    _validate_report,
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


def _commitments() -> dict[str, str]:
    return {
        name: f"{index:x}" * 64
        for index, name in enumerate(COMMITMENT_FIELDS, start=1)
    }


def _metric_rows() -> dict[str, dict[str, float]]:
    return {
        LANDMARK_RANDOM: {
            "auroc": 0.70, "average_precision": 0.71, "brier": 0.20,
            "balanced_accuracy": 0.65, "sensitivity": 0.75, "specificity": 0.55,
        },
        FUSION_RANDOM: {
            "auroc": 0.66, "average_precision": 0.69, "brier": 0.21,
            "balanced_accuracy": 0.70, "sensitivity": 0.80, "specificity": 0.60,
        },
        FUSION_SSL_WARMSTART: {
            "auroc": 0.70, "average_precision": 0.73, "brier": 0.19,
            "balanced_accuracy": 0.71, "sensitivity": 0.78, "specificity": 0.64,
        },
    }


def _report() -> dict[str, object]:
    return _build_report(
        accounting={
            "total_records": 49, "total_groups": 48,
            "development_records": 39, "development_groups": 38,
            "protected_records": 10, "protected_groups": 10,
        },
        commitments=_commitments(),
        candidate_metrics=_metric_rows(),
    )


def test_runner_parser_is_private_and_has_no_tuning_surface(c: Check):
    parser = _parser()
    args = parser.parse_args([
        "--ssl-pretraining-root", "/private/ssl",
        "--palsynet-cache-root", "/private/palsynet",
    ])
    c.eq(vars(args), {
        "ssl_pretraining_root": Path("/private/ssl"),
        "palsynet_cache_root": Path("/private/palsynet"),
    })
    c.eq((RUN_SEED, RUN_EPOCHS, RUN_OUTER_FOLD), (0, 12, 0))
    c.eq(
        DEFAULT_REPORT_PATH,
        ROOT / "outputs/dynamic_landmark/benchmarks/development/"
        "focused-ssl-transfer-smoke-v1/report.json",
    )
    for forbidden in ("--seed", "--epochs", "--fold", "--candidate",
                      "--output", "--outer"):
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            c.raises(
                lambda forbidden=forbidden: parser.parse_args([
                    "--ssl-pretraining-root", "/private/ssl",
                    "--palsynet-cache-root", "/private/palsynet",
                    forbidden, "/do/not/disclose",
                ]),
                SystemExit,
                f"{forbidden} must not be accepted",
            )
        c.true("/do/not/disclose" not in stderr.getvalue(),
               "argparse errors must redact supplied paths")


def test_closed_report_schema_order_metrics_and_decision(c: Check):
    report = _report()
    c.eq(tuple(report), (
        "schema_version", "status", "claim_scope", "dataset", "claim_unit",
        "identity_status", "protocol", "accounting", "commitments",
        "candidates", "decision",
    ))
    c.eq(tuple(report["commitments"]), COMMITMENT_FIELDS)
    protocol = report["protocol"]
    c.eq(protocol, {
        "seed": 0, "epochs": 12, "outer_fold": 0, "inner_folds": 4,
        "candidates": list(DEVELOPMENT_CANDIDATES), "optimizer": "AdamW",
        "learning_rate": 1e-3, "weight_decay": 1e-4,
        "mirror_probability": 0.5, "threshold": 0.5,
        "group_probability_aggregation": "mean", "outer_predictions": 0,
    })
    rows = report["candidates"]
    c.eq([row["candidate"] for row in rows], list(DEVELOPMENT_CANDIDATES))
    c.eq([row["initialization"] for row in rows], [
        "random", "random", "authenticated_fusion_ssl_seed0",
    ])
    c.eq(report["decision"], {
        "best_candidate": LANDMARK_RANDOM,
        "warmstart_minus_random_fusion_auroc": 0.04,
        "warmstart_minus_random_fusion_sensitivity": -0.02,
        "formal_expansion_gate": True,
        "recommendation": "expand_to_three_seed_development_evaluation_only",
    })
    c.eq(_validate_report(report), report)
    c.true(_validate_report(report) is not report,
           "validation reconstructs a closed report")

    invalid = copy.deepcopy(report)
    invalid["candidates"][0]["brier"] = float("nan")
    c.raises(lambda: _validate_report(invalid), ValueError,
             "nonfinite report numbers fail closed")
    invalid = copy.deepcopy(report)
    invalid["decision"]["recommendation"] = "/private/ssl/seed_0.pt"
    c.raises(
        lambda: _validate_report(
            invalid, forbidden_paths=(Path("/private/ssl"),),
        ),
        ValueError,
        "input paths cannot enter the report",
    )
    invalid = copy.deepcopy(report)
    invalid["status"] = "grp_" + "a" * 64
    c.raises(lambda: _validate_report(invalid), ValueError,
             "group identifiers cannot enter the report")
    for field in COMMITMENT_FIELDS:
        invalid = copy.deepcopy(report)
        invalid["commitments"][field] = "A" * 64
        c.raises(lambda invalid=invalid: _validate_report(invalid), ValueError,
                 f"{field} must be lowercase SHA-256")


def test_private_atomic_report_writer_has_modes_and_no_overwrite(c: Check):
    report = _report()
    with TemporaryDirectory() as temporary:
        path = Path(temporary) / "private" / "nested" / "report.json"
        _atomic_write_report(path, report)
        c.eq(path.read_bytes(), _canonical_json_bytes(report))
        c.eq(stat.S_IMODE(path.stat().st_mode), 0o600)
        c.eq(stat.S_IMODE(path.parent.stat().st_mode), 0o700)
        c.eq(stat.S_IMODE(path.parent.parent.stat().st_mode), 0o700)
        c.raises(lambda: _atomic_write_report(path, report), FileExistsError,
                 "report publication cannot overwrite")
        c.eq(path.read_bytes(), _canonical_json_bytes(report),
             "failed overwrite preserves the published report")


def _winner_chain() -> dict[str, object]:
    return {
        "trainer_sha256": "1" * 64,
        "bridge_generation_sha256": "2" * 64,
        "common_contract_sha256": "3" * 64,
        "winner_report_sha256": "4" * 64,
        "selected_arm": "fusion",
        "checkpoints": {
            seed: {
                "metadata": {
                    "phase": "winner", "arm": "fusion", "seed": seed,
                    "epochs": 30,
                },
                "model_state": OrderedDict({
                    "weight": torch.tensor([float(seed)], dtype=torch.float32),
                }),
                "checkpoint_fingerprint": f"{seed + 5:x}" * 64,
                "checkpoint_receipt_sha256": f"{seed + 8:x}" * 64,
            }
            for seed in (0, 1, 2)
        },
    }


def test_authenticated_winner_extraction_is_exact_and_cloned(c: Check):
    chain = _winner_chain()
    state, commitments = _extract_authenticated_winner(
        chain, current_trainer_sha256="1" * 64,
    )
    c.eq(tuple(commitments), COMMITMENT_FIELDS[:6])
    c.true(torch.equal(state["weight"], torch.tensor([0.0])))
    c.true(state["weight"].data_ptr()
           != chain["checkpoints"][0]["model_state"]["weight"].data_ptr())

    wrong_arm = _winner_chain()
    wrong_arm["selected_arm"] = "landmark_only"
    c.raises(lambda: _extract_authenticated_winner(
        wrong_arm, current_trainer_sha256="1" * 64,
    ), ValueError, "only the authenticated Fusion winner is eligible")
    wrong_seeds = _winner_chain()
    wrong_seeds["checkpoints"].pop(2)
    c.raises(lambda: _extract_authenticated_winner(
        wrong_seeds, current_trainer_sha256="1" * 64,
    ), ValueError, "winner seeds must be exactly 0, 1, 2")
    wrong_field = _winner_chain()
    wrong_field["checkpoints"][0]["pre_model_state"] = \
        wrong_field["checkpoints"][0].pop("model_state")
    c.raises(lambda: _extract_authenticated_winner(
        wrong_field, current_trainer_sha256="1" * 64,
    ), ValueError, "pre_model_state is never an eligible transfer source")


if __name__ == "__main__":
    run_all("test_dynamic_landmark_transfer_smoke", dict(globals()))
