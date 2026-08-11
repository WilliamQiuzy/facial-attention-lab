"""Contracts for the development-only Architecture Autoresearch v1 sprint."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _testlib import Check, run_all  # noqa: E402
from src.models.architecture_search_v1 import (  # noqa: E402
    CANDIDATE_ORDER,
    NEURAL_CANDIDATES,
    build_neural_candidate,
    count_trainable_parameters,
)
from src.training.architecture_search_v1 import (  # noqa: E402
    OuterSearchLockedError,
    SearchDataset,
    SearchConfig,
    candidate_rank_key,
    evaluate_fixed_ensembles,
    group_balanced_weights,
    require_development_only,
    run_confirmation,
    run_screening,
    select_screening_winner,
)
from scripts.run_architecture_search_v1 import build_aggregate_report  # noqa: E402


EXPECTED_CANDIDATES = (
    "logistic_110d",
    "extra_trees_110d",
    "hist_gradient_boosting_110d",
    "mlp_110d",
    "tcn_landmark23",
    "bigru_landmark23",
    "transformer_landmark23",
    "region_tcn_landmark23",
    "hybrid_110d_tcn",
)


def _batch(n: int = 3):
    generator = torch.Generator().manual_seed(811)
    raw = torch.randn(n, 4, 32, 95, generator=generator)
    mask = torch.ones(n, 4, 32, dtype=torch.bool)
    mask[0, 0, -3:] = False
    raw[~mask] = 1_000.0
    summary = torch.randn(n, 110, generator=generator)
    return raw, mask, summary


def test_candidate_registry_is_diverse_frozen_and_compact(c: Check):
    c.eq(CANDIDATE_ORDER, EXPECTED_CANDIDATES)
    c.eq(
        NEURAL_CANDIDATES,
        EXPECTED_CANDIDATES[3:],
        "six compact neural families follow three classical baselines",
    )
    raw, mask, summary = _batch()
    for name in NEURAL_CANDIDATES:
        torch.manual_seed(0)
        model = build_neural_candidate(name)
        count = count_trainable_parameters(model)
        c.true(0 < count < 300_000, f"{name} stays below the fixed capacity cap")
        logits = model(raw, mask, summary)
        c.eq(tuple(logits.shape), (3,), f"{name} emits one recording logit")
        c.true(bool(torch.isfinite(logits).all()), f"{name} logits remain finite")


def test_invalid_frames_are_masked_not_learned(c: Check):
    raw, mask, summary = _batch()
    clean = raw.clone()
    clean[~mask] = 0.0
    for name in NEURAL_CANDIDATES:
        torch.manual_seed(13)
        model = build_neural_candidate(name).eval()
        with torch.no_grad():
            dirty_logits = model(raw, mask, summary)
            clean_logits = model(clean, mask, summary)
        c.true(
            bool(torch.allclose(dirty_logits, clean_logits, atol=1e-6, rtol=1e-6)),
            f"{name} cannot observe invalid-frame sentinel values",
        )


def test_group_weights_and_development_gate_fail_closed(c: Check):
    groups = np.asarray(["a", "a", "b", "c", "c", "c"], dtype=object)
    weights = group_balanced_weights(groups)
    c.true(bool(np.allclose(weights, [0.5, 0.5, 1.0, 1 / 3, 1 / 3, 1 / 3])))
    for group in set(groups.tolist()):
        c.true(bool(np.isclose(weights[groups == group].sum(), 1.0)))

    development = np.asarray([0, 1, 2, 3], dtype=np.int64)
    protected = np.asarray([4, 5], dtype=np.int64)
    require_development_only(np.asarray([0, 2]), development, protected, "fit")
    c.raises(
        lambda: require_development_only(
            np.asarray([1, 4]), development, protected, "predict"
        ),
        OuterSearchLockedError,
        "protected rows cannot enter any research operation",
    )
    c.raises(
        lambda: require_development_only(
            np.asarray([0, 6]), development, protected, "scale"
        ),
        ValueError,
        "unknown rows cannot enter the development experiment",
    )


def test_short_budget_and_winner_policy_are_fixed(c: Check):
    config = SearchConfig()
    c.eq(config.screen_epochs, 40)
    c.eq(config.patience, 6)
    c.eq(config.screen_seed, 0)
    c.eq(config.confirmation_seeds, (0, 1, 2))
    c.raises(lambda: SearchConfig(screen_epochs=41), ValueError)
    c.raises(lambda: SearchConfig(patience=7), ValueError)

    metrics = {
        "logistic_110d": {
            "auroc": 0.90, "balanced_accuracy": 0.90, "brier": 0.12,
            "parameter_count": 111,
        },
        "tcn_landmark23": {
            "auroc": 0.92, "balanced_accuracy": 0.88, "brier": 0.11,
            "parameter_count": 10_000,
        },
        "mlp_110d": {
            "auroc": 0.92, "balanced_accuracy": 0.91, "brier": 0.13,
            "parameter_count": 5_000,
        },
        "hybrid_110d_tcn": {
            "auroc": 0.92, "balanced_accuracy": 0.91, "brier": 0.13,
            "parameter_count": 20_000,
        },
    }
    c.eq(select_screening_winner(metrics), "mlp_110d")
    c.true(candidate_rank_key("mlp_110d", metrics["mlp_110d"])
           < candidate_rank_key("hybrid_110d_tcn", metrics["hybrid_110d_tcn"]),
           "exact metric ties retain the smaller architecture")


def test_smoke_screen_is_group_oof_and_never_reads_protected_rows(c: Check):
    generator = np.random.default_rng(8112026)
    n_development, n_protected = 16, 2
    n = n_development + n_protected
    raw = generator.normal(size=(n, 4, 32, 95)).astype(np.float32)
    mirrored_raw = raw.copy()
    masks = np.ones((n, 4, 32), dtype=bool)
    summaries = generator.normal(size=(n, 110)).astype(np.float64)
    mirrored_summaries = summaries.copy()
    labels = np.asarray([index % 2 for index in range(n)], dtype=np.int64)
    groups = np.asarray([f"group-{index}" for index in range(n)], dtype=object)
    development = np.arange(n_development, dtype=np.int64)
    protected = np.arange(n_development, n, dtype=np.int64)
    fold_by_index = np.full(n, -1, dtype=np.int64)
    fold_by_index[development] = np.repeat(np.arange(4), 4)
    # Any eager whole-array check would trip on these protected sentinels.
    raw[protected] = np.nan
    mirrored_raw[protected] = np.nan
    summaries[protected] = np.nan
    mirrored_summaries[protected] = np.nan
    dataset = SearchDataset(
        raw_features=raw,
        mirrored_raw_features=mirrored_raw,
        valid_masks=masks,
        summary_features=summaries,
        mirrored_summary_features=mirrored_summaries,
        labels=labels,
        group_ids=groups,
        development_indices=development,
        protected_indices=protected,
        inner_fold_by_index=fold_by_index,
    )
    result = run_screening(
        dataset,
        config=SearchConfig(smoke=True),
        candidates=("logistic_110d", "tcn_landmark23"),
        device="cpu",
    )
    c.eq(tuple(result.candidate_metrics), ("logistic_110d", "tcn_landmark23"))
    c.eq(result.development_recordings, n_development)
    c.eq(result.protected_predictions, 0)
    for name, metrics in result.candidate_metrics.items():
        c.true(0.0 <= metrics["auroc"] <= 1.0, f"{name} AUROC is valid")
        c.true(0.0 <= metrics["balanced_accuracy"] <= 1.0,
               f"{name} balanced accuracy is valid")
        c.true(0.0 <= metrics["brier"] <= 1.0, f"{name} Brier is valid")
        c.eq(metrics["oof_groups"], n_development)
        c.eq(metrics["oof_recordings"], n_development)


def test_aggregate_report_is_development_only_and_identifier_free(c: Check):
    metrics = {
        "logistic_110d": {
            "auroc": 0.9, "average_precision": 0.91, "brier": 0.1,
            "balanced_accuracy": 0.85, "sensitivity": 0.8,
            "specificity": 0.9, "parameter_count": 111,
            "oof_recordings": 16, "oof_groups": 16, "elapsed_seconds": 1.0,
        }
    }
    result = type("Result", (), {
        "candidate_metrics": metrics,
        "candidate_fold_metrics": {"logistic_110d": ({"auroc": 0.9},)},
        "winner": "logistic_110d",
        "development_recordings": 16,
        "development_groups": 16,
        "protected_predictions": 0,
    })()
    report = build_aggregate_report(
        result,
        smoke=True,
        device="cpu",
        provenance={
            "source_collection_sha256": "a" * 64,
            "reviewed_manifest_sha256": "b" * 64,
            "review_ledger_sha256": "c" * 64,
            "split_registry_sha256": "d" * 64,
            "implementation_sha256": "e" * 64,
        },
        protected_groups=2,
        protected_recordings=2,
    )
    c.eq(report["claim_scope"], "identity_reviewed_palsynet_development_oof_only")
    c.eq(report["decision"]["outer_evaluation_authorized"], False)
    c.eq(report["audit"]["protected_predictions"], 0)
    encoded = str(report).lower()
    c.true("recording_id" not in encoded and "group_id" not in encoded,
           "aggregate report cannot expose identity keys")


def test_confirmation_uses_exact_three_seeds_and_keeps_outer_sealed(c: Check):
    generator = np.random.default_rng(1911)
    n = 18
    raw = generator.normal(size=(n, 4, 32, 95)).astype(np.float32)
    summaries = generator.normal(size=(n, 110)).astype(np.float64)
    labels = np.asarray([index % 2 for index in range(n)], dtype=np.int64)
    folds = np.full(n, -1, dtype=np.int64)
    folds[:16] = np.repeat(np.arange(4), 4)
    dataset = SearchDataset(
        raw_features=raw,
        mirrored_raw_features=raw.copy(),
        valid_masks=np.ones((n, 4, 32), dtype=bool),
        summary_features=summaries,
        mirrored_summary_features=summaries.copy(),
        labels=labels,
        group_ids=np.asarray([f"confirm-{index}" for index in range(n)], dtype=object),
        development_indices=np.arange(16, dtype=np.int64),
        protected_indices=np.arange(16, 18, dtype=np.int64),
        inner_fold_by_index=folds,
    )
    result = run_confirmation(
        dataset,
        winner="logistic_110d",
        config=SearchConfig(smoke=True),
        device="cpu",
    )
    c.eq(tuple(result.seed_metrics), (0, 1, 2))
    c.eq(result.winner, "logistic_110d")
    c.eq(result.protected_predictions, 0)
    c.true(0 <= result.ensemble_metrics["auroc"] <= 1)


def test_adaptive_ensemble_round_uses_only_aligned_oof_probabilities(c: Check):
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    groups = np.asarray(["e0", "e1", "e2", "e3"], dtype=object)
    probabilities = {
        "logistic_110d": np.asarray([0.10, 0.40, 0.60, 0.90]),
        "extra_trees_110d": np.asarray([0.20, 0.20, 0.80, 0.80]),
        "mlp_110d": np.asarray([0.15, 0.30, 0.70, 0.85]),
        "hybrid_110d_tcn": np.asarray([0.20, 0.20, 0.80, 0.80]),
    }
    metrics = evaluate_fixed_ensembles(labels, groups, probabilities, bootstrap_repeats=50)
    c.eq(tuple(metrics), (
        "logistic_extra_trees_mean",
        "logistic_mlp_mean",
        "logistic_hybrid_mean",
        "logistic_extra_hybrid_mean",
    ))
    for name, values in metrics.items():
        c.eq(values["auroc"], 1.0, f"{name} uses aligned OOF probabilities")
        c.eq(values["balanced_accuracy"], 1.0)
        c.eq(values["bootstrap"]["repeats"], 50)
        c.true("auroc_delta_vs_logistic" in values["bootstrap"])
    broken = dict(probabilities)
    broken["mlp_110d"] = broken["mlp_110d"][:-1]
    c.raises(
        lambda: evaluate_fixed_ensembles(labels, groups, broken, bootstrap_repeats=10),
        ValueError,
        "unaligned OOF candidates cannot be ensembled",
    )


if __name__ == "__main__":
    run_all("test_architecture_search_v1", dict(globals()))
