"""Nested selection contracts for Universal Phenotype Mixture v3."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src.evaluation.universal_phenotype_v3 import (  # noqa: E402
    CandidateSpec,
    FROZEN_CANDIDATES,
    nested_source_phenotype_folds,
    select_inner_global_threshold,
    select_worst_source_candidate,
    summarize_source_metrics,
)
from _testlib import Check, run_all  # noqa: E402


def _groups(count: int):
    return tuple(f"grp_{index:064x}" for index in range(count))


def _metadata():
    sources, phenotypes, labels = [], [], []
    cells = (
        ("palsynet", "healthy", 0, 12),
        ("palsynet", "palsy", 1, 12),
        ("neuroface", "healthy", 0, 11),
        ("neuroface", "als", 1, 11),
        ("neuroface", "post_stroke", 1, 14),
        ("meei", "healthy", 0, 10),
        ("meei", "palsy", 1, 50),
    )
    for source, phenotype, label, count in cells:
        sources.extend([source] * count)
        phenotypes.extend([phenotype] * count)
        labels.extend([label] * count)
    return (
        np.asarray(labels, dtype=np.int64), _groups(len(labels)),
        tuple(sources), tuple(phenotypes),
    )


def test_candidate_registry_is_small_diverse_and_frozen(c: Check):
    c.eq(len(FROZEN_CANDIDATES), 8, "formal nested search has eight candidates")
    c.eq(len({row.name for row in FROZEN_CANDIDATES}), 8,
         "candidate names are unique")
    c.true(any(row.family == "locked_landmark" for row in FROZEN_CANDIDATES),
           "the locked 110D comparator is retained")
    c.true({row.variant for row in FROZEN_CANDIDATES if row.variant}
           >= {"residual_mil", "hybrid_tcn_mil", "hybrid_set_transformer"},
           "major residual, temporal and set architectures are represented")
    c.eq({row.alignment_weight for row in FROZEN_CANDIDATES}, {0.0, 0.05},
         "the formal registry compares plain and control-aligned training")
    c.raises(lambda: CandidateSpec(
        name="bad", family="neural", variant="hybrid_tcn_mil", width=32,
        dropout=0.1, learning_rate=0.001, source="palsynet",
    ), TypeError, "source identity is not a candidate field")


def test_nested_folds_are_source_phenotype_stratified_and_disjoint(c: Check):
    labels, group_ids, sources, phenotypes = _metadata()
    folds = nested_source_phenotype_folds(labels, group_ids, sources, phenotypes)
    c.eq(len(folds), 6, "six outer folds are frozen")
    outer_seen = []
    all_indices = set(range(len(labels)))
    for fold in folds:
        train = set(fold.outer_train.tolist())
        held = set(fold.outer_held.tolist())
        c.true(not train.intersection(held), "outer train and held identities are disjoint")
        c.eq(train.union(held), all_indices, "each outer fold partitions all people")
        outer_seen.extend(held)
        c.eq(len(fold.inner_folds), 5, "each outer train has five inner folds")
        for inner_train, inner_held in fold.inner_folds:
            c.true(set(inner_train.tolist()).issubset(train),
                   "inner train remains inside the outer training boundary")
            c.true(set(inner_held.tolist()).issubset(train),
                   "inner validation remains inside the outer training boundary")
            c.true(not set(inner_train.tolist()).intersection(inner_held.tolist()),
                   "inner identities do not leak")
    c.eq(sorted(outer_seen), list(range(len(labels))),
         "each participant is outer-held exactly once")


def test_source_metrics_and_selection_use_worst_dataset_not_pooled_score(c: Check):
    labels, _, sources, _ = _metadata()
    probability = np.where(labels == 1, 0.8, 0.2).astype(np.float64)
    metrics = summarize_source_metrics(labels, probability, sources)
    c.eq(set(metrics), {"overall", "palsynet", "neuroface", "meei"},
         "metrics are reported separately for every dataset")
    summaries = {}
    for ordinal, candidate in enumerate(FROZEN_CANDIDATES):
        summaries[candidate.name] = {
            "worst_source_auroc": 0.70 + ordinal * 0.01,
            "worst_source_balanced_accuracy": 0.90,
            "worst_source_brier": 0.20,
            "pooled_auroc": 0.99 - ordinal * 0.05,
        }
    winner = select_worst_source_candidate(summaries)
    c.eq(winner, FROZEN_CANDIDATES[-1].name,
         "worst-source AUROC dominates a misleading pooled score")


def test_selection_tie_breaks_by_balanced_accuracy_then_brier_then_order(c: Check):
    summaries = {
        candidate.name: {
            "worst_source_auroc": 0.8,
            "worst_source_balanced_accuracy": 0.8,
            "worst_source_brier": 0.2,
        }
        for candidate in FROZEN_CANDIDATES
    }
    summaries[FROZEN_CANDIDATES[1].name]["worst_source_balanced_accuracy"] = 0.81
    summaries[FROZEN_CANDIDATES[2].name]["worst_source_balanced_accuracy"] = 0.81
    summaries[FROZEN_CANDIDATES[2].name]["worst_source_brier"] = 0.19
    c.eq(select_worst_source_candidate(summaries), FROZEN_CANDIDATES[2].name,
         "lower worst-source Brier resolves the AUROC and BA tie")


def test_one_inner_threshold_is_selected_without_source_identity(c: Check):
    labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    probabilities = np.asarray([0.10, 0.20, 0.60, 0.40, 0.70, 0.90])
    result = select_inner_global_threshold(labels, probabilities)
    c.eq(set(result), {"threshold", "balanced_accuracy"},
         "threshold selection returns a closed source-free result")
    c.true(np.isclose(result["threshold"], 0.3),
           "the optimal boundary is selected from adjacent OOF scores")
    c.true(np.isclose(result["balanced_accuracy"], 5.0 / 6.0),
           "the threshold objective is participant balanced accuracy")
    c.raises(lambda: select_inner_global_threshold(
        labels, np.asarray([0.1, 0.2, np.nan, 0.4, 0.7, 0.9])
    ), ValueError, "non-finite inner probabilities fail closed")


if __name__ == "__main__":
    run_all("test_universal_phenotype_evaluation_v3", dict(globals()))
