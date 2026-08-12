"""Aggregate-only Mayo Action-Aligned challenge contracts."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _testlib import Check, run_all  # noqa: E402
from scripts.run_mayo_action_aligned_challenge_v1 import _paired_transition  # noqa: E402


def test_paired_transition_exports_counts_and_aggregate_delta_only(c: Check):
    baseline = np.full(47, 0.7, dtype=np.float64)
    action = np.full(47, 0.8, dtype=np.float64)
    baseline[:2] = (0.40, 0.42)
    action[0] = 0.55
    action[1] = 0.45
    result = _paired_transition(baseline, action)
    c.eq(result["both_positive"], 45)
    c.eq(result["baseline_negative_to_action_positive"], 1)
    c.eq(result["baseline_positive_to_action_negative"], 0)
    c.eq(result["both_negative"], 1)
    c.eq(set(result), {
        "both_positive", "baseline_negative_to_action_positive",
        "baseline_positive_to_action_negative", "both_negative", "score_delta",
    })
    c.true("probabilities" not in result)


def test_paired_transition_rejects_unaligned_or_non47_vectors(c: Check):
    c.raises(lambda: _paired_transition(np.ones(46), np.ones(46)), ValueError)
    c.raises(lambda: _paired_transition(np.ones(47), np.ones(48)), ValueError)


if __name__ == "__main__":
    run_all("test_mayo_action_aligned_challenge_v1", dict(globals()))
