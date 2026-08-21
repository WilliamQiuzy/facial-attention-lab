from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _testlib import run_all
from test_medically_gated_shared_search_v2 import _dataset

from src.evaluation.literature_grounded_diagnostics_v9 import (
    diagnose_v8_shared_optimization,
    diagnostic_to_public_dict,
)
from src.evaluation.shared_clinical_encoder_v1 import SOURCES


def _diagnostic():
    return diagnose_v8_shared_optimization(
        _dataset(), audit_epochs=1, n_splits=2, seed=0, device="cpu"
    )


def test_diagnostic_is_fold_train_only_complete_and_finite(c):
    result = _diagnostic()
    c.eq(result.fold_count, 2)
    c.eq(tuple(row.source for row in result.source_summaries), SOURCES)
    c.eq(
        tuple(row.pair for row in result.pairwise_cosines),
        (
            "palsynet__neuroface",
            "palsynet__meei",
            "neuroface__meei",
        ),
    )
    for row in result.source_summaries:
        c.true(row.initial_loss > 0.0 and row.final_loss > 0.0)
        c.true(row.initial_gradient_norm > 0.0)
        c.true(math.isfinite(row.relative_remaining_loss))
    for row in result.pairwise_cosines:
        c.true(-1.0 <= row.median_cosine <= 1.0)


def test_authorization_is_derived_from_frozen_thresholds(c):
    result = _diagnostic()
    c.eq(
        result.cagrad_authorized,
        any(row.median_cosine < 0.0 for row in result.pairwise_cosines),
    )
    c.eq(
        result.gradnorm_authorized,
        result.gradient_norm_ratio >= 2.0
        or result.relative_training_rate_ratio >= 2.0,
    )


def test_public_payload_is_aggregate_closed_and_identifier_free(c):
    payload = diagnostic_to_public_dict(_diagnostic())
    c.eq(
        set(payload),
        {
            "schema_version",
            "fold_count",
            "source_summaries",
            "pairwise_gradient_cosines",
            "gradient_norm_ratio",
            "relative_training_rate_ratio",
            "cagrad_authorized",
            "gradnorm_authorized",
        },
    )
    rendered = repr(payload)
    c.true("grp_" not in rendered)
    c.true("participant" not in rendered.lower())
    c.true("/Users/" not in rendered and "/home/" not in rendered)


def test_invalid_configuration_fails_closed(c):
    c.raises(
        lambda: diagnose_v8_shared_optimization(
            _dataset(), audit_epochs=0, n_splits=2, seed=0, device="cpu"
        ),
        ValueError,
    )


if __name__ == "__main__":
    run_all("test_literature_grounded_diagnostics_v9", dict(globals()))
