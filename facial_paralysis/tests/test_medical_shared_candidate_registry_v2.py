from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _testlib import run_all

from src.models.medical_shared_candidate_registry_v2 import (
    COMPONENT_RATIONALES,
    SharedCandidateV2,
    candidate_registry,
)


def test_registry_is_exactly_the_frozen_32_candidate_cartesian_product(c):
    registry = candidate_registry()
    c.eq(len(registry), 32)
    c.eq(len({candidate.candidate_id for candidate in registry}), 32)
    c.true(all(type(candidate) is SharedCandidateV2 for candidate in registry))
    c.eq({candidate.view_mode for candidate in registry}, {
        "original_only", "bilateral_invariant",
    })
    c.eq({candidate.regional_mode for candidate in registry}, {
        "none", "all_excursion", "matched_excursion",
        "matched_excursion_velocity",
    })
    c.eq({candidate.pooling_mode for candidate in registry}, {
        "meanmax_set", "cross_action_transformer",
    })
    c.eq({candidate.fusion_mode for candidate in registry}, {
        "masked_concat", "reliability_gate",
    })
    c.eq(tuple(candidate.candidate_id for candidate in registry), tuple(
        f"MSC2-{index:03d}" for index in range(32)
    ))


def test_every_component_has_medical_or_methodological_authority_and_limit(c):
    c.eq(set(COMPONENT_RATIONALES), {
        "view_mode", "regional_mode", "pooling_mode", "fusion_mode",
    })
    for category, options in COMPONENT_RATIONALES.items():
        observed = {
            getattr(candidate, category) for candidate in candidate_registry()
        }
        c.eq(set(options), observed)
        for name, rationale in options.items():
            c.eq(set(rationale), {
                "phenomenon", "evidence", "valid_labels", "contraindication",
            })
            c.true(bool(rationale["phenomenon"].strip()))
            c.true(type(rationale["evidence"]) is tuple)
            c.true(any(item.startswith("https://") for item in rationale["evidence"]))
            c.true(type(rationale["valid_labels"]) is tuple)
            c.true(bool(rationale["valid_labels"]))
            c.true(bool(rationale["contraindication"].strip()))


def test_registry_forbids_unjustified_augmentation_and_source_routing(c):
    emitted = repr(candidate_registry()).lower()
    for forbidden in (
        "random_flip", "random_crop", "color_jitter", "source_adapter",
        "dataset_encoder", "source_encoder",
    ):
        c.true(forbidden not in emitted)


if __name__ == "__main__":
    run_all("test_medical_shared_candidate_registry_v2", dict(globals()))
