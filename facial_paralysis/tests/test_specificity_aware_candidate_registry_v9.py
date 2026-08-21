from __future__ import annotations

from _testlib import run_all

from src.models.specificity_aware_candidate_registry_v9 import (
    COMPONENT_RATIONALES_V9,
    SpecificityCandidateV9,
    candidate_registry_v9,
)


def test_registry_is_exact_unique_and_bounded(c):
    registry = candidate_registry_v9()
    c.eq(len(registry), 24)
    c.eq(len(set(registry)), 24)
    c.eq(tuple(row.candidate_id for row in registry), tuple(
        f"SSR9-{index:03d}" for index in range(24)
    ))
    c.eq({row.healthy_mode for row in registry}, {
        "off", "compact", "compact_margin",
    })
    c.eq({row.control_cost for row in registry}, {1.0, 1.5})
    c.eq({row.universal_blend for row in registry}, {0.25, 0.5})
    c.eq({row.control_alignment_weight for row in registry}, {0.0, 0.02})
    c.true(all(type(row) is SpecificityCandidateV9 for row in registry))


def test_every_search_component_has_a_medical_boundary(c):
    c.eq(set(COMPONENT_RATIONALES_V9), {
        "healthy_mode", "control_cost", "universal_blend",
        "control_alignment_weight",
    })
    for component, values in COMPONENT_RATIONALES_V9.items():
        c.true(values)
        for value, rationale in values.items():
            c.eq(set(rationale), {"phenomenon", "evidence", "contraindication"})
            c.true(bool(rationale["phenomenon"]))
            c.true(bool(rationale["evidence"]))
            c.true(bool(rationale["contraindication"]))


if __name__ == "__main__":
    run_all("test_specificity_aware_candidate_registry_v9", dict(globals()))
