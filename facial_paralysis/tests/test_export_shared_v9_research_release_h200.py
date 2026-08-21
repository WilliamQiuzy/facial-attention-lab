from __future__ import annotations

from _testlib import run_all

from scripts.export_shared_v9_research_release_h200 import (
    COUNTS,
    build_public_provenance,
    selected_candidate,
)


def test_exporter_selects_only_masked_clinical_reconstruction(c):
    candidate = selected_candidate()
    c.eq(candidate.candidate_id, "BLV9-009")
    c.eq(candidate.mechanism, "masked_clinical_reconstruction")
    c.eq(candidate.inference_change, "training_only")


def test_public_provenance_is_closed_and_identifier_free(c):
    provenance = build_public_provenance(
        git_commit="a" * 40,
        source_commitments={
            "palsynet": "1" * 64,
            "neuroface": "2" * 64,
            "meei": "3" * 64,
        },
    )
    c.eq(provenance["source_counts"], COUNTS)
    c.eq(provenance["training_seeds"], [0, 1, 2])
    c.eq(provenance["training_epochs"], 20)
    rendered = str(provenance).lower()
    c.true(all(token not in rendered for token in (
        "participant", "group_id", "/home/", "/users/", "manifest_path",
    )))


if __name__ == "__main__":
    run_all("test_export_shared_v9_research_release_h200", dict(globals()))
