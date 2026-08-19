"""Contracts for the inference-ready final PalsyNet 110D artifact."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from scripts.freeze_110d_generalization_v1_artifact import _parser  # noqa: E402
from scripts.run_110d_generalization_v1 import canonical_json_sha256  # noqa: E402
from src.evaluation.outer_release_110d_v1 import (  # noqa: E402
    FinalArtifactAudit,
    OuterReleaseAudit,
    authorize_final_artifact,
    freeze_final_artifact,
    predict_from_frozen_artifact,
    prepare_locked_views,
    run_protected_outer,
    validate_frozen_artifact,
    validate_outer_authorization,
)
from test_110d_outer_release_v1 import _fixture  # noqa: E402
from _testlib import Check, run_all  # noqa: E402


def _sealed_fixture():
    (
        dataset, _, _, _, gate, development_report, development_sha, _,
        authorization, authorization_sha,
    ) = _fixture()
    release_audit = OuterReleaseAudit()
    state = validate_outer_authorization(
        authorization,
        authorization_sha256=authorization_sha,
        pinned_authorization_sha256=authorization_sha,
        gate=gate,
        development_report=development_report,
        development_report_sha256=development_sha,
        audit=release_audit,
        expected_development_bootstrap_repeats=16,
    )
    release_audit.development_cache_records_loaded = gate.development_indices.size
    release_audit.protected_cache_records_loaded = gate.protected_indices.size
    views = prepare_locked_views(
        dataset, gate, state=state, audit=release_audit
    )
    outer = run_protected_outer(
        dataset,
        gate,
        views,
        state=state,
        audit=release_audit,
        bootstrap_repeats=16,
    )
    outer_sha = canonical_json_sha256(outer.report)
    return dataset, gate, state, views, outer.report, outer_sha


def test_outer_report_must_be_exactly_pinned_before_final_fit(c: Check):
    dataset, gate, state, views, outer_report, outer_sha = _sealed_fixture()
    audit = FinalArtifactAudit()
    sealed = authorize_final_artifact(
        outer_report,
        protected_report_sha256=outer_sha,
        pinned_protected_report_sha256=outer_sha,
        gate=gate,
        state=state,
        audit=audit,
        expected_bootstrap_repeats=16,
    )
    c.eq(audit.as_dict(), {
        "protected_report_attempts": 1,
        "protected_report_passes": 1,
        "scaler_fits": 0,
        "model_fits": 0,
    }, "sealed protected commitment performs no fit")

    mutations = []
    unsealed = copy.deepcopy(outer_report)
    unsealed["decision"]["sealed"] = False
    mutations.append(unsealed)
    changed = copy.deepcopy(outer_report)
    changed["decision"]["candidate"] = "landmark_mi_110d_action_proxy_168d"
    mutations.append(changed)
    stale = copy.deepcopy(outer_report)
    stale["provenance"]["person_split_registry_sha256"] = "f" * 64
    mutations.append(stale)
    for mutated in mutations:
        failed = FinalArtifactAudit()
        c.raises(lambda m=mutated, a=failed: authorize_final_artifact(
            m,
            protected_report_sha256=canonical_json_sha256(m),
            pinned_protected_report_sha256=outer_sha,
            gate=gate,
            state=state,
            audit=a,
            expected_bootstrap_repeats=16,
        ), ValueError, "unsealed, changed, or self-consistent-forged outer report fails")
        c.eq(failed.as_dict(), FinalArtifactAudit(
            protected_report_attempts=1
        ).as_dict(), "outer commitment failure precedes all final fits")
    c.true(sealed.protected_report_sha256 == outer_sha,
           "sealed state retains the exact protected report commitment")


def test_frozen_artifact_reproduces_sklearn_and_has_no_identifiers(c: Check):
    dataset, gate, state, views, outer_report, outer_sha = _sealed_fixture()
    audit = FinalArtifactAudit()
    sealed = authorize_final_artifact(
        outer_report,
        protected_report_sha256=outer_sha,
        pinned_protected_report_sha256=outer_sha,
        gate=gate,
        state=state,
        audit=audit,
        expected_bootstrap_repeats=16,
    )
    result = freeze_final_artifact(
        dataset,
        gate,
        views,
        state=state,
        sealed_outer=sealed,
        audit=audit,
    )
    validate_frozen_artifact(result.artifact, gate=gate, state=state,
                             sealed_outer=sealed)
    c.eq((audit.scaler_fits, audit.model_fits), (1, 1),
         "final artifact performs exactly one scaler and one model fit")
    index = int(gate.protected_indices[0])
    expected = 0.5 * (
        result.model.predict_proba(
            result.scaler.transform(views.original[[index]])
        )[0, 1]
        + result.model.predict_proba(
            result.scaler.transform(views.mirrored[[index]])
        )[0, 1]
    )
    observed = predict_from_frozen_artifact(
        result.artifact, views.original[index], views.mirrored[index]
    )
    c.true(abs(observed - expected) <= 1e-12,
           "serialized parameters reproduce sklearn symmetric inference")
    encoded = json.dumps(result.artifact, allow_nan=False)
    c.true("rec_" not in encoded and "grp_" not in encoded,
           "artifact contains no recording or group identifiers")
    c.true("/Users/" not in encoded and "\\" not in encoded,
           "artifact contains no local path")
    actions = {action.dest for action in _parser()._actions}
    c.eq(actions, {
        "help", "palsynet_cache_root", "reviewed_identity_manifest",
        "review_ledger", "split_registry", "locked_development_report",
        "sealed_outer_report", "authorization", "output",
    }, "freezer CLI exposes provenance locations but no model tuning")


if __name__ == "__main__":
    run_all("test_freeze_110d_generalization_v1_artifact", dict(globals()))
