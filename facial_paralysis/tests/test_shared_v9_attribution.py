from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _testlib import Check, run_all  # noqa: E402
from test_shared_v9_research_release import (  # noqa: E402
    _models,
    _provenance,
    _request,
)
from src.deployment.shared_v9_attribution import (  # noqa: E402
    ATTRIBUTION_BASELINE,
    ATTRIBUTION_SCHEMA,
    encode_attribution_request_npz,
    explain_prediction,
    load_attribution_request_npz,
)
from src.deployment.shared_v9_research_release import (  # noqa: E402
    load_release,
    write_release,
)


def _predictor(temporary: Path):
    root = temporary / "release"
    write_release(
        root,
        models=_models(),
        scaler_mean=np.zeros(110, dtype=np.float64),
        scaler_scale=np.ones(110, dtype=np.float64),
        provenance=_provenance(),
    )
    return load_release(root, device="cpu")


def _neutral(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array(arrays["clinical_original"], copy=True),
        np.array(arrays["clinical_mirrored"], copy=True),
    )


def test_action_token_integrated_gradients_are_faithful_and_deterministic(c: Check):
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        predictor = _predictor(Path(directory))
        arrays = _request("cue_aligned_action")
        original_copy = {name: np.array(value, copy=True) for name, value in arrays.items()}
        neutral_original, neutral_mirrored = _neutral(arrays)
        first = explain_prediction(
            predictor,
            "cue_aligned_action",
            arrays,
            neutral_original,
            neutral_mirrored,
        )
        second = explain_prediction(
            predictor,
            "cue_aligned_action",
            arrays,
            neutral_original,
            neutral_mirrored,
        )
        c.eq(first, second)
        c.eq(first.prediction, predictor.predict("cue_aligned_action", arrays))
        c.eq(first.attribution.schema_version, ATTRIBUTION_SCHEMA)
        c.eq(first.attribution.baseline, ATTRIBUTION_BASELINE)
        c.eq(first.attribution.method, "integrated_gradients_shared_action_tokens")
        c.eq(first.attribution.integration_steps, 32)
        c.eq(len(first.attribution.actions), 7)
        c.true(first.attribution.max_completeness_error <= 0.02)
        c.eq(
            tuple(row.action_code for row in first.attribution.actions),
            tuple(int(value) for value in arrays["action_codes"]),
        )
        c.true(all(row.mirror_consistent for row in first.attribution.actions))
        c.true(all(0.0 <= row.relative_magnitude <= 1.0 for row in first.attribution.actions))
        c.true(all(row.ensemble_sign_agreement in {0, 1, 2, 3} for row in first.attribution.actions))
        c.true(all(row.temporal_checks_passed in {0, 1, 2} for row in first.attribution.actions))
        for name, expected in original_copy.items():
            c.true(np.array_equal(arrays[name], expected), f"attribution mutated {name}")


def test_attribution_archive_is_exact_and_baseline_is_fail_closed(c: Check):
    arrays = _request("cue_aligned_action", actions_override=6)
    neutral_original, neutral_mirrored = _neutral(arrays)
    payload = encode_attribution_request_npz(
        "cue_aligned_action",
        arrays,
        neutral_original,
        neutral_mirrored,
    )
    restored, first, second = load_attribution_request_npz(
        payload,
        protocol="cue_aligned_action",
    )
    c.eq(set(restored), set(arrays))
    c.true(np.array_equal(first, neutral_original))
    c.true(np.array_equal(second, neutral_mirrored))

    wrong = np.zeros((5, 110), dtype=np.float32)
    c.raises(
        lambda: encode_attribution_request_npz(
            "cue_aligned_action", arrays, wrong, neutral_mirrored
        ),
        ValueError,
    )
    nonfinite = np.array(neutral_original, copy=True)
    nonfinite[0, 0] = np.nan
    c.raises(
        lambda: encode_attribution_request_npz(
            "cue_aligned_action", arrays, nonfinite, neutral_mirrored
        ),
        ValueError,
    )
    c.raises(
        lambda: load_attribution_request_npz(
            payload[:-10], protocol="cue_aligned_action"
        ),
        ValueError,
    )


def test_zero_response_does_not_fabricate_stable_influence(c: Check):
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        predictor = _predictor(Path(directory))
        arrays = _request("cue_aligned_action", actions_override=6)
        arrays["dense_original"].fill(0.0)
        arrays["dense_mirrored"].fill(0.0)
        neutral_original, neutral_mirrored = _neutral(arrays)
        explained = explain_prediction(
            predictor,
            "cue_aligned_action",
            arrays,
            neutral_original,
            neutral_mirrored,
        )
        c.true(all(not row.stable for row in explained.attribution.actions))
        c.true(all(row.direction == "not_reported" for row in explained.attribution.actions))
        c.true(all(row.strength == "not_reported" for row in explained.attribution.actions))


if __name__ == "__main__":
    run_all("test_shared_v9_attribution", dict(globals()))
