from __future__ import annotations

from pathlib import Path

from gaze_compare.cohort_simulate import load_cohort_config, simulate_independent_cohorts


PROJECT_ROOT = Path(__file__).parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "mock_independent_study.json"


def test_simulator_creates_two_independent_equal_sized_cohorts() -> None:
    config = load_cohort_config(CONFIG_PATH)
    config["participants_per_cohort"] = 12
    inputs = simulate_independent_cohorts(config)

    participants = inputs["participants"]
    assert participants["participant_id"].nunique() == 24
    assert participants.groupby("device")["participant_id"].nunique().to_dict() == {
        "professional": 12,
        "webcam": 12,
    }
    assert participants.groupby("participant_id")["device"].nunique().max() == 1
    assert not set(
        participants.loc[participants["device"].eq("webcam"), "participant_id"]
    ).intersection(
        participants.loc[participants["device"].eq("professional"), "participant_id"]
    )
    assert inputs["fixations"]["participant_id"].nunique() == 24


def test_simulator_is_deterministic() -> None:
    config = load_cohort_config(CONFIG_PATH)
    config["participants_per_cohort"] = 8

    first = simulate_independent_cohorts(config)
    second = simulate_independent_cohorts(config)

    assert first["participants"].equals(second["participants"])
    assert first["fixations"].equals(second["fixations"])
