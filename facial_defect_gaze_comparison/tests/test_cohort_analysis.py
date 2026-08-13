from __future__ import annotations

from pathlib import Path

from gaze_compare.cohort_analysis import run_independent_cohort_analysis
from gaze_compare.cohort_simulate import load_cohort_config, simulate_independent_cohorts


PROJECT_ROOT = Path(__file__).parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "mock_independent_study.json"


def test_independent_analysis_answers_the_four_primary_questions() -> None:
    config = load_cohort_config(CONFIG_PATH)
    config["participants_per_cohort"] = 40
    config["bootstrap_replicates"] = 120
    config["classifier_bootstrap_replicates"] = 120
    inputs = simulate_independent_cohorts(config)

    result = run_independent_cohort_analysis(
        inputs["participants"],
        inputs["fixations"],
        inputs["aois"],
        config=config,
    )

    assert set(result.tables) == {
        "protocol_gates",
        "cohort_characteristics",
        "covariate_balance",
        "participant_quality",
        "quality_comparison",
        "participant_aoi_profiles",
        "aoi_summary",
        "map_reliability",
        "domain_classifier",
        "decision_summary",
    }
    assert len(result.tables["quality_comparison"]) == 4
    assert set(result.tables["domain_classifier"]["feature_set"]) == {
        "attention_pattern",
        "technical_quality",
    }
    assert result.tables["map_reliability"]["stimulus_id"].nunique() == 3
    assert result.metadata["paired"] is False
    assert result.metadata["n_webcam"] == 40
    assert result.metadata["n_professional"] == 40


def test_map_analysis_compares_group_samples_not_people() -> None:
    config = load_cohort_config(CONFIG_PATH)
    config["participants_per_cohort"] = 30
    config["bootstrap_replicates"] = 100
    config["classifier_bootstrap_replicates"] = 100
    inputs = simulate_independent_cohorts(config)

    result = run_independent_cohort_analysis(
        inputs["participants"],
        inputs["fixations"],
        inputs["aois"],
        config=config,
    )

    reliability = result.tables["map_reliability"]
    assert reliability["comparison_unit"].eq("random_half_cohort").all()
    assert reliability["cross_domain_similarity"].between(0, 1).all()
    assert reliability["within_webcam_similarity"].between(0, 1).all()
    assert reliability["within_professional_similarity"].between(0, 1).all()
