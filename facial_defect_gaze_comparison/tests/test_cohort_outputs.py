from __future__ import annotations

import json
from pathlib import Path

from gaze_compare.cohort_analysis import run_independent_cohort_analysis
from gaze_compare.cohort_report import write_independent_outputs
from gaze_compare.cohort_simulate import load_cohort_config, simulate_independent_cohorts


PROJECT_ROOT = Path(__file__).parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "mock_independent_study.json"


def test_independent_report_and_visualizations_are_complete(tmp_path: Path) -> None:
    config = load_cohort_config(CONFIG_PATH)
    config["participants_per_cohort"] = 32
    config["bootstrap_replicates"] = 100
    config["classifier_bootstrap_replicates"] = 100
    inputs = simulate_independent_cohorts(config)
    result = run_independent_cohort_analysis(
        inputs["participants"], inputs["fixations"], inputs["aois"], config=config
    )

    manifest_path = write_independent_outputs(result, tmp_path, config=config)
    manifest = json.loads(manifest_path.read_text())

    expected_figures = {
        "01_covariate_balance.png",
        "02_quality_equivalence.png",
        "03_quality_distributions.png",
        "04_group_attention_maps.png",
        "05_map_reproducibility.png",
        "06_aoi_profile.png",
        "07_domain_classifier.png",
    }
    assert {path.name for path in (tmp_path / "figures").iterdir()} == expected_figures
    assert len(list((tmp_path / "tables").glob("*.csv"))) == 10
    assert manifest["analysis_mode"] == "independent_cohort_mock_demo"

    report = (tmp_path / "independent_cohort_report.md").read_text()
    for phrase in [
        "100% synthetic demonstration",
        "different participants",
        "cannot isolate a pure device effect",
        "same domain is not one yes/no property",
        "not observed Mayo",
        "not observed Prolific",
        "illustrative margins",
    ]:
        assert phrase.lower() in report.lower()
