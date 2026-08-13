from __future__ import annotations

import json
from pathlib import Path

from gaze_compare.cli import main


PROJECT_ROOT = Path(__file__).parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "mock_study.json"
COHORT_CONFIG_PATH = PROJECT_ROOT / "config" / "mock_independent_study.json"


def test_cli_simulate_and_analyze_end_to_end(tmp_path: Path, capsys) -> None:
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "outputs"

    assert main(["simulate", "--config", str(CONFIG_PATH), "--output", str(input_dir)]) == 0
    assert (input_dir / "mock_manifest.json").exists()
    assert "Synthetic inputs written" in capsys.readouterr().out

    assert (
        main(
            [
                "analyze",
                "--samples",
                str(input_dir / "gaze_samples.csv"),
                "--aois",
                str(input_dir / "aoi_definitions.csv"),
                "--config",
                str(CONFIG_PATH),
                "--output",
                str(output_dir),
                "--mode",
                "mock-paired",
            ]
        )
        == 0
    )
    assert (output_dir / "analysis_manifest.json").exists()
    assert (output_dir / "mock_analysis_report.md").exists()
    assert "Analysis written" in capsys.readouterr().out


def test_cli_rejects_real_mode_with_synthetic_config(tmp_path: Path) -> None:
    input_dir = tmp_path / "inputs"
    main(["simulate", "--config", str(CONFIG_PATH), "--output", str(input_dir)])

    try:
        main(
            [
                "analyze",
                "--samples",
                str(input_dir / "gaze_samples.csv"),
                "--aois",
                str(input_dir / "aoi_definitions.csv"),
                "--config",
                str(CONFIG_PATH),
                "--output",
                str(tmp_path / "real"),
                "--mode",
                "real-paired",
            ]
        )
    except ValueError as error:
        assert "synthetic config" in str(error)
    else:
        raise AssertionError("real mode must reject the synthetic mock configuration")


def test_independent_cohort_cli_end_to_end(tmp_path: Path, capsys) -> None:
    config = json.loads(COHORT_CONFIG_PATH.read_text())
    config["participants_per_cohort"] = 24
    config["bootstrap_replicates"] = 100
    config["classifier_bootstrap_replicates"] = 100
    config_path = tmp_path / "cohort_config.json"
    config_path.write_text(json.dumps(config))
    input_dir = tmp_path / "cohort_inputs"
    output_dir = tmp_path / "cohort_outputs"

    assert main(
        ["cohort-simulate", "--config", str(config_path), "--output", str(input_dir)]
    ) == 0
    assert main(
        [
            "cohort-analyze",
            "--participants",
            str(input_dir / "participant_summary.csv"),
            "--fixations",
            str(input_dir / "fixation_events.csv"),
            "--aois",
            str(input_dir / "aoi_definitions.csv"),
            "--config",
            str(config_path),
            "--output",
            str(output_dir),
        ]
    ) == 0
    assert (output_dir / "independent_cohort_report.md").exists()
    assert "Independent-cohort analysis written" in capsys.readouterr().out
