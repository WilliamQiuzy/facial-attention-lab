from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from gaze_compare.analysis import run_analysis
from gaze_compare.report import write_analysis_outputs
from gaze_compare.simulate import load_mock_config, simulate_study


PROJECT_ROOT = Path(__file__).parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "mock_study.json"

EXPECTED_TABLES = {
    "analysis_gates.csv",
    "aoi_agreement.csv",
    "aoi_dwell.csv",
    "data_quality.csv",
    "equivalence_summary.csv",
    "map_agreement.csv",
    "temporal_alignment.csv",
    "participant_endpoints.csv",
}
EXPECTED_FIGURES = {
    "01_data_quality.png",
    "02_accuracy_bland_altman.png",
    "03_map_similarity.png",
    "04_matched_heatmaps.png",
    "05_aoi_dwell.png",
    "06_equivalence_intervals.png",
    "07_participant_qc_matrix.png",
    "08_temporal_alignment.png",
}


@pytest.fixture(scope="module")
def written_outputs(tmp_path_factory: pytest.TempPathFactory) -> Path:
    config = load_mock_config(CONFIG_PATH)
    inputs = simulate_study(config)
    result = run_analysis(
        inputs["samples"],
        inputs["aois"],
        grid_size=config["map_grid_size"],
        margins=config["illustrative_margins"],
        mock_mode=True,
    )
    output_dir = tmp_path_factory.mktemp("analysis_outputs")
    write_analysis_outputs(result, output_dir, config=config)
    return output_dir


def test_output_manifest_is_exact_and_files_are_nonempty(written_outputs: Path) -> None:
    assert {path.name for path in (written_outputs / "tables").iterdir()} == EXPECTED_TABLES
    assert {path.name for path in (written_outputs / "figures").iterdir()} == EXPECTED_FIGURES

    manifest = json.loads((written_outputs / "analysis_manifest.json").read_text())
    assert manifest["synthetic"] is True
    assert manifest["analysis_mode"] == "paired_mock_demo"
    assert len(manifest["artifacts"]) == len(EXPECTED_TABLES | EXPECTED_FIGURES) + 1
    assert all(len(item["sha256"]) == 64 for item in manifest["artifacts"])
    assert all((written_outputs / item["path"]).stat().st_size > 0 for item in manifest["artifacts"])


def test_figure_files_are_rendered_not_placeholders(written_outputs: Path) -> None:
    for filename in EXPECTED_FIGURES:
        path = written_outputs / "figures" / filename
        assert path.stat().st_size > 20_000
        assert path.read_bytes().startswith(b"\x89PNG")


def test_tables_keep_interpretation_fields(written_outputs: Path) -> None:
    equivalence = pd.read_csv(written_outputs / "tables" / "equivalence_summary.csv")
    quality = pd.read_csv(written_outputs / "tables" / "data_quality.csv")

    assert {"ci90_lower", "ci90_upper", "margin", "margin_source", "outcome"}.issubset(
        equivalence.columns
    )
    assert set(equivalence["margin_source"]) == {"illustrative_mock_only"}
    assert set(quality["reference_role"]) == {"candidate", "reference_instrument"}


def test_report_has_mandatory_evidence_boundaries(written_outputs: Path) -> None:
    report = (written_outputs / "mock_analysis_report.md").read_text()
    required_phrases = [
        "100% synthetic demonstration",
        "not observed Mayo",
        "not observed Prolific",
        "not an iMotions result",
        "reference instrument, not error-free ground truth",
        "illustrative margins",
        "research-only and nonclinical",
        "cannot establish interchangeability",
    ]
    for phrase in required_phrases:
        assert phrase.lower() in report.lower()
