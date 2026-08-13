from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from gaze_compare.analysis import AnalysisGateError, run_analysis
from gaze_compare.simulate import load_mock_config, simulate_study


PROJECT_ROOT = Path(__file__).parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "mock_study.json"


@pytest.fixture(scope="module")
def mock_inputs() -> tuple[dict[str, object], dict[str, pd.DataFrame]]:
    config = load_mock_config(CONFIG_PATH)
    return config, simulate_study(config)


def test_analysis_returns_documented_tables_and_maps(mock_inputs) -> None:
    config, tables = mock_inputs

    result = run_analysis(
        tables["samples"],
        tables["aois"],
        grid_size=config["map_grid_size"],
        margins=config["illustrative_margins"],
        mock_mode=True,
    )

    assert set(result.tables) == {
        "data_quality",
        "map_agreement",
        "temporal_alignment",
        "aoi_dwell",
        "aoi_agreement",
        "participant_endpoints",
        "equivalence_summary",
        "analysis_gates",
    }
    assert not result.tables["data_quality"].empty
    assert not result.tables["map_agreement"].empty
    assert not result.tables["equivalence_summary"].empty
    assert result.maps
    assert all(key.count("|") == 2 for key in result.maps)


def test_repeated_stimuli_are_aggregated_to_one_participant_endpoint(mock_inputs) -> None:
    config, tables = mock_inputs
    result = run_analysis(
        tables["samples"],
        tables["aois"],
        grid_size=config["map_grid_size"],
        margins=config["illustrative_margins"],
        mock_mode=True,
    )
    endpoints = result.tables["participant_endpoints"]

    assert endpoints["comparison_unit_id"].is_unique
    assert len(endpoints) == tables["samples"]["comparison_unit_id"].nunique()
    assert {
        "accuracy_difference_deg",
        "data_loss_difference",
        "map_disagreement",
        "absolute_lag_ms",
    }.issubset(
        endpoints.columns
    )


def test_map_outputs_include_declared_center_bias_baseline(mock_inputs) -> None:
    config, tables = mock_inputs
    result = run_analysis(
        tables["samples"],
        tables["aois"],
        grid_size=config["map_grid_size"],
        margins=config["illustrative_margins"],
        mock_mode=True,
    )
    agreement = result.tables["map_agreement"]

    assert {"center_bias_webcam_cc", "center_bias_professional_cc"}.issubset(agreement.columns)
    assert agreement[["map_correlation", "histogram_intersection"]].notna().all(axis=None)
    scenario_sim = agreement.groupby("scenario")["histogram_intersection"].mean()
    assert scenario_sim["near_equivalent"] > scenario_sim["systematic_bias"]


def test_temporal_alignment_detects_the_delayed_scenario(mock_inputs) -> None:
    config, tables = mock_inputs
    result = run_analysis(
        tables["samples"],
        tables["aois"],
        grid_size=config["map_grid_size"],
        margins=config["illustrative_margins"],
        mock_mode=True,
    )
    timing = result.tables["temporal_alignment"]
    scenario_lag = timing.groupby("scenario")["estimated_lag_ms"].median().abs()

    assert scenario_lag["temporal_lag"] > 150
    assert scenario_lag["near_equivalent"] < 70


def test_mock_equivalence_is_labeled_illustrative_and_three_way(mock_inputs) -> None:
    config, tables = mock_inputs
    result = run_analysis(
        tables["samples"],
        tables["aois"],
        grid_size=config["map_grid_size"],
        margins=config["illustrative_margins"],
        mock_mode=True,
    )
    summary = result.tables["equivalence_summary"]

    assert set(summary["margin_source"]) == {"illustrative_mock_only"}
    assert set(summary["outcome"]).issubset({"equivalent", "not_equivalent", "inconclusive"})
    assert "interchangeable" not in " ".join(summary.astype(str).stack()).lower()


def test_real_mode_fails_closed_without_registered_margins(mock_inputs) -> None:
    _, tables = mock_inputs

    with pytest.raises(AnalysisGateError, match="preregistered margins"):
        run_analysis(tables["samples"], tables["aois"], mock_mode=False, margins=None)


def test_unpaired_mode_disables_paired_outputs(mock_inputs) -> None:
    config, tables = mock_inputs
    samples = tables["samples"]
    samples = samples[
        ~(
            samples["comparison_unit_id"].eq("SYN-P001")
            & samples["device"].eq("professional")
        )
    ]

    result = run_analysis(
        samples,
        tables["aois"],
        require_paired=False,
        grid_size=config["map_grid_size"],
        margins=config["illustrative_margins"],
        mock_mode=True,
    )

    assert not result.tables["data_quality"].empty
    for table_name in (
        "map_agreement",
        "temporal_alignment",
        "aoi_agreement",
        "participant_endpoints",
        "equivalence_summary",
    ):
        assert result.tables[table_name].empty
    gates = result.tables["analysis_gates"].set_index("analysis")
    assert gates.loc["paired_agreement", "status"] == "disabled"


def test_vendor_events_and_reliability_are_not_fabricated(mock_inputs) -> None:
    config, tables = mock_inputs
    result = run_analysis(
        tables["samples"],
        tables["aois"],
        grid_size=config["map_grid_size"],
        margins=config["illustrative_margins"],
        mock_mode=True,
    )

    all_columns = set().union(*(set(table.columns) for table in result.tables.values()))
    assert not {"fixation_count", "reliability_ceiling", "time_to_first_fixation"}.intersection(
        all_columns
    )
