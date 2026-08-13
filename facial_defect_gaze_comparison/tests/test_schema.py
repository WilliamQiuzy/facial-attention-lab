from __future__ import annotations

import pandas as pd
import pytest

from gaze_compare.schema import GazeSchemaError, validate_aoi_table, validate_samples


def valid_samples() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for participant_id in ("SYN-P001", "SYN-P002"):
        for device in ("webcam", "professional"):
            rows.append(
                {
                    "participant_id": participant_id,
                    "comparison_unit_id": participant_id,
                    "synthetic": True,
                    "scenario": "synthetic_test",
                    "recruitment_source": "synthetic_demo",
                    "cohort_site": "synthetic_site",
                    "session_id": f"{participant_id}-session",
                    "trial_id": f"{participant_id}-calibration-{device}",
                    "stimulus_id": "calibration-grid",
                    "stimulus_version": "v1",
                    "task_type": "calibration_grid",
                    "device": device,
                    "tracker_make_model": (
                        "Synthetic browser camera"
                        if device == "webcam"
                        else "Synthetic reference tracker"
                    ),
                    "acquisition_software": "synthetic_generator",
                    "acquisition_version": "1.0",
                    "display_width_px": 1920,
                    "display_height_px": 1080,
                    "display_width_mm": 531.0,
                    "display_height_mm": 299.0,
                    "viewport_width_px": 1536,
                    "viewport_height_px": 864,
                    "viewing_distance_mm": 650.0,
                    "coordinate_transform_version": "normalized-screen-v1",
                    "timebase": "trial_relative_ms",
                    "sample_index": 0,
                    "timestamp_ms": 0.0,
                    "gaze_x_norm": 0.5,
                    "gaze_y_norm": 0.5,
                    "gaze_x_deg": 0.1,
                    "gaze_y_deg": -0.1,
                    "target_x_norm": 0.5,
                    "target_y_norm": 0.5,
                    "target_x_deg": 0.0,
                    "target_y_deg": 0.0,
                    "valid": True,
                    "blink": False,
                    "validity_reason": "valid",
                    "fixation_id": "cal-1",
                    "fixation_duration_ms": 250.0,
                }
            )
    return pd.DataFrame(rows)


def test_samples_reject_missing_required_column() -> None:
    samples = valid_samples().drop(columns=["comparison_unit_id"])

    with pytest.raises(GazeSchemaError, match="comparison_unit_id"):
        validate_samples(samples)


@pytest.mark.parametrize(
    "column",
    [
        "recruitment_source",
        "scenario",
        "cohort_site",
        "tracker_make_model",
        "acquisition_software",
        "acquisition_version",
        "display_width_px",
        "display_width_mm",
        "viewing_distance_mm",
        "coordinate_transform_version",
        "timebase",
        "stimulus_version",
    ],
)
def test_samples_require_acquisition_and_geometry_metadata(column: str) -> None:
    with pytest.raises(GazeSchemaError, match=column):
        validate_samples(valid_samples().drop(columns=[column]))


def test_samples_reject_duplicate_device_sample_key() -> None:
    samples = pd.concat([valid_samples(), valid_samples().iloc[[0]]], ignore_index=True)

    with pytest.raises(GazeSchemaError, match="duplicate sample keys"):
        validate_samples(samples)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("device", "phone", "device"),
        ("task_type", "survey", "task_type"),
        ("gaze_x_norm", 1.2, "gaze_x_norm"),
        ("timestamp_ms", -1.0, "timestamp_ms"),
    ],
)
def test_samples_reject_invalid_values(
    column: str,
    value: object,
    message: str,
) -> None:
    samples = valid_samples()
    samples.loc[0, column] = value

    with pytest.raises(GazeSchemaError, match=message):
        validate_samples(samples)


def test_samples_reject_unpaired_comparison_units_by_default() -> None:
    samples = valid_samples()
    samples = samples[
        ~(
            (samples["participant_id"] == "SYN-P002")
            & (samples["device"] == "professional")
        )
    ]

    with pytest.raises(GazeSchemaError, match="paired device data"):
        validate_samples(samples)

    validated = validate_samples(samples, require_paired=False)
    assert len(validated) == 3


def test_invalid_samples_must_encode_coordinates_as_missing() -> None:
    samples = valid_samples()
    samples.loc[0, "valid"] = False

    with pytest.raises(GazeSchemaError, match="invalid samples.*missing"):
        validate_samples(samples)


def test_prolific_and_imotions_are_not_accepted_as_devices() -> None:
    for invalid_device in ("prolific", "imotions"):
        samples = valid_samples()
        samples.loc[0, "device"] = invalid_device

        with pytest.raises(GazeSchemaError, match="device"):
            validate_samples(samples)


def test_validated_rows_preserve_recruitment_and_platform_fields() -> None:
    samples = valid_samples()
    samples.loc[:, "recruitment_source"] = "prolific"
    samples.loc[:, "acquisition_software"] = "iMotions"

    validated = validate_samples(samples)

    assert set(validated["device"]) == {"webcam", "professional"}
    assert set(validated["recruitment_source"]) == {"prolific"}
    assert set(validated["acquisition_software"]) == {"iMotions"}


def test_aoi_rectangles_must_be_unique_and_bounded() -> None:
    aois = pd.DataFrame(
        [
            {
                "stimulus_id": "face-01",
                "stimulus_version": "v1",
                "aoi_version": "v1",
                "aoi_name": "left_eye",
                "x_min": 0.1,
                "x_max": 0.4,
                "y_min": 0.2,
                "y_max": 0.4,
            },
            {
                "stimulus_id": "face-01",
                "stimulus_version": "v1",
                "aoi_version": "v1",
                "aoi_name": "mouth",
                "x_min": 0.3,
                "x_max": 1.2,
                "y_min": 0.6,
                "y_max": 0.8,
            },
        ]
    )

    with pytest.raises(GazeSchemaError, match="AOI bounds"):
        validate_aoi_table(aois)
