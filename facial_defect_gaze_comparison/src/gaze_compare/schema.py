from __future__ import annotations

import pandas as pd


SAMPLE_REQUIRED_COLUMNS = (
    "participant_id",
    "comparison_unit_id",
    "synthetic",
    "scenario",
    "recruitment_source",
    "cohort_site",
    "session_id",
    "trial_id",
    "stimulus_id",
    "stimulus_version",
    "task_type",
    "device",
    "tracker_make_model",
    "acquisition_software",
    "acquisition_version",
    "display_width_px",
    "display_height_px",
    "display_width_mm",
    "display_height_mm",
    "viewport_width_px",
    "viewport_height_px",
    "viewing_distance_mm",
    "coordinate_transform_version",
    "timebase",
    "sample_index",
    "timestamp_ms",
    "gaze_x_norm",
    "gaze_y_norm",
    "gaze_x_deg",
    "gaze_y_deg",
    "target_x_norm",
    "target_y_norm",
    "target_x_deg",
    "target_y_deg",
    "valid",
    "blink",
    "validity_reason",
    "fixation_id",
    "fixation_duration_ms",
)

AOI_REQUIRED_COLUMNS = (
    "stimulus_id",
    "stimulus_version",
    "aoi_version",
    "aoi_name",
    "x_min",
    "x_max",
    "y_min",
    "y_max",
)

VALID_DEVICES = frozenset({"webcam", "professional"})
VALID_TASKS = frozenset({"calibration_grid", "free_view_face"})


class GazeSchemaError(ValueError):
    """Raised when an input table does not satisfy the study contract."""


def validate_samples(
    samples: pd.DataFrame,
    *,
    require_paired: bool = True,
) -> pd.DataFrame:
    if not isinstance(samples, pd.DataFrame) or samples.empty:
        raise GazeSchemaError("samples must be a non-empty pandas DataFrame")

    missing = [column for column in SAMPLE_REQUIRED_COLUMNS if column not in samples]
    if missing:
        raise GazeSchemaError(f"missing required sample columns: {', '.join(missing)}")

    validated = samples.copy(deep=True).reset_index(drop=True)
    duplicate_key = ["comparison_unit_id", "trial_id", "device", "sample_index"]
    if validated.duplicated(duplicate_key).any():
        raise GazeSchemaError(f"duplicate sample keys found for {duplicate_key}")

    invalid_devices = sorted(set(validated["device"].dropna()) - VALID_DEVICES)
    if invalid_devices:
        raise GazeSchemaError(f"invalid device values: {invalid_devices}")

    invalid_tasks = sorted(set(validated["task_type"].dropna()) - VALID_TASKS)
    if invalid_tasks:
        raise GazeSchemaError(f"invalid task_type values: {invalid_tasks}")

    for column in ("display_width_px", "display_height_px", "viewport_width_px", "viewport_height_px"):
        numeric = pd.to_numeric(validated[column], errors="coerce")
        if numeric.isna().any() or (numeric <= 0).any() or (numeric % 1 != 0).any():
            raise GazeSchemaError(f"{column} must contain positive integers")

    for column in ("display_width_mm", "display_height_mm", "viewing_distance_mm"):
        numeric = pd.to_numeric(validated[column], errors="coerce")
        if numeric.isna().any() or (numeric <= 0).any():
            raise GazeSchemaError(f"{column} must contain positive measurements")

    for column in ("sample_index", "timestamp_ms"):
        numeric = pd.to_numeric(validated[column], errors="coerce")
        if numeric.isna().any() or (numeric < 0).any():
            raise GazeSchemaError(f"{column} must be numeric and non-negative")

    for column in ("gaze_x_norm", "gaze_y_norm", "target_x_norm", "target_y_norm"):
        numeric = pd.to_numeric(validated[column], errors="coerce")
        present = numeric.notna()
        if ((numeric[present] < 0) | (numeric[present] > 1)).any():
            raise GazeSchemaError(f"{column} values must be within [0, 1]")

    if not validated["valid"].isin([True, False]).all():
        raise GazeSchemaError("valid must be boolean")
    if not validated["blink"].isin([True, False]).all():
        raise GazeSchemaError("blink must be boolean")

    gaze_columns = ["gaze_x_norm", "gaze_y_norm", "gaze_x_deg", "gaze_y_deg"]
    invalid_rows = ~validated["valid"].astype(bool)
    if validated.loc[invalid_rows, gaze_columns].notna().any(axis=None):
        raise GazeSchemaError("invalid samples must encode gaze coordinates as missing")
    if validated.loc[~invalid_rows, gaze_columns].isna().any(axis=None):
        raise GazeSchemaError("valid samples must include all gaze coordinates")

    calibration_rows = validated["task_type"].eq("calibration_grid")
    target_columns = ["target_x_norm", "target_y_norm", "target_x_deg", "target_y_deg"]
    if validated.loc[calibration_rows, target_columns].isna().any(axis=None):
        raise GazeSchemaError("calibration_grid samples require known target coordinates")

    metadata_columns = [
        "participant_id",
        "comparison_unit_id",
        "scenario",
        "recruitment_source",
        "cohort_site",
        "session_id",
        "stimulus_id",
        "stimulus_version",
        "tracker_make_model",
        "acquisition_software",
        "acquisition_version",
        "coordinate_transform_version",
        "timebase",
        "validity_reason",
    ]
    for column in metadata_columns:
        if validated[column].isna().any() or validated[column].astype(str).str.strip().eq("").any():
            raise GazeSchemaError(f"{column} must be populated")

    if require_paired:
        paired_keys = ["comparison_unit_id", "stimulus_id", "stimulus_version", "task_type"]
        device_sets = validated.groupby(paired_keys, dropna=False)["device"].agg(set)
        incomplete = device_sets[device_sets != VALID_DEVICES]
        if not incomplete.empty:
            raise GazeSchemaError(
                "paired device data are required for every comparison unit, stimulus, and task"
            )

    return validated


def validate_aoi_table(aois: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(aois, pd.DataFrame) or aois.empty:
        raise GazeSchemaError("AOI table must be a non-empty pandas DataFrame")
    missing = [column for column in AOI_REQUIRED_COLUMNS if column not in aois]
    if missing:
        raise GazeSchemaError(f"missing required AOI columns: {', '.join(missing)}")

    validated = aois.copy(deep=True).reset_index(drop=True)
    key = ["stimulus_id", "stimulus_version", "aoi_version", "aoi_name"]
    if validated.duplicated(key).any():
        raise GazeSchemaError(f"duplicate AOI definitions found for {key}")

    bounds = validated[["x_min", "x_max", "y_min", "y_max"]].apply(
        pd.to_numeric,
        errors="coerce",
    )
    invalid = (
        bounds.isna().any(axis=1)
        | bounds.lt(0).any(axis=1)
        | bounds.gt(1).any(axis=1)
        | bounds["x_min"].ge(bounds["x_max"])
        | bounds["y_min"].ge(bounds["y_max"])
    )
    if invalid.any():
        rows = validated.index[invalid].tolist()
        raise GazeSchemaError(f"AOI bounds must be ordered within [0, 1]; invalid rows: {rows}")

    return validated
