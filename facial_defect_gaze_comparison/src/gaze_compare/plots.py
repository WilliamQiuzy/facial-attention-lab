from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Patch
import numpy as np
import pandas as pd
import seaborn as sns

from .analysis import AnalysisResult
from .metrics import bland_altman


MAYO_BLUE = "#0057B8"
DEEP_BLUE = "#003B70"
SKY_BLUE = "#6CB4EE"
PALE_BLUE = "#EAF3FA"
REFERENCE_NAVY = "#243746"
CORAL = "#D9485F"
SCENARIO_ORDER = ["near_equivalent", "systematic_bias", "temporal_lag", "high_dropout"]
DEVICE_PALETTE = {"webcam": MAYO_BLUE, "professional": REFERENCE_NAVY}


def _set_style() -> None:
    # Seaborn stripplot jitter uses NumPy's legacy global RNG. Reset it here so
    # independently regenerated PNG artifacts have stable SHA-256 hashes.
    np.random.seed(20260813)
    sns.set_theme(
        style="whitegrid",
        context="notebook",
        rc={
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#B8C7D1",
            "grid.color": "#DFE8EE",
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
        },
    )


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": "gaze_compare synthetic demonstration"},
    )
    plt.close(fig)


def _friendly(value: str) -> str:
    return value.replace("_", " ").title()


def plot_data_quality(quality: pd.DataFrame, path: Path) -> None:
    metrics = [
        ("accuracy_deg", "Calibration accuracy error (°)"),
        ("rms_precision_deg", "RMS precision (°)"),
        ("data_loss", "Data loss (proportion)"),
        ("effective_sampling_rate_hz", "Effective sampling rate (Hz)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2))
    for axis, (column, label) in zip(axes.flat, metrics, strict=True):
        sns.boxplot(
            data=quality,
            x="scenario",
            y=column,
            hue="device",
            order=SCENARIO_ORDER,
            palette=DEVICE_PALETTE,
            width=0.7,
            fliersize=0,
            ax=axis,
        )
        sns.stripplot(
            data=quality,
            x="scenario",
            y=column,
            hue="device",
            order=SCENARIO_ORDER,
            palette=DEVICE_PALETTE,
            dodge=True,
            size=3.5,
            alpha=0.7,
            legend=False,
            ax=axis,
        )
        axis.set_xlabel("")
        axis.set_ylabel(label)
        axis.set_xticks(
            axis.get_xticks(),
            [_friendly(value) for value in SCENARIO_ORDER],
            rotation=18,
            ha="right",
        )
        if axis.get_legend() is not None:
            axis.get_legend().remove()
    fig.legend(
        handles=[
            Patch(facecolor=REFERENCE_NAVY, edgecolor="#2A3135", label="Professional reference"),
            Patch(facecolor=MAYO_BLUE, edgecolor="#2A3135", label="Webcam"),
        ],
        title="Measurement modality",
        frameon=False,
        ncol=2,
        loc="upper right",
        bbox_to_anchor=(0.97, 0.975),
    )
    fig.suptitle("Synthetic acquisition quality by failure scenario", fontsize=17, weight="bold", x=0.04, ha="left")
    fig.text(0.04, 0.94, "Metrics remain separate; no composite ‘better device’ score is used.", color="#4C6372")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    _save(fig, path)


def plot_bland_altman(endpoints: pd.DataFrame, path: Path) -> None:
    webcam = endpoints["accuracy_deg_webcam"].to_numpy()
    reference = endpoints["accuracy_deg_professional"].to_numpy()
    mean_values = (webcam + reference) / 2
    differences = webcam - reference
    summary = bland_altman(webcam, reference)
    palette = dict(zip(SCENARIO_ORDER, sns.color_palette("Blues", n_colors=5)[1:], strict=True))

    fig, axis = plt.subplots(figsize=(10, 6.3))
    for scenario in SCENARIO_ORDER:
        selected = endpoints["scenario"].eq(scenario)
        axis.scatter(
            mean_values[selected],
            differences[selected],
            s=58,
            color=palette[scenario],
            edgecolor="white",
            linewidth=0.8,
            label=_friendly(scenario),
            zorder=3,
        )
    axis.axhline(summary.mean_difference, color=DEEP_BLUE, linewidth=2, label="Mean difference")
    axis.axhline(summary.lower_limit, color=CORAL, linewidth=1.6, linestyle="--")
    axis.axhline(summary.upper_limit, color=CORAL, linewidth=1.6, linestyle="--", label="95% limits")
    axis.axhline(0, color="#738895", linewidth=1, linestyle=":")
    axis.set_title("Calibration accuracy agreement")
    axis.set_xlabel("Mean accuracy error across modalities (°)")
    axis.set_ylabel("Webcam − reference-instrument error (°)")
    axis.legend(frameon=False, ncol=1, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    axis.text(
        0.01,
        0.02,
        "One point per synthetic participant after aggregating repeated targets.",
        transform=axis.transAxes,
        color="#4C6372",
    )
    fig.tight_layout()
    _save(fig, path)


def plot_map_similarity(agreement: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    metrics = [
        ("histogram_intersection", "Histogram intersection (higher agreement)"),
        ("jensen_shannon_distance", "Jensen–Shannon distance (lower disagreement)"),
    ]
    for axis, (metric, label) in zip(axes, metrics, strict=True):
        sns.boxplot(
            data=agreement,
            x="scenario",
            y=metric,
            order=SCENARIO_ORDER,
            color=SKY_BLUE,
            width=0.62,
            fliersize=0,
            ax=axis,
        )
        sns.stripplot(
            data=agreement,
            x="scenario",
            y=metric,
            order=SCENARIO_ORDER,
            color=DEEP_BLUE,
            alpha=0.7,
            size=4,
            jitter=0.16,
            ax=axis,
        )
        axis.set_xlabel("")
        axis.set_ylabel(label)
        axis.set_xticks(
            axis.get_xticks(),
            [_friendly(value) for value in SCENARIO_ORDER],
            rotation=18,
            ha="right",
        )
    fig.suptitle("Paired free-viewing map agreement", fontsize=17, weight="bold", x=0.04, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, path)


def _face_outline(axis: plt.Axes, *, color: str = "white", alpha: float = 0.78) -> None:
    axis.add_patch(Ellipse((0.5, 0.52), 0.68, 0.84, fill=False, color=color, linewidth=1.4, alpha=alpha))
    for center_x in (0.34, 0.66):
        axis.add_patch(Ellipse((center_x, 0.39), 0.18, 0.075, fill=False, color=color, linewidth=1.1, alpha=alpha))
        axis.plot(
            [center_x - 0.10, center_x, center_x + 0.10],
            [0.32, 0.285, 0.32],
            color=color,
            linewidth=1.1,
            alpha=alpha,
        )
    axis.plot(
        [0.50, 0.47, 0.48, 0.52, 0.55],
        [0.43, 0.55, 0.61, 0.63, 0.60],
        color=color,
        linewidth=1.1,
        alpha=alpha,
    )
    axis.plot(
        [0.34, 0.42, 0.50, 0.58, 0.66, 0.58, 0.50, 0.42, 0.34],
        [0.73, 0.76, 0.77, 0.76, 0.73, 0.80, 0.82, 0.80, 0.73],
        color=color,
        linewidth=1.1,
        alpha=alpha,
    )


def plot_matched_heatmaps(result: AnalysisResult, path: Path) -> None:
    endpoints = result.tables["participant_endpoints"]
    selected = [
        endpoints[endpoints["scenario"].eq("near_equivalent")].iloc[0],
        endpoints[endpoints["scenario"].eq("systematic_bias")].iloc[0],
    ]
    stimulus_id = "SYN-FACE-01"
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 7.4))
    for row_index, participant in enumerate(selected):
        participant_id = participant["comparison_unit_id"]
        webcam = result.maps[f"{participant_id}|{stimulus_id}|webcam"]
        professional = result.maps[f"{participant_id}|{stimulus_id}|professional"]
        difference = np.abs(webcam - professional)
        vmax = max(float(webcam.max()), float(professional.max()))
        for column_index, (density, title, cmap, local_vmax) in enumerate(
            [
                (professional, "Reference instrument", "inferno", vmax),
                (webcam, "Webcam", "inferno", vmax),
                (difference, "Absolute difference", "Blues", float(difference.max())),
            ]
        ):
            axis = axes[row_index, column_index]
            image = axis.imshow(
                density,
                cmap=cmap,
                origin="upper",
                extent=(0, 1, 1, 0),
                vmin=0,
                vmax=local_vmax,
                interpolation="bilinear",
            )
            _face_outline(axis, color="white" if column_index < 2 else DEEP_BLUE)
            axis.set_xlim(0, 1)
            axis.set_ylim(1, 0)
            axis.set_xticks([])
            axis.set_yticks([])
            if row_index == 0:
                axis.set_title(title)
            if column_index == 0:
                axis.set_ylabel(_friendly(str(participant["scenario"])), weight="bold")
            fig.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
    fig.suptitle("Matched synthetic gaze-density examples", fontsize=17, weight="bold", x=0.04, ha="left")
    fig.text(0.04, 0.94, "Shared scale within each row; outline is the synthetic stimulus coordinate frame.", color="#4C6372")
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    _save(fig, path)


def plot_aoi_dwell(dwell: pd.DataFrame, path: Path) -> None:
    summary = (
        dwell.groupby(["scenario", "device", "aoi_name"], as_index=False)["dwell_share"]
        .mean()
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2), sharey=True)
    aoi_order = ["left_eye", "right_eye", "nose", "mouth", "other"]
    for axis, scenario in zip(axes.flat, SCENARIO_ORDER, strict=True):
        subset = summary[summary["scenario"].eq(scenario)]
        sns.barplot(
            data=subset,
            x="aoi_name",
            y="dwell_share",
            hue="device",
            order=aoi_order,
            palette=DEVICE_PALETTE,
            ax=axis,
        )
        axis.set_title(_friendly(scenario))
        axis.set_xlabel("")
        axis.set_ylabel("Mean valid-sample share")
        axis.set_xticks(
            axis.get_xticks(),
            [_friendly(value) for value in aoi_order],
            rotation=18,
            ha="right",
        )
        if axis is axes.flat[0]:
            axis.legend(title="Modality", frameon=False)
        elif axis.get_legend() is not None:
            axis.get_legend().remove()
    fig.suptitle("Versioned facial AOI dwell distributions", fontsize=17, weight="bold", x=0.04, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, path)


def plot_equivalence(summary: pd.DataFrame, path: Path) -> None:
    display = summary.copy()
    display["scenario"] = pd.Categorical(
        display["scenario"], categories=SCENARIO_ORDER, ordered=True
    )
    endpoint_order = [
        "accuracy_difference_deg",
        "data_loss_difference",
        "map_disagreement",
        "absolute_lag_ms",
    ]
    display["endpoint"] = pd.Categorical(
        display["endpoint"], categories=endpoint_order, ordered=True
    )
    display = display.sort_values(["scenario", "endpoint"]).reset_index(drop=True)
    display["estimate_scaled"] = display["mean_difference"] / display["margin"]
    display["lower_scaled"] = display["ci90_lower"] / display["margin"]
    display["upper_scaled"] = display["ci90_upper"] / display["margin"]
    endpoint_labels = {
        "accuracy_difference_deg": "Accuracy error Δ (°)",
        "data_loss_difference": "Data loss Δ",
        "map_disagreement": "Map disagreement",
        "absolute_lag_ms": "Absolute lag (ms)",
    }
    display["label"] = display.apply(
        lambda row: f"{_friendly(str(row['scenario']))} · {endpoint_labels[str(row['endpoint'])]}",
        axis=1,
    )
    colors = {"equivalent": MAYO_BLUE, "not_equivalent": CORAL, "inconclusive": "#7A8891"}
    fig, axis = plt.subplots(figsize=(11, 8))
    axis.axvspan(-1, 1, color=PALE_BLUE, zorder=0, label="Illustrative equivalence region")
    for y, row in display.reset_index(drop=True).iterrows():
        color = colors[row["outcome"]]
        axis.plot([row["lower_scaled"], row["upper_scaled"]], [y, y], color=color, linewidth=2.2)
        axis.scatter(row["estimate_scaled"], y, color=color, s=46, zorder=3)
    axis.axvline(-1, color=DEEP_BLUE, linestyle="--", linewidth=1)
    axis.axvline(1, color=DEEP_BLUE, linestyle="--", linewidth=1)
    axis.axvline(0, color="#738895", linewidth=1)
    axis.set_yticks(range(len(display)), display["label"])
    axis.invert_yaxis()
    axis.set_xlabel("Estimate and 90% CI, divided by illustrative margin")
    axis.set_title("Equivalence decisions remain endpoint-specific")
    axis.legend(
        handles=[
            Patch(facecolor=PALE_BLUE, edgecolor="none", label="Equivalence region"),
            Line2D([0], [0], marker="o", color=MAYO_BLUE, label="Equivalent", linewidth=2),
            Line2D([0], [0], marker="o", color="#7A8891", label="Inconclusive", linewidth=2),
            Line2D([0], [0], marker="o", color=CORAL, label="Not equivalent", linewidth=2),
        ],
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
    )
    axis.text(
        0.01,
        -0.09,
        "Inside both dashed lines = equivalent; crossing a line = inconclusive unless fully outside.",
        transform=axis.transAxes,
        color="#4C6372",
    )
    fig.tight_layout()
    _save(fig, path)


def plot_qc_matrix(endpoints: pd.DataFrame, path: Path) -> None:
    columns = [
        "accuracy_difference_deg",
        "data_loss_difference",
        "map_disagreement",
        "absolute_lag_ms",
    ]
    matrix = endpoints.set_index("comparison_unit_id")[columns].copy()
    scales = matrix.abs().quantile(0.95).replace(0, 1)
    scaled = matrix.divide(scales)
    fig_height = max(5.5, 0.38 * len(matrix))
    fig, axis = plt.subplots(figsize=(8.8, fig_height))
    sns.heatmap(
        scaled,
        cmap=sns.diverging_palette(220, 12, as_cmap=True),
        center=0,
        vmin=-1.2,
        vmax=1.2,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Value scaled by endpoint 95th percentile"},
        ax=axis,
    )
    axis.set_xlabel("")
    axis.set_ylabel("Synthetic comparison unit")
    axis.set_xticklabels(
        ["Accuracy error Δ (°)", "Data loss Δ", "Map disagreement", "Absolute lag (ms)"],
        rotation=12,
        ha="right",
    )
    axis.set_title("Participant-level quality-control matrix")
    fig.tight_layout()
    _save(fig, path)


def plot_temporal_alignment(timing: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    for axis, metric, label in (
        (axes[0], "estimated_lag_ms", "Estimated lag (ms; positive = webcam delayed)"),
        (
            axes[1],
            "peak_position_correlation",
            "Peak position correlation after lag search",
        ),
    ):
        sns.boxplot(
            data=timing,
            x="scenario",
            y=metric,
            order=SCENARIO_ORDER,
            color=SKY_BLUE,
            fliersize=0,
            ax=axis,
        )
        sns.stripplot(
            data=timing,
            x="scenario",
            y=metric,
            order=SCENARIO_ORDER,
            color=DEEP_BLUE,
            size=4,
            alpha=0.7,
            jitter=0.16,
            ax=axis,
        )
        axis.set_ylabel(label)
        axis.set_xlabel("")
        axis.set_xticks(
            axis.get_xticks(),
            [_friendly(value) for value in SCENARIO_ORDER],
            rotation=18,
            ha="right",
        )
    axes[0].axhline(0, color="#738895", linestyle=":", linewidth=1)
    fig.suptitle(
        "Temporal alignment after common 30 Hz resampling",
        fontsize=16,
        weight="bold",
        x=0.04,
        y=0.99,
        ha="left",
    )
    fig.text(
        0.04,
        0.91,
        "Lag is estimated and reported; it is never silently corrected.",
        color="#4C6372",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    _save(fig, path)


def render_all_figures(result: AnalysisResult, figure_dir: str | Path) -> list[Path]:
    _set_style()
    output = Path(figure_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = [
        output / "01_data_quality.png",
        output / "02_accuracy_bland_altman.png",
        output / "03_map_similarity.png",
        output / "04_matched_heatmaps.png",
        output / "05_aoi_dwell.png",
        output / "06_equivalence_intervals.png",
        output / "07_participant_qc_matrix.png",
        output / "08_temporal_alignment.png",
    ]
    plot_data_quality(result.tables["data_quality"], paths[0])
    plot_bland_altman(result.tables["participant_endpoints"], paths[1])
    plot_map_similarity(result.tables["map_agreement"], paths[2])
    plot_matched_heatmaps(result, paths[3])
    plot_aoi_dwell(result.tables["aoi_dwell"], paths[4])
    plot_equivalence(result.tables["equivalence_summary"], paths[5])
    plot_qc_matrix(result.tables["participant_endpoints"], paths[6])
    plot_temporal_alignment(result.tables["temporal_alignment"], paths[7])
    return paths
