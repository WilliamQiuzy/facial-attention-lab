from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from .cohort_analysis import IndependentCohortResult


MAYO_BLUE = "#0057B8"
DEEP_BLUE = "#003B71"
SKY_BLUE = "#6CB4EE"
PALE_BLUE = "#EAF3FA"
CHARCOAL = "#233746"
SLATE = "#627786"
RED = "#C94242"
GREEN = "#2A7F62"
GOLD = "#B9770E"
PROFESSIONAL = "#3F6C8F"
WEBCAM = MAYO_BLUE


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelcolor": CHARCOAL,
            "axes.edgecolor": "#B8C7D1",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": SLATE,
            "ytick.color": SLATE,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_covariate_balance(result: IndependentCohortResult, output: Path) -> Path:
    table = result.tables["covariate_balance"].sort_values("absolute_smd")
    colors = [MAYO_BLUE if role == "participant_characteristic" else GOLD for role in table["role"]]
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.axvspan(-0.10, 0.10, color=PALE_BLUE, zorder=0)
    ax.axvline(0, color=SLATE, linewidth=1)
    ax.axvline(-0.10, color=SKY_BLUE, linestyle="--", linewidth=1)
    ax.axvline(0.10, color=SKY_BLUE, linestyle="--", linewidth=1)
    y = np.arange(len(table))
    ax.hlines(y, 0, table["standardized_mean_difference"], color="#CBD8E1", linewidth=2)
    ax.scatter(table["standardized_mean_difference"], y, color=colors, s=65, zorder=3)
    ax.set_yticks(y, table["label"])
    ax.set_xlabel("Standardized mean difference (Webcam − Professional)")
    fig.suptitle(
        "Before outcomes: are the two cohorts comparable?",
        x=0.06,
        y=0.97,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.06,
        0.90,
        "The blue band (|SMD| < 0.10) is a review aid, not proof of randomization.",
        color=SLATE,
    )
    limit = max(0.35, float(table["absolute_smd"].max()) * 1.18)
    ax.set_xlim(-limit, limit)
    ax.grid(axis="x", color="#E4EBF0", linewidth=0.8)
    ax.scatter([], [], color=MAYO_BLUE, label="Participant characteristic")
    ax.scatter([], [], color=GOLD, label="Acquisition context")
    ax.legend(frameon=False, loc="lower right")
    fig.subplots_adjust(top=0.82, left=0.25, right=0.98, bottom=0.16)
    return _save(fig, output / "01_covariate_balance.png")


def plot_quality_equivalence(result: IndependentCohortResult, output: Path) -> Path:
    table = result.tables["quality_comparison"].copy()
    table["estimate_scaled"] = table["mean_difference_webcam_minus_professional"] / table[
        "equivalence_margin"
    ]
    table["lower_scaled"] = table["ci90_lower"] / table["equivalence_margin"]
    table["upper_scaled"] = table["ci90_upper"] / table["equivalence_margin"]
    table = table.iloc[::-1].reset_index(drop=True)
    decision_colors = {
        "similar_within_margin": GREEN,
        "meaningfully_different": RED,
        "inconclusive": GOLD,
    }
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    ax.axvspan(-1, 1, color="#E8F4EF", zorder=0)
    ax.axvline(0, color=SLATE, linewidth=1)
    ax.axvline(-1, color=GREEN, linestyle="--", linewidth=1)
    ax.axvline(1, color=GREEN, linestyle="--", linewidth=1)
    for index, row in table.iterrows():
        color = decision_colors[row["decision"]]
        ax.plot([row["lower_scaled"], row["upper_scaled"]], [index, index], color=color, linewidth=3)
        ax.scatter(row["estimate_scaled"], index, color=color, s=75, zorder=3)
        raw = row["mean_difference_webcam_minus_professional"]
        margin = row["equivalence_margin"]
        ax.text(
            2.12,
            index,
            f"Δ={raw:+.3f}; margin ±{margin:g}",
            va="center",
            color=CHARCOAL,
            fontsize=9,
        )
    ax.set_yticks(np.arange(len(table)), table["label"])
    ax.set_xlim(-2.1, 4.0)
    ax.set_xlabel("Difference divided by the prespecified margin")
    fig.suptitle(
        "Are technical differences small enough for the intended endpoint?",
        x=0.06,
        y=0.97,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.06,
        0.90,
        "Dots are Webcam − Professional; bars are Welch 90% CIs. Entirely inside ±1 supports equivalence.",
        color=SLATE,
    )
    ax.grid(axis="x", color="#E4EBF0", linewidth=0.8)
    fig.subplots_adjust(top=0.82, left=0.28, right=0.98, bottom=0.16)
    return _save(fig, output / "02_quality_equivalence.png")


def _ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(values)
    return ordered, np.arange(1, len(ordered) + 1) / len(ordered)


def plot_quality_distributions(result: IndependentCohortResult, output: Path) -> Path:
    data = result.tables["participant_quality"]
    specifications = [
        ("accuracy_deg", "Calibration accuracy error", "Degrees"),
        ("rms_precision_deg", "RMS precision", "Degrees"),
        ("data_loss", "Data loss", "Proportion"),
        ("valid_trial_share", "Valid-trial share", "Proportion"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.4))
    for ax, (column, title, unit) in zip(axes.ravel(), specifications, strict=True):
        for device, color, label in [
            ("professional", PROFESSIONAL, "Professional"),
            ("webcam", WEBCAM, "Webcam"),
        ]:
            values = data.loc[data["device"].eq(device), column].to_numpy(float)
            x, y = _ecdf(values)
            ax.step(x, y, where="post", color=color, linewidth=2.2, label=label)
        ax.set_title(title, loc="left")
        ax.set_xlabel(unit)
        ax.set_ylabel("Participants at or below value")
        ax.grid(color="#E7EDF1", linewidth=0.8)
    axes[0, 0].legend(frameon=False, loc="lower right")
    fig.suptitle("See the full participant distributions—not only the averages", x=0.07, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.07, 0.93, "ECDFs show overlap, tails, and whether a small mean difference hides poor participants.", color=SLATE)
    fig.tight_layout(rect=(0.04, 0.03, 0.98, 0.91))
    return _save(fig, output / "03_quality_distributions.png")


def plot_group_attention_maps(result: IndependentCohortResult, output: Path) -> Path:
    stimulus_ids = sorted(result.tables["map_reliability"]["stimulus_id"])
    attention_cmap = LinearSegmentedColormap.from_list(
        "attention", ["#F7FBFF", "#74A9CF", "#0570B0", "#FFB000", "#D7301F"]
    )
    fig, axes = plt.subplots(len(stimulus_ids), 3, figsize=(10.8, 10.2), constrained_layout=False)
    for row, stimulus_id in enumerate(stimulus_ids):
        professional = result.maps[f"{stimulus_id}|professional"]
        webcam = result.maps[f"{stimulus_id}|webcam"]
        difference = result.maps[f"{stimulus_id}|difference"]
        vmax = max(float(professional.max()), float(webcam.max()))
        diff_max = max(float(np.abs(difference).max()), 1e-8)
        for column, (values, title) in enumerate(
            [(professional, "Professional"), (webcam, "Webcam"), (difference, "Webcam − Professional")]
        ):
            ax = axes[row, column]
            if column < 2:
                image = ax.imshow(values, origin="upper", cmap=attention_cmap, vmin=0, vmax=vmax)
            else:
                image = ax.imshow(values, origin="upper", cmap="RdBu_r", vmin=-diff_max, vmax=diff_max)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(title)
            if column == 0:
                ax.set_ylabel(stimulus_id, color=CHARCOAL, fontweight="bold")
            if row == len(stimulus_ids) - 1:
                fig.colorbar(image, ax=ax, orientation="horizontal", fraction=0.055, pad=0.035)
    fig.suptitle(
        "Same stimuli, independent people: compare the group attention field",
        y=0.985,
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.945,
        "Each row shares one scale for the first two maps. Red in the difference map means more Webcam dwell density.",
        ha="center",
        color=SLATE,
    )
    fig.subplots_adjust(top=0.88, bottom=0.08, left=0.07, right=0.98, hspace=0.08, wspace=0.08)
    return _save(fig, output / "04_group_attention_maps.png")


def plot_map_reproducibility(result: IndependentCohortResult, output: Path) -> Path:
    data = result.tables["map_reliability"]
    fig, axes = plt.subplots(1, len(data), figsize=(12.2, 4.6), sharex=True, sharey=True)
    comparisons = [
        ("within_professional_similarity", "within_professional", "Professional split-half", PROFESSIONAL),
        ("within_webcam_similarity", "within_webcam", "Webcam split-half", WEBCAM),
        ("cross_domain_similarity", "cross", "Cross-domain halves", GOLD),
    ]
    for ax, (_, row) in zip(axes, data.iterrows(), strict=True):
        for y, (estimate_column, prefix, label, color) in enumerate(comparisons):
            lower_column = f"{prefix}_ci90_lower"
            upper_column = f"{prefix}_ci90_upper"
            estimate = row[estimate_column]
            ax.plot([row[lower_column], row[upper_column]], [y, y], color=color, linewidth=3)
            ax.scatter(estimate, y, color=color, s=65, zorder=3)
        ax.set_title(row["stimulus_id"])
        ax.grid(axis="x", color="#E7EDF1", linewidth=0.8)
        ax.set_xlim(0.93, 0.98)
        ax.set_xlabel("Map similarity (SIM)")
    axes[0].set_yticks(np.arange(3), [item[2] for item in comparisons])
    fig.suptitle("Is cross-domain similarity close to ordinary sampling variation?", x=0.06, ha="left", fontsize=15, fontweight="bold")
    fig.text(
        0.06,
        0.91,
        "Bars are 90% intervals across repeated random half-cohort splits. Cross-domain is judged against the lower within-cohort benchmark.",
        color=SLATE,
    )
    fig.tight_layout(rect=(0.04, 0.04, 0.99, 0.86))
    return _save(fig, output / "05_map_reproducibility.png")


def plot_aoi_profile(result: IndependentCohortResult, output: Path) -> Path:
    data = result.tables["aoi_summary"]
    stimulus_ids = sorted(data["stimulus_id"].unique())
    aoi_order = ["left_eye", "right_eye", "nose", "mouth", "other"]
    fig, axes = plt.subplots(1, len(stimulus_ids), figsize=(12.4, 5.2), sharey=True)
    for ax, stimulus_id in zip(axes, stimulus_ids, strict=True):
        subset = data[data["stimulus_id"].eq(stimulus_id)]
        for index, aoi in enumerate(aoi_order):
            values = subset[subset["aoi_name"].eq(aoi)].set_index("device")
            professional = values.loc["professional"]
            webcam = values.loc["webcam"]
            ax.plot(
                [professional["mean_dwell_share"], webcam["mean_dwell_share"]],
                [index, index],
                color="#C9D5DD",
                linewidth=2,
            )
            for row, color, offset in [(professional, PROFESSIONAL, -0.06), (webcam, WEBCAM, 0.06)]:
                ax.errorbar(
                    row["mean_dwell_share"],
                    index + offset,
                    xerr=[
                        [row["mean_dwell_share"] - row["ci95_lower"]],
                        [row["ci95_upper"] - row["mean_dwell_share"]],
                    ],
                    fmt="o",
                    color=color,
                    capsize=2,
                )
        ax.set_title(stimulus_id)
        ax.set_xlabel("Mean dwell share")
        ax.grid(axis="x", color="#E7EDF1", linewidth=0.8)
        ax.set_xlim(0, 0.42)
    axes[0].set_yticks(np.arange(len(aoi_order)), [name.replace("_", " ").title() for name in aoi_order])
    axes[0].invert_yaxis()
    axes[0].scatter([], [], color=PROFESSIONAL, label="Professional")
    axes[0].scatter([], [], color=WEBCAM, label="Webcam")
    axes[0].legend(frameon=False, loc="lower right")
    fig.suptitle("Which facial regions receive attention?", x=0.06, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.06, 0.91, "Dots are cohort means; bars are participant-level 95% CIs. AOIs are versioned and stimulus-aligned.", color=SLATE)
    fig.tight_layout(rect=(0.04, 0.04, 0.99, 0.86))
    return _save(fig, output / "06_aoi_profile.png")


def plot_domain_classifier(result: IndependentCohortResult, output: Path) -> Path:
    data = result.tables["domain_classifier"].copy()
    label_map = {
        "technical_quality": "Technical quality features",
        "attention_pattern": "Attention-pattern features",
    }
    data["label"] = data["feature_set"].map(label_map)
    data = data.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    low = float(data["low_detectability_threshold"].iloc[0])
    clear = float(data["clear_difference_threshold"].iloc[0])
    ax.axvspan(0.5, low, color="#E8F4EF")
    ax.axvspan(clear, 1.0, color="#FBECEC")
    ax.axvline(0.5, color=SLATE, linestyle="--", linewidth=1.2)
    for index, row in data.iterrows():
        color = RED if row["interpretation"] == "clearly_distinguishable" else MAYO_BLUE
        ax.plot([row["ci95_lower"], row["ci95_upper"]], [index, index], color=color, linewidth=3)
        ax.scatter(row["auc"], index, color=color, s=80, zorder=3)
        if row["auc"] > 0.90:
            label_x, alignment = row["auc"] - 0.012, "right"
        else:
            label_x, alignment = row["ci95_upper"] + 0.018, "left"
        ax.text(
            label_x,
            index,
            f"AUC {row['auc']:.2f}",
            va="center",
            ha=alignment,
            color=CHARCOAL,
        )
    ax.set_yticks(np.arange(len(data)), data["label"])
    ax.set_xlim(0.45, 1.0)
    ax.set_xlabel("Repeated cross-validated AUC (0.50 = chance)")
    fig.suptitle(
        "Can a simple model tell which workflow produced a participant record?",
        x=0.06,
        y=0.97,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.06,
        0.89,
        "Technical and attention domains are tested separately. Low AUC is supportive, but never proves equality.",
        color=SLATE,
    )
    ax.grid(axis="x", color="#E7EDF1", linewidth=0.8)
    fig.subplots_adjust(top=0.78, left=0.32, right=0.97, bottom=0.20)
    return _save(fig, output / "07_domain_classifier.png")


def render_independent_figures(
    result: IndependentCohortResult, output_dir: str | Path
) -> list[Path]:
    _style()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    functions = [
        plot_covariate_balance,
        plot_quality_equivalence,
        plot_quality_distributions,
        plot_group_attention_maps,
        plot_map_reproducibility,
        plot_aoi_profile,
        plot_domain_classifier,
    ]
    return [function(result, output) for function in functions]
