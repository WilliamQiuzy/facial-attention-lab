from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import pandas as pd

from .analysis import AnalysisResult
from .plots import render_all_figures


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _scenario_summary(result: AnalysisResult) -> str:
    quality = result.tables["data_quality"]
    maps = result.tables["map_agreement"]
    lines = [
        "| Scenario | Webcam accuracy error (°) | Reference accuracy error (°) | Map SIM |",
        "|---|---:|---:|---:|",
    ]
    for scenario in sorted(quality["scenario"].unique()):
        scenario_quality = quality[quality["scenario"].eq(scenario)]
        accuracy = scenario_quality.groupby("device")["accuracy_deg"].mean()
        sim = maps[maps["scenario"].eq(scenario)]["histogram_intersection"].mean()
        lines.append(
            f"| {scenario.replace('_', ' ').title()} | {accuracy['webcam']:.2f} | "
            f"{accuracy['professional']:.2f} | {sim:.2f} |"
        )
    return "\n".join(lines)


def render_mock_report(result: AnalysisResult, config: Mapping[str, object]) -> str:
    outcomes = result.tables["equivalence_summary"]["outcome"].value_counts().to_dict()
    return f"""# Webcam vs professional gaze comparison

> **100% synthetic demonstration. Research-only and nonclinical.** These are not observed Mayo data, not observed Prolific data, and not an iMotions result. No patient images, recordings, gaze exports, or identifiers are included.

## What this demonstration answers

This mock run shows how a future paired study can compare a webcam gaze estimator with a professional **reference instrument, not error-free ground truth**. It keeps recruitment source, cohort/site, tracker modality and model, and acquisition software/version as separate factors.

It does not answer whether a real webcam workflow is clinically acceptable. The simulated scenarios exist to verify that the pipeline reacts differently to near agreement, systematic spatial bias, temporal lag, and high data loss.

## Synthetic results at a glance

{_scenario_summary(result)}

The demo produced {outcomes.get('equivalent', 0)} equivalent, {outcomes.get('not_equivalent', 0)} not-equivalent, and {outcomes.get('inconclusive', 0)} inconclusive endpoint/scenario decisions. All use **illustrative margins** from `config/mock_study.json`; they are not acceptance thresholds for a real study. Failure to show equivalence is not proof of inequivalence.

## How to read the figures

1. `01_data_quality.png` keeps calibration error, precision, data loss, and sampling rate separate.
2. `02_accuracy_bland_altman.png` uses one aggregated point per synthetic participant, avoiding pseudoreplication across targets.
3. `03_map_similarity.png` shows two complementary density-map metrics rather than a single universal score.
4. `04_matched_heatmaps.png` uses the same scale within each row and overlays the known synthetic face coordinate frame.
5. `05_aoi_dwell.png` compares versioned eye, nose, mouth, and outside-AOI shares.
6. `06_equivalence_intervals.png` visualizes the exact three-way decision rule.
7. `07_participant_qc_matrix.png` helps locate synthetic comparison units that drive disagreement.
8. `08_temporal_alignment.png` reports estimated lag after common 30 Hz resampling; no silent time shift is applied.

## Evidence gates

Paired map, AOI, Bland–Altman, and equivalence analysis require the same comparison unit and versioned stimulus under both modalities, a common coordinate/time frame, sufficient valid samples, and externally justified preregistered margins for real observations. If the existing Prolific and Mayo exports are separate cohorts rather than paired observations, this pipeline can provide descriptive summaries but **cannot establish interchangeability**.

Vendor-derived fixation events are not compared in this version because algorithms can differ. Reliability ceilings are also deferred until independent repeated sessions are available. The professional instrument is never labeled as truth.

## Recommended next collection

- Record both streams simultaneously for the same consented participant and exact stimulus presentation.
- Preserve display geometry, viewing distance, viewport-to-stimulus transform, timestamps, validity/blink flags, tracker model, acquisition software version, and AOI/stimulus versions.
- Include known-location calibration and validation targets before and after free viewing.
- Preregister primary endpoints, exclusion rules, bootstrap unit, multiplicity family, and clinically/scientifically justified equivalence margins before unblinding results.

## Reproducibility

- Synthetic project ID: `{config['project_id']}`
- Random seed: `{config['seed']}`
- Analysis mode: paired mock demonstration
- Raw and real export directories are ignored by Git in this public project.

See `docs/metric_spec.md` for formulas and decision rules and `docs/study_protocol.md` for the proposed real acquisition design.
"""


def write_analysis_outputs(
    result: AnalysisResult,
    output_dir: str | Path,
    *,
    config: Mapping[str, object],
) -> Path:
    output = Path(output_dir)
    table_dir = output / "tables"
    figure_dir = output / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    artifacts: list[Path] = []
    for name, table in sorted(result.tables.items()):
        table_path = table_dir / f"{name}.csv"
        table.to_csv(table_path, index=False, lineterminator="\n")
        artifacts.append(table_path)
    artifacts.extend(render_all_figures(result, figure_dir))

    report_path = output / "mock_analysis_report.md"
    report_path.write_text(render_mock_report(result, config), encoding="utf-8")
    artifacts.append(report_path)

    manifest = {
        "project_id": config["project_id"],
        "synthetic": True,
        "analysis_mode": "paired_mock_demo",
        "seed": int(config["seed"]),
        "artifacts": [
            {
                "path": str(path.relative_to(output)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in sorted(artifacts)
        ],
    }
    manifest_path = output / "analysis_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path
