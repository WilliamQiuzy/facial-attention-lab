from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

from .cohort_analysis import IndependentCohortResult
from .cohort_plots import render_independent_figures


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _decision_table(result: IndependentCohortResult) -> str:
    lines = [
        "| Priority | Research question | Method | Mock result |",
        "|---:|---|---|---|",
    ]
    for row in result.tables["decision_summary"].itertuples(index=False):
        lines.append(f"| {row.priority} | {row.question} | {row.method} | `{row.decision}` |")
    return "\n".join(lines)


def _quality_table(result: IndependentCohortResult) -> str:
    lines = [
        "| Endpoint | Why it matters | Webcam | Professional | Difference (90% CI) | Mock decision |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in result.tables["quality_comparison"].itertuples(index=False):
        lines.append(
            f"| {row.label} | {row.why_important} | {row.webcam_mean:.3f} | "
            f"{row.professional_mean:.3f} | {row.mean_difference_webcam_minus_professional:+.3f} "
            f"({row.ci90_lower:+.3f}, {row.ci90_upper:+.3f}) | `{row.decision}` |"
        )
    return "\n".join(lines)


def _map_table(result: IndependentCohortResult) -> str:
    lines = [
        "| Stimulus | Cross-domain SIM | Webcam split-half | Professional split-half | Gap decision |",
        "|---|---:|---:|---:|---|",
    ]
    for row in result.tables["map_reliability"].itertuples(index=False):
        lines.append(
            f"| {row.stimulus_id} | {row.cross_domain_similarity:.3f} | "
            f"{row.within_webcam_similarity:.3f} | {row.within_professional_similarity:.3f} | "
            f"`{row.decision}` |"
        )
    return "\n".join(lines)


def _classifier_table(result: IndependentCohortResult) -> str:
    lines = ["| Feature set | AUC (95% bootstrap interval) | Interpretation |", "|---|---:|---|"]
    for row in result.tables["domain_classifier"].itertuples(index=False):
        lines.append(
            f"| {row.feature_set.replace('_', ' ').title()} | {row.auc:.3f} "
            f"({row.ci95_lower:.3f}, {row.ci95_upper:.3f}) | `{row.interpretation}` |"
        )
    return "\n".join(lines)


def render_independent_report(
    result: IndependentCohortResult, config: Mapping[str, object]
) -> str:
    return f"""# Independent-cohort Webcam vs professional gaze comparison

> **100% synthetic demonstration; research-only and nonclinical.** These are not observed Mayo data, not observed Prolific data, and not an iMotions result. There are no patient images, recordings, raw gaze exports, or identifiers in this demo.

## The design we are actually analyzing

This demonstration has **{result.metadata['n_webcam']} Webcam participants and {result.metadata['n_professional']} professional-camera participants, with different participants in the two groups**. No person-level pairing is performed or implied.

Because participant, recruitment source, setting, device, lighting, and display can all differ together, this design **cannot isolate a pure device effect**. It estimates whether the *complete Webcam/Prolific workflow-and-cohort distribution* is sufficiently similar to the *professional/Mayo workflow-and-cohort distribution* for a prespecified group-level endpoint, provided both cohorts saw the same versioned stimuli under harmonized instructions.

**“Same domain is not one yes/no property.”** Technical acquisition features may be distinguishable while group attention patterns remain close. The decision must therefore name the endpoint.

## Results at a glance

{_decision_table(result)}

These results are deliberately illustrative. All tolerances in `config/mock_independent_study.json` are **illustrative margins**, not Mayo acceptance thresholds and not clinical criteria.

## 1. First gate: are the cohorts and tasks comparable?

Use `01_covariate_balance.png` before looking at outcomes. Standardized mean differences (SMDs) place continuous and binary characteristics on one scale. SMD is more useful than a large-sample p-value here: with 500+500 people, a tiny unimportant difference can be statistically significant. The |SMD| < 0.10 band is a review convention, not evidence of randomization.

- Participant characteristics help reveal recruitment differences.
- Display size and lighting are acquisition-context variables; they describe the real workflow difference and should not be silently “adjusted away.”
- If stimuli, task version, timing, transform, or exclusion rules differ, stop the primary comparison or restrict it to the common harmonized subset.

## 2. Technical data quality: estimate differences and test practical similarity

{_quality_table(result)}

### How each primary quality metric is used

| Metric | Importance | Calculation | Visualization | Decision rule |
|---|---|---|---|---|
| Calibration accuracy error | A map can look smooth but be spatially wrong. | Mean angular distance from gaze to independently known calibration targets, one value per participant. | `02_quality_equivalence.png`, `03_quality_distributions.png` | Welch 90% CI for Webcam − Professional must fall inside a preregistered ±margin. |
| RMS precision | Separates jitter from systematic offset. | RMS sample-to-sample displacement during stable calibration targets. | Same two figures | Same independent-sample equivalence rule. |
| Data loss | Missingness can alter maps and reduce usable participants. | 1 − valid expected samples / all expected samples. | Same two figures | Same rule; inspect the upper tail as well as the mean. |
| Valid-trial share | A good mean among survivors can hide many failed trials. | Valid trials / expected trials per participant. | Same two figures | Same rule; higher is better. |

Why a usual “no significant difference” test is not enough: failure to reject zero difference does not demonstrate practical equivalence. Equivalence requires a scientifically justified smallest difference of interest chosen before examining the outcomes.

## 3. Group attention: compare against ordinary sampling variation

{_map_table(result)}

The group maps in `04_group_attention_maps.png` show Professional, Webcam, and a signed difference map for every common stimulus. The first two panels share a scale within each row; the difference panel uses a zero-centered scale.

The stronger check is `05_map_reproducibility.png`:

1. Randomly split the 500 Webcam participants into two halves and compare their dwell-density maps.
2. Do the same within the 500 professional participants.
3. Compare one random Webcam half with one random professional half.
4. Repeat the procedure {config['bootstrap_replicates']} times for each stimulus.

Histogram intersection (SIM, 0 to 1) asks how much normalized dwell density the maps share. Cross-domain similarity is interpreted relative to the lower within-cohort split-half similarity, not against an unrealistic perfect score of 1. The mock noninferiority margin is {config['illustrative_margins']['map_similarity_gap']:.2f} SIM units.

`06_aoi_profile.png` then makes the spatial result clinically readable by summarizing dwell share in versioned left-eye, right-eye, nose, mouth, and outside-AOI regions. AOI results are secondary to the full map because a broad AOI can hide within-region shifts.

## 4. “Same domain?” Use an interpretable source classifier

{_classifier_table(result)}

`07_domain_classifier.png` uses repeated stratified cross-validation. A simple logistic model is asked whether it can identify Webcam versus Professional records:

- **Technical-quality model:** calibration accuracy, precision, loss, and valid-trial share.
- **Attention-pattern model:** stimulus-by-AOI dwell shares only.

AUC = 0.50 is chance; higher AUC means the two samples are more separable in that feature space. This is a diagnostic, not a universal same-domain test. Chance-level AUC does not prove equality, and a high technical AUC does not automatically mean group attention maps are unusable.

## What we can and cannot conclude

This independent design can support statements such as: “for the same stimuli and this protocol, group-level Webcam attention maps were within the preregistered loss margin relative to cohort split-half repeatability.” It cannot support person-level agreement, measurement interchangeability for an individual, or a pure causal device effect.

For a later real study:

1. Freeze the common stimulus set, instructions, exposure time, coordinate transform, and exclusions.
2. Define one primary use case: group heatmap, AOI ranking, or individual gaze location. Do not reuse one margin for all three.
3. Set equivalence/noninferiority margins with clinicians and eye-tracking experts before unblinding.
4. Report age/glasses and acquisition context; consider stratification or weighting as sensitivity analyses when participant composition differs.
5. If the goal changes to individual device interchangeability, collect both devices on the same people, preferably simultaneously and in randomized order where possible.

## Reproducibility and evidence boundary

- Project ID: `{config['project_id']}`
- Random seed: `{config['seed']}`
- Design: two independent cohorts, not paired
- Mock cohort size: {config['participants_per_cohort']} per workflow
- Map unit: participant-resampled dwell-density field
- Professional tracker role: reference workflow, not error-free truth
- Pure device effect identified: no

The public repository ignores real/raw export directories. Real participant data, videos, images, and linkage keys belong only in approved access-controlled research storage.
"""


def write_independent_outputs(
    result: IndependentCohortResult,
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
        path = table_dir / f"{name}.csv"
        table.to_csv(path, index=False, lineterminator="\n")
        artifacts.append(path)
    artifacts.extend(render_independent_figures(result, figure_dir))
    report_path = output / "independent_cohort_report.md"
    report_path.write_text(render_independent_report(result, config), encoding="utf-8")
    artifacts.append(report_path)
    manifest = {
        "project_id": config["project_id"],
        "synthetic": bool(config.get("synthetic", False)),
        "analysis_mode": "independent_cohort_mock_demo",
        "seed": int(config["seed"]),
        "n_webcam": int(result.metadata["n_webcam"]),
        "n_professional": int(result.metadata["n_professional"]),
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
