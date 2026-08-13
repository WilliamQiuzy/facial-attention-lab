# Independent-cohort Webcam vs professional gaze comparison

> **Research-only, nonclinical, and currently 100% synthetic.** This project contains no observed Mayo, Prolific, or iMotions data; no patient images or identifiers; and no result about real Webcam or professional eye-tracker performance.

This project is built for the actual first-stage design:

- 500 participants measured through the Prolific/Webcam workflow;
- 500 different participants measured through the professional/Mayo workflow;
- the same versioned face stimuli and task protocol in both cohorts;
- no person-level pairing.

The central question is not “do the same two devices agree on the same person?” It is:

> Are the two independent workflow-and-cohort distributions sufficiently similar for a named group-level gaze endpoint?

Because device, recruitment, site, home/lab environment, display, and participant composition change together, this design cannot identify a pure causal device effect or individual-level interchangeability. Equal sample sizes improve precision; they do not remove confounding.

## Start here

Requirements: Python 3.12–3.14 and [uv](https://docs.astral.sh/uv/).

```bash
cd facial_defect_gaze_comparison
uv sync --extra dev
uv run gaze-compare cohort-simulate
uv run gaze-compare cohort-analyze
```

Open the main report:

- [`outputs/independent_cohort_demo/independent_cohort_report.md`](outputs/independent_cohort_demo/independent_cohort_report.md)
- Chinese analysis guide: [`docs/analysis_guide_zh.md`](docs/analysis_guide_zh.md)
- Concise English PDF: [`output/pdf/independent_cohort_gaze_comparison_brief.pdf`](output/pdf/independent_cohort_gaze_comparison_brief.pdf)

Regenerate the PDF:

```bash
uv run python scripts/build_method_brief.py
```

Run all tests:

```bash
uv run pytest -q
```

Execute the notebook from a clean kernel:

```bash
uv run jupyter execute notebooks/webcam_vs_professional_comparison.ipynb --timeout=240
```

## The four primary questions

| Priority | Question | Main method | Why it matters |
|---:|---|---|---|
| 1 | Were collection conditions comparable? | Common-stimulus/protocol gates and SMD balance plot | A numerical comparison is uninterpretable when tasks or stimulus versions differ. |
| 2 | Are technical-quality differences acceptably small? | Welch 90% CIs and independent-sample equivalence tests | “Not statistically different” does not demonstrate practical similarity. |
| 3 | Are group attention maps close relative to ordinary cohort sampling variation? | Repeated within-cohort and cross-domain split-half bootstrap | A perfect score of 1 is unrealistic even for two random halves from one workflow. |
| 4 | Can a simple model identify the source domain? | Repeated cross-validated logistic-regression AUC | Separates a technical domain shift from a behavioral attention-pattern shift. |

“Same domain” is deliberately not reduced to one score. The technical feature space can be separable while the group-level attention endpoint remains close enough for a prespecified use.

## Outputs designed for interpretation

| Figure | What to look for |
|---|---|
| `01_covariate_balance.png` | Participant mix and acquisition-context differences before outcomes. |
| `02_quality_equivalence.png` | Whether the entire 90% CI falls inside a prespecified practical margin. |
| `03_quality_distributions.png` | Cohort overlap and tails, not only means. |
| `04_group_attention_maps.png` | Professional, Webcam, and signed difference maps on the same stimuli. |
| `05_map_reproducibility.png` | Cross-domain similarity compared with Webcam and professional split-half baselines. |
| `06_aoi_profile.png` | Interpretable eye, nose, mouth, and outside-AOI dwell shares. |
| `07_domain_classifier.png` | Technical-quality and attention-pattern source AUCs shown separately. |

The report explains for every primary metric: its importance, calculation, visualization, decision rule, and limitation.

## Synthetic input contract

The mock generator creates a compact 500+500 example rather than millions of raw frame rows:

- `participant_summary.csv`: one row per independent participant, with acquisition quality and declared covariates;
- `fixation_events.csv`: device-neutral fixation-like dwell events on common versioned stimuli;
- `aoi_definitions.csv`: versioned normalized facial AOIs;
- `stimuli.csv`: versioned synthetic stimulus metadata;
- `mock_manifest.json`: row counts, bytes, seeds, and SHA-256 hashes.

The future real adapter should preserve recruitment source, site, hardware model, acquisition software/version, display geometry, lighting, glasses, stimulus version, task version, coordinate transform, validity, and calibration evidence as distinct fields. Prolific and iMotions are not device names.

## Decision language

For a quality endpoint with a preregistered symmetric margin:

- `similar_within_margin`: the full Welch 90% CI for Webcam − Professional is inside the margin;
- `meaningfully_different`: the full interval lies beyond a margin;
- `inconclusive`: every overlap case.

For maps, cross-domain SIM is compared with the lower of the two within-cohort split-half similarities. The analysis asks whether the loss is no larger than a preregistered noninferiority margin.

The real conclusion should be endpoint-specific, for example:

> “For these versioned stimuli and this workflow, the 500-person Webcam group map was within the prespecified SIM-loss margin relative to cohort split-half repeatability.”

It should not say “the devices are the same,” “Webcam equals the professional tracker,” or “individual records are interchangeable.”

## Real-data safety boundary

Real/raw directories are ignored by Git. Keep facial video, patient/participant images, raw gaze, linkage keys, and identifiable metadata only in approved access-controlled research storage. A real analysis must use externally justified, versioned, preregistered margins; the mock values in `config/mock_independent_study.json` are demonstrations only.

## Project layout

```text
facial_defect_gaze_comparison/
├── config/mock_independent_study.json
├── data/mock_independent/
├── docs/analysis_guide_zh.md
├── docs/metric_spec.md
├── docs/study_protocol.md
├── notebooks/webcam_vs_professional_comparison.ipynb
├── outputs/independent_cohort_demo/
├── src/gaze_compare/cohort_*.py
└── tests/test_cohort_*.py
```

The earlier paired-acquisition prototype remains in `analysis.py`, `metrics.py`, `plots.py`, `report.py`, and `simulate.py` as a secondary method for a future same-person study. It is not the default answer to the current independent-cohort question.

## Methodological sources

- Holmqvist et al., [Eye-tracking data quality: what it is and how to measure it](https://pmc.ncbi.nlm.nih.gov/articles/PMC3996543/).
- Yang and Krajbich, [Webcam-based online eye-tracking for behavioral research](https://pmc.ncbi.nlm.nih.gov/articles/PMC11289017/).
- Semmelmann and Weigelt, [Online Webcam-based eye tracking in cognitive science](https://pmc.ncbi.nlm.nih.gov/articles/PMC8787048/).
- Lakens, [Equivalence tests: a practical primer](https://pmc.ncbi.nlm.nih.gov/articles/PMC5502906/).
- Lopez-Paz and Oquab, [Revisiting classifier two-sample tests](https://arxiv.org/abs/1610.06545).
- Kümmerer et al., [Saliency Benchmarking Made Easy](https://openaccess.thecvf.com/content_ECCV_2018/html/Matthias_Kummerer_Saliency_Benchmarking_Made_ECCV_2018_paper.html).

These sources motivate the framework. They do not validate this code or any Webcam workflow on Mayo data.
