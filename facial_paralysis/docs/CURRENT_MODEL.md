# Current Model: Universal Clinical Router v4

Universal Clinical Router v4 is the sole canonical facial-paralysis research
model as of 2026-08-16. New code must import `src.models.current`; new reports
must bind the artifact registered in `docs/model_registry.json`.

## Architecture

The artifact routes authenticated task, timing, and modality evidence to one of
three frozen clinical experts. Dataset and institution identity are never model
inputs.

| Evidence profile | Frozen expert |
|---|---|
| Free recording | mirror-invariant 110D Landmark asymmetry head |
| Three named scripted actions | Landmark + Py-Feat AU clinical heads, with an 18-head frozen MARLIN median gate for low-confidence cases |
| Externally timed seven-action recording | two-head cue-aligned Landmark sequence ensemble |

The 110D model is an internal v4 expert, not a separate current model. Historical
MARLIN/GRU, HB, SSL, architecture-search, Universal Orofacial v1, and Universal
Phenotype v3 code is retained only for explicit reproducibility work.

## Current development evidence

All primary estimates are participant-disjoint. They are development evidence,
not clinical validation.

| Cohort/profile | People | AUROC | Accuracy | Balanced accuracy |
|---|---:|---:|---:|---:|
| PalsyNet development / free recording | 38 | 0.980 | 0.947 | 0.952 |
| PalsyNet sealed outer / free recording | 10 | 1.000 | 0.900 | 0.900 |
| NeuroFace / scripted multimechanism | 36 | 0.931 | 0.917 | 0.889 |
| MEEI / cue-aligned action geometry | 56 | 0.911 | 0.875 | 0.885 |

Mayo has no verified negative/control class or HB grades. The locked 110D
expert called 45 of 47 quality-eligible assumed-positive videos positive
(95.74% positive-call rate); this is not binary accuracy. The Mayo action timing
gate is not eligible, so the action expert has made zero Mayo predictions.

## Authoritative files

- Registry: `docs/model_registry.json`
- Runtime facade: `src/models/current.py`
- Runtime implementation: `src/models/universal_clinical_router_v4.py`
- Model artifact: `docs/results/artifacts/universal_clinical_router_v4/model.json`
- Aggregate report: `docs/results/artifacts/universal_clinical_router_v4/report.json`
- Full method/result: `docs/results/universal_clinical_router_v4.md`

## Maintenance and promotion policy

1. Never change the meaning of v4 in place; a successor receives a new version.
2. Keep dataset/institution identity out of routing and prediction.
3. Select only with participant-disjoint development data and preregistered
   gates; protected or external labels never tune a candidate.
4. Report each cohort separately. Do not turn Mayo positive-call rate into
   accuracy, or development metrics into a clinical claim.
5. A successor replaces v4 only after it improves the intended endpoint without
   degrading the other evidence profiles and passes an untouched validation
   gate. Until then, v4 remains canonical.

Historical model narratives are indexed under `docs/archive/` and must not be
used for present-tense model reporting.
