# Vitestro Phlebotomy Safety

English-language research package for evaluating live presyncope and discomfort
sensors during automated venipuncture.

## Artifacts

| Artifact | Purpose | Status |
|---|---|---|
| [`docs/wearable_device_evaluation.md`](docs/wearable_device_evaluation.md) | Ten-device `✓ / △ / ✕` live feature matrix, evidence, shortlist, and bench gates | Current |
| [`data/device_feature_matrix.csv`](data/device_feature_matrix.csv) | Broader 18-device screen | Background |
| [`data/modality_evidence_matrix.csv`](data/modality_evidence_matrix.csv) | Modality evidence and failure modes | Background |
| [`sources/evidence_registry.csv`](sources/evidence_registry.csv) | Source registry for the broader screen | Background |
| `outputs/019f8cc8-9802-7b01-8b3a-7fe5ef10eaa5/` | Legacy Excel scorecard | Background |

## Boundaries

| Rule | Requirement |
|---|---|
| Device selection | Documented live sensor-to-host path |
| Primary evaluation | Accuracy, continuity, latency, synchronization, and workflow stability |
| Not scored | Historical CSV export, wellness summaries, or app-only display |
| Data | No participant identifiers, clinical recordings, credentials, or raw study exports in this repository |
| Status | Research recommendation only; not a medical device or patient-alert authorization |
