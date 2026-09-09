# Vitestro Phlebotomy Safety

This project reviews live sensing devices for presyncope and discomfort monitoring during automated venipuncture.

## Artifacts

| Artifact | Purpose | Status |
|---|---|---|
| [`docs/device_evaluation.md`](docs/device_evaluation.md) | Seven categories with ranked candidates and other screened products. Live transmission, price, patient fit and interface conditions are listed separately. | Current |
| [`data/device_feature_matrix.csv`](data/device_feature_matrix.csv) | Broader 18-device screen | Background |
| [`data/modality_evidence_matrix.csv`](data/modality_evidence_matrix.csv) | Modality evidence and failure modes | Background |
| [`sources/evidence_registry.csv`](sources/evidence_registry.csv) | Source registry for the broader screen | Background |
| `outputs/019f8cc8-9802-7b01-8b3a-7fe5ef10eaa5/` | Legacy Excel scorecard | Background |

## Boundaries

| Rule | Requirement |
|---|---|
| Device selection | Documented live sensor-to-host interfaces and Conditional interfaces remain in the comparison. Conditional entries state their access or integration requirements. |
| Primary evaluation | Signal quality, continuity, latency, synchronization and workflow reliability. |
| Patient fit | Sizing, skin contact, comfort, cleaning and handling across shared-patient use. No single contact device provides universal fit. |
| Ranking | Proposed evaluation order balances acquisition quality, patient workflow, integration, price and availability. It does not rank measured clinical accuracy. Apple is a baseline. Finger EDA is a separate add-on. |
| Camera comparison | Working distance, field of view, nominal facial pixels and synchronization are compared separately. Priced subtotals identify included hardware; reference lenses are not validated assemblies. |
| Not scored | Historical downloads, wellness summaries and vendor-app-only displays do not qualify as live access. |
| Data | Do not store patient identifiers, clinical recordings, credentials or raw study exports in this repository. |
| Status | This is a research review. It does not authorize patient alerts or clinical use. |
