# Facial Attention Lab

An independent, research-only frontend for exploring how facial-scar visual-attention evidence could eventually support clinician and patient conversations around facial reconstruction.

The visual system is inspired by Mayo Clinic's clarity, restraint, editorial typography, blue palette, and generous whitespace. It does not use Mayo Clinic logos, proprietary fonts, or patient photography, and it is not an official Mayo Clinic product.

## Current status

This build is an interface prototype—not a diagnostic tool, a clinical decision aid, an eye-tracking result, or a deployed model.

| Capability | Current state |
| --- | --- |
| Clinician workflow | Interactive synthetic demo |
| Patient explanation | Printable plain-language view |
| Human gaze | Not present |
| Model inference | Not connected |
| Patient records / PHI | Not present |
| Upload | Intentionally unavailable |
| Clinical use | Blocked pending validation and governance |

## Product surfaces

- `/` — research overview and clinician/patient pathways
- `/cases` — searchable synthetic worklist and upload boundary
- `/analysis?case=demo-001` — attention-map controls, proposal-aligned metrics, provenance, QC gates, and clinician-to-patient handoff
- `/patient-report?case=demo-001` — printable, plain-language conversation guide
- `/model` — future service boundary, origin taxonomy, endpoints, and promotion gates
- `/methods` — metric definitions, provenance requirements, interpretation limits, privacy, and governance

Every simulated map is visibly labeled `SIMULATED — NOT HUMAN GAZE`. The two images are different AI-generated identities and are always described as an unpaired interface demonstration—not as a before/after result.

## Run locally

Requirements: Node.js 22.12–24.x and pnpm 11.x. The repository records the
development package manager as `pnpm@11.9.0`.

```bash
cd facial_paralysis/facial_defect_web
pnpm install --frozen-lockfile
pnpm dev
```

Open the local URL printed by Vite. The default build is deterministic, makes no network requests, and does not use browser storage.

## Verification

```bash
pnpm typecheck
pnpm test:run
pnpm build
```

The Vite configuration verifies the actual SHA-256 bytes of exactly ten approved synthetic source files both when configuration loads and when a production build starts. A missing, changed, additional, or non-synthetic source fails the build.

## Future model connection

Copy `.env.example` to `.env.local` only when an approved research service exists:

```dotenv
VITE_ENABLE_CONNECTED_MODE=true
VITE_ATTENTION_API_URL=https://approved-research-service.example
```

Connected mode is opt-in and fail-closed. The current analysis page refuses to substitute its mock fixture once connected mode is enabled. The typed service seam keeps empirical gaze and model predictions separate:

| Operation | Endpoint | Required origin |
| --- | --- | --- |
| Observed aggregate | `POST /api/v1/attention-analyses` | `observed_gaze` |
| Salience prediction | `POST /api/v1/salience-predictions` | `model_prediction` |

Each request carries an allowlisted synthetic `assetId` and its registered SHA-256. The response must echo that exact asset binding and include a non-empty `analysisId`, the expected `origin`, `capabilityStatus: "research_unvalidated"`, a versioned ROI with `reviewStatus: "reviewed"`, and `quality.status: "eligible"`. Observed-gaze responses must meet their declared `protocolMinimum`; prediction responses must name an explicit model and version. Any unknown asset, changed hash, missing field, ineligible result, underpowered sample, or operation mismatch throws instead of rendering a result.

Before a connected result is displayable, the production contract should also require immutable input hashes, pairing status, ROI version/review state, model and analysis versions, uncertainty, calibration/QC evidence, cohort metadata, and governance approval. These are documented as promotion gates in the Model & data and Methods pages.

## Safety and scope

- Attention describes where and when gaze landed under a defined protocol. It does not reveal emotion, judgment, stigma, attractiveness, or procedure success.
- No patient image, identifier, clinical record, real-image synthesis output, human gaze, or model output is bundled.
- The UI does not diagnose scar severity, recommend treatment, or replace patient-reported outcomes.
- Human-study data remains subject to institutional protocol, consent, privacy, retention, access-control, and cohort-quality decisions.

Implementation and verification decisions, including the exact synthetic-asset provenance allowlist, are recorded in `audits/`.
