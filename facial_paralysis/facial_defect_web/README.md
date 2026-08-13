# Facial Reconstruction Imaging

A clinician-facing, session-only frontend prototype for a facial
reconstruction photo workflow. The primary path lets the team rehearse
creating a synthetic/test patient record, adding a dated photo visit, taking
or uploading one standardized frontal photograph, confirming photo quality,
running a local simulation, reviewing image-first results, and recording a
simple clinician decision. Legacy case, batch, model-comparison, and research
review tools remain available from Help. No trained facial-defect spatial
attention model or real patient backend exists in this repository yet.

The interface uses a restrained Mayo-inspired visual system, but it is an independent research prototype. It does not use Mayo Clinic logos or proprietary fonts and is not an official clinical product.

## Current capability boundary

| Surface | Current state |
| --- | --- |
| Clinician workflow | Patient record → photo visit → frontal capture → four quality checks → local simulation → image-first result → review |
| Patient records | Synthetic/test records in React memory only; no authentication, persistence, server record, or real-PHI support |
| Camera and upload | Native camera/file input; validated JPEG/PNG/WebP bytes remain in an in-memory session vault |
| Patient result | Original, simulated overlay, density field with a photo-matched face contour, and face-relative AOI summary on one scrolling page |
| Clinician review | `Reviewed` or `Repeat photo`; the current simulated result remains exact-bound to its capture |
| Case catalog | 10 hash-locked AI-generated synthetic identities |
| Legacy research workbench | Case-to-Run, batch, model comparison, structured research review, and gated patient-explanation rehearsal |
| Source image binding | Verified full-image identity retained only for provenance and fail-closed validation; one-click recovery if malformed |
| Single inference | Deterministic local mock engine by default |
| Mock AOI summary | Fixed-template simulated point-weight shares; center-assigned, radius ignored, and UI-rehearsal only |
| Connected AOI summary | Unavailable and fail-closed in version 1; the response explicitly declares registration geometry unavailable |
| Batch jobs | Session-only review, explicit exclusions, atomic submission, cancellation, and retry lineage |
| Simulation comparison | Two mock profiles on the same exact synthetic asset and full-image source binding |
| Result review | Structured research-demo review with independent actor roles |
| Patient explanation | Available only for an approved, current mock result |
| Export | Client-side, non-PHI JSON whitelist; never persisted |
| Current `facial_paralysis` system | Palsy probability and eyes/mouth ordinal outputs plus separate label-free FACES research measures; not connected |
| Connected seam | Explicit opt-in synthetic spatial contract rehearsal; no attention checkpoint or patient-media integration exists |
| Human gaze | Not present |
| Real patient backend and media gateway | Not present |
| Clinical use | Blocked |

Every mock output carries the permanent label `SIMULATED — NOT HUMAN GAZE`. A connected result must instead carry `MODEL PREDICTION — RESEARCH UNVALIDATED — NOT HUMAN GAZE — CLINICAL USE BLOCKED` and can never unlock the patient preview in the current governance model.

The patient-visit simulation performs same-origin, on-device face-landmark
preprocessing with the pinned MediaPipe Face Landmarker bundle. It requires
exactly one valid face and binds the resulting contour to the decoded image
SHA-256, intrinsic dimensions, and non-mirrored capture protocol. The
illustrative attention anchors are transformed into that same source-image
coordinate system before the overlay, density view, and face-relative summary
are rendered. No generic face outline is used as a fallback. Failed,
multi-face, invalid, or mismatched registration asks the user to replace the
photograph and records no result. This contour is a display-location reference,
not a defect boundary, clinical segmentation, diagnosis, or attention-model
output.

The intended future connected contract is a population-level predicted
observer-attention spatial density field. The current browser generates a deterministic
synthetic field solely to test that interface. Mock AOIs are fixed-template simulated
point-weight shares: each point's intensity weight is assigned by its center, and radius
is ignored. They are not raster or density-kernel integrals. The facial-subsite rows
(including outside-template) and the hemiface rows are two separate 100% partitions;
the central facial triangle is an overlapping, non-additive reference. These summaries
do not modify the simulation and are not gaze duration, fixation count, severity,
defect localization, or treatment effect. Patient left appears on the viewer's right
in a frontal image.

Connected version 1 requires
`registration_geometry_unavailable_v1`. It does not claim that landmarks were
supplied and carries no landmarks or polygons, source dimensions, orientation or
mirror metadata, or registration quality control. Connected AOI reporting is
therefore unavailable and fails closed until a later contract supplies and validates
that geometry.

The current `facial_paralysis` system is separate. Its checked-in scoring path returns
palsy probability and eyes/mouth ordinal outputs. Separate analyses derive
landmark-based left-right asymmetry, eye-closure dynamics, and a Mayo FACES label-free
research measurement summary. No checked-in checkpoint includes an HB task; the
architecture can support one, but Mayo HB calibration has not started. The
FACES-action-derived regional research measures are not a validated eFACE, Sunnybrook,
or HB composite or grade. None of these outputs is a `HeatmapPoint[]`, and the browser
never converts them into one.
`warmstart_v2_attention` uses temporal frame-pooling attention, not spatial facial
attention. `warmstart_v4_expanded` contains a `coarse3` head, but the current
`scripts/predict.py` path does not export that output.

## Image-first Run result

After a successful Run, Analysis and exact Run Detail show one continuous vertical
story in this order:

1. source image;
2. signal field; legacy or connected results without exact registered geometry
   explicitly show that a face outline is unavailable rather than drawing a
   generic template;
3. source-plus-signal overlay;
4. mock-only fixed-template point-weight AOI summary with patient laterality (a
   connected result shows AOI unavailable);
5. written interpretation boundaries, next action, and technical details.

All visual methods are present on the same page and are read by scrolling. No tab,
carousel, crop control, or display-mode choice is required in the clinician result
flow. The compact patient explanation keeps its shorter Summary/Separate/Overlay
choice.

## Operational routes

| Route | Purpose |
| --- | --- |
| `/` | Redirects to the clinician patient list |
| `/patients` | Search session-only synthetic/test patient records |
| `/patients/new` | Create one session-only synthetic/test record and initial photo visit |
| `/patients/:patientId` | Patient identity and chronological photo-visit timeline |
| `/patients/:patientId/visits/new` | Add a dated preoperative, postoperative, or follow-up photo visit |
| `/patients/:patientId/visits/:visitId` | Capture, quality confirmation, processing, image result, and simple review |
| `/reviews` | Simulated patient results awaiting a clinician decision |
| `/cases` | Legacy research-only synthetic case catalog |
| `/cases/:caseId/roi` | Source image binding status and one-click full-image recovery (legacy route name retained for compatibility) |
| `/analysis?case=:caseId[&run=:runId]` | Configure one case or reopen one exact in-session result |
| `/runs` | Case-led recent simulations |
| `/runs/:runId` | Immutable attempt lineage and provenance |
| `/jobs` | Multi-case preflight and batch job controls |
| `/models[?case=:caseId]` | Compare two fixed mock profiles and their grouped point-weight summaries on one exact synthetic asset |
| `/research/reviews` | Legacy structured research-result queue and blockers |
| `/research/reviews/new?run=:runId&attempt=:attemptId` | Create a research review for one exact result |
| `/research/reviews/:reviewId` | Request changes, resubmit, approve, or revoke a research review |
| `/patient-report?review=:reviewId` | Gated single-image simulated explanation and safe JSON export |
| `/about` | Clinician Help and links to research-only tools |
| `/methods`, `/integration` | Research method and integration documentation |

The former research-review deep link `/reviews/:reviewId` redirects into the
`/research/reviews/:reviewId` namespace. Dynamic IDs are authoritative.
Missing, duplicate, unknown, prototype-key, stale, or refresh-lost IDs fail
closed; no route substitutes a default fixture.

## Interactive state machines

```text
Source image binding
missing | malformed | partial | unapproved | superseded
  -> restoreFullImage -> verified approved full image
verified approved full image -> restoreFullImage -> no-op

Run attempt
draft -> validating -> blocked
                  -> queued -> blocked
                            -> running -> succeeded | failed | cancelled

Batch attempt
draft -> preflighting -> ready | blocked
ready -> queued -> running -> completed | completed_with_failures | failed | cancelled

Result review
awaiting_review -> approved_for_research -> revoked
                -> changes_requested -> awaiting_review
```

Retries create new immutable attempts with parent lineage; they never overwrite a failed or cancelled attempt. Every single-run, retry, and batch launch revalidates the current verified full-image source binding immediately before gateway work; a changed binding blocks the attempt without a request. Restoring a source binding never starts inference. It increments the binding version when recovery is needed, preserves prior run history, and marks incompatible current results stale. Review approval means approval for this synthetic research demonstration only—not clinical approval.

## Run locally

Requirements: Node.js 22.12–24.x and pnpm 11.x. The lockfile records `pnpm@11.9.0`.

```bash
cd facial_paralysis/facial_defect_web
pnpm install --frozen-lockfile
pnpm dev
```

Open [http://127.0.0.1:5173/](http://127.0.0.1:5173/). The default mode is
deterministic and local. It performs no external model inference, writes
nothing to browser persistence, and keeps selected test-image bytes inside
the current in-memory session.

## Verification

```bash
pnpm typecheck
pnpm test:run
pnpm build
```

Vite verifies the source bytes of the exact 10 approved assets when configuration loads and again before production build. Missing, changed, non-synthetic, or unapproved assets fail the build. The final bundle must contain exactly those 10 PNG byte hashes and no real-image or `facial_paralysis` research asset.

## Future model gateway

The pages and reducers use one `WorkbenchGateway` port. The local
`MockWorkbenchGateway` keeps the internal simulation binding. The opt-in
`HttpWorkbenchGateway` converts that verified full-image binding into a distinct,
minimal connected wire contract.

Connected mode is deliberately explicit:

```dotenv
VITE_ENABLE_CONNECTED_MODE=true
VITE_ATTENTION_API_URL=https://approved-research-service.example
```

The HTTP gateway calls `POST /api/v1/workbench/inference`. This is a synthetic
spatial contract rehearsal, not an implemented model integration. Its exact
`ConnectedAttentionRequestV1` sends only the request-profile version, run and attempt
IDs, case and asset IDs, source SHA-256, and the full-image source-binding
ID/version/geometry. It does not send the internal mock model version or mode,
threshold, smoothing, configuration/hash/fingerprint, mock metrics, or ROI approval.

The response must echo that exact connected request identity and report a strict
model identity: model ID and version, 64-hex artifact SHA-256, preprocessing version,
calibration version, and display-scale ID. It must also pass origin, capability,
watermark, explicit spatial-point, attention-semantics, source-binding-integrity,
registration-unavailable, quality, and provenance validation. The gateway caps the
display representation at 4,096 points and maps the original verified internal
binding back into the in-memory result only after validation. The result payload
declares whether observed gaze is included separately from training-data provenance;
this rehearsal requires `observedGazePayloadIncluded: false` and
`trainingDataProvenance: not_disclosed`. Timeout, abort, malformed output, or any
mismatch fails closed; connected mode never falls back to the mock engine.

The current `HeatmapPoint[]` is only a provisional display-points wire
representation, not a frozen scientific raster or radial-basis-field contract.
Patient-media reference and authorization are also unimplemented. The model and
backend teams must jointly freeze the media reference, coordinate frame, spatial
representation, normalization, display scale, and production request/response
schema before integration. No actual observer-attention checkpoint or output exists
in this repository today.

This seam is reserved for a separately defined spatial-attention model/API. A response
shaped like the current functional-assessment system—palsy probability or ordinal
eyes/mouth outputs without valid spatial points—fails as `MALFORMED_RESPONSE`; the web
app never invents a heatmap from severity scores, logits, temporal pooling weights, or
occlusion analyses.

A clinician-defined surgical-site mask is not part of the current request. If added
later, it must be a separate, versioned contextual annotation. It may summarize how
much completed spatial density falls inside the marked site, but it does not stand in
for attention or severity and does not alter prediction unless a future validated
contract explicitly declares the mask as an input.

Connected results remain `research_unvalidated`, not human gaze, and clinically blocked. Internal research review does not make them eligible for patient preview or export. A future patient-use path requires a separate validated model, governance policy, and server-side authorization boundary.

## Patient-preview and export gate

The patient explanation is a simulation of a future conversation surface, not an individual prediction. It requires all of the following in the current in-memory session:

- one exact active succeeded attempt whose result is still current;
- canonical synthetic asset and current verified full-image source binding;
- `origin: mock_simulation` and `capabilityStatus: simulated_ui_only`;
- valid offline deterministic provenance and all research display quality gates;
- a structured, role-separated review ending in `approved_for_research`.

The downloadable `application/json` manifest uses an explicit whitelist. It includes only the asset hash, model and source-binding versions, result digest, safe review decision/event, quality status, and disclaimers. It excludes review notes, actors, operational run/review IDs, names, dates, medical-record fields, and all other free text. The Blob URL is revoked immediately after the browser download is triggered. No PDF, print workflow, browser persistence, or network export exists.

## Safety and asset scope

- All 10 images are separate AI-generated identities and standalone, unpaired demonstrations.
- The UI must never describe them as before/after, the same patient, postoperative change, treatment outcome, or a scientific cross-case comparison.
- No source under `output/real`, no facial-paralysis research image, no
  patient-derived feature, no clinical label, and no trained clinical or
  observer-attention model weight is imported. The only checked-in model bundle
  is the public MediaPipe Face Landmarker used for on-device registration.
- Attention output cannot infer emotion, judgment, stigma, attractiveness, surgical success, diagnosis, or treatment need.
- Workspace state is React memory only and resets on refresh or the explicit session-reset action.

## Method references

These sources inform terminology and transparency boundaries; they do not validate this
prototype or its synthetic output:

- [Facial palsy eye-tracking study (PMID 40242878)](https://pubmed.ncbi.nlm.nih.gov/40242878/)
- [Visual saliency and acquired facial differences (PMC10118307)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10118307/)
- [FDA transparency principles for machine-learning-enabled medical devices](https://www.fda.gov/medical-devices/software-medical-device-samd/transparency-machine-learning-enabled-medical-devices-guiding-principles)

Approved provenance and browser/build evidence live in [`audits/`](./audits/).
