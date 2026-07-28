# Release verification — 2026-07-23

This record covers the local synthetic clinician-interface build in
`facial_paralysis/facial_defect_web/`. It is not evidence of model validation,
clinical authorization, patient-data readiness, surgical outcome prediction, or
public deployment.

## Delivered interaction

- The default clinician path no longer asks a doctor to draw, approve, filter, or
  interpret an ROI.
- Every default case retains an immutable full-image internal bound only for exact
  input identity, provenance, stale-result detection, review integrity, and export
  policy. It is not a model finding and is not shown as a clinician control.
- A successful Run now presents one normal vertically scrollable story in this
  order: source image, signal-only field, overlay, regional visual summary, safety
  explanation, then the next lifecycle action.
- Analysis and exact Run Detail use the same image-first stack. Patient explanation
  retains the shorter Summary-first choice among Summary, Separate, and Overlay.
- The connected spatial-output labels follow the validated output origin. A valid
  all-zero connected field does not fall back to simulated-result wording.

## Model and result boundary

The separate functional-assessment research system emits non-spatial,
severity-oriented outputs. It does not emit the web workbench's
`HeatmapPoint[]` spatial contract and is not connected to this interface.

The current web result is deterministic synthetic demonstration output containing a
mock heatmap, four display metrics, and provenance. A connected response without the
strict spatial heatmap envelope, including a severity-shaped response, fails closed
as `MALFORMED_RESPONSE`; the interface never manufactures spatial points from
severity logits. A future lesion mask or spatial-attention result requires a new,
explicit model/API contract.

All ten images are AI-generated, standalone, unpaired synthetic identities. The
interface does not make before/after, healing, treatment-effect, or surgical-outcome
claims.

## Fresh automated gates

| Gate | Fresh result |
| --- | --- |
| Focused provenance and Run Detail regression | 2 files, 44 tests passed, 0 failed |
| Focused Analysis routing regression | 1 file, 38 tests passed, 0 failed |
| `CI=1 pnpm exec vitest run --maxWorkers=4` | 37 files, 546 tests passed, 0 failed |
| `pnpm typecheck` | Passed |
| `pnpm build` | Passed; 1,845 modules transformed |
| `git diff --check` | Passed |
| Local service | `http://127.0.0.1:5173/` returned HTTP 200 |

The test suite includes exact-target rejection for stale, revoked, malformed,
historical, and deterministic-tampered evidence; connected all-zero provenance
labelling; and rejection of the current severity-model-shaped output at the spatial
gateway boundary.

## Chromium interaction verification

A fresh Playwright Chromium session exercised:

`Cases → Run → completed Analysis → exact Run Detail → Review → approved patient explanation`

The audit passed **32 of 32 checks** at `1440 × 1050`, `768 × 900`, and
`390 × 844`.

- Source, signal, overlay, and regional summary shared one vertical result story and
  preceded safety text and actions.
- No result-mode tabs or visible ROI workflow appeared in the clinician story.
- No horizontal overflow occurred at any checked viewport.
- There were no console warnings/errors, page exceptions, failed/error responses,
  external requests, cookies, IndexedDB entries, or browser-storage writes.
- The browser made 86 requests, all to the local service.

Structured evidence is `/tmp/facial_image_first_browser_audit.json`; detailed
artifact paths are recorded in `audits/browser-verification.md`.

## Production asset audit

- Built PNG files: **10**
- Unique built PNG hashes: **10**
- Approved hashes: **10**
- Missing or extra hashes: **0**
- Exact hash-set match: **yes**
- Forbidden build-string hits: **0**

The forbidden scan covered real-output paths, the facial-paralysis research-output
path, retired demo IDs, and retired inference endpoints.

## Protected dirty-tree boundary

The final exact-file-filter fingerprints match the amended protected-scope baseline
byte for byte:

- Protected porcelain entries: **134**
- Status SHA-256:
  `300087bb410221e7eff97b74becbf15ff76fac5aa735973d045913331ed9f489`
- Tracked binary-diff SHA-256:
  `5aa4016b6c184dfb6089ea8717cabd961163404f451b3c3a26352135dbf7e860`
- Recursive untracked-content SHA-256:
  `8ced19d11fc89ab6da6d13df8993802aeb6d9b1a6e3919352ab377433245a458`

No unrelated dirty-tree change is attributed to this implementation. No file was
staged or committed.

## Independent reviews

- Final specification review: **APPROVED**, no remaining Critical or Important
  finding.
- Final code/safety review: one connected-origin wording edge was found, repaired
  with red/green regressions, and re-reviewed as **READY** with no remaining Critical
  or Important finding.
- Final visual review: **VISUAL READY**, no Critical or Important finding. The
  reviewer noted only non-blocking mobile summary density and the intentional long
  clinician scroll created by showing every visual before the review action.
