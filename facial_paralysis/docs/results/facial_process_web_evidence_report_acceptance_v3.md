# Facial Process Web — Evidence Report Acceptance v3

**Date:** 2026-08-25
**Route:** Facial Process Web → raw-video gateway → MediaPipe → pinned Shared V9 BLV9-009
**Local acceptance URL:** `http://127.0.0.1:8080`

## Outcome

The percentage-only result has been replaced by a formal, browser-session
research report. A value such as `48 / 100` is now presented directly on the
MEEI facial-palsy-versus-healthy-control classification scale: higher values
lean toward the facial-palsy class, lower values toward the healthy-control
class, and the fixed cutpoint is 50.

A successful inference is immutable for the session. `Run research analysis`
is removed from the DOM and replaced by the prominent `View full research
report` action. Rapid repeated activation produces one browser request, and the
gateway also enforces a content-bound, bounded in-memory idempotency key.

The post-implementation product review approved the desktop report, mobile
report, locked-success interaction, evidence framing, clinical-claim boundary,
and no-face recovery path with no P0, P1, or P2 product blockers. This approval
applies to the Shared V9 research prototype, not Mayo clinical validation or a
diagnostic product.

## Report contents

1. Score meaning, fixed cutpoint, distance from the cutpoint, and current output
   class; individual ensemble-member scores are no longer shown.
2. Concise processing path: recorded actions → MediaPipe facial geometry →
   shared encoder → MEEI classification score.
3. One local context frame at the registered midpoint of every active hold.
4. Descriptive brow, eye, and oral geometry in inter-eye-distance units,
   including action-to-rest change where medically applicable. The report
   explains that `0.010` equals 1% of the inter-eye distance and that these
   values have no clinical normal range or severity meaning.
5. Plain-language recording coverage: one neutral baseline plus every active
   movement used in the score, usable face-tracking checkpoints, and optional
   Step 8 status. The 1–7 route therefore uses all seven recorded steps rather
   than “six of seven”; Step 1 supplies the resting reference.
6. Interpretation limits. The empty clinical-scale status table and technical
   model-provenance card were removed; the report now states once that the
   named clinical scales require separate clinician or patient assessment.

## Closed interaction and privacy behavior

| Boundary | Verified behavior |
|---|---|
| Rapid mouse/keyboard-equivalent activation | Synchronous client lock plus gateway single-flight; one inference only. |
| Same key / same bytes | Concurrent callers wait for or replay the same response. |
| Same key / different bytes | Rejected before preprocessing. |
| Successful session | Run control disappears; report remains available through Back/Forward without resubmission. |
| Save PDF | Opens the browser print/save flow exactly once; all available action context images are included and report controls are excluded. |
| Download recorded video | Saves the in-memory source recording to the current device with a fixed de-identified filename, then revokes the temporary download URL. |
| Back to summary | Original browser recording preview and the locked report remain in memory. |
| Reload/direct report URL | Shows `Report not retained`; never restores media or reruns inference. |
| Transient network/5xx/timeout | Same recording can be retried. |
| Permanent timing/format/tracking/geometry 4xx/422 | Resubmission is disabled; user is directed to correct and record again. |
| Context frames | One additional revocable Blob URL while the report is open; it is revoked on report close. |
| Print | Identifiable context frames are retained and a prominent secure-storage warning is printed. |
| Network boundary | Browser acceptance observed no HTTP(S) origin other than the local application origin. |
| Container media | No persistent media volume; request media remains tmpfs/request-scoped. |

## Verification evidence

- Frontend unit/component/strict-contract suite: **113/113 passed**.
- TypeScript check and production build: **passed**.
- Raw-video pipeline: **12/12 passed**.
- Gateway including idempotency/concurrency: **9/9 passed**.
- Deployment contract: **9/9 passed**.
- Docker image build reran the same **113/113** frontend tests and all three
  services became healthy.
- Browser/Docker acceptance with public synthetic media:
  - steps 1–7, strict v2 success fixture: report, six context frames, rapid
    double activation, one-call Save PDF with embedded context images, a
    de-identified local recording download, Back/Forward, reload privacy, Blob
    URL lifecycle, and same-origin network gate all passed;
  - steps 1–8, strict v2 success fixture at a 390 px mobile viewport: seven
    action cards and responsive report passed;
  - steps 1–7 through the real gateway with a no-face synthetic camera:
    permanent action-level tracking rejection and retained recovery controls
    passed. This also verifies that the browser and gateway agree on the live
    idempotency-key construction.

Strict synthetic success responses validate the complete browser report
contract and interaction flow; they are not model-performance evidence. The
real model/pipeline path is covered by the Python contract suite and the live
no-face rejection. No identifiable person was retransmitted for this report
acceptance pass.

The final four-page synthetic A4 print acceptance retains all six action context images.
Poppler image inspection verifies that raster evidence is embedded in the PDF,
and rendered-page review checks every page for clipping, overlap, or orphaned
section headings. The printed warning explicitly states that the report
contains identifiable facial context images and must be stored securely.

## Defects found and fixed

1. The old label made a score look like patient probability or confidence.
   The v2 response now binds the exact MEEI development endpoint semantics and
   the report explains both classes and the external-validity limit.
2. Success originally left the Run button available. The result is now
   immutable, the run control is removed, and a formal report entry replaces it.
3. Client-only duplicate protection could not cover response loss or two
   callers. A content-bound gateway single-flight/replay layer now complements
   the synchronous UI lock.
4. Returning from the report initially remounted the capture workspace and hid
   the prior preview. The session summary now remains mounted but hidden while
   the report is open, preserving the page-scoped recording and report.
5. The first browser acceptance run was interrupted by the local keep-running
   supervisor reconciling different image tags after a base-only manual build.
   Current images were retagged to the supervisor's fixed local tags and the
   stack was started with the same Compose override. The maintenance tool was
   audited separately: it does not call Compose or restart containers.
6. Initial screenshots were captured before asynchronous context-frame
   extraction completed. Browser acceptance now requires every expected image
   before capture and fails if the frame lifecycle does not complete.
7. A normally off-screen skip link appeared as a black floating control in
   stitched screenshots. It is now clipped to 1 px until true keyboard focus;
   opening a report moves focus to its title, and browser acceptance verifies
   both states.
8. Black-frame detection originally inspected only the upper-left corner and
   could discard a valid image with dark corners. It now samples four corners
   and the center, with a regression case for a valid center and dark edges.
9. The inter-eye-normalized evidence values lacked a user-readable unit. The
   report now states the percentage conversion and explicitly disclaims any
   clinical range or severity interpretation.
10. The report had no persistent handoff action. A prominent `Save PDF` control
    invokes the browser print flow, includes the recorded context images, and
    uses print-specific pagination that keeps compact section headings with
    their content.
11. The source recording could only be replayed in the transient browser
    session. `Download recorded video` now exports the exact in-memory file to
    the current device using a fixed filename that cannot expose an uploaded
    patient/MRN filename. The temporary Blob URL is revoked immediately after
    the download is initiated; no server-side media storage was added.

## Public-safe visual evidence

- [Desktop seven-step research report](artifacts/facial_process_web_ui/evidence-report-synthetic-desktop.png)
- [Mobile eight-step research report](artifacts/facial_process_web_ui/evidence-report-synthetic-mobile.png)
- [Synthetic report PDF with context images](artifacts/facial_process_web_ui/facial-process-shared-v9-synthetic-report-with-images.pdf)
- [Real-gateway no-face rejection](artifacts/facial_process_web_ui/evidence-report-no-face-rejection.png)

All three files use a non-person synthetic camera source. They contain no
patient or user face.

## Remaining boundary

This is a strong research-prototype interaction and deployment acceptance, not
a clinical product release. Institutional authentication/TLS, approved PHI
governance, monitoring/high availability, and participant-disjoint Mayo labels
plus controls are still required before clinical validation or patient-facing
diagnostic use.
