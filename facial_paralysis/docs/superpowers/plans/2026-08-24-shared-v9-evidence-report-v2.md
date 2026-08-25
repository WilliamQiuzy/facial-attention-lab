# Shared V9 Evidence Report v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use test-driven-development for every behavior change. This plan is executed in the existing isolated `codex/facial-process-web-v9-integration` worktree; preserve all unrelated dirty changes.

**Goal:** Replace the ambiguous percentage-only result with a patient-safe, evidence-rich research report and make a successful inference impossible to submit twice.

**Architecture:** Keep the locked Shared V9 prediction unchanged. Extend only the browser-facing gateway contract with descriptive, inter-eye-normalized geometry already present in the model input; display browser-session keyframes from the original local video; and explicitly separate model score, descriptive evidence, collection quality, and unsupported clinical grades. A successful inference becomes an immutable session result with one prominent `View full report` action until the user starts a new session.

**Tech Stack:** Python 3.11 gateway and script-style `Check` tests; React 19, TypeScript, Vitest, browser-session Blob video; Docker Compose; Playwright/Chrome acceptance testing.

---

## Frozen scientific and privacy boundaries

- The displayed number is **Shared V9 class-1 research score: 48 / 100**: the arithmetic mean of three frozen members, compared with the fixed research cutpoint 50. The locked public release does not define class 1 as a patient-facing clinical label, so the UI must never call it patient probability, risk, severity, confidence, or accuracy.
- Descriptive brow, eye, and oral asymmetry values come from the same sampled MediaPipe geometry used by the 110D clinical branch, but they are not causal feature attribution, regional severity, House-Brackmann, Sunnybrook, or eFACE scores.
- Every evidence block states: **Measured movement observation — not a cause of the model score or a clinical severity grade.** UI copy and tests prohibit `reason`, `caused`, `contributed`, `abnormal`, and `affected side` claims.
- Action context frames are rendered from the already-held browser Blob at the registered hold midpoint and explicitly labelled recorded context, not a model-selected or causal frame. They are never returned by the server, logged, downloaded, or persisted.
- After success, no second inference is allowed for that session. Retry remains available only after an error, using the retained recording.
- Clinical scales appear only as House-Brackmann, Sunnybrook, and eFACE `Not assessed`; FaCE patient-reported outcome is `Not collected`. The product must never synthesize clinician or patient-reported scores.
- Retry policy is typed: transport/timeout/5xx keeps the recording and permits retry; permanent evidence/format/tracking/timing 4xx/422 requires correction or a new recording and cannot loop indefinitely.

## Task 1: Browser-safe evidence contract

**Files:**
- Modify: `src/preprocessing/faces_shared_v9_pipeline.py`
- Modify: `src/deployment/faces_shared_v9_gateway.py`
- Modify: `tests/test_faces_shared_v9_pipeline.py`
- Modify: `tests/test_faces_shared_v9_gateway.py`

- [ ] Write failing tests that require one closed, finite descriptive geometry row for every active action. Compute on the original-view geometry after the same centering, inter-eye scaling, and eye-axis rotation used by the 110D clinical branch. Include static absolute eye-aperture, brow-height, and oral-commissure asymmetry plus action-to-neutral movement measures: brow displacement, residual eye aperture/closure change, smile commissure displacement, pucker width change, and lower-lip movement. Never emit side, normal range, abnormality, or severity.
- [ ] Verify RED with the repository's approved Python interpreter and `tests/test_faces_shared_v9_pipeline.py`.
- [ ] Compute medians only from valid original-view sampled frames already admitted by the 26-of-32 gate; keep capture-side labels absent and round only during JSON serialization.
- [ ] Extend the strict browser response to schema v2 with frozen class-1 score semantics, descriptive-only evidence, and numeric context-frame midpoint timestamps; reject extra, missing, nonfinite, action-incompatible, or action-misaligned evidence.
- [ ] Verify GREEN with the pipeline and gateway direct test harnesses.

## Task 1b: End-to-end idempotency

**Files:**
- Modify: `facial_paralysis_web/src/model/inference.ts`
- Modify: `src/deployment/faces_shared_v9_gateway.py`
- Modify: both inference/gateway test files

- [ ] Derive an `Idempotency-Key` before fetch from the exact video SHA-256, canonical timeline bytes, candidate ID, release-manifest SHA, and preprocessing version. Set the synchronous `inFlightRef` before any asynchronous work.
- [ ] Add a bounded, expiring, in-memory gateway single-flight registry that stores only payload commitment plus the non-identifying response, never video bytes. Same key plus same payload waits for/replays the one result; same key plus different payload is rejected; failures do not become successful cached values.
- [ ] Test simultaneous equal requests, key/payload conflicts, rapid mouse/Enter/touch-equivalent activation, and server-completed/client-response-lost retry. Document that a service restart can recompute but cannot silently bind the key to different bytes.

## Task 2: Strict TypeScript contract and understandable score semantics

**Files:**
- Modify: `facial_paralysis_web/src/model/inference.ts`
- Modify: `facial_paralysis_web/src/model/inference.test.ts`

- [ ] Write failing tests for exact v2 evidence fields, class-1 score semantics, action alignment, finite geometry, and rejection of causal/HB/region-severity additions. Explicitly test scores `0`, `0.499`, `0.5`, and `1`, plus NaN, out-of-range values, member-mean drift, and release identity drift.
- [ ] Verify RED with `pnpm test:run -- src/model/inference.test.ts`.
- [ ] Implement strict parsing into typed report evidence.
- [ ] Verify GREEN and confirm malformed/expanded server payloads remain closed.

## Task 3: Immutable success state and report navigation

**Files:**
- Modify: `facial_paralysis_web/src/App.tsx`
- Modify: `facial_paralysis_web/src/App.test.tsx`

- [ ] Write failing tests proving a success removes `Run research analysis` from the DOM and replaces it with one prominent `View full research report` navigation link; rapid activation can call the endpoint only once; and only a confirmed `Start a new session` clears the immutable result and browser video.
- [ ] Preserve retry-after-error and stale-response invalidation tests.
- [ ] Add one canonical `#research-report` route with browser Back/Forward support, a report-specific document title, `<main>` landmark and focused report heading. `Back to session summary` returns to `#analysis`; confirmed `Start a new session` deletes the browser recording/report. Reloading with no in-memory result shows a private `Report not retained` empty state and never reruns inference.
- [ ] Verify GREEN with the App test suite.

## Task 4: Formal evidence report and browser-session keyframes

**Files:**
- Replace/expand: `facial_paralysis_web/src/components/ResultsView.tsx`
- Create: `facial_paralysis_web/src/components/ResultsView.test.tsx`
- Modify: `facial_paralysis_web/src/styles/app.css`

- [ ] Write failing component tests for six- and seven-action reports, midpoint context timestamps, score/cutpoint/member-mean explanation, relevant static and action-to-neutral geometry, missing Step 8, quality/provenance, HB/Sunnybrook/eFACE `Not assessed`, and FaCE `Not collected`.
- [ ] Implement report sections: executive summary; what the score means; action evidence gallery; descriptive geometry; capture quality; clinical-scale status; model provenance and limitations.
- [ ] Render context frames with one shared, revocable object URL and a controlled canvas extraction queue. Handle `loadedmetadata`, WebM `duration=Infinity`, clamped seek, `seeked`, black/empty frames, decode/seek timeout, and stale seek completion; never autoplay. Provide accessible action/time descriptions and a clear fallback. Revoke URL, zero canvases, and drop references on report close or new session.
- [ ] Add responsive, print-private clinical editorial styling, visible report navigation, keyboard focus, screen-reader headings/status, 200% zoom support, reduced-motion behavior, and no hidden horizontal overflow. Printing hides every context frame and displays an identifiable-video warning.
- [ ] Verify component and full frontend suites.

## Task 5: Deployment and edge acceptance

**Files:**
- Modify: `facial_paralysis_web/tests/browser/live_full_loop.py`
- Modify: `docs/results/facial_process_web_patient_safe_edge_acceptance_v2.md`
- Add screenshots only under: `docs/results/artifacts/facial_process_web_ui/`

- [ ] Build the web and gateway images and recreate only the affected services.
- [ ] Exercise rapid mouse double-click, Enter repeat, touch-equivalent activation, same-key concurrency/replay/conflict, server-finished/client-missed retry, permanent 4xx/422 versus retryable 5xx/network/timeout, stale responses, changing/clearing recording, six and seven active actions, missing context-frame decode, mobile/desktop, 200% zoom, screen reader landmarks, keyboard navigation, reduced motion, Back/Forward, and Reload.
- [ ] Confirm the report survives Back to session summary, disappears only after confirmed new session, reload never reruns, print hides video frames, there are no external requests, and every Blob URL/canvas plus all container/disk temporary artifacts are released.
- [ ] Capture desktop and mobile report screenshots with public/synthetic media only.
- [ ] Run Python, frontend, build, Docker health, `py_compile`, `git diff --check`, and secret/path scans.

## Task 6: Product review

- [ ] Give the product manager the live report, screenshots, exact model/clinical boundaries, and edge-test evidence.
- [ ] Resolve every blocking product or safety issue and rerun the affected tests.
- [ ] Leave the locally running application on the approved report page for user acceptance.
