# Facial Process End-to-End Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the FACES capture site from a small set of happy-path checks into an auditable, repeatable end-to-end release gate covering every user-reachable journey state and recovery path.

**Architecture:** Keep Vitest for component/state contracts and native Python Playwright for browser, permissions, files, camera, network, responsive, and full-loop verification. Add a machine-readable coverage matrix and a single suite runner so a release cannot silently omit a required scenario. Any product change follows red-green TDD and is deployed only to the isolated `127.0.0.1:8081` candidate.

**Tech Stack:** React 19, TypeScript, Vitest, Python Playwright, Docker Compose/Nginx, Chromium/Firefox/WebKit.

---

### Task 1: Inventory and coverage contract

**Files:**
- Create: `facial_paralysis/docs/testing/facial-process-e2e-matrix.md`
- Create: `facial_paralysis/facial_paralysis_web/tests/browser/acceptance_manifest.json`
- Inspect: `facial_paralysis/facial_paralysis_web/src/App.tsx`
- Inspect: `facial_paralysis/facial_paralysis_web/src/components/GuidedCaptureWorkspace.tsx`
- Inspect: `facial_paralysis/facial_paralysis_web/src/components/MediaCapture.tsx`

- [x] List preparation, camera setup, source switching, recording, analysis, report, refresh, and reset states.
- [x] Define required cases for both seven- and eight-step scripts.
- [x] Define desktop, tablet, phone portrait, phone landscape, and three-browser coverage.
- [x] Map every case to one executable test module and mark real-backend versus mocked-boundary coverage.

### Task 2: Journey and recovery edge suite

**Files:**
- Create: `facial_paralysis/facial_paralysis_web/tests/browser/journey_edge_acceptance.py`
- Modify: `facial_paralysis/facial_paralysis_web/tests/browser/capture_source_recovery_acceptance.py`
- Test: `facial_paralysis/facial_paralysis_web/src/App.test.tsx`
- Test: `facial_paralysis/facial_paralysis_web/src/components/GuidedCaptureWorkspace.test.tsx`

- [x] Add assertions for missing Step 8, both Step 8 choices, back/forward preservation, repeat switching, invalid/cancelled uploads, and refresh reset.
- [x] Add camera permission denied, missing device, unsupported speech, double-click, stop/discard, retry, and focus recovery cases.
- [x] Run the new suite against current code and record each genuine failure before any production edit.
- [x] Implement only the minimum UI/state fixes required by failing tests.
- [x] Re-run focused component and browser tests until green.

### Task 3: Upload and network failure matrix

**Files:**
- Create: `facial_paralysis/facial_paralysis_web/tests/browser/upload_network_edge_acceptance.py`
- Test: `facial_paralysis/facial_paralysis_web/src/components/MediaCapture.test.tsx`
- Test: `facial_paralysis/facial_paralysis_web/src/model/inference.test.ts`

- [x] Cover accepted extensions/MIME pairs, misleading MIME, empty/oversized video, replacement, cancel, valid seven/eight-step sidecars, malformed JSON, missing actions, digest mismatch, and oversized sidecar.
- [x] Cover readiness delay, connection abort, readiness 500/retry, inference 400/422/500, retryable/non-retryable responses, malformed success response, and rapid duplicate submit.
- [x] Assert every failure retains the appropriate recoverable input and exposes one clear recovery action.

### Task 4: Real browser recording and report loop

**Files:**
- Modify: `facial_paralysis/facial_paralysis_web/tests/browser/live_full_loop.py`
- Create: `facial_paralysis/facial_paralysis_web/tests/browser/run_release_acceptance.py`

- [x] Update the live loop to traverse the five-stage wizard before enabling the camera.
- [x] Generate a non-clinical Y4M fixture outside Git and exercise real `MediaRecorder` for seven and eight steps.
- [x] Verify one and only one inference request, idempotency key, report navigation, PDF download, video download, Blob URL cleanup, refresh disposal, and no external origins.
- [x] Run both strict mocked-success report paths and at least one real-backend request path.

### Task 5: Cross-browser, accessibility, stability, and release gate

**Files:**
- Modify: `facial_paralysis/facial_paralysis_web/tests/browser/responsive_capture_acceptance.py`
- Create: `facial_paralysis/facial_paralysis_web/tests/browser/accessibility_runtime_acceptance.py`
- Modify: `facial_paralysis/facial_paralysis_web/README.md`

- [x] Fail on unexpected console errors and uncaught page errors throughout every browser test.
- [x] Verify keyboard-only source switching, visible focus, live status announcements, touch target size, no horizontal overflow, reduced motion, and high-contrast visibility.
- [x] Run Chromium/Firefox/WebKit across all six existing viewports and repeat critical recovery flows five times.
- [x] Run all Vitest tests, typecheck/build, Docker build, live `8081` suite, health/readiness, and legacy `8080` identity/hash invariance.
- [x] Commit and push only after the full gate is green and the worktree is clean.
