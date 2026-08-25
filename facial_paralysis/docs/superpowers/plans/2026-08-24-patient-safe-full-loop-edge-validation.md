# Patient-Safe Full-Loop Edge Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ensure every defined Facial Process Web capture, upload, preprocessing, inference, cleanup, and retry outcome either produces an accepted Shared V9 result or gives the participant a specific, safe, actionable recovery message.

**Architecture:** Keep the model and medically meaningful action gates frozen. Replace generic transport/preprocessing failures with a closed, non-identifying error taxonomy; expose action-level tracking context without paths or patient identifiers; reject impossible uploads in the browser before transfer; and verify success and failure routes through unit, HTTP-boundary, browser, container, and storage tests.

**Tech Stack:** React 19, TypeScript, Vitest, FastAPI, Python direct `Check` harness, MediaPipe, Docker Compose, Playwright.

---

### Task 1: Freeze the patient-visible failure contract

**Files:**
- Modify: `src/preprocessing/faces_shared_v9_pipeline.py`
- Modify: `src/deployment/faces_shared_v9_gateway.py`
- Test: `tests/test_faces_shared_v9_pipeline.py`
- Test: `tests/test_faces_shared_v9_gateway.py`

- [ ] Add failing tests for all safe preprocessing categories, all seven/eight tracking-action identities, 25/26/32 sample boundaries, and prohibition of paths or raw exception text.
- [ ] Run both direct test files and confirm the new cases fail for missing typed evidence.
- [ ] Add a typed tracking failure carrying only canonical action, observed count, and required count; implement a closed gateway error classifier with a generic fallback.
- [ ] Re-run both direct test files and confirm all cases pass.

### Task 2: Give actionable feedback for every HTTP/runtime outcome

**Files:**
- Modify: `facial_paralysis_web/src/model/inference.ts`
- Test: `facial_paralysis_web/src/model/inference.test.ts`
- Modify: `facial_paralysis_web/src/App.tsx`
- Test: `facial_paralysis_web/src/App.test.tsx`

- [ ] Add failing table-driven tests for closed 400/413/415/422/502/503 categories, tracking-action guidance, malformed/oversized/unknown error bodies, network failure, and stale responses.
- [ ] Verify the tests fail because current handling covers only four 422 codes.
- [ ] Implement strict parsing of bounded allowlisted errors and local patient guidance; never render caller-controlled error text.
- [ ] Ensure an error keeps Clear/Retry available and never produces demonstration/model output.
- [ ] Re-run the focused frontend suites.

### Task 3: Reject unusable uploads before transfer

**Files:**
- Modify: `facial_paralysis_web/src/components/MediaCapture.tsx`
- Test: `facial_paralysis_web/src/components/MediaCapture.test.tsx`

- [ ] Add failing tests for empty files, files over 512 MiB, misleading MIME/extension pairs, replacement after rejection, and exact boundary acceptance.
- [ ] Verify RED.
- [ ] Implement the browser-side exact size/type contract and preserve the upload fallback and object-URL cleanup.
- [ ] Re-run the component and App tests.

### Task 4: Exhaust the HTTP and preprocessing matrices

**Files:**
- Modify: `tests/test_faces_shared_v9_gateway.py`
- Modify: `tests/test_faces_shared_v9_pipeline.py`
- Modify: `tests/test_facial_process_shared_v9_deployment.py`

- [ ] Cover missing/duplicate/extra multipart fields, wrong content type, empty/oversized media, invalid UTF-8/JSON/duplicate keys/hash, both scripts, unsupported suffix, corrupt container, low FPS, incomplete hold coverage, duration drift, tracking thresholds, original/flip failure, downstream timeout/drift, and readiness failure.
- [ ] Verify each expected status and closed error schema and confirm no patient/path/media bytes appear.
- [ ] Confirm request-scoped files and multipart handles are removed on every branch.

### Task 5: Browser and live-container acceptance

**Files:**
- Modify: `facial_paralysis_web/tests/browser/acceptance.py`
- Create: `facial_paralysis_web/tests/browser/live_full_loop_acceptance.py`
- Create: `docs/results/facial_process_web_patient_safe_edge_acceptance_v2.md`

- [ ] Run desktop and mobile browser checks for camera denied/missing, speech unavailable/failing, start/finalize timeout, stop/discard, pagehide/refresh, clear/retry, no external requests, and no button overlap.
- [ ] Run real virtual-camera 1–7 and 1–8 success flows through Nginx, gateway, MediaPipe, Shared V9, and result rendering.
- [ ] Run a no-face virtual-camera failure through the same stack and verify action-specific guidance plus successful clear/retry.
- [ ] Recheck tmpfs/browser residue after success, rejection, abort, concurrency, timeout, and container restart.
- [ ] Record exact commands, counts, timings, limitations, and screenshots in the acceptance report.

### Task 6: Final verification and handoff

**Files:**
- Modify: `deploy/facial-process-shared-v9/README.md`

- [ ] Run all frontend tests, direct Python contract tests, typecheck/build, `py_compile`, `git diff --check`, and a secret/path scan.
- [ ] Rebuild and restart only the web and gateway containers with the existing local override.
- [ ] Verify all three containers healthy and the live success/failure contracts.
- [ ] Update the deployment README with the closed error codes and participant recovery behavior.
