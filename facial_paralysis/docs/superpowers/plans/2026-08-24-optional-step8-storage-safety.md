# Optional Step 8 and Storage Safety Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Support both medically valid FACES variants (steps 1–7 and steps 1–8) without inventing a missing movement, while proving that identifiable recordings do not persist or accumulate across browser, gateway, container, failure, concurrency, refresh, and restart boundaries.

**Architecture:** Neutral repose remains a baseline rather than a classifier action. The six mandatory active actions and optional reanimated smile become a variable-length action bag; a missing optional action is omitted, while a prompted but flat response remains present. Browser media remains session-only, gateway decode files remain request-scoped inside bounded tmpfs, and Docker/LaunchAgent logs receive explicit size bounds.

**Tech Stack:** React 19, TypeScript, MediaRecorder, Vitest, Playwright, FastAPI/Starlette, NumPy/OpenCV, Docker Compose, Nginx, macOS launchd.

**Completed:** 2026-08-24. The implementation, focused tests, live browser/Docker
edge matrix, aggregate Mayo readiness audit, and acceptance documentation are
complete. Detailed evidence is in
`docs/results/facial_process_web_script_and_storage_acceptance_v1.md`.

---

### Task 1: Freeze the two medically valid FACES contracts

**Files:**
- Modify: `tests/test_faces_shared_v9_pipeline.py`
- Modify: `src/preprocessing/faces_shared_v9_pipeline.py`
- Modify: `facial_paralysis_web/src/App.test.tsx`
- Modify: `facial_paralysis_web/src/App.tsx`
- Modify: `facial_paralysis_web/README.md`

- [ ] Add failing tests proving 1–7 yields six active action tensors and 1–8 yields seven, with no zero-imputed Step 8.
- [ ] Run the direct Python and frontend tests and confirm RED at the old hard rejection/button disable.
- [ ] Implement present-action filtering and response provenance for either six or seven active actions.
- [ ] Enable inference for a completed 1–7 guided capture and label the missing optional action explicitly.
- [ ] Run focused suites to GREEN.

### Task 2: Make browser recording disposal visible and deterministic

**Files:**
- Modify: `facial_paralysis_web/src/hooks/useCameraRecorder.test.ts`
- Modify: `facial_paralysis_web/src/hooks/useCameraRecorder.ts`
- Modify: `facial_paralysis_web/src/components/MediaCapture.test.tsx`
- Modify: `facial_paralysis_web/src/components/MediaCapture.tsx`
- Modify: `facial_paralysis_web/src/App.test.tsx`
- Modify: `facial_paralysis_web/src/App.tsx`

- [ ] Add failing tests for replace, record-again, explicit clear, component unmount, `pagehide`, refresh/remount, recorder error, and interrupted recording.
- [ ] Prove object URLs are revoked, chunks/file state are released, camera tracks stop, and no browser persistence API is used.
- [ ] Add an always-visible `Clear recording and start over` action whenever a recording exists.
- [ ] Add `pagehide` cleanup without attempting to delete user-owned uploaded source files.
- [ ] Run all frontend unit tests to GREEN.

### Task 3: Close request-temporary-file lifetimes

**Files:**
- Modify: `tests/test_faces_shared_v9_gateway.py`
- Modify: `tests/test_faces_shared_v9_pipeline.py`
- Modify: `src/deployment/faces_shared_v9_gateway.py`

- [ ] Add failing tests proving multipart spools close on success, validation failure, processor failure, downstream failure, oversized input, and cancellation.
- [ ] Add failing tests proving decode staging directories are empty after successful decode and every exception path.
- [ ] Explicitly close Starlette form uploads immediately after bounded bytes are captured.
- [ ] Retain `TemporaryDirectory` request scoping and verify no raw-video path survives.
- [ ] Run gateway and pipeline suites to GREEN.

### Task 4: Bound 24/7 Docker and launchd storage

**Files:**
- Modify: `tests/test_facial_process_shared_v9_deployment.py`
- Modify: `deploy/facial-process-shared-v9/compose.yaml`
- Modify: `deploy/facial-process-shared-v9/README.md`
- Modify: local `~/Library/Application Support/FacialProcessV9/keep-running.zsh`

- [ ] Add a failing deployment test requiring no persistent volumes, bounded tmpfs, localhost-only exposure, and `json-file` rotation (`10m`, three files) on every service.
- [ ] Add the Compose log limits and recreate the three local containers.
- [ ] Suppress successful watchdog reconciliation noise and cap local service logs while retaining failures.
- [ ] Test success, invalid multipart, oversized request, client abort, concurrency, container restart, and watchdog restart; compare tmpfs contents before and after.
- [ ] Verify no raw video file, multipart spool, or decode directory remains and log growth is bounded.

### Task 5: Audit existing Mayo script variants without supervised leakage

**Files:**
- Create: `scripts/audit_mayo_faces_script_variants.py`
- Create: `tests/test_audit_mayo_faces_script_variants.py`
- Create: private local aggregate report under `outputs/` only; do not commit media, identifiers, paths, or per-record rows.

- [ ] Add failing synthetic tests for exact 1–7/1–8 classification, duplicate content, missing/ambiguous timelines, prompted-flat preservation, and identifier-free aggregate output.
- [ ] Implement a read-only aggregate auditor over authenticated timeline/manifest metadata.
- [ ] Run it on locally available Mayo evidence only if exact prompt timing is already authenticated; otherwise report the precise eligibility gap and make zero predictions.
- [ ] Use existing Mayo only for label-free script/preprocessing coverage; do not fit a binary classifier or claim accuracy without labels and controls.

### Task 6: Full acceptance and release hygiene

**Files:**
- Modify: `docs/results/facial_process_web_shared_v9_acceptance.md`

- [ ] Run all focused Python direct-test suites.
- [ ] Run all frontend tests, TypeScript checking, and production build.
- [ ] Run native Playwright against the live Docker stack for both script variants, refresh, clear, repeated recordings, and mobile layout.
- [ ] Inspect container tmpfs, mounts, writable layers, rotated logging configuration, health, and automatic recovery.
- [ ] Run `git diff --check`, secret/raw-media scans, and verify the worktree diff before any commit or push.
