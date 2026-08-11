# Guided Voice Recording Implementation Plan

> **For Codex:** Implement this plan test-first. Keep the browser recording fail-closed: interrupted or voice-failed sessions must not become analyzable recordings.

**Goal:** Replace the separate camera-recording and voice-play actions with one guided workflow that records the full FACES sequence, advances automatically after every three-second hold, and safely stops or discards incomplete capture.

**Architecture:** Add a dedicated speech-sequence hook that owns the ordered protocol queue, speech events, hold timers, cancellation generation, and optional-step filtering. Add a workspace controller above the existing capture and voice cards so camera state gates speech start and speech completion gates camera stop. Keep upload/manual voice review available, while browser-camera recording uses the combined workflow.

Each run gets a unique generation and snapshots an immutable 7- or 8-action capture plan plus the clinician's Step 8 choice. A recorder is considered active only after the matching recorder instance reaches `recording`; only that run may start speech. Finalization and discard are separate paths: only a fully completed matching voice plan may commit a file, while cancellation/error invalidates the generation before speech, timers, chunks, and recorder callbacks are cleaned up.

**Tech Stack:** React 19, TypeScript, Web Speech API, MediaRecorder API, Vitest, Testing Library, Vite, Python Playwright browser acceptance.

---

## Task 1: Lock the guided speech sequence with failing tests

**Files:**

- Create: `src/hooks/useGuidedVoiceSequence.test.ts`
- Create: `src/hooks/useGuidedVoiceSequence.ts`

1. Test that a non-reanimation session speaks steps 1-7, waits exactly three seconds after each utterance, and completes without step 8.
2. Test that an applicable reanimation session includes step 8.
3. Test that cancel invalidates pending speech/timer callbacks and returns to idle.
4. Test that an unavailable or failed speech API transitions to an explicit error.
5. Test that the plan cannot change during a run, 2,999 ms cannot advance a hold, 3,000 ms can, and late callbacks cannot advance or complete a generation twice.
6. Run the targeted test before implementation and confirm it fails because the hook does not exist.
7. Implement the smallest generation-safe speech/timer state machine and rerun the targeted test.

## Task 2: Lock camera/voice orchestration with failing tests

**Files:**

- Create: `src/components/GuidedCaptureWorkspace.test.tsx`
- Create: `src/components/GuidedCaptureWorkspace.tsx`
- Modify: `src/components/MediaCapture.tsx`
- Modify: `src/components/VoiceGuide.tsx`
- Modify: `src/App.tsx`

1. Test that the combined start action is unavailable until the front camera is ready and step 8 has an explicit clinician choice.
2. Test that one click starts MediaRecorder first, then starts the voice sequence only after camera status becomes `recording`.
3. Test that voice completion automatically stops MediaRecorder.
4. Test that a clinician interruption cancels speech and discards the incomplete recording.
5. Test that voice failure also discards the incomplete recording and displays a clear recovery message.
6. Test that browser-camera completion preserves the step-8 choice for analysis, while a new uploaded recording still resets it.
7. Publish guided completion as one atomic capture object containing the matching generation, file, exact action IDs, source, and the Step 8 snapshot; stale completion must not replace a newer run.
8. Run the targeted component test before implementation and confirm the missing workflow fails.
9. Implement a workspace-level controller; make capture and voice panels controlled while preserving the existing upload/manual-instruction behavior.

## Task 3: Make incomplete camera capture fail closed

**Files:**

- Modify: `src/hooks/useCameraRecorder.test.ts`
- Modify: `src/hooks/useCameraRecorder.ts`

1. Add a failing test that `discardRecording` stops MediaRecorder but produces no `File` and returns to a safe non-recorded state.
2. Ensure closing the camera during recording also discards instead of finalizing a partial session.
3. Keep finalize and discard as distinct two-phase paths. `onstop` may publish only when the matching generation has a commit flag, a complete immutable plan, and non-empty chunks.
4. Test discard followed by late `dataavailable`, `onstop`, and `onerror`; none may create a preview, invoke the recording callback, or enable analysis.
5. Implement the discard flag, generation checks, and lifecycle cleanup.
6. Rerun camera-hook tests.

## Task 4: Polish the integrated recording UI and hero statistic

**Files:**

- Modify: `src/styles/app.css`
- Modify: `src/App.test.tsx`

1. Add one prominent full-width guided-recording control above the capture and instruction panels.
2. Show four concise stages: camera ready, voice cue, three-second hold, automatic advance/finish.
3. Disable source switching and manual instruction controls during a guided run; expose one explicit `Stop and discard` action.
4. Add accessible live status for start, current step, hold, completion, cancellation, and error.
5. Fix the hero `8 / guided movements / 3-second holds` block with a non-overlapping grid, bounded text width, and responsive placement.
6. Preserve the Mayo-inspired blue/white visual system and small-screen stacking.

## Task 5: Verify with unit, build, and real browser evidence

**Files:**

- Modify: `tests/browser/acceptance.py`

1. Run `python tests/browser/acceptance.py --help` before using the helper.
2. Run all Vitest tests, TypeScript checking, and production build from a clean command invocation.
3. Exercise the live page with a mocked browser camera/MediaRecorder and Web Speech events: choose device, enable camera, resolve step 8, start once, observe automatic voice/hold progression, completion, and generated preview.
4. Exercise interruption and confirm no partial recording is exposed to analysis.
5. After interruption, simulate late recorder callbacks and confirm there is still no preview/file and analysis remains disabled.
6. Capture desktop and mobile screenshots and inspect the hero statistic and integrated workflow visually.
7. Keep the local Vite preview running for clinician review.
