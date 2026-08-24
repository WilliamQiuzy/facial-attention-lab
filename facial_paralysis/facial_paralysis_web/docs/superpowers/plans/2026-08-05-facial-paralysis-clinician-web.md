# Facial Paralysis Clinician Web Implementation Plan

> **Historical plan, superseded.** The supported deployment is Shared V9 and
> its current contract is documented in
> `../../../../docs/plans/facial_process_web_shared_v9_integration.md` and
> `../../../../deploy/facial-process-shared-v9/README.md`. The V4 contract below
> is retained only to explain the original frontend implementation.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a polished, responsive research clinician web app that accepts a LifeLink Face video or records one in-browser with the eight-step FACES voice protocol, then presents only the severity outputs supported by the current Facial Paralysis model contract.

**Architecture:** A new React/Vite/TypeScript application lives at `facial_paralysis/facial_paralysis_web/`, isolated from the existing synthetic `facial_defect_web`. Browser media and speech APIs are wrapped in focused hooks/components. A fail-closed inference gateway posts video plus a protocol manifest to an optional configured HTTP endpoint and accepts results only when the service returns complete versioned eight-action segmentation and exact checkpoint provenance. Demonstration mode is a separate, explicitly enabled user action and is never an automatic fallback for missing configuration, network failure, incomplete segmentation, or malformed model output.

**Tech Stack:** React 19, strict TypeScript, Vite 8, Vitest, Testing Library, Lucide React, native MediaRecorder/getUserMedia, Web Speech synthesis, native CSS, Python Playwright for browser acceptance.

---

## Confirmed model and protocol constraints

- Latest serialized checkpoint: internal `outputs/checkpoints/warmstart_v4_expanded.pt`, SHA-256 `6310052121ed8a9a9e746716cb9c0d178eb252b438b6de7d33160eb555f6417b`.
- Supported checkpoint tasks: `binary`, `coarse3`, `eyes`, `mouth`; the current scorer exports `palsy_probability`, `eyes`, and `mouth` only.
- The checkpoint has no House-Brackmann head and no spatial heatmap output. The UI must not infer or fabricate either.
- The public app must not copy the private checkpoint, patient recordings, derived features, or clinical labels.
- The one-page FACES Script contains eight spoken actions, each held for three seconds; reanimated smile is optional.
- Warm-start checkpoints were trained as one-action inputs. A continuous session is not eligible for research inference unless the HTTP service returns seven required completed segments plus step 8 as `completed` or clinician-confirmed `not_applicable`. `skipped` or missing required segments fail closed.
- The app may borrow layout rhythm and blue/white restraint from the referenced Mayo page, but must not use the Mayo logo, shields, proprietary fonts, or imply that this research prototype is an official clinical product.

## Versioned inference contract

The multipart request contains `video` plus a `manifest` JSON field:

```json
{
  "schema_version": "faces-capture-manifest/v1",
  "protocol_version": "FACES-v0.01",
  "recording_source": "livelink-upload",
  "reanimated_smile_applicable": false
}
```

Before upload the UI must warn that facial video is identifiable and may only be sent to an authorized research endpoint. The allowed `recording_source` values are `livelink-upload` and `browser-camera`.

The only accepted response shape is:

```json
{
  "schema_version": "facial-palsy-research-inference/v1",
  "provenance": {
    "model_file": "warmstart_v4_expanded.pt",
    "model_sha256": "6310052121ed8a9a9e746716cb9c0d178eb252b438b6de7d33160eb555f6417b",
    "preprocessing_version": "predict-pipeline/v1",
    "segmentation_version": "faces-segmentation/v1"
  },
  "segmentation": {
    "duration_ms": 42000,
    "actions": [
      {"id": "repose", "status": "completed", "start_ms": 0, "end_ms": 3000},
      {"id": "eyebrow_raise", "status": "completed", "start_ms": 5000, "end_ms": 8000},
      {"id": "gentle_eye_closure", "status": "completed", "start_ms": 10000, "end_ms": 13000},
      {"id": "tight_eye_squeeze", "status": "completed", "start_ms": 15000, "end_ms": 18000},
      {"id": "relaxed_smile", "status": "completed", "start_ms": 20000, "end_ms": 23000},
      {"id": "lip_pucker", "status": "completed", "start_ms": 25000, "end_ms": 28000},
      {"id": "lower_teeth_show", "status": "completed", "start_ms": 30000, "end_ms": 33000},
      {"id": "reanimated_smile", "status": "not_applicable", "start_ms": null, "end_ms": null}
    ]
  },
  "scores": {
    "palsy_probability": 0.73,
    "eyes": {"level": 1, "expected": 1.2, "p_gt": [0.82, 0.38], "label": "Slight"},
    "mouth": {"level": 2, "expected": 1.7, "p_gt": [0.91, 0.79], "label": "Strong"}
  }
}
```

No additional fields are allowed. `palsy_probability` and both `p_gt` values must be in `[0,1]`; `p_gt` must contain exactly two non-increasing values. `level` must be integer `0..2`, `expected` must be `0..2`, and `label` must exactly match `Normal/Slight/Strong`. Segments must follow protocol order, lie within `duration_ms`, have positive non-overlapping intervals, and steps 1-7 must be `completed`. Step 8 may be `completed` with a valid interval or `not_applicable` with both timestamps null; `skipped` is rejected as incomplete.

## File map

- Create `facial_paralysis/facial_paralysis_web/package.json` and tool configuration for an independent app.
- Create `facial_paralysis/facial_paralysis_web/src/protocol/facesProtocol.ts` for the exact eight-step script and preparation guidance.
- Create `facial_paralysis/facial_paralysis_web/src/model/inference.ts` for strict response validation and HTTP upload.
- Create `facial_paralysis/facial_paralysis_web/src/model/demonstration.ts` for the separate, explicitly labelled deterministic interface preview.
- Create `facial_paralysis/facial_paralysis_web/src/hooks/useCameraRecorder.ts` for camera/MediaRecorder lifecycle.
- Create `facial_paralysis/facial_paralysis_web/src/hooks/useVoiceInstructions.ts` for speech synthesis and protocol navigation.
- Create focused components under `src/components/` for the shell, workflow, capture/import, voice guide, analysis state, and regional results.
- Create `facial_paralysis/facial_paralysis_web/src/styles/app.css` for the Mayo-inspired blue/white editorial-clinical design system and responsive states.
- Create `facial_paralysis/facial_paralysis_web/README.md` and `.env.example` documenting the endpoint contract and research boundary.
- Modify repository `README.md` to list the new project without changing the existing project descriptions.

### Task 1: Scaffold and protocol contract

- [ ] Create the package/tool configuration and test setup.
- [ ] Write `src/protocol/facesProtocol.test.ts` first, asserting every source-accurate preparation instruction and all eight exact English voice strings, ordered IDs, three-second holds, and only the reanimated-smile optional flag.
- [ ] Run `pnpm test:run src/protocol/facesProtocol.test.ts` and confirm the expected missing-module failure.
- [ ] Implement `facesProtocol.ts` with source-accurate English voice instructions and concise clinician labels.
- [ ] Re-run the targeted test and confirm it passes.

### Task 2: Fail-closed inference gateway

- [ ] Write `src/model/inference.test.ts` first from the exact JSON contract above, including the seven-required-plus-one-conditional rule, timestamp order/bounds, and rejection of missing/incomplete coverage, out-of-range numbers, non-monotonic or wrong-length ordinal thresholds, label/level mismatch, wrong model SHA, wrong preprocessing/segmentation versions, HB/heatmap/coarse3 fields, unknown fields, and malformed payloads.
- [ ] Test that HTTP upload uses multipart video plus a versioned protocol manifest and that network/HTTP/schema failures never return demonstration data.
- [ ] Run the targeted test and confirm failures are caused by the missing implementation.
- [ ] Implement strict type guards and `analyzeRecording(file, options)`; never turn severity into spatial data and never accept a continuous-session result without complete action segmentation.
- [ ] Write and implement separate tests for `createDemonstrationResult`, requiring explicit invocation and permanent `DEMONSTRATION - NOT MODEL OUTPUT` provenance.
- [ ] Re-run the targeted tests and confirm they pass.

### Task 3: Capture and voice workflow

- [ ] Write component tests first for upload validation, source preview, upload/camera mode switching, voice step navigation, optional-step skip, protocol-manifest timestamps, research-analysis eligibility, and explicit demonstration invocation.
- [ ] Run the tests and confirm the expected missing-component failures.
- [ ] Implement the browser media and speech hooks with supported MIME selection plus stream-track, MediaRecorder, object-URL, and speech cleanup on stop/unmount; include useful unsupported and permission-denied states.
- [ ] Implement the capture/import and voice guide components with keyboard-operable controls, live regions, visible focus, and motion-reduction support.
- [ ] Re-run the component tests and confirm they pass.

### Task 4: Results and Mayo-inspired shell

- [ ] Write application tests first for the research-only notice, model provenance, demo/API status, results rendering, retry/reset, and absence of unsupported HB/heatmap claims.
- [ ] Run the tests and confirm they fail for the expected missing UI.
- [ ] Implement a restrained blue/white clinical editorial shell inspired by the official Mayo patient-centered-care page: deep-blue masthead, generous whitespace, strong typographic hierarchy, slim rules, large clear actions, and calm card surfaces. Do not use Mayo marks, shields, proprietary typefaces, or official-product language.
- [ ] Render supported outputs as **uncalibrated research probability** and eye/mouth ordinal severity with `P(y > k)` threshold bars. Do not call these confidence, uncertainty, clinical grade, or diagnosis. Keep demonstration provenance visible before the run, in the result header, and in any copied/exported text.
- [ ] Re-run tests and refactor only while green.

### Task 5: Documentation and repository integration

- [ ] Add `.env.example` for `VITE_FACIAL_PARALYSIS_API_URL`, the pinned checkpoint SHA/version, required preprocessing/segmentation versions, and explicit demo-mode control.
- [ ] Add a README with local commands, browser permissions, the multipart request and strict versioned JSON response contract, segmentation responsibility, checkpoint provenance, and the explicit non-clinical boundary.
- [ ] Add the new app to the public repository project table and run instructions.

### Task 6: Verification

- [ ] Run `pnpm install --offline` when possible; use the network only if the existing store lacks a package.
- [ ] Run `pnpm test:run`, `pnpm typecheck`, `pnpm build`, and `git diff --check` with zero failures.
- [ ] Add a non-PHI generated synthetic video fixture or generate one in `/tmp` for browser acceptance; never use a patient recording.
- [ ] Run the webapp-testing helper with `tests/browser/acceptance.py` against the local Vite app using Chromium fake-media flags.
- [ ] Verify desktop, tablet, and mobile layouts, file upload, mode switching, voice-step controls, explicit demo/results, research API failure without fallback, reset, keyboard navigation, fake-camera permission success/denial, supported MediaRecorder MIME behavior, no persistence after reload, cleanup of media streams/object URLs/speech, no horizontal overflow, no console errors, and no unexpected external requests.
- [ ] Capture desktop and mobile screenshots and visually inspect them before completion.
