# Facial Process Web — Patient-safe edge acceptance v2

**Date:** 2026-08-24
**Route:** Facial Process Web → raw-video gateway → MediaPipe → pinned Shared V9 BLV9-009
**Local acceptance URL:** `http://127.0.0.1:8080`

## Outcome

Both medically valid guided scripts now complete through the real Docker and
browser stack: steps 1–7 and steps 1–8 each produced an accepted, provenance-
checked BLV9-009 response. A complete no-face recording produced an actionable
action-level rejection (`Neutral Expression`, 0/26 usable samples), while the
recording preview and clear/re-record controls remained available.

These tests validate software behavior, not model accuracy or clinical use.

## Closed edge matrix

| Boundary | Cases covered | Required behavior |
|---|---|---|
| Before recording | endpoint pending, offline, wrong model, wrong preprocessing, retry | Never label a configured URL online; lock guided start until exact Shared V9 readiness passes. |
| Local capture | permission denied, late permission after unmount, camera interruption, voice failure, stop/discard, refresh/unmount | Give visible feedback; stop tracks; never publish an incomplete recording. |
| Script | exact 1–7, exact 1–8, missing/extra/reordered action, optional Step 8 absent | Accept both real scripts; never require or zero-impute unavailable Step 8. |
| Browser upload | empty, unsupported/misleading type, over 512 MiB, exact 512 MiB | Reject before upload when invalid; accept the exact boundary. |
| Proxy/API | multipart overhead, missing/duplicate/extra fields, wrong transport, 400/413/415/422/500/502/503 | Return a closed JSON code and non-identifying recovery guidance. |
| Video decode | bad container, unsupported suffix, below 10.67 fps, frame-size drift, incomplete final hold, duration drift | Reject with the specific recording problem, not a generic 422. |
| Face geometry | original or flipped detection absent, 25/32 and 26/32 boundary, degenerate geometry | Require both real detections; 25 fails, 26 passes; identify the affected action. |
| Inference | downstream timeout/failure, malformed/unknown/expanded error body, stale result after replacement | Retain the same recording for retry; never echo server paths or accept stale/unknown output. |
| Storage | success, preprocessing failure, clear, replacement, refresh, tab close, container restart | Revoke browser URLs, remove request files, and create no persistent media volume. |
| Operations | image/cache cleanup drift, Docker unavailable, concurrent cleanup, symlink/hardlink logs, log rotation | Fail closed and remain project-scoped; never run global prune or delete a volume. |

## Verification evidence

- Frontend unit/component/contract suite: **96/96 passed**; TypeScript and production build passed.
- Gateway: **8/8**; raw-video pipeline: **11/11**; deployment: **6/6**.
- Script/action segmentation: **14/14**; Docker maintenance: **13/13**.
- Shared V9 service/public image/release regressions: **9/9**.
- Live Chromium + Docker:
  - 1–7 guided capture → accepted result;
  - 1–8 guided capture → accepted result;
  - no-face 1–7 capture → action-specific feedback, recording retained.
- After the live success and failure requests, the gateway tmpfs contained no
  request video/decode directory; the web proxy temp directories were empty.
  Web and gateway containers had no mounted media volumes. Docker logs are
  capped at 10 MiB × 3 files per service.

## Defects found and fixed by this pass

1. Nginx's old 512 MiB request limit rejected an otherwise valid 512 MiB video
   because multipart framing adds bytes; the proxy limit is now 513 MiB while
   the gateway keeps the exact 512 MiB video limit.
2. The initial readiness test fixture omitted the live `preprocessing` field,
   causing the real ready service to be rejected; the frontend now validates
   the exact five-field live response.
3. Generic HTTP 422/500 responses gave no recovery path; expected failure
   classes now have closed codes and patient-facing instructions.
4. A hung inference could spin indefinitely; it now stops after five minutes,
   preserves the recording, and allows retry.

## Visual evidence

- [Seven-step accepted result](artifacts/facial_process_web_ui/live-full-loop-7-step.png)
- [Eight-step accepted result](artifacts/facial_process_web_ui/live-full-loop-8-step.png)
- [No-face recoverable rejection](artifacts/facial_process_web_ui/live-full-loop-no-face-recovery.png)
- [Desktop button spacing](artifacts/facial_process_web_ui/analysis-actions-spacing-desktop.png)
- [Mobile button spacing](artifacts/facial_process_web_ui/analysis-actions-spacing-mobile.png)

The remaining acceptance boundary is real supported camera/browser hardware
under the institution's deployment network. Hardware-specific autofocus,
driver failure, and network policy cannot be proven with a synthetic camera;
the software response and cleanup paths for those failures are covered.
