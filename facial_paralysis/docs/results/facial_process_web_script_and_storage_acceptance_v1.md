# Facial Process Web: Script Compatibility and Storage Acceptance v1

Date: 2026-08-24
Scope: local research deployment; no clinical-performance claim

## Protocol decision

The previous hard requirement for Step 8 was removed. FACES steps 1–7 are a
complete clinical route: neutral repose plus six mandatory active movements.
Step 8, reanimated smile, is included only when facial reanimation applies.
The same frozen Shared V9 ensemble now accepts either six or seven active
action tokens. An inapplicable Step 8 is omitted; it is never represented as a
zero-motion action. A prompted but flat mandatory movement remains an eligible
low-motion observation when tracking is adequate.

Imported historical recordings may retain an authenticated
`capture_event_log`, audited `audio_forced_alignment`, or `blinded_manual`
timeline. The timing source is preserved end to end. Visual motion peaks cannot
invent an attempted action or choose its hold interval.

## Current Mayo readiness

The current local historical Mayo media root contains 53 source media files.
The frozen content audit reports 52 unique contents (one exact duplicate), 51
audio-bearing files, and two audio-free files. A fresh aggregate scan found 145
recording metadata JSON files but zero canonical `faces-action-timeline/v1`
sidecars. Therefore these videos are not silently treated as 1–7 or 1–8 and
are not used to train Shared V9. They become eligible only after each exact
video is bound to a capture-event, audited audio-aligned, or blinded-manual
timeline; both script variants are already supported once that evidence exists.

## Storage architecture

| Layer | Recording lifetime | Bound |
|---|---|---|
| Browser | Page-owned `File`/`Blob` only | Clear, replace, refresh, tab close, unmount, or `pagehide` releases the app reference and object URL |
| Nginx | Streams request body; no media volume | 512 MiB body ceiling; 600 MiB `/tmp` tmpfs |
| Gateway multipart | Request-owned upload object | Explicitly closed on success, preprocessing failure, and size rejection |
| Gateway decoder | Exact request bytes in a private temporary directory | 1.2 GiB `/tmp` tmpfs; context cleanup on success or failure |
| Shared V9 | Receives only an in-memory NPZ tensor request | 64 MiB `/tmp` tmpfs; no raw video |
| Docker logs | JSON logs only, no video or landmark payload | 10 MiB per file, three files per service |
| macOS watchdog | Reconciliation status only | Successful output suppressed; launchd logs truncated if they exceed 1 MiB |

All three containers have read-only root filesystems, no bind mounts, named
volumes, or anonymous volumes, and publish only `127.0.0.1:8080`.

## Edge-test evidence

| Case | Result |
|---|---|
| Browser 1–7 upload + timeline + inference | HTTP 200; six active actions; optional Step 8 explicitly unavailable |
| Browser 1–8 upload + timeline + inference | HTTP 200; seven active actions |
| Clear and start over | File preview and result removed; capture subtree remounted |
| Refresh after completed inference | No recording, preview, or result restored |
| Refresh during inference | Browser state cleared; gateway returned to its non-video baseline after the request window |
| Browser persistence stores | `localStorage=0`, `sessionStorage=0`, IndexedDB databases `=0` before and after |
| Object URLs | Revoked on replacement and component teardown; zero Blob video elements after clear/refresh |
| Active camera on page hide | Media tracks stopped and recorder discarded |
| Interrupted upload | Client abort left gateway tmpfs at its pre-request file count |
| Three concurrent valid requests | All three completed; tmpfs returned to its pre-request file count |
| Multipart upload forced to spool (>7 MiB) | HTTP 200; spool and decode files removed |
| Oversized upload | HTTP 413 in the bounded unit contract; processor not called; upload closed |
| Corrupt video | HTTP 422; no residual request video |
| Wrong video digest | HTTP 422; no residual request video |
| Non-multipart request | HTTP 415 |
| Downstream model timeout | HTTP 502; no raw video remained while waiting or after failure |
| Container restart | Deliberate `/tmp` markers disappeared from all three services |
| Error responses | No filename, host path, traceback, or subject identifier exposed |
| Mobile 390 x 844 | No horizontal overflow, console error, or external request |
| One-minute 24/7 reconciliation | LaunchAgent remained running; stdout/stderr byte counts did not grow |

Normal non-video tmpfs entries may still exist, such as the Nginx PID file or
MediaPipe/Matplotlib cache. Acceptance compares file counts before and after and
also verifies that no MOV, MP4, M4V, AVI, or WebM remains.

## Boundary

This establishes protocol compatibility and bounded media lifetime, not Mayo
accuracy. Shared V9 has not been trained on the current unlabeled Mayo cohort.
When Mayo HB labels arrive, participant-disjoint evaluation must report 1–7 and
1–8 coverage and performance separately before any combined result or model
selection.
