# Facial Process Web × Shared V9 Integration

Status: approved for implementation on 2026-08-24.

## Boundary

The target browser application is `facial_paralysis/facial_paralysis_web/`.
It is independent from `facial_paralysis/facial_defect_web/`, which is outside
this integration.

The existing Shared V9 image remains the immutable tensor inference service.
A separate gateway converts an authorized FACES video and externally anchored
action timeline into its checksum-bound NPZ request contract.

## Data flow

1. The browser supplies the video, capture manifest, and `faces-action-timeline/v1`.
2. The gateway verifies the video digest and the exact FACES action order.
3. Browser capture and imported recordings use a digest-bound
   `capture_event_log`; unsupported timing sources fail closed in this release.
4. The gateway samples 32 positions from every three-second hold. Repose is the
   baseline and is not an action token.
5. MediaPipe runs on the original image and on an actual horizontal image flip.
6. The gateway builds the exact 110D clinical and 478-point dense action tensors.
7. The existing Shared V9 service returns the three-member ensemble result.
8. The browser displays only the research score, quality evidence, and pinned
   provenance; it does not display HB grade or unsupported regional heads.

## FACES to V9 action mapping

| FACES step | V9 role |
| --- | --- |
| Neutral repose | baseline only |
| Eyebrow raise | `BROW_RAISE` |
| Gentle eye closure | `EYE_GENTLE` |
| Tight eye squeeze | `EYE_FORCEFUL` |
| Relaxed smile | `SMILE_GENTLE` |
| Lip pucker | `LIP_PUCKER` |
| Lower teeth show | `SHOW_BOTTOM_TEETH` |
| Reanimated smile | `SMILE_FULL` |

The current release has not validated a six-action FACES bag. When the final
reanimated-smile action is not applicable, capture remains available but V9
inference fails closed until a separately validated missing-action release is
available.

## Deployment

The deployment has three services: static web, video preprocessing gateway,
and the existing digest-pinned Shared V9 image. Only the same-origin web
entrypoint is published; an institutional deployment must terminate HTTPS at
its approved ingress. Raw video is request-scoped and confined to tmpfs; the
model service is reachable only on the internal container network.

## Acceptance

- exact request, response, timing, action, and provenance contracts;
- original/true-flip MediaPipe extraction and 26-of-32 tracking QC;
- malformed, oversized, unsupported, unanchored, and incomplete input rejection;
- frontend unit, type, build, and desktop/mobile browser acceptance;
- CPU/GPU model parity and gateway-to-direct-NPZ parity;
- clean-machine Docker Compose startup and no raw-video persistence.
