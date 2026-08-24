# Facial Process Web × Shared V9 Acceptance

Date: 2026-08-24

Scope: research deployment integration, not clinical validation

## Release identity

- Frontend: `facial_paralysis_web` (Facial Process Web).
- Model: Shared V9 candidate `BLV9-009`.
- Model ID: `broad_literature_shared_v9_blv9_009_ensemble`.
- Model image: `ghcr.io/williamqiuzy/facial-attention-lab-shared-v9@sha256:ec0e2b34e2233e159d555ab3761fe113f5b768562ba9d9d7bf7c2d7a27d42c95`.
- Release-manifest SHA-256: `c4fdaf054f3076a2e31b0e1ae93d1e91a45212817eb39d1c4a53620a4007b18f`.
- Face Landmarker SHA-256: `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`.

## Verified path

```text
Browser -> Nginx -> raw-video gateway -> MediaPipe original/true-flip
        -> 110D + 478-point action tensors -> Shared V9 -> strict binary response
```

The local Compose stack started with three healthy containers. Only
`127.0.0.1:8080` was published. All containers ran non-root with read-only root
filesystems, dropped capabilities, and `no-new-privileges`.

A 32-second synthetic public face video was sent through the published Nginx
route. The request completed with HTTP 200, used all seven active movements,
and retained 32/32 paired samples for every movement. The returned model,
release, preprocessing, and MediaPipe identities matched the frozen contract.
The synthetic prediction value is intentionally not reported because it is not
a scientific or clinical performance measurement.

## Test evidence

- Frontend: 58/58 unit tests, TypeScript check, and production build passed.
- New gateway/deployment path: 15/15 tests passed.
- Affected feature/data contracts: 52/52 tests passed.
- Existing Shared V9 release/service gates: 16/16 tests passed.
- Total fresh automated checks in this acceptance run: 141/141.
- Desktop DOM and 390 × 844 mobile layout checks passed with no console error
  and no horizontal overflow.
- Malformed content type returned HTTP 415; closed multipart and downstream
  identity failures are covered by the gateway tests.

## Local 24/7 and edge acceptance

- A macOS LaunchAgent was installed locally to open Docker, reconcile Compose
  once per minute, and prevent idle system sleep while the user is logged in.
- A manually stopped Web container recovered in 50 seconds; a killed model
  container recovered in 60 seconds; all three services returned to healthy.
- Seventeen malformed-request cases returned the frozen 400, 415, 405, or 422
  boundary without paths, hashes, tracebacks, or filenames in the response.
- A burst of 200 health requests returned 200/200 HTTP 200; a burst of 50 JSON
  requests to the multipart endpoint returned 50/50 HTTP 415.
- One valid synthetic public-face request and three simultaneous requests all
  returned the exact Shared V9 contract. Concurrent completion times were
  6.16, 12.36, and 18.29 seconds, showing bounded serial preprocessing rather
  than request loss.
- Live no-face, 10 fps, and two-second recordings each failed preprocessing
  with HTTP 422. A flat but trackable synthetic face remained eligible, which
  preserves the prompted-flat clinical behavior.
- Desktop and 390 × 844 mobile browser flows passed a real inference, reset,
  no-overflow, no-console-error, and no-external-request check.
- Port 8080 was reachable on loopback and unreachable through the Mac's LAN
  address. No MOV, MP4, M4V, AVI, or WebM file remained in container tmpfs.

## Boundary

This verifies software integration and reproducible research deployment. It
does not establish Mayo accuracy, House–Brackmann grading performance, clinical
safety, or medical-device readiness. The current response is one binary
research probability; it does not expose regional eye/mouth severity or an HB
grade. Institutional use still requires approved HTTPS ingress, authentication,
authorization, PHI governance, audit logging, monitoring, and a labeled Mayo
validation study.
