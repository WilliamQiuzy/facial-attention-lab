# Facial Process Web × Shared V9 Acceptance

Date: 2026-08-24

Scope: research deployment integration, not clinical validation

## Release identity

- Frontend: `facial_paralysis_web` (Facial Process Web).
- Model: Shared V9 candidate `BLV9-009`.
- Model ID: `broad_literature_shared_v9_blv9_009_ensemble`.
- Model image: source-built from `environment/shared_v9_public_v1.Dockerfile`
  with the release-manifest commitment below.
- Release-manifest SHA-256: `81e396954090a0da6b99519909c1af15b6df5d1585ba27a642539352fe0a0c64`.
- Face Landmarker SHA-256: `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`.

## Verified path

```text
Browser -> Nginx -> raw-video gateway -> MediaPipe original/true-flip
        -> 110D + 478-point action tensors -> Shared V9 -> strict binary response
```

The local Compose stack started with three healthy containers. Only
`127.0.0.1:8080` was published. All containers ran non-root with read-only root
filesystems, dropped capabilities, and `no-new-privileges`.

A 28-second seven-step and a 32-second eight-step synthetic public face video
were sent through the published Nginx route. Both requests completed with HTTP
200, used six and seven active movements respectively, and did not impute the
optional movement. The returned model,
release, preprocessing, and MediaPipe identities matched the frozen contract.
The synthetic prediction value is intentionally not reported because it is not
a scientific or clinical performance measurement.

## Test evidence

- Frontend: 64/64 unit tests, TypeScript check, and production build passed.
- Focused gateway, pipeline, deployment, release, service, registry, and claim
  gates: 46/46 tests passed.
- Total fresh focused automated checks in this acceptance run: 110/110.
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
- Both script variants and three simultaneous valid requests returned the exact
  Shared V9 contract without request loss.
- Live no-face, 10 fps, and two-second recordings each failed preprocessing
  with HTTP 422. A flat but trackable synthetic face remained eligible, which
  preserves the prompted-flat clinical behavior.
- Desktop and 390 × 844 mobile browser flows passed real 1–7 and 1–8 inference,
  explicit clear, refresh, no-persistence-store, no-overflow,
  no-console-error, and no-external-request checks.
- Port 8080 was reachable on loopback and unreachable through the Mac's LAN
  address. No MOV, MP4, M4V, AVI, or WebM file remained in container tmpfs.
- Client abort, corrupt video, wrong digest, a greater-than-7-MiB spooled
  upload, downstream timeout, three concurrent requests, and container restart
  all returned tmpfs to its non-video baseline. All three services have no
  media volume and rotate JSON logs at 10 MiB × three files.

## Boundary

This verifies software integration and reproducible research deployment. It
does not establish Mayo accuracy, House–Brackmann grading performance, clinical
safety, or medical-device readiness. The current response is one binary
research probability; it does not expose regional eye/mouth severity or an HB
grade. Institutional use still requires approved HTTPS ingress, authentication,
authorization, PHI governance, audit logging, monitoring, and a labeled Mayo
validation study.
