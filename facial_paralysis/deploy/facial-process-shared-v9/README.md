# Facial Process Web + Shared V9

This Compose release connects the correct facial-paralysis frontend to the
pinned Shared V9 research model through a raw-video preprocessing gateway.

```text
Browser
  -> Nginx web (:8080, only published port)
  -> raw-video gateway (private network)
     -> SHA-bound FACES timeline
     -> 32 samples per three-second action hold
     -> MediaPipe 478-point mesh on original image
     -> horizontal image flip + independent MediaPipe re-detection
     -> 110D clinical geometry + dense action tensors
  -> Shared V9 BLV9-009 ensemble (private network)
  -> strict binary research response
```

## Requirements

- Docker Engine with Compose v2;
- at least 6 GB free memory for a CPU deployment;
- at least 8 GB free disk space for source builds, images, and build cache;
- outbound access during the first build/pull.

The MediaPipe gateway and the published Shared V9 image run as
`linux/amd64`. They run natively on x86-64 servers; Docker Desktop can emulate
them on Apple Silicon, but preprocessing is slower.

## Start

```bash
git clone https://github.com/WilliamQiuzy/facial-attention-lab.git
cd facial-attention-lab
git switch codex/facial-process-web-v9-integration
cd facial_paralysis
docker compose -f deploy/facial-process-shared-v9/compose.yaml up --build -d
```

Then open <http://127.0.0.1:8080>.

The model container is pinned to:

```text
ghcr.io/williamqiuzy/facial-attention-lab-shared-v9@sha256:ec0e2b34e2233e159d555ab3761fe113f5b768562ba9d9d7bf7c2d7a27d42c95
```

The gateway independently pins the official MediaPipe Face Landmarker asset to
SHA-256 `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`.

## Verify and stop

```bash
curl --fail http://127.0.0.1:8080/healthz
docker compose -f deploy/facial-process-shared-v9/compose.yaml ps
docker compose -f deploy/facial-process-shared-v9/compose.yaml logs --tail=100
docker compose -f deploy/facial-process-shared-v9/compose.yaml down
```

Healthy containers prove that the UI, gateway, exact model identity, and
internal readiness route are available. They do not prove clinical validity.

## HTTP API

`POST /api/v1/facial-paralysis/infer` accepts exactly three multipart fields:

- `video`: MOV, MP4, M4V, AVI, or WebM, at most 512 MiB and 180 seconds;
- `manifest`: canonical `faces-v9-capture-manifest/v1` JSON;
- `timeline`: canonical `faces-action-timeline/v1` JSON.

The manifest binds the video SHA-256, FACES protocol version, capture source,
and step-8 applicability. The timeline binds the same video hash and the eight
ordered prompt/hold/completion intervals. Every hold is exactly three seconds.
The server rejects missing actions, digest drift, duplicate or extra form
fields, unsupported media, insufficient frame rate, incomplete time coverage,
and fewer than 26 valid paired MediaPipe samples in any active action.

The success response has schema
`facial-paralysis-shared-v9-inference/v1`. It contains only:

- pinned model/release identity;
- pinned preprocessing and MediaPipe identity;
- action-level tracking counts;
- ensemble probability, three member probabilities, 0.5-threshold class;
- `clinical_use_eligible: false`.

No raw landmarks, video path, patient identifier, HB grade, regional severity,
or label is returned.

## Security and operational boundary

All containers run as non-root, with read-only root filesystems, dropped Linux
capabilities, `no-new-privileges`, bounded processes, and tmpfs working space.
Only port 8080 is published; model and gateway ports remain internal.

This source release does not provide TLS termination, user authentication,
authorization, audit-log storage, PHI retention policy, backup, monitoring, or
high availability. Add those controls in the institution's approved ingress
and data-governance environment before any participant use. This remains a
research prototype, not a medical device or deployable clinical product.
