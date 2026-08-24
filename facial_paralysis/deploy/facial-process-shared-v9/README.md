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
python3 scripts/maintain_facial_process_docker.py init-builder
docker compose -f deploy/facial-process-shared-v9/compose.yaml build \
  --builder facial-process-v9-builder
docker compose -f deploy/facial-process-shared-v9/compose.yaml up --no-build -d
```

Then open <http://127.0.0.1:8080>.

The full stack builds its model container from the checked-in, digest-pinned
Dockerfile and the Git-tracked release weights. Its release-manifest commitment
is:

```text
81e396954090a0da6b99519909c1af15b6df5d1585ba27a642539352fe0a0c64
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
and step-8 applicability. The timeline binds the same video hash, an external
timing source (`capture_event_log`, audited `audio_forced_alignment`, or
`blinded_manual`), and either
seven ordered intervals (steps 1–7) or eight (steps 1–8). Every hold is exactly
three seconds. Step 8 is omitted when not clinically applicable; it is never
zero-imputed. The server rejects missing mandatory actions, digest drift, duplicate or extra form
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
Only `127.0.0.1:8080` is published; model and gateway ports remain internal.
There are no persistent media volumes. Browser recording state is page-scoped;
the visible **Clear recording and start over** action, page refresh, or tab close
releases it. Gateway decode files are request-scoped inside a 1.2 GiB tmpfs and
are removed after success or failure. Docker JSON logs rotate at 10 MiB with
three files per service.
An approved same-host reverse proxy may provide authenticated TLS access when
remote research use is required.

This source release does not provide TLS termination, user authentication,
authorization, audit-log storage, PHI retention policy, backup, monitoring, or
high availability. Add those controls in the institution's approved ingress
and data-governance environment before any participant use. This remains a
research prototype, not a medical device or deployable clinical product.

## Project-scoped storage maintenance

Builds use the dedicated `facial-process-v9-builder`, so new cache is isolated
from other Docker projects. Maintenance is a two-step, fail-closed operation.
Audit does not delete anything and writes a plan valid for 15 minutes:

```bash
python3 scripts/maintain_facial_process_docker.py audit \
  --plan /tmp/facial-process-v9-cleanup-plan.json
python3 scripts/maintain_facial_process_docker.py apply \
  --plan /tmp/facial-process-v9-cleanup-plan.json
```

Only untagged images older than seven days that carry the exact Facial Process
ownership label and are unused by every running or stopped container are
eligible. Apply rereads the complete image/container inventory and refuses to
run if it changed after audit. It prunes cache only from the named project
builder, keeps at least 2 GB of that cache, and targets at most 8 GB. It never
removes volumes or tagged images. `docker system prune`, global
`docker image prune`, global `docker builder prune`, and `docker volume prune`
must not be used for this project because this host contains unrelated Docker
workloads and mixed historical cache.

For the local LaunchAgent, the same tool caps all three watchdog logs at 1 MiB
without following symbolic links:

```bash
python3 scripts/maintain_facial_process_docker.py cap-logs
```
