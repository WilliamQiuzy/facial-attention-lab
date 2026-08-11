# FACES Research Capture

A research-only React application for importing a LifeLink Face recording or
recording a standardized eight-step FACES session with the browser camera. The
browser can send the identifiable video to an explicitly authorized research
endpoint and displays a result only after the response passes the exact,
versioned, fail-closed contract below.

This application is independent from
[`../facial_defect_web/`](../facial_defect_web/), the synthetic facial-attention
workbench. It does not include or run a model in the browser.

## Run locally

Requirements: Node.js 22.12–24.x and pnpm 11.x.

```bash
pnpm install --frozen-lockfile
cp .env.example .env.local
pnpm dev
```

Open the local URL printed by Vite. Camera capture requires browser permission
and a secure context; `http://127.0.0.1` and `http://localhost` are accepted by
modern browsers for local development. If the camera is unavailable or access
is denied, use a LifeLink Face video instead.

Other commands:

```bash
pnpm test:run
pnpm typecheck
pnpm build
pnpm preview
```

## Browser acceptance

The checked-in Playwright script uses an inline 16 x 16 synthetic WebM with no
face, voice, identifiers, or PHI. Start the demonstration-enabled development
server, then run the script with a Python environment that provides
Playwright/Chromium:

```bash
VITE_ENABLE_DEMONSTRATION=true pnpm dev --port 4173
```

```bash
ACCEPTANCE_BASE_URL=http://127.0.0.1:4173 \
ACCEPTANCE_SCREENSHOT_DIR=/tmp/faces-browser-acceptance \
python tests/browser/acceptance.py
```

It verifies desktop, tablet, and mobile layouts; upload, explicit demo, reset,
and reload behavior; keyboard focus; fake-camera recording and track release;
horizontal overflow; runtime errors; and unexpected external requests.

## Configuration

Copy `.env.example` to `.env.local`, then set only the capabilities that should
be available in that environment:

```dotenv
VITE_FACIAL_PARALYSIS_API_URL=https://authorized-research.example.org/infer
VITE_ENABLE_DEMONSTRATION=false
```

- `VITE_FACIAL_PARALYSIS_API_URL` enables research inference. Production-like
  endpoints must use HTTPS; plain HTTP is accepted only for `localhost` or
  `127.0.0.1` development endpoints.
- `VITE_ENABLE_DEMONSTRATION=true` exposes a separate **Preview demonstration
  results** action. Demonstration values are deterministic interface-preview
  values derived locally from file metadata and are permanently labelled
  `DEMONSTRATION - NOT MODEL OUTPUT`.

Demonstration mode is never automatic fallback. A missing endpoint, network or
HTTP failure, incomplete action segmentation, wrong checkpoint provenance,
malformed JSON, or any contract mismatch produces an error and no result. A
demonstration appears only when it is enabled and the user invokes its separate
button explicitly.

## Recording workflow

The capture card supports:

- LifeLink Face upload: MOV, MP4, M4V, AVI, or WebM containing one complete
  protocol session.
- Browser camera: a front-facing, video-only `MediaRecorder` session. The user
  must grant camera access and explicitly start and stop recording.

The selected file and preview URL remain session-only in the browser. No model
weights are embedded, downloaded, or executed by this app. Research inference
requires a server that is separately authorized to receive the video, segment
the continuous recording, run the pinned checkpoint, and return the accepted
schema.

Facial video is identifiable. Before upload, the interface requires the user to
confirm that the configured endpoint is an authorized research endpoint under
the approved protocol. Do not configure a general-purpose, unapproved, or
third-party endpoint for participant recordings.

## FACES v0.01 protocol

Each spoken movement includes a three-second hold:

| Step | Movement | Requirement |
| ---: | --- | --- |
| 1 | Neutral Expression (Repose) | Required |
| 2 | Eyebrow Raise | Required |
| 3 | Gentle Eye Closure | Required |
| 4 | Tight Eye Squeeze | Required |
| 5 | Relaxed Smile | Required |
| 6 | Lip Pucker | Required |
| 7 | Lower Teeth Show | Required |
| 8 | Reanimated Smile | Conditional; clinician confirms applicability |

The voice guide helps conduct the recording; it does not itself claim that the
continuous video has been segmented correctly. The authorized service is
responsible for temporal segmentation. Research inference is accepted only
when steps 1–7 are returned as `completed` and step 8 is either `completed` or
clinician-confirmed `not_applicable`. Missing or `skipped` required actions,
and an unresolved or `skipped` step 8, fail closed.

The browser begins with step 8 unresolved and requires the clinician to choose
either **Step 8 not applicable** or **Include step 8** before research
inference. The returned step-8 segmentation status must match that explicit
choice; a contradictory response is rejected.

## Multipart request contract

The browser sends one `POST` request as `multipart/form-data`. The browser sets
the multipart boundary; clients should not replace the generated
`Content-Type`. The form contains exactly these application fields:

- `video`: the selected LifeLink file or browser-camera `File`.
- `manifest`: a JSON string using `faces-capture-manifest/v1`.

```json
{
  "schema_version": "faces-capture-manifest/v1",
  "protocol_version": "FACES-v0.01",
  "recording_source": "livelink-upload",
  "reanimated_smile_applicable": false
}
```

`recording_source` must be `livelink-upload` or `browser-camera`. The request
omits credentials, disables caching, and rejects redirects. The manifest is not
a segmentation claim; it records capture provenance and whether the clinician
confirmed that conditional step 8 applies.

## Strict response contract

The endpoint must return JSON matching
`facial-palsy-research-inference/v1` exactly. Unknown, missing, renamed, or
additional fields are rejected, including unsupported HB, heatmap, or `coarse3`
outputs.

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

The response gate enforces all of the following:

- Exact provenance for the v4 checkpoint `warmstart_v4_expanded.pt`, SHA-256
  `6310052121ed8a9a9e746716cb9c0d178eb252b438b6de7d33160eb555f6417b`,
  preprocessing `predict-pipeline/v1`, and segmentation
  `faces-segmentation/v1`.
- Exactly eight ordered action records. Completed intervals use integer
  milliseconds, have positive duration, do not overlap, and remain within
  `duration_ms`. A `not_applicable` step 8 has both timestamps set to `null`.
- `palsy_probability` and every `p_gt` value are finite values in `[0,1]`.
  Each `p_gt` array contains exactly two non-increasing ordinal thresholds.
- Eye and mouth `level` values are integers `0..2`, `expected` values are in
  `0..2`, and `label` matches the level exactly: `Normal`, `Slight`, or
  `Strong`.

The public browser does not contain the private checkpoint, patient recordings,
derived features, or clinical labels. It accepts only the displayed
uncalibrated research probability plus eye and mouth ordinal outputs. The
checkpoint has no House–Brackmann (HB) output and no spatial heatmap output;
the interface does not derive, fabricate, or display either.

## Research-only boundary

This is a research prototype, not a diagnostic tool, clinical decision aid,
validated patient assessment, or treatment recommendation. Its accepted values
must not be described as a clinical grade, diagnosis, calibrated confidence,
or validated accuracy. A clinician must review the source recording and decide
whether the research output is useful under the approved study protocol.
