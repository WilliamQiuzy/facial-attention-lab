# Facial Process Web

Research-only React interface for a standardized eight-step FACES recording.
It can record a session in the browser or import a LifeLink Face video plus its
capture-event timeline. It sends the raw video to the local preprocessing
gateway and displays only the pinned Shared V9 binary research output.

This is the facial-paralysis capture application. It is intentionally separate
from `facial_defect_web`, which is a different synthetic-attention project.

## Run the complete stack

From `facial_paralysis/`:

```bash
docker compose -f deploy/facial-process-shared-v9/compose.yaml up --build -d
curl --fail http://127.0.0.1:8080/healthz
```

Open <http://127.0.0.1:8080>. Only the web container is exposed. Nginx sends
the same-origin inference request to the private gateway, which extracts
MediaPipe landmarks and calls the private Shared V9 model container.

See [`../deploy/facial-process-shared-v9/README.md`](../deploy/facial-process-shared-v9/README.md)
for deployment, verification, and API details.

## Frontend development

Requires Node.js 22 and pnpm 11:

```bash
pnpm install --frozen-lockfile
pnpm test:run
pnpm typecheck
pnpm build
pnpm dev
```

The default API path is `/api/v1/facial-paralysis/infer`. An explicit
`VITE_FACIAL_PARALYSIS_API_URL` may override it for an authorized HTTPS
research endpoint; localhost HTTP is accepted for development only.

`VITE_ENABLE_DEMONSTRATION=true` exposes a separately labelled interface
preview. It never falls back from a failed model request and is never model
output.

## Capture contract

Every movement has an externally recorded prompt, an exact three-second hold,
and a completion event:

1. neutral repose;
2. eyebrow raise;
3. gentle eye closure;
4. tight eye squeeze;
5. relaxed smile;
6. lip pucker;
7. lower teeth show;
8. reanimated/full smile.

This frozen Shared V9 route requires all seven active movements (steps 2–8).
When step 8 is not applicable, the interface may still capture a study video,
but V9 inference is disabled rather than imputing the missing action.

An imported video therefore also requires its canonical
`faces-action-timeline/v1` JSON sidecar. The browser hashes the selected video
and rejects a sidecar whose `recording_sha256` does not match.

## Output boundary

The accepted response is `facial-paralysis-shared-v9-inference/v1` and is
fail-closed on model identity, release-manifest hash, preprocessing identity,
action order, tracking support, ensemble arithmetic, and threshold.

The interface shows:

- one binary research probability;
- the frozen 0.5-threshold research class;
- the three ensemble-member probabilities;
- action-level landmark tracking support.

It does **not** show or infer House–Brackmann grade, eye or mouth severity,
facial laterality, treatment advice, or a clinically validated diagnosis.
The model has not been trained on the current unlabeled Mayo videos.

## Privacy boundary

Facial video is identifiable. The browser requires an explicit confirmation
that the endpoint is authorized, but that control does not replace study
governance, authentication, encryption, retention rules, or access control.
The browser keeps the selected file in the current session; the checked-in
gateway performs request-scoped preprocessing and does not implement a media
database.
