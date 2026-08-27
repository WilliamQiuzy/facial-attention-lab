# Facial Process Web

Research-only React interface for standardized seven- or eight-step FACES recordings.
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

Steps 2–7 are the six mandatory active movements. Step 8 is included only when
facial reanimation applies. Shared V9 accepts either six or seven active action
tokens; an inapplicable Step 8 is omitted rather than imputed.

An imported video therefore also requires its canonical
`faces-action-timeline/v1` JSON sidecar. The browser hashes the selected video
and rejects a sidecar whose `recording_sha256` does not match. Browser-guided
sessions use `capture_event_log`; audited historical imports may retain
`audio_forced_alignment` or `blinded_manual` rather than being relabelled.

## Output boundary

The accepted response is `facial-paralysis-shared-v9-inference/v3` and is
fail-closed on model identity, release-manifest hash, preprocessing identity,
action order, tracking support, ensemble arithmetic, and threshold.

The interface shows:

- one MEEI facial-palsy-versus-healthy-control classification score;
- the frozen 0.5-threshold research class;
- recorded action context frames and descriptive movement geometry;
- action-region model influence computed at the shared action-token layer,
  released only after three-member, true-mirror, and two timing-shift checks;
- plain-language action and face-tracking coverage.

Each evidence card and PDF action page keeps three layers separate: measured
movement, model influence, and stability checks. Unstable influence is hidden
rather than assigned a direction. The report's **Save PDF** flow generates and
directly downloads a PDF containing the recorded action context images and an identifiable-media warning, without
opening the browser print dialog. **Download recorded video** exports the
exact page-scoped source file to the current device with a fixed filename that
does not reuse a potentially identifying upload name. Both are local browser
exports; neither adds server-side media persistence.

It does **not** show or infer House–Brackmann grade, eye or mouth severity,
facial laterality, treatment advice, or a clinically validated diagnosis.
The model has not been trained on the current unlabeled Mayo videos.

## Privacy boundary

Facial video is identifiable. The browser requires an explicit confirmation
that the endpoint is authorized, but that control does not replace study
governance, authentication, encryption, retention rules, or access control.
The browser keeps the selected file in the current session; the checked-in
gateway performs request-scoped preprocessing and does not implement a media
database. **Clear recording and start over**, page refresh, and tab close release
the page-owned recording; gateway temporary decode files are deleted on both
successful and failed requests.
