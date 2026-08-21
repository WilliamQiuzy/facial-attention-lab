# Shared V9 public Docker deployment

This public image contains the complete three-member BLV9-009 ensemble, common
110D scaler, API code, and checksum manifest. It requires no private model
mount, H200 directory, GitHub token, or Docker registry login.

The service is a research inference API, not clinically validated software. It
accepts authenticated, preprocessed MediaPipe clinical-action tensors in the
frozen binary NPZ contract; it does not accept a raw video or return an HB grade.

## CPU quickstart

```bash
git clone --depth 1 --branch codex/shared-v9-public-release \
  https://github.com/WilliamQiuzy/facial-attention-lab.git
cd facial-attention-lab/facial_paralysis/deploy/shared-v9
docker compose pull
docker compose up -d
curl --fail http://127.0.0.1:18090/readyz
```

The image is public and the default service uses CPU, so this path works on an
x86-64 Linux Docker server without an NVIDIA runtime. The port is intentionally
bound to localhost; add authenticated TLS ingress separately if remote clients
must call it.

## NVIDIA GPU

Install NVIDIA Container Toolkit, then start with the GPU override:

```bash
docker compose -f compose.yaml -f compose.gpu.yaml pull
docker compose -f compose.yaml -f compose.gpu.yaml up -d
curl --fail http://127.0.0.1:18090/readyz
```

Readiness must report model
`broad_literature_shared_v9_blv9_009_ensemble`, candidate `BLV9-009`, three
ensemble members, and device `cpu` or `cuda` as selected.

Prediction endpoint:

```text
POST /v1/predict/{free_motion_four_window|scripted_three_action|cue_aligned_action}
Content-Type: application/octet-stream
Body: validated NPZ request payload
```

Stop the service with `docker compose down` (include both `-f` arguments if the
GPU override was used).
