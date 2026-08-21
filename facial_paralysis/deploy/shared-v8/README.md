# Shared V8 portable deployment

This quickstart runs the exact private, weight-bundled image accepted on H200.
It targets an x86-64 Linux host with an NVIDIA GPU, NVIDIA Container Toolkit,
Docker Engine, and Docker Compose v2. It is an internal research service, not clinical software or an HB grading product.

The GHCR package is private. Obtain package access from the repository owner,
create a GitHub token with `read:packages`, and authenticate without saving the
token in this repository:

```bash
export GHCR_USER=your-github-user
export GHCR_TOKEN=your-read-packages-token
printf '%s' "$GHCR_TOKEN" | docker login ghcr.io -u "$GHCR_USER" --password-stdin
unset GHCR_TOKEN
```

From a checkout of this repository:

```bash
cd facial_paralysis/deploy/shared-v8
docker compose pull
docker compose up -d
curl --fail http://127.0.0.1:18080/readyz
```

Expected identity:

```json
{"status":"ready","model_id":"residual_shared_router_v8_rsr8_001","weights_sha256":"72e40ea7b127b6768e931665df622550f06cc5a1bbad20070a42614c5b9901ab","device":"cuda"}
```

The port is intentionally bound to localhost. Put authenticated TLS ingress in
front of it if another machine must call the service. Stop it with:

```bash
docker compose down
```
