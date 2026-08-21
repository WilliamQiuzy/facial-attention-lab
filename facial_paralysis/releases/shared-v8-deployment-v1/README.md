# Shared V8 deployment v1

This is the locked research deployment of `ResidualSharedRouterV8 / RSR8-001`.
It is the current deployable shared model, while Universal Clinical Router v4
remains the current scientific comparator. This distinction prevents deployment
readiness from being mistaken for independent clinical validation.

The public release contains the model/API code, exact model and image
commitments, and aggregate acceptance evidence. The fitted `weights.npz` is a
restricted artifact and is intentionally not stored in public Git. Authorized
deployments must obtain the two-file `model/` directory through the restricted
project handoff and verify the weight SHA-256 in `model_manifest.json`.

## Runtime boundary

- Input: one pre-extracted, identifier-free clinical action bag encoded as NPZ.
- Output: binary probability and fixed-threshold class for one protocol.
- Protocols: free-motion four-window, scripted three-action, and cue-aligned
  seven/eight-action.
- Not included: raw-video decoding, MediaPipe extraction, Mayo HB grading, or a
  clinical diagnosis endpoint.

Build from a clean committed tree, then launch only by immutable image ID:

```bash
docker build \
  -f environment/shared_v8_deployment_v1.Dockerfile \
  -t facial-paralysis-shared-v8:1.0.0 .

python scripts/launch_shared_v8_container_v1.py \
  --image-id sha256:d5f1de3c57ab5b080ab30907f114b764b67bc3acc897ff29ceda426cb44296ea \
  --release-root /restricted/path/to/model \
  --port 18080
```

The service binds only to `127.0.0.1`; place authenticated TLS ingress in front
of it rather than exposing port 8080 directly.
