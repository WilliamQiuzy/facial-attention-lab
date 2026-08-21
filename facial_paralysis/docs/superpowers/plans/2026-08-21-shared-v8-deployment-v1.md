# Shared V8 Deployment V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze `ResidualSharedRouterV8 / RSR8-001`, create a reproducible feature-level inference service, and prove it in a hardened Docker container on the H200.

**Architecture:** Train one deterministic full-development fit using the already frozen V8 loss and data loaders, then export only tensor weights and the fold-independent 110D scaler into a checksum-bound release. Serve strict pre-extracted action bags through a small FastAPI process; raw clinical videos and participant identifiers never enter the container or Git. Git contains code, tests, the public release manifest, and image commitment, while restricted fitted weights remain in the H200 release directory.

**Tech Stack:** Python 3.10+, PyTorch 2.7.1, NumPy NPZ with `allow_pickle=False`, FastAPI/Uvicorn, Docker, NVIDIA Container Toolkit, H200.

---

### Task 1: Freeze the model and artifact contract

**Files:**
- Create: `src/deployment/shared_v8_release.py`
- Create: `src/deployment/__init__.py`
- Test: `tests/test_shared_v8_release.py`

- [ ] Write failing tests for exact candidate/config, closed NPZ fields, checksum validation, corrupt/duplicate archive rejection, strict model-state loading, scaler validation, deterministic prediction, and protocol names that do not encode institutions.
- [ ] Run `PYTHONPATH=. python tests/test_shared_v8_release.py` and confirm RED because the deployment module is absent.
- [ ] Implement the minimal release loader/exporter and strict prediction input validation.
- [ ] Rerun the test and confirm GREEN.

### Task 2: Add deterministic full-data fitting and release finalization

**Files:**
- Create: `scripts/freeze_shared_v8_deployment_v1.py`
- Test: `tests/test_freeze_shared_v8_deployment_v1.py`

- [ ] Write failing tests for the exact 20-update seed-0 RSR8-001 fit, all-three-source gradient audit, source-class balancing, no-overwrite release publication, aggregate-only manifest, and absence of participant identifiers.
- [ ] Confirm RED, implement the frozen full-data fit by reusing authenticated loaders, and confirm GREEN.
- [ ] Export `weights.npz`, `manifest.json`, and a deidentified synthetic acceptance fixture; publish by atomic directory rename.

### Task 3: Build the inference API

**Files:**
- Create: `src/deployment/shared_v8_service.py`
- Create: `scripts/serve_shared_v8.py`
- Test: `tests/test_shared_v8_service.py`

- [ ] Write failing tests for `/healthz`, `/readyz`, `/v1/model`, and binary-NPZ `/v1/predict/{protocol}`.
- [ ] Cover malformed ZIP, duplicate members, excessive expansion, missing fields, wrong dtype/shape, NaN/Inf, invalid protocol, oversized body, deterministic response, no request data in errors, and concurrent calls.
- [ ] Confirm RED, implement the app with startup-only model loading and inference locking, then confirm GREEN.

### Task 4: Add hardened Docker packaging

**Files:**
- Create: `environment/shared_v8_deployment_v1.Dockerfile`
- Create: `environment/shared_v8_requirements.lock`
- Create: `scripts/accept_shared_v8_deployment_v1.py`
- Modify: `.dockerignore`
- Test: `tests/test_shared_v8_image_contract.py`

- [ ] Write failing static tests for a pinned CUDA base, non-root UID, read-only model mount, closed build context, health check, no training/data modules in the runtime image, and pinned service dependencies.
- [ ] Confirm RED, implement the image and acceptance runner, then confirm GREEN.
- [ ] Acceptance runner must test model/image SHA, GPU identity, readiness, 100 warm requests, 1,000 sequential requests, bounded concurrency, restart determinism, malformed input handling, CPU/GPU probability tolerance, latency distribution, and peak GPU memory.

### Task 5: Execute H200 release and container acceptance

**Files:**
- Generate on H200 only: `/home/ssh-ziyue/facial-paralysis-h200/releases/shared-v8-deployment-v1-<id>/weights.npz`
- Generate on H200 only: `/home/ssh-ziyue/facial-paralysis-h200/releases/shared-v8-deployment-v1-<id>/manifest.json`
- Create public aggregate record: `docs/results/artifacts/shared_v8_deployment_v1/acceptance.json`
- Create: `docs/deployment/shared_v8_deployment_v1.md`

- [ ] Transfer a clean Git archive to H200 and verify its commit SHA.
- [ ] Run the frozen full-data fit on H200 without reading Mayo.
- [ ] Build the image from the clean archive and record the immutable image ID.
- [ ] Start with non-root user, read-only root filesystem, read-only model mount, tmpfs `/tmp`, dropped capabilities, and no-new-privileges.
- [ ] Run the full acceptance suite and copy back only aggregate evidence and commitments.

### Task 6: Version, verify, and publish

**Files:**
- Modify: `docs/model_registry.json`
- Modify: `docs/CURRENT_MODEL.md`
- Create: `docs/results/artifacts/shared_v8_deployment_v1/release_manifest.json`

- [ ] Register V8 as the locked deployment research model while preserving the statement that it is not clinically validated and does not replace the scientific UCR4 evidence boundary.
- [ ] Run all new tests, V8 model/evaluation regressions, current-model contract tests, `compileall`, secret/private-path/raw-data scans, and `git diff --check`.
- [ ] Stage only deployment paths, commit, push `agent/shared-v8-deployment-v1`, and update/reuse the existing public PR rather than opening a conflicting model-history branch.
