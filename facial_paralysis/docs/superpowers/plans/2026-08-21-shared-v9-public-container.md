# Shared V9 Public Container Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish an anonymous-pull GHCR image containing the complete Shared V9 ensemble and make a fresh repository clone start a hardened inference API with Docker Compose.

**Architecture:** Wrap the checksum-bound `SharedV9Predictor` with a small FastAPI service. Build a closed-context, non-root image containing code and all three weights; default Compose uses CPU and an override selects one NVIDIA GPU.

**Tech Stack:** Python, FastAPI, PyTorch, Docker Compose, GHCR.

---

### Task 1: V9 HTTP service

- [ ] Add failing health, readiness, prediction, and malformed-request tests.
- [ ] Implement `src/deployment/shared_v9_service.py` and `scripts/serve_shared_v9.py`.
- [ ] Run V9 and V8 service regressions.

### Task 2: Closed public image and Compose

- [ ] Add a failing Docker/Compose contract test.
- [ ] Add `environment/shared_v9_public_v1.Dockerfile` with exact weight checks.
- [ ] Add CPU-default and optional-GPU Compose plus an anonymous quickstart.
- [ ] Run V9 and V8 container regressions and commit the buildable source.

### Task 3: Immutable public activation

- [ ] Build the committed context on H200 and push to public GHCR.
- [ ] Anonymously pull the image by digest and test CPU/CUDA service paths.
- [ ] Pin Compose, registry, OCI manifest, and docs to the observed digest.
- [ ] Commit and push activation metadata to the existing draft PR.

### Task 4: Fresh-clone acceptance

- [ ] Clone the public branch anonymously.
- [ ] Run `docker compose pull` and `docker compose up -d` without login.
- [ ] Verify readiness, prediction, restart, shutdown, tests, and scans.
