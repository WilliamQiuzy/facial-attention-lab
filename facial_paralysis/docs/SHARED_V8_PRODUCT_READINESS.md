# Shared V8 product-readiness boundary

## Current decision

Shared V8 deployment v1 is suitable for an authenticated internal research
pilot with clinician oversight. It is not ready to be marketed or operated as
an autonomous clinical product.

## What is ready

- Immutable model, source, dependency, Docker, and weight commitments.
- H200 GPU serving with restart-deterministic synthetic acceptance.
- Non-root container, read-only root, dropped capabilities, bounded binary
  input, localhost-only publication, and fail-closed malformed-input handling.
- A versioned rollback target and a private, weight-bundled OCI image for
  authorized collaborators.

## Required before clinical production

1. Validate binary and HB endpoints on participant-disjoint Mayo labels with
   confirmed negative controls, then perform an untouched external validation.
2. Freeze and validate the raw-video-to-feature preprocessing service; the
   current API starts from pre-extracted action bags.
3. Add authenticated TLS ingress, authorization, rate limiting, audit logs,
   privacy review, secret management, and vulnerability scanning.
4. Add calibration and abstention rules, quality-control failure behavior,
   subgroup and acquisition-shift evaluation, and prospective drift monitoring.
5. Add replicated serving, uptime and latency SLOs, backup/restore drills,
   rollout/rollback automation, incident response, and on-call ownership.
6. Complete clinician-facing intended-use, human-oversight, regulatory, and
   change-control documentation.

Deployment performance is engineering evidence. It does not establish clinical
accuracy, clinical utility, HB grading validity, or regulatory readiness.
