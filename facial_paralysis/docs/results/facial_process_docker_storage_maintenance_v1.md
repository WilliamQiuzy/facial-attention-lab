# Facial Process Docker Storage Maintenance v1

Date: 2026-08-24
Scope: Shared V9 local/portable Docker deployment only

## Outcome

Docker storage maintenance is project-scoped and fail-closed. It never calls a
global image/cache prune and has no volume-delete path. The local 24/7 watchdog
runs the two-phase audit/apply once per day; manual and portable commands are in
`deploy/facial-process-shared-v9/README.md`.

| Resource | Bound and cleanup rule |
|---|---|
| Historical images | Delete only untagged images older than 7 days with exact `shared-v9` ownership label and no running or stopped container reference |
| Build cache | New builds use only `facial-process-v9-builder`; automatic BuildKit GC keeps 2 GB, caps at 8 GB, and preserves at least 20 GB free; daily apply additionally prunes only this builder's cache older than 7 days |
| Container logs | Every app service uses `json-file`, `max-size=10m`, `max-file=3` |
| BuildKit daemon logs | Frozen BuildKit config lowers daemon logging to `warn`; current daemon log was 492 bytes after bootstrap |
| LaunchAgent logs | `service.log`, `service.err.log`, and `last-reconcile-error.log` are checked every minute and truncated above 1 MiB without following symlinks or hardlinks |
| Volumes | Never pruned; application media has no persistent Docker volume |

The existing host-wide Docker inventory remains mixed across unrelated
projects: about 119.6 GB of images with 68.54 GB reported reclaimable, and
51.7 GB of cache with 20.73 GB reported reclaimable. Those global reclaimable
numbers were **not deleted**, because Docker cannot prove that they belong to
Facial Process. The dedicated Facial Process builder currently reports 0 B of
cache, and the first audited image plan contained zero eligible old project
images.

## Edge acceptance

The direct maintenance suite passes 13/13 cases:

- current/running, tagged, young, unknown-age, or unrelated images are retained;
- stopped-container references are treated as in use;
- an inventory change between audit and apply blocks every mutation;
- plans expire after 15 minutes;
- concurrent maintenance fails on a process lock;
- Docker unavailable fails before inventory or mutation;
- a missing, wrong-driver, inactive, or policy-drifted builder fails before
  image deletion;
- only exact `docker image rm <sha256>` and named-builder cache-prune commands
  are constructible; no global prune, `--all`, or volume command exists;
- symlink and hardlink log targets are rejected before any truncation;
- all three oversized watchdog logs are bounded while small logs are unchanged.

A disposable Docker container emitted 42,000,000 log bytes using the production
`10m`/three-file policy. Docker retained 22,077,056 bytes, confirming real
rotation rather than only static Compose configuration; the disposable
container was then removed.

Two live-only bugs were found and fixed during acceptance:

1. Docker Desktop exposed one tagged third-party image with `Created=null`.
   The first audit correctly stopped; the fix records unknown age but makes it
   permanently ineligible for deletion.
2. A newly created Buildx builder was initially inactive, so cache prune failed.
   Apply now bootstraps and verifies the exact frozen BuildKit log/GC policy
   before any cache or image mutation.

Related regression evidence: 43 Python deployment/gateway/model/maintenance
checks and 64 frontend checks pass (107 total); the production web build,
Compose validation, Dockerfile build check, live three-container health checks,
and current localhost `/healthz` also pass.

## Claim boundary

This proves storage bounds and deletion isolation for this release. It does not
authorize deletion of the host's pre-existing mixed Docker cache, establish an
institutional PHI retention policy, or change Shared V9 clinical validity.
