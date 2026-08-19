# Mayo Source Attestation Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the repeated 48-GB Mayo raw-source hashing bottleneck without weakening the existing fail-closed provenance contract.

**Architecture:** The cold builder retains every defense-in-depth raw-byte verification required by the current pin/decode/provenance transaction, then—only after every temporary source hard link is removed—captures a final descriptor-anchored source identity sweep and publishes a private HMAC-authenticated attestation as an exact member of the committed Mayo generation. Normal authorizers hold that generation and compare the exact canonical source set plus POSIX identities (`dev`, `inode`, `mode`, `uid`, `gid`, `nlink`, `size`, `mtime_ns`, `ctime_ns`) against the attestation; any mismatch fails closed and requires an explicit full rebuild. This plan guarantees zero raw-byte hashing on a warm authorization, not a single read per cold build.

**Tech Stack:** Python 3.9, SHA-256/HMAC, descriptor-anchored POSIX file operations, canonical JSON, the repository's script-style tests, and the existing transaction/lock machinery in `scripts/build_mayo_ssl_cache.py`.

---

## Frozen security boundary

- The threat boundary is the current trusted local kernel and current project owner. A malicious root process or an attacker who can rewrite the committed generation and read the HMAC key requires an external trust root and is out of scope.
- The source attestation is `source_digest_attestation.json`, exact mode `0600`, one hard link, current owner, inside the exact mode-`0700` Mayo generation.
- Persisted paths are domain-separated HMAC tokens. Private raw source SHA-256 values may exist only inside this owner-only attestation; they remain forbidden from public manifests, NPZs, logs, stdout, reports, bridge receipts, and checkpoints.
- The attestation exact schema binds the two approved root tokens, every canonical video and ARKit member (`50 + 8 = 58` under the frozen inventory), path token, content token, private source digest, exact final stat identity, entry-set digest, and whole-object HMAC. The legacy export root remains topology/classification evidence rather than content-hashed input: a separate HMAC aggregate binds all 65 session lookup outcomes, the exact four `EXISTING_EXPORT_FILES` (`done.json`, `landmarks.csv`, `blendshapes_wide.csv`, `transform_matrices.npy`) for each complete export, and their descriptor-validated type/stat identity without reading legacy-export payload bytes.
- The independent private writer accepts at most 128 source entries, 65 session classifications, 1 MiB of canonical JSON, depth 4, 255-byte relative components, and 128-byte ASCII tokens. It does not call the public-manifest writer. Validation scans all public artifacts and errors for raw path components, raw SHA-256, hex/Base64/reversible encodings, and key material.
- Generation commitment becomes `mayo_cache_generation_commitment_v4` and binds the attestation SHA-256, entry count, and source-identity aggregate HMAC. A v3 generation is never accepted by the fast path.
- A warm authorization performs zero raw-source byte hashes. Missing attestation, file-set drift, root/path-token drift, any stat identity drift, symlink/hardlink anomaly, key rotation, schema drift, or HMAC mismatch fails closed. No mtime-only hit, TTL, process-global memo, or automatic refresh is permitted.
- Entry and publication-edge authorizations remain independent. Both hold and revalidate the same immutable generation; the optimization changes the raw-source resolver only.

## Task 1: Freeze the attestation schema with RED tests

**Files:**

- Modify: `tests/test_build_mayo_ssl_cache.py`
- Modify: `scripts/build_mayo_ssl_cache.py`

- [ ] Add fixtures for an exact private attestation and a v4 generation commitment.
- [ ] Require exact keys and exact primitive types; reject duplicate JSON keys, extra fields, malformed digests/tokens, unsafe permissions/ownership, extra hard links, and oversized input before parsing.
- [ ] Require path-free public artifacts and prove the private digest never reaches manifests, NPZs, stdout, or stderr.
- [ ] Run the focused test and observe RED because the attestation API and v4 schema do not exist.

```bash
python3 tests/test_build_mayo_ssl_cache.py
```

## Task 2: Implement descriptor-held source identities and canonical attestation

**Files:**

- Modify: `scripts/build_mayo_ssl_cache.py`
- Test: `tests/test_build_mayo_ssl_cache.py`

- [ ] Add a held-source descriptor type and helpers that reject symlinks/special files, require current ownership and one hard link, snapshot the exact nine stat fields before/after reads, and close every descriptor on all paths.
- [ ] Add domain-separated HMAC helpers for approved root, relative path, content, entry-set, source-identity aggregate, and whole-object authentication.
- [ ] Build canonical JSON with sorted entries and type-exact validation using the frozen limits above; use an independent exact private writer and bounded duplicate-key rejecting reader.
- [ ] Keep `inventory_mayo_sources()` API-compatible by injecting a strict digest resolver: full hashing during build; attested lookup during authorization.
- [ ] Treat the builder's temporary source-hardlink pin phase as a controlled internal exception to `nlink=1`; no attestation identity may be captured or accepted while a pin exists.
- [ ] Run the focused test to GREEN.

## Task 3: Publish the attestation inside the committed generation

**Files:**

- Modify: `scripts/build_mayo_ssl_cache.py`
- Modify: `tests/test_build_mayo_ssl_cache.py`

- [ ] Let the cold builder perform its existing inventory, provenance, pin verification, decode, and final unchanged-byte checks; do not claim these defensive passes collapse to one raw read in this version.
- [ ] After decode/validation succeeds and `.source_snapshots` is removed, re-open and keep all 58 canonical raw descriptors held through publication, require final `nlink=1`, capture the nine-field identity, sign the attestation from the already verified content digests, and revalidate every held descriptor and live name immediately before and immediately after no-replace publication.
- [ ] Add the file to the exact top-level generation set and to `_hold_committed_mayo_generation()` / `_assert_held_mayo_generation()`.
- [ ] Upgrade `_validate_staging()` and `_validate_generation_commitment()` to v4 and bind the attestation bytes and aggregate identities.
- [ ] Prove atomic publication, staging retention on failure, no in-place refresh, and rejection of v3/mixed generations.
- [ ] Add a regression that a newly built v4 generation immediately warm-authorizes with zero raw-byte hashes; this must catch stale ctime/nlink identities caused by the pin phase.
- [ ] Add a barrier mutation after the final identity sweep but before publication and prove the held-descriptor recheck blocks promotion without cleanup.
- [ ] Run focused tests to GREEN.

## Task 4: Authorize through the warm resolver without weakening two-edge checks

**Files:**

- Modify: `scripts/build_mayo_ssl_cache.py`
- Modify: `scripts/prepare_dynamic_landmark_ssl_inputs.py`
- Modify: `tests/test_build_mayo_ssl_cache.py`
- Modify: `tests/test_dynamic_landmark_ssl_bridge.py`
- Modify: `tests/test_dynamic_landmark_ssl.py`

- [ ] Load the attestation only through its held committed-generation descriptor and verify its HMAC before consulting entries.
- [ ] Enumerate both live roots descriptor-anchored, recompute canonical classification/metadata, validate the exact legacy-export topology aggregate, and resolve each of the 58 content digests only after exact identity and path-token matches.
- [ ] Reuse the same resolver in `_live_privacy_inventories()` so bridge privacy checks do not trigger a hidden second 48-GB scan.
- [ ] Keep the entry and publication-edge authorizer calls separate; prove both run while the warm path performs zero raw-source byte hashes.
- [ ] Prove same-size/same-mtime replacement is caught by ctime, and prove inode/root swaps, add/remove, symlink, key rotation, stat-after-read drift, and attestation replacement fail closed.

## Task 5: Rebuild and verify the real generation

- [ ] Under the existing exclusive output lock, archive the validated v3 cache and coupled exposure through a crash-recoverable two-object journal into no-replace sibling names `mayo_ssl_cache.retired-v3-<aggregate16>` and `mayo_exposure_manifest.retired-v3-<aggregate16>.json`; fsync both parents, prohibit deletion, and block all builds on partial archive residue.
- [ ] Fault-inject crashes after the first rename, after the second rename, after each parent fsync, and before journal retirement; recovery must be idempotent, yield one complete paired archive, never delete either old object, and never expose a mixed canonical generation.
- [ ] Run one explicit full builder transaction to publish v4 Mayo cache plus its coupled exposure manifest.
- [ ] Rebuild the bridge and all frozen inputs because the upstream generation commitment changes.
- [ ] Time one cold build and two independent warm authorizations. Require identical authorized recording closure and zero raw-source hashing on the warm paths.
- [ ] Run the complete Mayo, bridge, and trainer regression suites, followed by one real one-epoch two-stage smoke run.

```bash
python3 tests/test_build_mayo_ssl_cache.py
python3 tests/test_dynamic_landmark_ssl_bridge.py
python3 tests/test_dynamic_landmark_ssl.py
git diff --check
git status --short
```

Expected: all tests pass; a warm authorization reads source metadata but hashes zero raw video/ARKit bytes; the smoke produces two bound checkpoints and one execution-only report without a medical performance claim.

## Task 6: Independent security and code-quality review

- [ ] Ask a spec reviewer to verify every frozen boundary and failure mode above.
- [ ] Ask a code-quality reviewer to inspect descriptor lifetime, transaction atomicity, schema complexity, and compatibility.
- [ ] Apply evidence-backed corrections and repeat the focused and full verification commands before committing.
