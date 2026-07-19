# Dynamic Landmark SSL Bridge and Real Pretraining Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert every authorized RAVDESS semantic23 trajectory and every retained Mayo MediaPipe trajectory into provenance-bound canonical 30-Hz SSL packets, authorize the exact bundles, and run the frozen three-seed RAVDESS then Mayo pretraining experiment.

**Architecture:** A new bridge module sits between the already-reviewed source caches and the already-reviewed SSL trainer. It validates the upstream HMAC-bound manifests and exact NPZ schemas, applies a content-independent uniform window policy, emits one exact five-field bundle per source into a shared mode-neutral generation, then has `freeze-stage` emit one mode-bound owner-only HMAC receipt per source. SSL authorization is upgraded so checkpoints are cryptographically bound to the live upstream generation, canonical key, bridge policy, and run mode. RAVDESS and Mayo keep separate feature adapters/scalers while sharing only the intended temporal representation; ARKit remains auxiliary-only and is rejected by this bridge.

**Tech Stack:** Tests, analysis, RAVDESS authorization, bridge construction, and training use `/Users/williamqiu/opt/anaconda3/bin/python3` exactly: Python `3.9.12`, NumPy `1.26.4`, PyTorch `2.2.1`, OpenCV `4.8.1`, with MediaPipe intentionally absent. Mayo extraction alone uses `/Users/williamqiu/.cache/facial-paralysis/mediapipe-py310/bin/python` exactly: Python `3.10.2`, NumPy `1.26.4`, PyTorch `2.2.1`, MediaPipe `0.10.35`, OpenCV `4.11.0`. The implementation also uses existing script-style `Check` tests, SHA-256/HMAC provenance, and transactional local-only outputs.

> **2026-07-18 execution amendment:** Real smoke exposed five Mayo windows with exactly eight valid frames: enough for two non-overlapping four-frame masks, but no observed context after masking. `valid_quantile_span4_context_v2` supersedes `valid_quantile_span4_v1` and requires every selected window to retain at least one valid unmasked frame in addition to the two mask spans. The trainer's fail-closed context requirement is unchanged; bridge and frozen artifacts produced under v1 are stale and must not be reused.

---

## Frozen scientific and data contracts

- RAVDESS input is `semantic23_v1`, all 2,452 trials and all 24 actor groups. No trial is excluded for being shorter than 128 frames: observed canonical lengths are 88–191 frames.
- RAVDESS scientific identity and member/content topology are frozen: each validated archive member is one source unit. The verified inventory has exactly 2,452 unique archive member names, 2,451 unique member-byte SHA-256 digests, one duplicate-content group, one member beyond unique content, maximum content multiplicity two, and zero cross-actor duplicate-content groups. The sole duplicate-content group contains two distinct valid members from the same actor; both members are retained as separate trials with no content deduplication or exclusion. Any drift in those aggregate counts, multiplicities, or actor topology fails closed and requires renewed review.
- RAVDESS v2 trial identity is generated only by one shared helper used by both the generator and committed-generation authorizer. `member_name` is the exact `ZipInfo.filename` string, with no `Path` coercion, Unicode normalization, case-folding, or other rewrite; `type(member_name) is str`, `member_name.isascii()`, and `re.fullmatch(r"[0-9]{2}(?:-[0-9]{2}){6}\.csv", member_name)` must all succeed. `source_content_sha256` is a type-exact string satisfying `re.fullmatch(r"[0-9a-f]{64}", source_content_sha256)`, and `key` is the exact canonical RAVDESS key with `type(key) is bytes` and `len(key) == 32`. The helper is exactly:

  ```python
  body = json.dumps(
      {"archive_member_name": member_name, "source_content_sha256": source_content_sha256},
      sort_keys=True,
      separators=(",", ":"),
      ensure_ascii=True,
      allow_nan=False,
  ).encode("ascii")
  mac = hmac.new(
      key,
      b"ravdess-semantic23-trial-id-v2\x00" + body,
      hashlib.sha256,
  ).digest()
  token = base64.b32encode(mac).decode("ascii").lower().rstrip("=")[:16]
  trial_id = "trial_" + token
  ```

  Each caller recomputes the helper from the exact validated member name and the SHA-256 of the same single-read verified member bytes used for validation and parsing. Caller-provided trial IDs or source digests, every v1 ID or provenance policy, content-only identity, alternate serialization, and alternate normalization are rejected. The required known-answer vector is key `b"k" * 32`, member name `01-01-01-01-01-01-01.csv`, and source digest of 64 lowercase zeroes, producing `trial_o457alx6gmxoxyak`. Raw member names and source digests are prohibited from persisted artifacts, logs, stdout, and stderr.
- The RAVDESS manifest has type-exact integer `format_version=2` (`type(format_version) is int`, so `bool` is rejected) and exact type-preserving equality with `provenance_policy={"actor_id":"private_hmac_sha256_base32","cache_integrity_id":"private_hmac_trial_id_actor_id_cache_sha256_base32","raw_paths_or_filenames_in_manifest":false,"raw_source_content_sha256_in_manifest":false,"source_binding":"verified_archive_member_name_and_bytes_single_read","trial_id":"private_hmac_archive_member_name_source_content_sha256_base32_v2"}`. Its v2 inventory retains every existing frozen aggregate field and adds exactly the six frozen topology fields, yielding this exact field union and values:

  ```json
  {
    "archive_size_bytes": 417163019,
    "archive_md5": "5753bbc64a9a790f8a8d3e03cba526ee",
    "csv_trials": 2452,
    "actors": 24,
    "source_frames": 299854,
    "header_sha256": "d89e2164e4c4e8d60393f88365ef0e87a10bef227dc90dc1d431117a74991b4e",
    "empty_trials": 0,
    "repeated_headers": 0,
    "unique_archive_member_names": 2452,
    "unique_source_content_sha256s": 2451,
    "duplicate_content_groups": 1,
    "members_beyond_unique_content": 1,
    "max_content_multiplicity": 2,
    "cross_actor_duplicate_content_groups": 0
  }
  ```

  All numeric values are type-exact integers and both digests are type-exact lowercase strings. Missing or extra fields, different names/types/values, member-name maps or arrays, and member digests fail closed. The generator and authorizer require the identical exact field union; no archive member name or member-byte digest may persist.
- RAVDESS emits one `(4,32,23)` packet per trial. Window starts are `floor(i * (T - 32) / 3)` for `i=0..3`; overlap is allowed for short trials and must be recorded, not hidden.
- Mayo authorization and generation closure retain all 48 unique long recordings from the homogeneous MediaPipe VIDEO-mode cache. The 13 legacy exports remain audit-only and are never reused. A post-generation, mask-only quality gate is frozen from the complete 48-recording cache: 46 recordings are eligible and two are explicitly excluded because they contain fewer than 64 distinct 32-frame windows with two non-overlapping valid spans of length four. The exclusion decision depends only on the committed boolean detector mask, is recomputed during every live authorization, and is bound by the complete 48-recording upstream commitment; no cache is deleted.
- Each eligible Mayo recording emits exactly 16 `(4,32,95)` packets, for 736 samples total. Let `F` be the sorted set of all 32-frame start positions whose committed mask contains at least two non-overlapping contiguous valid spans of length four. Eligibility requires `|F| >= 64`; the exact 64 starts are `F[floor(j * (|F| - 1) / 63)]` for `j=0..63`, which are strictly increasing because `|F| >= 64`. Packet `k` uses selected starts `k`, `k+16`, `k+32`, `k+48`. A recording with fewer than 64 eligible starts contributes no packet and increments the exact quality-exclusion count; duplicate windows, gap compression, cross-gap span construction, and threshold relaxation are forbidden.
- RAVDESS window selection depends only on canonical trajectory length. Mayo quality selection depends only on canonical trajectory length and the committed boolean detector mask. Feature values, movement amplitude, asymmetry, labels, and future evaluation results cannot influence a window or exclusion.
- Both stages use canonical 30-Hz windows with expected step `1`. In every bundle window, `timestamps` is exactly `float32([0..31] / 30)` and `source_frame_indices` is exactly `int64([0..31])`; neither array carries a recording offset. Original canonical 30-Hz indices, upstream source/target indices, and source timestamps stay only in the private receipt.
- Missing detector rows remain zero-valued with `valid_mask=False`; no interpolation, nearest fill, compression, or gap bridging is permitted.
- RAVDESS uses exact `semantic23_v1` names/order. Mayo uses exact `72 + clinical23_v2`; its final 23 values must be explicitly checked through `clinical23_v2_to_semantic23`, never accepted by width alone.
- RAVDESS split unit is actor. Mayo split unit is recording; all 16 packets from one recording remain together. Mayo claims remain `recording_held_out_not_patient_held_out`.
- ARKit 52d is rejected by the main bridge and remains auxiliary-only.
- Frozen training configuration for both formal stages: seeds `0,1,2`; AdamW; learning rate `0.001`; weight decay `0.0001`; 30 epochs; full train partition; span length `4`; two spans per window; CPU. The exact batch policy is `deterministic_microbatch_full_partition_64`: each epoch preserves the complete full train partition in its fixed row order, with no shuffling or runtime override, and splits it into consecutive chunks of at most 64 rows. Each chunk computes its masked SmoothL1 sum; each backward contribution is normalized by the total number of masked feature elements across the complete full train partition, all chunk gradients accumulate, and exactly one `optimizer.step()` occurs per epoch. Every mode-bound config, receipt, and checkpoint lineage must bind the exact policy string `deterministic_microbatch_full_partition_64`. A separately HMAC-attested one-epoch smoke config may test execution in an exclusive disposable namespace but cannot mint or initialize formal checkpoints.
- Existing Mayo recordings remain development-only. No Mayo HB accuracy or patient-held-out generalization claim is allowed.
- Every operation that authorizes, builds from, freezes, verifies, or trains against the Mayo bridge must receive both the live Mayo source root and the legacy-export audit root explicitly. The deidentified cache/exposure manifests intentionally cannot reveal or reconstruct these paths; omitting either root must fail closed rather than reuse a caller-supplied inventory commitment or a hard-coded user path.
- Those two Mayo roots are transient authorization inputs only. Their absolute values, basenames, relative forms, and reversible encodings must never enter a persisted bundle, artifact, receipt, stage evidence, checkpoint metadata, checkpoint receipt, report, summary, outer result, registry, stdout, or stderr. Only path-independent opaque identities and keyed commitments may persist. Every Mayo-consuming CLI named below (`inventory`, `build-bundles`, `freeze-stage`, `verify-determinism`, `two-stage`, and the locked outer evaluator) must reject a missing root before creating a staging directory, bundle, input artifact, result, report, or outer prediction; the independent `initialize-mayo-key` command is the only exception because it does not authorize Mayo data.

## Frozen security and transaction contracts

- RAVDESS authorization reads `manifest.json` through a held regular-file descriptor with an exact 4-MiB limit checked before the first read. Its nested JSON comparisons are type exact: `bool` never aliases `int`, including inside inventory, timeline, provenance, and quality-control objects. NPZ/ZIP structure and declared resource sizes are bounded before NumPy loading; any directory member in a privacy-scanned artifact ZIP is rejected rather than treated as an empty payload.
- Every exact private-tree snapshot is descriptor anchored and shares one hard budget across the complete traversal: depth at most `4`, at most `64` entries, and at most `256 MiB` of regular-file payload. The 256-MiB bound is required by the frozen 48-video clinical23 source-rate/30-Hz/transform generation plus eight ARKit caches; it remains a hard aggregate limit rather than a per-file allowance. It rejects links, special files, unsafe ownership/modes, changed identities, extra entries, and budget overruns before authorization. Every registered descriptor and every opened root chain is given a close attempt even if an earlier close fails; the validation error and any cleanup error remain preserved in the exception chain.
- Bridge, freeze, key-staging, and run-namespace transaction parents must be current-euid private directories with exact mode `0700`. Ordinary run preparation may create only one missing private level. The sole additional creation exception is `initialize-mayo-key` at the exact canonical suffix `outputs/dynamic_landmark/pretraining/.mayo_ssl_hmac.key`: beneath an already-existing, descriptor-anchored, current-euid, non-group/world-writable, non-symlinked `outputs/` namespace, it may create exactly `dynamic_landmark/` and then `pretraining/`, each current-euid mode `0700` and parent-fsynced. It may create no other missing chain, never changes the existing `outputs/` mode, and rejects every unsafe pre-existing component; directories created before a failure are retained.
- The RAVDESS canonical data root is the deliberate parent-mode exception: its held output parent must be current-euid, non-symlinked, and non-group/world-writable, so reviewed mode `0755` is allowed but mode `0777` and foreign ownership are rejected and rechecked throughout the transaction. The publisher binds the exact staging device/inode captured from its held descriptor and refuses to publish a same-name replacement.
- Every committed Mayo-derived private directory is descriptor-bound, current-euid, and exact mode `0700`: `outputs/dynamic_landmark/`, `pretraining/`, the Mayo generation root, and its `mediapipe/` and `arkit/` directories. Every committed derived NPZ, internal manifest, external exposure manifest, transaction journal, lock, temporary, and backup is a current-euid, single-link regular file with exact mode `0600`; these facts are checked before parsing and continuously through held descriptors. The only ancestor exception is an already-existing real, current-euid, non-group/world-writable `outputs/` directory, whose mode is never changed. The transient `.source_snapshots/` directory is `0700`; its raw-source hard links are never chmodded and must be removed before promotion. Builder-created private files/directories are explicitly fchmodded after creation, so even hostile umasks cannot weaken the contract. Unsafe existing storage is rejected, never silently repaired.
- Producer provenance is `dynamic-landmark-bridge-producer-v3`: it binds normalized marshalled code plus positional defaults, sorted keyword defaults, closure cells, referenced behavior-bearing globals/builtins, and referenced imported callable/module dispatch from every already-loaded producer component. Its strict canonical semantic graph supports only audited scalar/container/path/regex/dataclass/function/code/type/module/builtin forms, uses deterministic cycle ordinals rather than object IDs, rejects unsupported referenced values without `repr` or pickle, and is bounded to 16 producer modules, 2,048 executable components, 65,536 graph nodes, depth 64, 8,192 items per container, 4 MiB per leaf, and 32 MiB total. Two independently encoded live-semantic passes must match around the held source-file closure. The same digest also binds current-owner, single-link, non-group/world-writable, no-follow source-file snapshots: each source is bounded to `4 MiB`, the set to `32 MiB`, every file descriptor and lexical path is rechecked after the full set is read, and any live default/closure/global/import mutation or imported-code/on-disk mismatch fails closed or changes the digest. Direct script execution must bootstrap the canonical `scripts.prepare_dynamic_landmark_ssl_inputs` module before defining or dispatching producer logic, so CLI execution and runner import use one logical module identity and yield the identical producer digest; an entrypoint regression must execute the real script through `runpy` with a sentinel canonical module and prove exactly one canonical `main()` call before local argument parsing.
- Bridge and frozen-input namespaces use persistent, destination-named, zero-byte lock files: `.bridge.lock` beside `bridge/` and `.inputs.lock` beside `inputs/`. Each lock is a descriptor/path identity-stable, current-euid, single-link regular mode-`0600` file; writers may create it once, fsync it and its parent, then take an exclusive lock before the first destination/residue check and retain it through final publication validation and descriptor closure. Verifiers take a shared lock on the already-existing file, never create or fsync storage, and fail before authorization if the lock is missing or unsafe. The global acquisition order is bridge lock then inputs lock; no path upgrades or reverses that order. Frozen verification explicitly rejects every sibling `.inputs.staging-*` residue. Persistent locks are never unlinked, avoiding ABA/lost-wakeup races; waiting callers recheck the held descriptor against the live name after acquisition. Barrier-based two-caller tests must prove one publication, one pre-authorization destination collision, and zero losing staging residue for both bridge build and freeze.
- No bridge/freeze/key or RAVDESS source transaction performs pathname-based destructive cleanup after private staging exists. A prepublication failure retains owner-only staging as indeterminate evidence, and any staging residue blocks before new staging or secret generation until explicit offline review. A postpublication failure retains canonical storage. Bridge, freeze, and RAVDESS canonical storage cannot be republished in place. The sole postpublication retry exception is the canonical key initializer: after rejecting staging residue and acquiring the same parent lock, a later call may only revalidate the already-present canonical key through a held no-follow descriptor and returns `False` only if it remains an identity-stable, current-owner, single-link, regular mode-`0600`, exactly 32-byte file. That retry generates no secret and never stages, deletes, replaces, or chmods the key; invalid or unstable canonical storage fails closed and remains untouched. Publication is no-replace and never exposes a partially validated canonical generation.

## Canonical local outputs

All bridge and checkpoint outputs remain ignored and owner-local under:

```text
facial_paralysis/outputs/dynamic_landmark/pretraining/
  .bridge.lock
  bridge/
    bundles/{ravdess_bundle,mayo_bundle}.npz
    bundle_generation.json
  smoke/<exclusive-run-id>/
    .inputs.lock
    inputs/receipts/{ravdess,mayo}.json
    inputs/artifacts/{ravdess,mayo}/{manifest,config,split,scaler}.json
    results/checkpoints/{ravdess_only,ravdess_then_mayo}.pt
    results/checkpoints/{ravdess_only,ravdess_then_mayo}.pt.receipt.json
    results/reports/execution_only.json
  formal/
    .inputs.lock
    inputs/receipts/{ravdess,mayo}.json
    inputs/artifacts/{ravdess,mayo}/{manifest,config,split,scaler}.json
    results/checkpoints/seed_{0,1,2}/{ravdess_only,ravdess_then_mayo}.pt
    results/checkpoints/seed_{0,1,2}/{ravdess_only,ravdess_then_mayo}.pt.receipt.json
    results/reports/formal_pretraining_results.json
```

The RAVDESS source generation remains canonical at `<ravdess-data-root>/derived_semantic23/`; the Mayo generation remains canonical at the reviewed worktree-local `outputs/dynamic_landmark/pretraining/mayo_ssl_cache/` plus `outputs/dynamic_landmark/mayo_exposure_manifest.json`.

## Plan approval checkpoint

Before Task 1, this plan must receive `APPROVED` from the plan-document reviewer. Commit this exact plan together with the 2026-07-13 amendment, record the commit hash, and require a clean worktree. Any later plan change invalidates that approval and requires full plan re-review before execution continues. If an independent review requires a security-contract amendment while an already-enumerated task is WIP, freeze and report the exact in-scope WIP path list, review the current plan bytes, and commit only the two plan files; those already-enumerated task paths may remain WIP, but no unlisted path may be present. This narrow re-approval exception does not relax the clean-worktree gate before real data or Task 4 approval.

```bash
git add facial_paralysis/docs/superpowers/plans/2026-07-13-dynamic-landmark-pretraining.md \
  facial_paralysis/docs/superpowers/plans/2026-07-15-dynamic-landmark-ssl-bridge.md
git commit -m "docs(ssl): freeze bridge execution plan"
git status --short
```

Expected: commit succeeds; `git status --short` prints nothing.

---

## Task 1: Freeze canonical time and uniform packet construction with RED tests

**Files:**

- Create: `facial_paralysis/tests/test_dynamic_landmark_ssl_bridge.py`
- Create: `facial_paralysis/src/pretraining/dynamic_landmark_ssl_bridge.py`
- Modify: `facial_paralysis/tests/test_dynamic_landmark_ssl.py`
- Modify: `facial_paralysis/src/pretraining/dynamic_landmark_ssl.py`

- [ ] **Step 1: Write the failing SSL time-contract tests.**

  Replace the synthetic Mayo `source_step=2` assumption with canonical step `1`. Add a long-recording regression proving that every window uses the exact local float32 timeline and that derivatives/masks do not cross a missing canonical index.

  ```python
  expected_t = np.arange(32, dtype=np.float32) / np.float32(30.0)
  expected_i = np.arange(32, dtype=np.int64)
  assert np.array_equal(bundle.timestamps[0, 0], expected_t)
  assert np.array_equal(bundle.source_frame_indices[0, 0], expected_i)
  ```

- [ ] **Step 2: Run the SSL test and observe RED.**

  Run:

  ```bash
  /Users/williamqiu/opt/anaconda3/bin/python3 facial_paralysis/tests/test_dynamic_landmark_ssl.py
  ```

  Expected: failure at the two current Mayo `expected_source_step=2` paths in `DynamicLandmarkSSLModel.build_gru_input()` and `train_ssl_stage()`.

- [ ] **Step 3: Write failing pure bridge-policy tests.**

  Require:

  - `uniform_floor_v1(T=88, window=32, count=4)` returns four deterministic starts and retains the trial;
  - all 2,452-trial length values can produce one RAVDESS packet;
  - Mayo `K=16` selects exactly 64 strictly increasing eligible starts by frozen valid-window quantiles and preserves the exact quartile-interleaved packet layout;
  - changing features, labels, or movement values does not change Mayo starts; changing only the validity mask changes starts solely through the exact frozen span-capacity rule;
  - fewer than 64 eligible Mayo starts produces one explicit quality exclusion rather than duplicate packets, gap compression, threshold relaxation, or a partial source;
  - gaps remain fixed grid positions with `False/zero` rather than compressed rows;
  - a trajectory shorter than 32 rows fails closed;
  - wrong schemas, names, widths, dtypes, nonfinite values, or ARKit 52d fail closed.

- [ ] **Step 4: Run the bridge test and observe RED.**

  ```bash
  /Users/williamqiu/opt/anaconda3/bin/python3 facial_paralysis/tests/test_dynamic_landmark_ssl_bridge.py
  ```

  Expected: import failure because `dynamic_landmark_ssl_bridge.py` does not exist.

- [ ] **Step 5: Implement the minimal pure bridge API.**

  Implement focused, side-effect-free units:

  ```python
  @dataclass(frozen=True)
  class BridgePolicy:
      sample_rate_hz: float = 30.0
      window_length: int = 32
      ravdess_packets_per_trial: int = 1
      mayo_packets_per_recording: int = 16
      ravdess_selection: str = "uniform_floor_v1"
      mayo_selection: str = "valid_quantile_span4_context_v2"

  def uniform_floor_starts(length: int, *, count: int, window: int = 32) -> np.ndarray: ...
  def packetize_ravdess_trajectory(...): ...
  def packetize_mayo_trajectory(...): ...
  ```

  Use integer floor arithmetic. Emit the exact local `0..31` timestamp/index axes, fixed-grid masks, and zero invalid features. Return the original canonical/source mapping separately for the private receipt. Validate Mayo `clinical23_v2` through the explicit semantic adapter while retaining the full 95d values.

- [ ] **Step 6: Change both SSL trainer call sites to canonical step `1`.**

  Remove the source-specific Mayo step branch. The stored bundle index is the window-local canonical 30-Hz position for both sources; original canonical/source indices are private provenance only. Make the cache parser require exact broadcast local timestamps/indices and zero values wherever `valid_mask=False`.

- [ ] **Step 7: Run both focused tests to GREEN and refactor without changing behavior.**

  ```bash
  /Users/williamqiu/opt/anaconda3/bin/python3 facial_paralysis/tests/test_dynamic_landmark_ssl_bridge.py
  /Users/williamqiu/opt/anaconda3/bin/python3 facial_paralysis/tests/test_dynamic_landmark_ssl.py
  ```

- [ ] **Step 8: Commit Task 1.**

  ```bash
  git add facial_paralysis/tests/test_dynamic_landmark_ssl_bridge.py \
    facial_paralysis/src/pretraining/dynamic_landmark_ssl_bridge.py \
    facial_paralysis/tests/test_dynamic_landmark_ssl.py \
    facial_paralysis/src/pretraining/dynamic_landmark_ssl.py
  git commit -m "feat(ssl): freeze canonical trajectory packets"
  ```

---

## Task 2: Authorize upstream caches and publish exact private bridge bundles

**Files:**

- Modify: `facial_paralysis/tests/test_dynamic_landmark_ssl_bridge.py`
- Modify: `facial_paralysis/src/pretraining/dynamic_landmark_ssl_bridge.py`
- Create: `facial_paralysis/scripts/prepare_dynamic_landmark_ssl_inputs.py`
- Modify: `facial_paralysis/scripts/prepare_ravdess_semantic23.py`
- Modify: `facial_paralysis/tests/test_openface68_semantic.py`
- Modify: `facial_paralysis/scripts/build_mayo_ssl_cache.py`
- Modify: `facial_paralysis/tests/test_build_mayo_ssl_cache.py`

- [ ] **Step 1: Write failing RAVDESS authorization tests.**

  Given `derived_semantic23/manifest.json`, its owner-only `.semantic23_private_id_key`, and `trials/*.npz`, require exact manifest fields, type-exact integer `format_version=2`, exact type-preserving provenance-policy equality, `semantic23_v1` names/order, 2,452 unique opaque trials, 24 actor groups, exact cache filenames, recomputed private-key cache integrity IDs, safe NPZ fields, and the exact inventory union frozen above. Assert the exact six topology keys, integer types, and values in addition to all eight existing aggregate fields. Reject a changed manifest, cache byte, key identity/mode, trial/group join, duplicate ID, raw filename/path, missing/extra/wrongly typed inventory field, or partial generation. Require the held-descriptor 4-MiB manifest pre-read limit, type-exact nested JSON comparison, pre-NumPy NPZ resource bounds, and rejection of payload-bearing directory records.

  Require the read-only RAVDESS inventory CLI to emit one JSON document whose parsed value is exactly the following object, with no missing or extra output fields:

  ```json
  {
    "status": "audit_ok",
    "archive_size_bytes": 417163019,
    "archive_md5": "5753bbc64a9a790f8a8d3e03cba526ee",
    "csv_trials": 2452,
    "actors": 24,
    "source_frames": 299854,
    "header_sha256": "d89e2164e4c4e8d60393f88365ef0e87a10bef227dc90dc1d431117a74991b4e",
    "empty_trials": 0,
    "repeated_headers": 0,
    "unique_archive_member_names": 2452,
    "unique_source_content_sha256s": 2451,
    "duplicate_content_groups": 1,
    "members_beyond_unique_content": 1,
    "max_content_multiplicity": 2,
    "cross_actor_duplicate_content_groups": 0
  }
  ```

  Add a synthetic same-actor archive fixture containing two distinct exact validated `ZipInfo.filename` values with identical CSV bytes. Assert the known-answer vector `b"k" * 32`, `01-01-01-01-01-01-01.csv`, and 64 lowercase zeroes gives `trial_o457alx6gmxoxyak`; reject non-ASCII, Unicode-normalized/coerced, path-like, wrong-case/shape names or digests, and a noncanonical/wrong-length key. Generator followed by committed authorization must call the same shared v2 helper and independently recompute from each exact name plus the digest of the same single-read verified bytes, retaining two unique opaque trial IDs and two unique cache IDs joined to the same opaque actor, with no deduplication or exclusion. Reject a valid-shaped v1 manifest/ID/policy and a nominal v2 manifest that uses the old content-only ID, wrong v2 serialization/prefix/policy, a caller-provided ID/digest, or any other construction.

  Force two distinct validated members to receive the same final 80-bit token and require a generic fail-closed collision error before any cache filename is opened, any manifest is written, or any staging or canonical generation is created. Never overwrite, deduplicate, skip, or continue after the collision. Any already-existing transaction residue must remain owner-only and unchanged, and it must block retry before new staging or output work.

  Scan every artifact filename and payload, including NPZ/ZIP entry names and decompressed payloads, plus exceptions, captured logs, stdout, and stderr for each raw member name and source digest as text; the raw 32 digest bytes; lower- and upper-case hexadecimal; Base64; and reversible hex encodings of the raw name, digest text, and digest bytes. Inject each representation into every applicable surface and require rejection. Only the opaque HMAC-derived IDs may survive; zero raw or reversibly encoded matches are allowed.

- [ ] **Step 2: Write failing Mayo authorization tests.**

  Expose a narrow, public, read-only committed-generation authorizer from `build_mayo_ssl_cache.py` rather than implementing a weaker parallel validator. Its required inputs include the live Mayo source root and the legacy-export audit root as well as cache/exposure/key paths. It must take the output lock, reject unresolved journal/staging/backup state, rebuild expected counts/classification commitments from the frozen inventory, run the full committed cache/exposure validation, and return the recomputed v3 commitment without recovery or mutation. Require the canonical 0600 Mayo key, exact 48 retained MediaPipe caches, all identity/governance joins, and reject duplicate/short/ARKit records as main bridge inputs. Require exact current-euid `0700` for every committed Mayo private directory and exact current-euid/single-link `0600` for every committed cache, internal manifest, and external exposure manifest; chmod-to-`0777`/`0666`, added hard links, hostile umasks, and held-open permission mutation must fail closed. Add an equivalent narrow RAVDESS authorizer that verifies the exact manifest/file set, keyed cache IDs, frozen inventory, and same-descriptor snapshots.

- [ ] **Step 3: Write failing bundle and receipt tests.**

  Each stage bundle must contain exactly:

  ```text
  features              float32 (N,4,32,W)
  valid_mask            bool    (N,4,32)
  timestamps            float32 (N,4,32)
  source_frame_indices  int64   (N,4,32)
  group_ids             unicode (N,)
  ```

  Require RAVDESS `N=2452,W=23` and Mayo `N=736,W=95`; ordered opaque packet `sample_ids` live in the private receipt/manifest, not as a sixth NPZ field. Require RAVDESS 24 unique groups and 2,452 source units; Mayo 46 eligible unique groups/source units with every eligible source unit repeated exactly 16 times, two exact mask-quality exclusions, and the complete 48-recording upstream generation commitment.

  The owner-only receipt must bind:

  - schema and producer code digest;
  - run mode (`smoke` or `formal`), upstream manifest and keyed generation-closure digests;
  - ordered packet sample IDs, upstream source-unit IDs, group IDs, upstream cache integrity IDs, and window starts;
  - exact original-canonical/source-target mapping digest aligned with every bundle slot;
  - feature schema/name digest and explicit clinical23 adapter digest;
  - `bundle_file_count=1`, sample count, source-unit count, unique-group count, upstream-cache count, packet policy, overlap/coverage aggregates, exclusions, output bundle SHA-256 and byte size;
  - a domain-separated `receipt_hmac` over every preceding field, including run mode and producer digest, using the canonical stage key.

  It must contain no raw path, filename, source SHA, key material, Mayo session name, or patient identifier.

  Add sentinels for both Mayo roots and scan the shared bridge generation plus every frozen `inputs/` receipt and manifest/config/split/scaler artifact for the exact absolute values, basename components, relative forms, hexadecimal encoding, and Base64 encoding. Each representation must be absent. For `inventory`, `build-bundles`, `freeze-stage`, and `verify-determinism`, omitting either Mayo root must fail before any output or staging path is created, and stdout/stderr must not echo either supplied root.

- [ ] **Step 4: Run all three changed suites and observe RED for missing authorizers/publisher.**

  ```bash
  /Users/williamqiu/opt/anaconda3/bin/python3 facial_paralysis/tests/test_openface68_semantic.py
  /Users/williamqiu/opt/anaconda3/bin/python3 facial_paralysis/tests/test_build_mayo_ssl_cache.py
  /Users/williamqiu/opt/anaconda3/bin/python3 facial_paralysis/tests/test_dynamic_landmark_ssl_bridge.py
  ```

  Expected: the newly added RAVDESS authorizer, Mayo committed-authorizer, and bridge publication tests fail for the intended missing APIs; all pre-existing tests remain green.

- [ ] **Step 5: Implement same-file-descriptor snapshot authorization.**

  Authorize manifests, keys, and every cache from the bytes actually parsed; recheck identity before publication. Use the new narrow public committed-generation wrappers. The wrapper itself must derive its commitment from live bytes and the canonical key; it cannot accept a caller-provided commitment. Apply the shared exact-tree depth/entry/aggregate-byte budgets, require current-owner private transaction parents, enforce the exact Mayo `0700/0600` tree contract through central stat guards and hostile-umask-safe creators, and make every multi-descriptor/multi-root close path attempt all closes while preserving validation and cleanup failures in the exception chain. Compute producer v3 from two matching, strictly bounded canonical live-semantic graph passes around the complete held source-file snapshot closure; tests must mutate positional/keyword defaults, closure cells, referenced globals, and imported dispatch, and must cover deterministic cycles, unsupported referenced values, bounds, and between-pass mutation.

  Implement the single shared byte-exact RAVDESS v2 trial-ID helper frozen above and call that helper from both generation and committed authorization. Each path must validate the exact `ZipInfo.filename`, compute the lowercase source digest from the same single read of the verified member bytes, and reject caller-provided IDs/digests and every v1/content-only policy. Before creating staging or opening a cache filename, precompute the complete opaque-ID set and reject any final 80-bit collision generically. Generator and authorizer must both require type-exact `format_version=2`, the exact provenance policy, and the identical exact eight-existing-plus-six-topology inventory field union.

  Harden the RAVDESS source publisher before the bridge consumes it: require and continuously recheck a current-euid, non-symlinked, non-group/world-writable held output parent (the canonical data root may remain mode `0755`); while holding its output lock, reject every stale staging/backup/tmp/journal state before creating a new stage; bind publication to the original held staging device/inode; never delete a staging or canonical generation by pathname after failure; retain prepublication staging and postpublication canonical storage as indeterminate evidence; and make that retained state block the next generation attempt. Its committed-generation authorizer remains strictly read-only.

- [ ] **Step 6: Implement the mode-neutral bundle transaction.**

  Stage under the ignored bridge parent, write both bundles and `bundle_generation.json` mode `0600`, fsync files/directories, and validate the two staged bundles plus the dual-stage keyed bundle-generation closure from the bytes actually staged. Before the first canonical/residue check, acquire the persistent `.bridge.lock` exclusively and continuously bind its descriptor/name identity plus the parent `0700` anchor through publication; release only after final canonical validation and descriptor closure. Atomically promote that complete shared generation and never overwrite an existing committed generation implicitly. This transaction has no run mode, config, split, scaler, or mode-bound receipt. Python/macOS path APIs do not offer an atomic condition-by-inode deletion primitive, so a failed prepublication transaction must retain its owner-only staging residue rather than perform a path-based destructive cleanup; the residue makes the outcome auditable and blocks every retry until explicit offline review. It must never publish a partial canonical generation. Exact owner-only `0700` canonical parent directories and persistent locks created before failure may remain, and any prior committed generation remains untouched.

  `freeze-stage` is a separate transaction: acquire the existing bridge lock shared, then acquire/create `<run-root>/.inputs.lock` exclusively before residue checks or live authorization. While holding both in that fixed order, reauthorize the live upstreams, canonical keys, and committed shared bundle; derive the requested mode/config/split/scaler closure; stage the two mode-bound receipts plus all manifest/config/split/scaler artifacts under `<run-root>/.inputs.staging-*`; validate their HMACs and cross-links from staged bytes; fsync; and atomically promote the complete directory to `<run-root>/inputs/`. It must fail closed if `inputs/` already exists or any live authorization changed; it never mutates the shared bundle generation. As above, a failed prepublication freeze retains its owner-only staging residue and blocks retry instead of deleting by pathname; only a fully validated staging tree can become canonical, and a post-publication inconsistency retains the canonical tree as indeterminate evidence. Two concurrent callers must serialize before authorization so the loser sees the committed destination without creating staging.

- [ ] **Step 7: Implement the CLI with canonical, fail-closed paths.**

  The CLI uses explicit subcommands:

  - `initialize-mayo-key`: creates an owner-only staged 32-byte key with `O_EXCL`, no-follow checks, fsync, and mode `0600`, validates the staged bytes and identity, then no-replace renames that valid staging file to the exact canonical key; existing keys are validated and never replaced. Any prepublication failure after staging creation retains the owned staging residue as indeterminate evidence and never publishes a partial canonical key. It is the sole postpublication retry exception: after rejecting staging residue and acquiring the same parent lock, a subsequent call may only revalidate the already-present canonical key and returns `False` if exact, without secret generation, staging, deletion, replacement, or chmod; it fails closed otherwise. Every stale canonical `..mayo_ssl_hmac.key.staging-*` residue blocks before secret generation until explicit offline review;
  - `inventory`: requires `--mayo-data-root` and `--mayo-existing-export-root`, performs read-only live authorization, and prints aggregate counts;
  - `build-bundles`: requires both upstream roots/manifests and canonical keys, including the two explicit live Mayo roots, and atomically publishes the two shared bundles plus a keyed `bundle_generation.json` closure;
  - `freeze-stage --mode smoke|formal`: requires the two explicit live Mayo roots, reauthorizes the live upstreams and shared bundle, then writes the mode-bound HMAC receipts and exact config/split/scaler/manifest artifacts into only that mode namespace;
  - `verify-determinism`: requires the two explicit live Mayo roots, acquires the already-existing bridge lock shared without creating or fsyncing anything, performs two independent live generation prepares entirely in memory, compares their keyed commitments, and validates the exact committed tree through held descriptors before and after those prepares. It performs zero filesystem writes, creates no sibling, and calls no fsync; any missing/unsafe lock or pre-existing `.bridge.verify-*` residue is rejected before authorization. With an optional canonical `--run-root`, it acquires the existing inputs lock shared only after the bridge lock, explicitly rejects sibling `.inputs.staging-*`, and additionally scans the committed smoke/formal `inputs/` and `results/` trees without changing the same seven-field aggregate JSON output.

  It prints aggregate deidentified counts only. Every persisted producer digest uses the frozen v3 combination of the strict two-pass live semantic graph and the complete bounded/rechecked held source-file closure; neither only imported code, only code objects, nor only later disk bytes can authorize a generation.

- [ ] **Step 8: Run tests, privacy scan, and transaction fault injection to GREEN.**

  ```bash
  /Users/williamqiu/opt/anaconda3/bin/python3 facial_paralysis/tests/test_openface68_semantic.py
  /Users/williamqiu/opt/anaconda3/bin/python3 facial_paralysis/tests/test_build_mayo_ssl_cache.py
  /Users/williamqiu/opt/anaconda3/bin/python3 facial_paralysis/tests/test_dynamic_landmark_ssl_bridge.py
  ```

  Expected: every suite reports zero failures. Synthetic receipt/privacy tests must explicitly reject raw paths, filenames, source SHA values, keys, Mayo session identifiers, patient fields, and every raw-root representation defined above across the shared generation and frozen inputs; captured stdout/stderr must also be clean. Transaction fault injection must leave no published partial generation, must retain any private residue whose outcome cannot be proven absent, and must make that residue block retry.

- [ ] **Step 9: Commit Task 2.**

  ```bash
  git add facial_paralysis/tests/test_dynamic_landmark_ssl_bridge.py \
    facial_paralysis/src/pretraining/dynamic_landmark_ssl_bridge.py \
    facial_paralysis/scripts/prepare_dynamic_landmark_ssl_inputs.py \
    facial_paralysis/scripts/prepare_ravdess_semantic23.py \
    facial_paralysis/tests/test_openface68_semantic.py \
    facial_paralysis/scripts/build_mayo_ssl_cache.py \
    facial_paralysis/tests/test_build_mayo_ssl_cache.py
  git commit -m "feat(ssl): bind bridge bundles to source provenance"
  ```

---

## Task 3: Upgrade SSL authorization to receipt-bound manifest v2 and unlock only the real runner

**Files:**

- Modify: `facial_paralysis/tests/test_dynamic_landmark_ssl.py`
- Modify: `facial_paralysis/src/pretraining/dynamic_landmark_ssl.py`
- Modify: `facial_paralysis/scripts/pretrain_dynamic_landmarks.py`

- [ ] **Step 1: Write failing manifest/evidence v2 tamper tests.**

  Upgrade the exact manifest and stage-evidence contracts to bind:

  - `bridge_receipt_sha256`, `receipt_hmac`, run mode, canonical-key file identity, and bridge receipt file identity;
  - ordered opaque packet IDs, source-unit IDs, group IDs, and original canonical mapping digests;
  - `bundle_file_count=1`, `sample_count=2452|736`, `source_unit_count=2452|46`, `unique_group_count=24|46`, `upstream_cache_count=2452|46`, and `exclusion_count=0|2` separately;
  - frozen-stage authorization is stage exact: it requires `exclusion_count=0` for RAVDESS and `exclusion_count=2` for Mayo, carries that claim through the HMAC-bound bridge receipt, and rejects a swapped, missing, boolean, negative, or otherwise mismatched exclusion count before optimizer construction;
  - feature schema/name digest;
  - canonical temporal policy digest;
  - bundle cache commitment/count;
  - upstream aggregate commitment already carried by the receipt.

  Deleting, replacing, chmod-changing, or changing one byte in the canonical key, receipt, manifest, live upstream generation, bundle, split, scaler, config, or prior checkpoint must fail before optimizer construction. A v1 artifact, self-consistent forged JSON/SHA chain, copied public cache ID, or smoke-mode HMAC must not authorize formal training.

- [ ] **Step 2: Write failing unique-frame scaler tests.**

  The private receipt must provide an ordered `source_unit_id` and original canonical 30-Hz index for every bundle slot. Recompute the train-only scaler after deduplicating `(source_unit_id, original_canonical_30hz_index)`; never use packet ID or the bundle-local `0..31` index. A Mayo source unit repeats 16 times, while each packet ID remains unique. Heldout source-unit changes must not affect the scaler. Actor/recording group isolation remains the split authority.

- [ ] **Step 3: Write failing real-runner integration tests.**

  `pretrain_dynamic_landmarks.py` must stay locked when any required file is missing or mismatched. With two exact stage directories, both upstream roots/keys, and a one-epoch smoke config, it must authorize RAVDESS, train one seed, authorize the resulting prior checkpoint, run Mayo, save both checkpoint receipts, and write one execution-only report inside `smoke/<exclusive-run-id>`. Smoke must verify finite train loss, serialization/reload, and lineage without computing or emitting heldout performance. A smoke config/receipt/path cannot be accepted as a formal checkpoint input.

  Require a root-privacy regression over manifest/config/split/scaler artifacts, stage evidence, public checkpoint metadata, private checkpoint receipts, smoke/formal reports, summaries, and captured stdout/stderr. The exact absolute Mayo roots, basenames, relative forms, hexadecimal encodings, and Base64 encodings must all be absent. Omitting either Mayo root from `two-stage` must fail before optimizer construction and before any result staging path is created.

- [ ] **Step 4: Write failing source-boundary parameter tests.**

  Require that RAVDESS training leaves all Mayo projections/decoder byte-identical, Mayo training leaves the RAVDESS adapter/decoder byte-identical, and only the registered shared temporal/attention/pooling modules continue across stages. Verify the downstream transfer allowlist exactly: RAVDESS-only transfers only the allowed landmark half/shared modules; RAVDESS-then-Mayo adds only registered MediaPipe projections; extra, missing, or wrong-shape state fails closed. No source scaler crosses a stage or enters PalsyNet.

- [ ] **Step 5: Run SSL tests and observe RED.**

- [ ] **Step 6: Implement manifest/stage-evidence v2.**

  Extend the immutable authorization object to retain the exact bridge receipt/key/upstream snapshots. Before optimizer construction and before checkpoint mint/load/transfer, reopen the canonical key and live upstream generation through the narrow authorizer, recompute the domain-separated receipt HMAC, and require the same keyed generation closure. Keep the five-field NPZ schema exact and require exactly one aggregate bundle file.

- [ ] **Step 7: Implement train-only unique-frame scaling.**

  Deduplicate valid observations by receipt-provided source-unit ID plus original canonical frame index before computing mean/scale. Preserve current source-specific scaler boundaries and exact artifact recomputation.

- [ ] **Step 8: Replace the unconditional runner lock with exact file authorization.**

  The CLI uses one `two-stage` subcommand and accepts stage artifact directories, bridge receipts, both live upstream roots/manifests, both canonical keys, output root, and `--mode smoke|formal`. For Mayo, "live upstream roots" explicitly means both `--mayo-data-root` and `--mayo-existing-export-root` in addition to cache/exposure/key paths. It reconstructs split/scaler/group/sample order only from exact artifacts. `freeze-stage` first atomically commits an `inputs/` generation. The runner requires that exact committed `inputs/` generation and that `results/` does not exist; it writes checkpoints, checkpoint receipts, and the report under `.results.staging-*`, fsyncs and fully revalidates them, then atomically promotes the complete `results/` tree. Failure publishes no result and preserves the immutable inputs. Formal mode requires the frozen 30-epoch config, all three seeds, and a mode-bound formal receipt. Smoke mode requires one epoch/seed 0 and a mode-bound smoke receipt. No CLI flag may override epochs, optimizer, seeds, packet policy, or output namespace.

- [ ] **Step 9: Run focused tests and deterministic two-stage synthetic integration to GREEN.**

  ```bash
  /Users/williamqiu/opt/anaconda3/bin/python3 facial_paralysis/tests/test_dynamic_landmark_ssl.py
  /Users/williamqiu/opt/anaconda3/bin/python3 facial_paralysis/tests/test_dynamic_landmark_ssl_bridge.py
  ```

  Expected: the integration also scans every synthetic `inputs/` and `results/` file plus captured stdout/stderr for both raw-root sentinels and all required reversible representations; zero matches are allowed.

- [ ] **Step 10: Commit Task 3.**

  ```bash
  git add facial_paralysis/tests/test_dynamic_landmark_ssl.py \
    facial_paralysis/src/pretraining/dynamic_landmark_ssl.py \
    facial_paralysis/scripts/pretrain_dynamic_landmarks.py
  git commit -m "feat(ssl): authorize receipt-bound real pretraining"
  ```

---

## 2026-07-15 post-audit Task 2 re-entry amendment

The current approved-code parent HEAD is `23cfc27bc841480ab0bde8fd8ff984830fc8aebd`. Task 2 and Task 3 APIs already exist on that parent, so the original historical RED expectations for missing APIs must not be rerun or claimed. The existing failed real-data staging remains untouched evidence throughout all plan and code work below.

1. The current plan-only bytes must receive a full plan review. After approval, commit only this lower plan with `docs(ssl): harden bridge execution plan`, verify that the worktree is clean, and record the resulting plan commit hash.

   ```bash
   git add facial_paralysis/docs/superpowers/plans/2026-07-15-dynamic-landmark-ssl-bridge.md
   git commit -m "docs(ssl): harden bridge execution plan"
   git status --short
   git rev-parse HEAD
   ```

   Expected: the plan-only commit succeeds, `git status --short` prints nothing, and the exact resulting hash is recorded before Task 2 re-entry begins.

2. Modify only the already-enumerated Task 2 paths `facial_paralysis/scripts/prepare_ravdess_semantic23.py` and `facial_paralysis/tests/test_openface68_semantic.py`. No other file may change unless this plan is re-amended and reapproved. Add the v2 identity, frozen-topology, known-answer-vector (KAT), collision-preflight, privacy, and CLI tests first.

3. Run the OpenFace semantic suite and observe RED for the exact remaining defects: content-only trial identity, a v1 manifest, and the missing six topology aggregates, KAT, and collision preflight. Record those failures; do not substitute or claim the historical missing-API RED expectations. Existing unrelated tests must stay green.

   ```bash
   /Users/williamqiu/opt/anaconda3/bin/python3 facial_paralysis/tests/test_openface68_semantic.py
   ```

4. Implement the minimal v2 repair in those same two files. Then run all four exact suites, compile the six exact Python files, and run `git diff --check`. Commit only the two authorized files with `fix(ravdess): bind trial identity to archive members`, verify that the worktree is clean, and record the resulting code commit hash.

   ```bash
   /Users/williamqiu/opt/anaconda3/bin/python3 facial_paralysis/tests/test_openface68_semantic.py
   /Users/williamqiu/opt/anaconda3/bin/python3 facial_paralysis/tests/test_build_mayo_ssl_cache.py
   /Users/williamqiu/opt/anaconda3/bin/python3 facial_paralysis/tests/test_dynamic_landmark_ssl_bridge.py
   /Users/williamqiu/opt/anaconda3/bin/python3 facial_paralysis/tests/test_dynamic_landmark_ssl.py
   /Users/williamqiu/opt/anaconda3/bin/python3 -m py_compile \
     facial_paralysis/src/pretraining/dynamic_landmark_ssl_bridge.py \
     facial_paralysis/src/pretraining/dynamic_landmark_ssl.py \
     facial_paralysis/scripts/prepare_dynamic_landmark_ssl_inputs.py \
     facial_paralysis/scripts/pretrain_dynamic_landmarks.py \
     facial_paralysis/scripts/prepare_ravdess_semantic23.py \
     facial_paralysis/scripts/build_mayo_ssl_cache.py
   git diff --check
   git add facial_paralysis/scripts/prepare_ravdess_semantic23.py \
     facial_paralysis/tests/test_openface68_semantic.py
   git commit -m "fix(ravdess): bind trial identity to archive members"
   git status --short
   git rev-parse HEAD
   ```

   Expected: all four suites pass, compilation and `git diff --check` succeed, the code commit contains only the two authorized paths, `git status --short` prints nothing, and the exact resulting hash is recorded.

5. Restart the entire Task 4 exact gate from its first command on the new clean HEAD. Only after that exact gate passes, obtain a fresh specification review and then a security/code review against that exact new HEAD. No real-data mutation, quarantine rename, or retry is allowed until Task 4 is approved. Any further code or scientific change invalidates the gate and both reviews and requires the entire sequence again.

---

## Task 4: Mandatory pre-data specification and security review gate

**Files:**

- Review: `facial_paralysis/tests/test_dynamic_landmark_ssl_bridge.py`
- Review: `facial_paralysis/src/pretraining/dynamic_landmark_ssl_bridge.py`
- Review: `facial_paralysis/scripts/prepare_dynamic_landmark_ssl_inputs.py`
- Review: `facial_paralysis/tests/test_dynamic_landmark_ssl.py`
- Review: `facial_paralysis/src/pretraining/dynamic_landmark_ssl.py`
- Review: `facial_paralysis/scripts/pretrain_dynamic_landmarks.py`
- Review: `facial_paralysis/scripts/prepare_ravdess_semantic23.py`
- Review: `facial_paralysis/scripts/build_mayo_ssl_cache.py`

- [ ] Run the exact pre-data verification gate:

  ```bash
  /Users/williamqiu/opt/anaconda3/bin/python3 -c 'import importlib.util,sys,numpy,torch,cv2; assert sys.version_info[:3] == (3,9,12); assert numpy.__version__ == "1.26.4"; assert torch.__version__ == "2.2.1"; assert cv2.__version__ == "4.8.1"; assert importlib.util.find_spec("mediapipe") is None'
  /Users/williamqiu/.cache/facial-paralysis/mediapipe-py310/bin/python -c 'import sys,numpy,torch,mediapipe,cv2; assert sys.version_info[:3] == (3,10,2); assert numpy.__version__ == "1.26.4"; assert torch.__version__ == "2.2.1"; assert mediapipe.__version__ == "0.10.35"; assert cv2.__version__ == "4.11.0"'
  /Users/williamqiu/opt/anaconda3/bin/python3 facial_paralysis/tests/test_openface68_semantic.py
  /Users/williamqiu/opt/anaconda3/bin/python3 facial_paralysis/tests/test_build_mayo_ssl_cache.py
  /Users/williamqiu/opt/anaconda3/bin/python3 facial_paralysis/tests/test_dynamic_landmark_ssl_bridge.py
  /Users/williamqiu/opt/anaconda3/bin/python3 facial_paralysis/tests/test_dynamic_landmark_ssl.py
  /Users/williamqiu/opt/anaconda3/bin/python3 -m py_compile \
    facial_paralysis/src/pretraining/dynamic_landmark_ssl_bridge.py \
    facial_paralysis/src/pretraining/dynamic_landmark_ssl.py \
    facial_paralysis/scripts/prepare_dynamic_landmark_ssl_inputs.py \
    facial_paralysis/scripts/pretrain_dynamic_landmarks.py \
    facial_paralysis/scripts/prepare_ravdess_semantic23.py \
    facial_paralysis/scripts/build_mayo_ssl_cache.py
  git diff --check
  test -z "$(git status --porcelain)"
  PLAN_COMMIT="$(git log -1 --format=%H --extended-regexp --grep='^docs\(ssl\): (freeze|harden) bridge execution plan$')"
  test -n "$PLAN_COMMIT"
  git merge-base --is-ancestor "$PLAN_COMMIT" HEAD
  git diff --quiet "$PLAN_COMMIT" HEAD -- \
    facial_paralysis/docs/superpowers/plans/2026-07-13-dynamic-landmark-pretraining.md \
    facial_paralysis/docs/superpowers/plans/2026-07-15-dynamic-landmark-ssl-bridge.md
  command -v rg >/dev/null
  if git ls-files | rg '(^|/)(\.semantic23_private_id_key|\.mayo_ssl_hmac\.key)$|facial_paralysis/outputs/dynamic_landmark/pretraining/|/derived_semantic23/'; then
    exit 1
  fi
  ```

  Expected: every command exits 0; both runtime tuples match exactly, including MediaPipe being absent from the analysis runtime; all four suites report zero failures; worktree is clean; the approved plan commit is an ancestor and both plan files are byte-unchanged; the tracked-path privacy scan prints nothing.
- [ ] Obtain one independent specification review against this complete plan. Fix every Critical/Important and repeat the whole review.
- [ ] Only after spec approval, obtain an independent code-quality/security review covering forged bundles/receipts, key/path/FD races, Mayo exact private modes/hard links/hostile umasks, persistent bridge/inputs lock identity and global lock order, concurrent writer/verifier serialization, explicit sibling-residue rejection, exact tree/manifest/ZIP resource bounds, producer v3 defaults/closures/globals/import dispatch and semantic-graph bounds, close-all error handling, retained recovery state, mode confusion, source/group joins, scaler leakage, time/gap semantics, transfer allowlists, and output transactions.
- [ ] Fix every Critical/Important and repeat review until both reviewers return READY. Record the exact clean `HEAD` approved by both reviewers. Any producer/trainer/bridge code change or scientific-contract plan change after approval invalidates the gate and requires both reviews again; if formal results already exist, quarantine or remove them and repeat Task 4, `freeze-stage`, and training before they may be cited. The evidence-only documentation updates explicitly scheduled in Task 7 do not alter the producer digest, but they still require the plan-document re-review and final exact-diff reviews defined there before handoff.
- [ ] **Hard gate:** do not create a real key, RAVDESS derived generation, Mayo cache, bridge bundle, smoke output, or formal checkpoint before this task is fully approved.

---

## Task 5: Generate and verify the real source caches and bridge bundles

**Files generated outside Git:**

- `/Users/williamqiu/Desktop/Harvard/Mayo-Clinic/facial_paralysis/data/external/ravdess_facial_tracking/derived_semantic23/`
- `facial_paralysis/outputs/dynamic_landmark/pretraining/.mayo_ssl_hmac.key`
- `facial_paralysis/outputs/dynamic_landmark/pretraining/mayo_ssl_cache/`
- `facial_paralysis/outputs/dynamic_landmark/mayo_exposure_manifest.json`
- `facial_paralysis/outputs/dynamic_landmark/pretraining/bridge/`

- [ ] **Step 1: Re-run read-only frozen inventories immediately before mutation.**

  Require RAVDESS `2452/24/299854` plus exact member/content aggregates `unique_archive_member_names=2452`, `unique_source_content_sha256s=2451`, `duplicate_content_groups=1`, `members_beyond_unique_content=1`, `max_content_multiplicity=2`, and `cross_actor_duplicate_content_groups=0`; require Mayo `65/50/48`, plus the already-reviewed duplicate, short-clip, ARKit, frame, and gap counts.

  ```bash
  /Users/williamqiu/opt/anaconda3/bin/python3 \
    facial_paralysis/scripts/prepare_ravdess_semantic23.py \
    --data-root /Users/williamqiu/Desktop/Harvard/Mayo-Clinic/facial_paralysis/data/external/ravdess_facial_tracking

  /Users/williamqiu/.cache/facial-paralysis/mediapipe-py310/bin/python \
    facial_paralysis/scripts/build_mayo_ssl_cache.py \
    --data-root /Users/williamqiu/Desktop/Harvard/Mayo-Clinic/facial_paralysis/data/livelinkface_data \
    --existing-export-root /Users/williamqiu/Desktop/Harvard/Mayo-Clinic/facial_paralysis/data/mediapipe_out \
    --inventory-only
  ```

  Expected: exit 0; RAVDESS `2452/24/299854`, `unique_archive_member_names=2452`, `unique_source_content_sha256s=2451`, `duplicate_content_groups=1`, `members_beyond_unique_content=1`, `max_content_multiplicity=2`, and `cross_actor_duplicate_content_groups=0`; Mayo `65 sessions`, `50 video`, `48 long unique`, `13 legacy`, `35 pending`, `221121 pending frames`, `8 ARKit trajectories`, `58054 ARKit rows`, `24 gaps`.

- [ ] **Step 2: Generate the canonical RAVDESS semantic23 generation.**

  ```bash
  /Users/williamqiu/opt/anaconda3/bin/python3 \
    facial_paralysis/scripts/prepare_ravdess_semantic23.py \
    --data-root /Users/williamqiu/Desktop/Harvard/Mayo-Clinic/facial_paralysis/data/external/ravdess_facial_tracking \
    --execute
  ```

  Stop if an existing output is present but not byte-for-byte authorized; never overwrite implicitly.

- [ ] **Step 3: Create the canonical Mayo HMAC key owner-only, then re-extract all 48 videos.**

  Use the tested atomic initializer; do not create key bytes with shell redirection or print them.

  ```bash
  /Users/williamqiu/opt/anaconda3/bin/python3 \
    facial_paralysis/scripts/prepare_dynamic_landmark_ssl_inputs.py initialize-mayo-key \
    --key-path /Users/williamqiu/.config/superpowers/worktrees/Mayo-Clinic/landmark-fusion/facial_paralysis/outputs/dynamic_landmark/pretraining/.mayo_ssl_hmac.key

  /Users/williamqiu/.cache/facial-paralysis/mediapipe-py310/bin/python \
    facial_paralysis/scripts/build_mayo_ssl_cache.py \
    --data-root /Users/williamqiu/Desktop/Harvard/Mayo-Clinic/facial_paralysis/data/livelinkface_data \
    --existing-export-root /Users/williamqiu/Desktop/Harvard/Mayo-Clinic/facial_paralysis/data/mediapipe_out \
    --model-path /Users/williamqiu/Desktop/Harvard/Mayo-Clinic/facial_paralysis/data/mediapipe_out/_models/face_landmarker.task \
    --salt-file /Users/williamqiu/.config/superpowers/worktrees/Mayo-Clinic/landmark-fusion/facial_paralysis/outputs/dynamic_landmark/pretraining/.mayo_ssl_hmac.key \
    --output-root /Users/williamqiu/.config/superpowers/worktrees/Mayo-Clinic/landmark-fusion/facial_paralysis/outputs/dynamic_landmark/pretraining/mayo_ssl_cache \
    --exposure-manifest /Users/williamqiu/.config/superpowers/worktrees/Mayo-Clinic/landmark-fusion/facial_paralysis/outputs/dynamic_landmark/mayo_exposure_manifest.json
  ```

  Expected: exit 0; exact deidentified `65/50/48` aggregate; 48 MediaPipe NPZ caches; 8 separate ARKit NPZ caches; no preview video. The key is exactly 32 bytes, regular, owner-owned, non-symlink, single-link, and mode `0600`. Every committed Mayo-derived directory is current-euid mode `0700`; every cache, internal manifest, external exposure manifest, transaction journal, and output lock is current-euid, single-link, regular mode `0600`, independent of umask.

- [ ] **Step 4: Build the two bridge bundles.**

  ```bash
  /Users/williamqiu/opt/anaconda3/bin/python3 \
    facial_paralysis/scripts/prepare_dynamic_landmark_ssl_inputs.py build-bundles \
    --ravdess-data-root /Users/williamqiu/Desktop/Harvard/Mayo-Clinic/facial_paralysis/data/external/ravdess_facial_tracking \
    --ravdess-key /Users/williamqiu/Desktop/Harvard/Mayo-Clinic/facial_paralysis/data/external/ravdess_facial_tracking/.semantic23_private_id_key \
    --mayo-data-root /Users/williamqiu/Desktop/Harvard/Mayo-Clinic/facial_paralysis/data/livelinkface_data \
    --mayo-existing-export-root /Users/williamqiu/Desktop/Harvard/Mayo-Clinic/facial_paralysis/data/mediapipe_out \
    --mayo-cache-root /Users/williamqiu/.config/superpowers/worktrees/Mayo-Clinic/landmark-fusion/facial_paralysis/outputs/dynamic_landmark/pretraining/mayo_ssl_cache \
    --mayo-exposure-manifest /Users/williamqiu/.config/superpowers/worktrees/Mayo-Clinic/landmark-fusion/facial_paralysis/outputs/dynamic_landmark/mayo_exposure_manifest.json \
    --mayo-key /Users/williamqiu/.config/superpowers/worktrees/Mayo-Clinic/landmark-fusion/facial_paralysis/outputs/dynamic_landmark/pretraining/.mayo_ssl_hmac.key \
    --output-root /Users/williamqiu/.config/superpowers/worktrees/Mayo-Clinic/landmark-fusion/facial_paralysis/outputs/dynamic_landmark/pretraining/bridge
  ```

  Require exact output aggregates: one RAVDESS bundle with 2,452 samples/2,452 source units/24 groups/2,452 upstream caches/0 exclusions; one Mayo bundle with 736 samples/46 eligible source units/46 groups/46 upstream caches/2 exact mask-quality exclusions, while its upstream generation commitment still closes over all 48 MediaPipe and eight ARKit caches. The sibling `.bridge.lock` must remain an empty current-euid, single-link regular mode-`0600` file and no `.bridge.staging-*` may remain. In every one of the four windows of every retained packet, the frozen mask must contain at least two non-overlapping contiguous valid spans of length four. Recompute the eligible-start set from the live committed mask and require exactly 46 eligible and two excluded recordings before staging; any other count, any retained packet failure, any duplicate selected start, or any partial source fails the entire generation and publishes nothing.

- [ ] **Step 5: Validate privacy, hashes, modes, disk size, and deterministic rebuild equivalence.**

  ```bash
  VERIFY_JSON="$(/Users/williamqiu/opt/anaconda3/bin/python3 \
    facial_paralysis/scripts/prepare_dynamic_landmark_ssl_inputs.py verify-determinism \
    --ravdess-data-root /Users/williamqiu/Desktop/Harvard/Mayo-Clinic/facial_paralysis/data/external/ravdess_facial_tracking \
    --ravdess-key /Users/williamqiu/Desktop/Harvard/Mayo-Clinic/facial_paralysis/data/external/ravdess_facial_tracking/.semantic23_private_id_key \
    --mayo-data-root /Users/williamqiu/Desktop/Harvard/Mayo-Clinic/facial_paralysis/data/livelinkface_data \
    --mayo-existing-export-root /Users/williamqiu/Desktop/Harvard/Mayo-Clinic/facial_paralysis/data/mediapipe_out \
    --mayo-cache-root /Users/williamqiu/.config/superpowers/worktrees/Mayo-Clinic/landmark-fusion/facial_paralysis/outputs/dynamic_landmark/pretraining/mayo_ssl_cache \
    --mayo-exposure-manifest /Users/williamqiu/.config/superpowers/worktrees/Mayo-Clinic/landmark-fusion/facial_paralysis/outputs/dynamic_landmark/mayo_exposure_manifest.json \
    --mayo-key /Users/williamqiu/.config/superpowers/worktrees/Mayo-Clinic/landmark-fusion/facial_paralysis/outputs/dynamic_landmark/pretraining/.mayo_ssl_hmac.key \
    --bridge-root /Users/williamqiu/.config/superpowers/worktrees/Mayo-Clinic/landmark-fusion/facial_paralysis/outputs/dynamic_landmark/pretraining/bridge)"
  /Users/williamqiu/opt/anaconda3/bin/python3 -c \
    'import json,sys; r=json.loads(sys.argv[1]); assert r == {"bundle_count":2,"bundle_total_bytes":r["bundle_total_bytes"],"deterministic":True,"modes_ok":True,"non_0600_private_file_count":0,"privacy_ok":True,"size_ok":True}; assert 0 < r["bundle_total_bytes"] <= 104857600' \
    "$VERIFY_JSON"
  ```

  The CLI itself must fail nonzero unless all seven claims are true. `privacy_ok` covers the committed generation and emitted JSON and rejects raw paths, filenames, source SHA values, key bytes, patient/session identifiers, and patient fields; `modes_ok` covers every private bundle/closure file; `size_ok` enforces the same 100 MiB bound checked above. Expected: both commands exit 0, the JSON contains only those exact aggregate fields, and verification proves determinism with two independent in-memory prepares plus held-descriptor committed-tree validation while holding the existing `.bridge.lock` shared and performing zero filesystem writes, lock creation, sibling creation, or fsync.

- [ ] **Step 6: Run the authorized one-epoch smoke path.**

  First freeze mode-bound smoke receipts/artifacts, then execute the isolated smoke runner:

  ```bash
  /Users/williamqiu/opt/anaconda3/bin/python3 \
    facial_paralysis/scripts/prepare_dynamic_landmark_ssl_inputs.py freeze-stage \
    --mode smoke --run-id preflight-seed0 \
    --bridge-root /Users/williamqiu/.config/superpowers/worktrees/Mayo-Clinic/landmark-fusion/facial_paralysis/outputs/dynamic_landmark/pretraining/bridge \
    --ravdess-data-root /Users/williamqiu/Desktop/Harvard/Mayo-Clinic/facial_paralysis/data/external/ravdess_facial_tracking \
    --ravdess-key /Users/williamqiu/Desktop/Harvard/Mayo-Clinic/facial_paralysis/data/external/ravdess_facial_tracking/.semantic23_private_id_key \
    --mayo-data-root /Users/williamqiu/Desktop/Harvard/Mayo-Clinic/facial_paralysis/data/livelinkface_data \
    --mayo-existing-export-root /Users/williamqiu/Desktop/Harvard/Mayo-Clinic/facial_paralysis/data/mediapipe_out \
    --mayo-cache-root /Users/williamqiu/.config/superpowers/worktrees/Mayo-Clinic/landmark-fusion/facial_paralysis/outputs/dynamic_landmark/pretraining/mayo_ssl_cache \
    --mayo-exposure-manifest /Users/williamqiu/.config/superpowers/worktrees/Mayo-Clinic/landmark-fusion/facial_paralysis/outputs/dynamic_landmark/mayo_exposure_manifest.json \
    --mayo-key /Users/williamqiu/.config/superpowers/worktrees/Mayo-Clinic/landmark-fusion/facial_paralysis/outputs/dynamic_landmark/pretraining/.mayo_ssl_hmac.key

  /Users/williamqiu/opt/anaconda3/bin/python3 \
    facial_paralysis/scripts/pretrain_dynamic_landmarks.py two-stage \
    --mode smoke \
    --run-root /Users/williamqiu/.config/superpowers/worktrees/Mayo-Clinic/landmark-fusion/facial_paralysis/outputs/dynamic_landmark/pretraining/smoke/preflight-seed0 \
    --bridge-root /Users/williamqiu/.config/superpowers/worktrees/Mayo-Clinic/landmark-fusion/facial_paralysis/outputs/dynamic_landmark/pretraining/bridge \
    --ravdess-data-root /Users/williamqiu/Desktop/Harvard/Mayo-Clinic/facial_paralysis/data/external/ravdess_facial_tracking \
    --ravdess-key /Users/williamqiu/Desktop/Harvard/Mayo-Clinic/facial_paralysis/data/external/ravdess_facial_tracking/.semantic23_private_id_key \
    --mayo-data-root /Users/williamqiu/Desktop/Harvard/Mayo-Clinic/facial_paralysis/data/livelinkface_data \
    --mayo-existing-export-root /Users/williamqiu/Desktop/Harvard/Mayo-Clinic/facial_paralysis/data/mediapipe_out \
    --mayo-cache-root /Users/williamqiu/.config/superpowers/worktrees/Mayo-Clinic/landmark-fusion/facial_paralysis/outputs/dynamic_landmark/pretraining/mayo_ssl_cache \
    --mayo-exposure-manifest /Users/williamqiu/.config/superpowers/worktrees/Mayo-Clinic/landmark-fusion/facial_paralysis/outputs/dynamic_landmark/mayo_exposure_manifest.json \
    --mayo-key /Users/williamqiu/.config/superpowers/worktrees/Mayo-Clinic/landmark-fusion/facial_paralysis/outputs/dynamic_landmark/pretraining/.mayo_ssl_hmac.key
  ```

  Expected: exit 0; freeze leaves one persistent empty current-euid/single-link mode-`0600` `.inputs.lock`, one canonical `inputs/`, and zero `.inputs.staging-*`; execution verifies the exact `deterministic_microbatch_full_partition_64` policy and exactly one optimizer step per epoch for each stage (therefore one step per stage in this one-epoch smoke run); finite train loss; both checkpoints serialize/reload with exact lineage; no heldout loss or model-selection metric is computed or emitted. Stop on missing mask span, lineage mismatch, unsafe/missing lock, sibling staging residue, or unexpected runtime/memory pressure.
  The two `.pt` files, two `.pt.receipt.json` files, and one execution-only report must appear together only after atomic `results/` promotion; a fault-injected run must leave `results/` absent.

- [ ] **Step 7: Re-run privacy verification over the committed smoke inputs and results.**

  Repeat the Step 5 `verify-determinism` command with:

  ```bash
  --run-root /Users/williamqiu/.config/superpowers/worktrees/Mayo-Clinic/landmark-fusion/facial_paralysis/outputs/dynamic_landmark/pretraining/smoke/preflight-seed0
  ```

  Expected: the same exact seven-field JSON and `privacy_ok=true`; the scan covers every shared bundle/closure, smoke manifest/config/split/scaler artifact, bridge receipt, checkpoint metadata/receipt, report, and captured persisted log. Any absolute root, basename, relative form, hexadecimal encoding, Base64 encoding, source SHA, key material, session/patient identifier, or non-0600 private file fails nonzero.

---

## Task 6: Freeze artifacts and run the formal three-seed RAVDESS then Mayo experiment

**Files generated outside Git:**

- `facial_paralysis/outputs/dynamic_landmark/pretraining/formal/inputs/receipts/`
- `facial_paralysis/outputs/dynamic_landmark/pretraining/formal/inputs/artifacts/`
- `facial_paralysis/outputs/dynamic_landmark/pretraining/formal/results/checkpoints/`
- `facial_paralysis/outputs/dynamic_landmark/pretraining/formal/results/reports/formal_pretraining_results.json`

- [ ] **Step 1: Freeze deterministic 20% group-held-out splits with split seed 0.**

  RAVDESS must keep all trials from an actor together. Mayo must keep all 16 packets from a recording together. Record and verify train/heldout group sets before training.

- [ ] **Step 2: Freeze exact 30-epoch configs and train-only unique-frame scalers.**

  Require the formal target to be absent/empty, then write exact mode-bound HMAC receipts and JSON artifacts, fsync them, and build stage evidence v2. No subsequent edit is allowed.

  ```bash
  /Users/williamqiu/opt/anaconda3/bin/python3 \
    facial_paralysis/scripts/prepare_dynamic_landmark_ssl_inputs.py freeze-stage \
    --mode formal \
    --bridge-root /Users/williamqiu/.config/superpowers/worktrees/Mayo-Clinic/landmark-fusion/facial_paralysis/outputs/dynamic_landmark/pretraining/bridge \
    --ravdess-data-root /Users/williamqiu/Desktop/Harvard/Mayo-Clinic/facial_paralysis/data/external/ravdess_facial_tracking \
    --ravdess-key /Users/williamqiu/Desktop/Harvard/Mayo-Clinic/facial_paralysis/data/external/ravdess_facial_tracking/.semantic23_private_id_key \
    --mayo-data-root /Users/williamqiu/Desktop/Harvard/Mayo-Clinic/facial_paralysis/data/livelinkface_data \
    --mayo-existing-export-root /Users/williamqiu/Desktop/Harvard/Mayo-Clinic/facial_paralysis/data/mediapipe_out \
    --mayo-cache-root /Users/williamqiu/.config/superpowers/worktrees/Mayo-Clinic/landmark-fusion/facial_paralysis/outputs/dynamic_landmark/pretraining/mayo_ssl_cache \
    --mayo-exposure-manifest /Users/williamqiu/.config/superpowers/worktrees/Mayo-Clinic/landmark-fusion/facial_paralysis/outputs/dynamic_landmark/mayo_exposure_manifest.json \
    --mayo-key /Users/williamqiu/.config/superpowers/worktrees/Mayo-Clinic/landmark-fusion/facial_paralysis/outputs/dynamic_landmark/pretraining/.mayo_ssl_hmac.key
  ```

  Expected: RAVDESS/Mayo 20% deterministic group-held-out splits with seed 0; exact 30-epoch configs; train-only source-unit/index-deduplicated scalers; no checkpoint yet.

- [ ] **Step 3: Run formal seeds 0, 1, and 2.**

  For each seed, mint `ravdess_only`, authorize its exact receipt, initialize Mayo from that exact checkpoint, then mint `ravdess_then_mayo`. Do not mix a RAVDESS checkpoint from another seed.

  ```bash
  /Users/williamqiu/opt/anaconda3/bin/python3 \
    facial_paralysis/scripts/pretrain_dynamic_landmarks.py two-stage \
    --mode formal \
    --run-root /Users/williamqiu/.config/superpowers/worktrees/Mayo-Clinic/landmark-fusion/facial_paralysis/outputs/dynamic_landmark/pretraining/formal \
    --bridge-root /Users/williamqiu/.config/superpowers/worktrees/Mayo-Clinic/landmark-fusion/facial_paralysis/outputs/dynamic_landmark/pretraining/bridge \
    --ravdess-data-root /Users/williamqiu/Desktop/Harvard/Mayo-Clinic/facial_paralysis/data/external/ravdess_facial_tracking \
    --ravdess-key /Users/williamqiu/Desktop/Harvard/Mayo-Clinic/facial_paralysis/data/external/ravdess_facial_tracking/.semantic23_private_id_key \
    --mayo-data-root /Users/williamqiu/Desktop/Harvard/Mayo-Clinic/facial_paralysis/data/livelinkface_data \
    --mayo-existing-export-root /Users/williamqiu/Desktop/Harvard/Mayo-Clinic/facial_paralysis/data/mediapipe_out \
    --mayo-cache-root /Users/williamqiu/.config/superpowers/worktrees/Mayo-Clinic/landmark-fusion/facial_paralysis/outputs/dynamic_landmark/pretraining/mayo_ssl_cache \
    --mayo-exposure-manifest /Users/williamqiu/.config/superpowers/worktrees/Mayo-Clinic/landmark-fusion/facial_paralysis/outputs/dynamic_landmark/mayo_exposure_manifest.json \
    --mayo-key /Users/williamqiu/.config/superpowers/worktrees/Mayo-Clinic/landmark-fusion/facial_paralysis/outputs/dynamic_landmark/pretraining/.mayo_ssl_hmac.key
  ```

  Expected: exit 0; exactly six formal checkpoints and six adjacent checkpoint receipts; three seed-matched RAVDESS-to-Mayo lineages. All thirteen result files (six checkpoints, six receipts, one report) appear only through one atomic `results/` promotion; no overwrite or partial result tree is possible.

- [ ] **Step 4: Aggregate only preregistered reconstruction evidence.**

  Report actor-held-out RAVDESS and recording-held-out Mayo masked SmoothL1 versus untrained and train-mean baselines, separately by seed and as mean/SD. Do not choose a seed or epoch by the heldout result.

- [ ] **Step 5: Verify every checkpoint and receipt from disk.**

  Reload with exact authorization, confirm state schemas and lineage, and confirm all validation actions leave caller RNG unchanged. Repeat Task 5 Step 5 `verify-determinism` with `--run-root /Users/williamqiu/.config/superpowers/worktrees/Mayo-Clinic/landmark-fusion/facial_paralysis/outputs/dynamic_landmark/pretraining/formal`; require the same exact seven-field JSON and scan every formal `inputs/` and `results/` artifact, checkpoint, receipt, report, and persisted log for both raw Mayo roots and all forbidden representations defined in the frozen contract.

- [ ] **Step 6: Commit only code, tests, plan, and small deidentified aggregate metadata.**

  Write the trackable aggregate-only result to `facial_paralysis/outputs/dynamic_landmark/pretraining_summary.json`. It may contain stage/seed metrics, checkpoint fingerprints, code/schema versions, and claim boundaries, but no local path, key, receipt mapping, raw cache SHA, source name, or PHI. Never stage keys, raw data, bridge receipts, bundles, checkpoints, or ignored formal reports.

---

## Task 7: Post-training verification, documentation, and non-executable handoff to the locked PalsyNet outer experiment

**Files:**

- Create: `facial_paralysis/outputs/dynamic_landmark/pretraining_summary.json`
- Modify: `facial_paralysis/docs/superpowers/plans/2026-07-13-dynamic-landmark-pretraining.md`
- Modify: `facial_paralysis/docs/landmark_research_20260713.md`
- Modify: `facial_paralysis/docs/model_design.md`
- Modify: `facial_paralysis/docs/training_runs.md`
- Modify: `facial_paralysis/autoresearch_fp/FINDINGS.md`

- [ ] Generate the deidentified summary, then update `2026-07-13-dynamic-landmark-pretraining.md`, research documentation, and findings with exact checkpoint hashes and claim boundaries. These are evidence-only updates: do not alter this frozen scientific contract or producer/trainer behavior.
- [ ] Run Task 1–3 focused tests plus all previously completed source-builder/SSL tests on that exact final diff.
- [ ] Run `py_compile`, static checks, `git diff --check`, `git status --short`, and tracked-file privacy/size scans on that exact final diff.
- [ ] Obtain a fresh plan-document review of the evidence-only upper-plan amendment, then independent final specification review and independent final code-quality/security review of the exact complete diff. Fix every Critical/Important and repeat all affected verification and reviews after every change, including a pure documentation correction.
- [ ] If any review requires a producer, trainer, bridge, or scientific-contract change, declare every formal result made by the prior producer invalid, move it to an explicitly non-citable quarantine or remove it, and return to Task 4 for both gate reviews followed by a new `freeze-stage` and complete three-seed training. Never retain the old checkpoint or summary as current evidence.
- [ ] Freeze only the non-executable handoff contract in the upper plan: its later Task 7 must implement the single shared Task 3 checkpoint load/transfer authorizer, accept the RAVDESS live root/key, both Mayo live roots plus cache/exposure/key, the committed bridge root, and the immutable formal run root, and run adversarial/privacy tests before any registry is created. It must also create a persistent path-free `O_EXCL` one-shot claim before any checkpoint load/prediction and never delete that claim after success, failure, or interruption. This bridge task must not modify evaluator code, create the final registry/claim, or run/view outer predictions.
- [ ] Only after the exact evidence diff passes all reviews and is committed cleanly may the deidentified summary be handed to upper-plan Task 7. Upper-plan Task 7 then owns the separate reviewed evaluator-code commit, subsequent path-free registry commit, and one-shot outer run; none of those actions are part of this bridge-plan commit.
- [ ] Stage only the exact files listed above after a tracked-file privacy/size audit, then commit the verified implementation and deidentified summary.

  ```bash
  git add facial_paralysis/outputs/dynamic_landmark/pretraining_summary.json \
    facial_paralysis/docs/superpowers/plans/2026-07-13-dynamic-landmark-pretraining.md \
    facial_paralysis/docs/landmark_research_20260713.md \
    facial_paralysis/docs/model_design.md \
    facial_paralysis/docs/training_runs.md \
    facial_paralysis/autoresearch_fp/FINDINGS.md
  git commit -m "docs(ssl): freeze dynamic pretraining evidence"
  ```

Focused verification:

```bash
/Users/williamqiu/opt/anaconda3/bin/python3 facial_paralysis/tests/test_openface68_semantic.py
/Users/williamqiu/opt/anaconda3/bin/python3 facial_paralysis/tests/test_build_mayo_ssl_cache.py
/Users/williamqiu/opt/anaconda3/bin/python3 facial_paralysis/tests/test_dynamic_landmark_ssl_bridge.py
/Users/williamqiu/opt/anaconda3/bin/python3 facial_paralysis/tests/test_dynamic_landmark_ssl.py
git diff --check
git status --short
```
