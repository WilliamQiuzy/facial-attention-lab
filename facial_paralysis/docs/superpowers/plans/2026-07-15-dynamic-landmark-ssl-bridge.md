# Dynamic Landmark SSL Bridge and Real Pretraining Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert every authorized RAVDESS semantic23 trajectory and every retained Mayo MediaPipe trajectory into provenance-bound canonical 30-Hz SSL packets, authorize the exact bundles, and run the frozen three-seed RAVDESS then Mayo pretraining experiment.

**Architecture:** A new bridge module sits between the already-reviewed source caches and the already-reviewed SSL trainer. It validates the upstream HMAC-bound manifests and exact NPZ schemas, applies a content-independent uniform window policy, emits one exact five-field bundle plus a mode-bound owner-only HMAC receipt per stage, and upgrades SSL authorization so checkpoints are cryptographically bound to the live upstream generation, canonical key, bridge policy, and run mode. RAVDESS and Mayo keep separate feature adapters/scalers while sharing only the intended temporal representation; ARKit remains auxiliary-only and is rejected by this bridge.

**Tech Stack:** Python 3.10; NumPy; PyTorch; existing script-style `Check` tests; MediaPipe `0.10.35` extraction environment for Mayo; Anaconda analysis runtime; SHA-256/HMAC provenance; transactional local-only outputs.

---

## Frozen scientific and data contracts

- RAVDESS input is `semantic23_v1`, all 2,452 trials and all 24 actor groups. No trial is excluded for being shorter than 128 frames: observed canonical lengths are 88–191 frames.
- RAVDESS emits one `(4,32,23)` packet per trial. Window starts are `floor(i * (T - 32) / 3)` for `i=0..3`; overlap is allowed for short trials and must be recorded, not hidden.
- Mayo input is the full retained set of 48 unique long recordings from the homogeneous MediaPipe VIDEO-mode cache. The 13 legacy exports remain audit-only and are never reused.
- Mayo emits exactly 16 `(4,32,95)` packets per recording, or 768 samples total. For `M=64`, starts are `floor(j * (T - 32) / 63)` for `j=0..63`; packet `k` uses starts `k`, `k+16`, `k+32`, `k+48`.
- Window selection depends only on canonical trajectory length. Features, detector validity, movement amplitude, asymmetry, labels, and future evaluation results cannot move a window.
- Both stages use canonical 30-Hz windows with expected step `1`. In every bundle window, `timestamps` is exactly `float32([0..31] / 30)` and `source_frame_indices` is exactly `int64([0..31])`; neither array carries a recording offset. Original canonical 30-Hz indices, upstream source/target indices, and source timestamps stay only in the private receipt.
- Missing detector rows remain zero-valued with `valid_mask=False`; no interpolation, nearest fill, compression, or gap bridging is permitted.
- RAVDESS uses exact `semantic23_v1` names/order. Mayo uses exact `72 + clinical23_v2`; its final 23 values must be explicitly checked through `clinical23_v2_to_semantic23`, never accepted by width alone.
- RAVDESS split unit is actor. Mayo split unit is recording; all 16 packets from one recording remain together. Mayo claims remain `recording_held_out_not_patient_held_out`.
- ARKit 52d is rejected by the main bridge and remains auxiliary-only.
- Frozen training configuration for both formal stages: seeds `0,1,2`; AdamW; learning rate `0.001`; weight decay `0.0001`; 30 epochs; full train partition; span length `4`; two spans per window; CPU. A separately HMAC-attested one-epoch smoke config may test execution in an exclusive disposable namespace but cannot mint or initialize formal checkpoints.
- Existing Mayo recordings remain development-only. No Mayo HB accuracy or patient-held-out generalization claim is allowed.

## Canonical local outputs

All bridge and checkpoint outputs remain ignored and owner-local under:

```text
facial_paralysis/outputs/dynamic_landmark/pretraining/
  bridge/
    bundles/{ravdess_bundle,mayo_bundle}.npz
    bundle_generation.json
  smoke/<exclusive-run-id>/
    inputs/receipts/{ravdess,mayo}.json
    inputs/artifacts/{ravdess,mayo}/{manifest,config,split,scaler}.json
    results/checkpoints/{ravdess_only,ravdess_then_mayo}.pt
    results/checkpoints/{ravdess_only,ravdess_then_mayo}.pt.receipt.json
    results/reports/execution_only.json
  formal/
    inputs/receipts/{ravdess,mayo}.json
    inputs/artifacts/{ravdess,mayo}/{manifest,config,split,scaler}.json
    results/checkpoints/seed_{0,1,2}/{ravdess_only,ravdess_then_mayo}.pt
    results/checkpoints/seed_{0,1,2}/{ravdess_only,ravdess_then_mayo}.pt.receipt.json
    results/reports/formal_pretraining_results.json
```

The RAVDESS source generation remains canonical at `<ravdess-data-root>/derived_semantic23/`; the Mayo generation remains canonical at the reviewed worktree-local `outputs/dynamic_landmark/pretraining/mayo_ssl_cache/` plus `outputs/dynamic_landmark/mayo_exposure_manifest.json`.

## Plan approval checkpoint

Before Task 1, this plan must receive `APPROVED` from the plan-document reviewer. Commit this exact plan together with the 2026-07-13 amendment, record the commit hash, and require a clean worktree. Any later plan change invalidates that approval and requires full plan re-review before execution continues.

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
  - Mayo `K=16` produces exactly 64 starts and the exact quartile-interleaved packet layout;
  - changing features, masks, labels, or movement values does not change starts;
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
      selection: str = "uniform_floor_v1"

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

  Given `derived_semantic23/manifest.json`, its owner-only `.semantic23_private_id_key`, and `trials/*.npz`, require exact manifest fields, `semantic23_v1` names/order, 2,452 unique opaque trials, 24 actor groups, exact cache filenames, recomputed private-key cache integrity IDs, safe NPZ fields, and the frozen aggregate counts. Reject a changed manifest, cache byte, key identity/mode, trial/group join, duplicate ID, raw filename/path, or partial generation.

- [ ] **Step 2: Write failing Mayo authorization tests.**

  Expose a narrow, public, read-only committed-generation authorizer from `build_mayo_ssl_cache.py` rather than implementing a weaker parallel validator. It must take the output lock, reject unresolved journal/staging/backup state, rebuild expected counts/classification commitments from the frozen inventory, run the full committed cache/exposure validation, and return the recomputed v3 commitment without recovery or mutation. Require the canonical 0600 Mayo key, exact 48 retained MediaPipe caches, all identity/governance joins, and reject duplicate/short/ARKit records as main bridge inputs. Add an equivalent narrow RAVDESS authorizer that verifies the exact manifest/file set, keyed cache IDs, frozen inventory, and same-descriptor snapshots.

- [ ] **Step 3: Write failing bundle and receipt tests.**

  Each stage bundle must contain exactly:

  ```text
  features              float32 (N,4,32,W)
  valid_mask            bool    (N,4,32)
  timestamps            float32 (N,4,32)
  source_frame_indices  int64   (N,4,32)
  group_ids             unicode (N,)
  ```

  Require RAVDESS `N=2452,W=23` and Mayo `N=768,W=95`; ordered opaque packet `sample_ids` live in the private receipt/manifest, not as a sixth NPZ field. Require RAVDESS 24 unique groups and 2,452 source units; Mayo 48 unique groups/source units with every source unit repeated exactly 16 times.

  The owner-only receipt must bind:

  - schema and producer code digest;
  - run mode (`smoke` or `formal`), upstream manifest and keyed generation-closure digests;
  - ordered packet sample IDs, upstream source-unit IDs, group IDs, upstream cache integrity IDs, and window starts;
  - exact original-canonical/source-target mapping digest aligned with every bundle slot;
  - feature schema/name digest and explicit clinical23 adapter digest;
  - `bundle_file_count=1`, sample count, source-unit count, unique-group count, upstream-cache count, packet policy, overlap/coverage aggregates, exclusions, output bundle SHA-256 and byte size;
  - a domain-separated `receipt_hmac` over every preceding field, including run mode and producer digest, using the canonical stage key.

  It must contain no raw path, filename, source SHA, key material, Mayo session name, or patient identifier.

- [ ] **Step 4: Run all three changed suites and observe RED for missing authorizers/publisher.**

  ```bash
  /Users/williamqiu/opt/anaconda3/bin/python3 facial_paralysis/tests/test_openface68_semantic.py
  /Users/williamqiu/opt/anaconda3/bin/python3 facial_paralysis/tests/test_build_mayo_ssl_cache.py
  /Users/williamqiu/opt/anaconda3/bin/python3 facial_paralysis/tests/test_dynamic_landmark_ssl_bridge.py
  ```

  Expected: the newly added RAVDESS authorizer, Mayo committed-authorizer, and bridge publication tests fail for the intended missing APIs; all pre-existing tests remain green.

- [ ] **Step 5: Implement same-file-descriptor snapshot authorization.**

  Authorize manifests, keys, and every cache from the bytes actually parsed; recheck identity before publication. Use the new narrow public committed-generation wrappers. The wrapper itself must derive its commitment from live bytes and the canonical key; it cannot accept a caller-provided commitment.

- [ ] **Step 6: Implement the mode-neutral bundle transaction.**

  Stage under the ignored bridge parent, write both bundles and `bundle_generation.json` mode `0600`, fsync files/directories, and validate the two staged bundles plus the dual-stage keyed bundle-generation closure from the bytes actually staged. Atomically promote that complete shared generation and never overwrite an existing committed generation implicitly. This transaction has no run mode, config, split, scaler, or mode-bound receipt. On failure, remove staging and preserve any prior committed generation.

  `freeze-stage` is a separate transaction: after reauthorizing the live upstreams, canonical keys, and committed shared bundle, it derives the requested mode/config/split/scaler closure, stages the two mode-bound receipts plus all manifest/config/split/scaler artifacts under `<run-root>/.inputs.staging-*`, validates their HMACs and cross-links from staged bytes, fsyncs, and atomically promotes the directory to `<run-root>/inputs/`. It must fail closed if `inputs/` already exists or any live authorization changed; it never mutates the shared bundle generation.

- [ ] **Step 7: Implement the CLI with canonical, fail-closed paths.**

  The CLI uses explicit subcommands:

  - `initialize-mayo-key`: atomically creates the exact canonical 32-byte key with `O_EXCL`, no-follow checks, fsync, and mode `0600`; existing keys are validated and never replaced;
  - `inventory`: read-only authorization and aggregate counts;
  - `build-bundles`: requires both upstream roots/manifests and canonical keys and atomically publishes the two shared bundles plus a keyed `bundle_generation.json` closure;
  - `freeze-stage --mode smoke|formal`: reauthorizes the live upstreams and shared bundle, then writes the mode-bound HMAC receipts and exact config/split/scaler/manifest artifacts into only that mode namespace;
  - `verify-determinism`: builds into an internal safe temporary sibling, compares keyed commitments, then removes it; it never accepts an arbitrary output path.

  It prints aggregate deidentified counts only.

- [ ] **Step 8: Run tests, privacy scan, and transaction fault injection to GREEN.**

  ```bash
  /Users/williamqiu/opt/anaconda3/bin/python3 facial_paralysis/tests/test_openface68_semantic.py
  /Users/williamqiu/opt/anaconda3/bin/python3 facial_paralysis/tests/test_build_mayo_ssl_cache.py
  /Users/williamqiu/opt/anaconda3/bin/python3 facial_paralysis/tests/test_dynamic_landmark_ssl_bridge.py
  ```

  Expected: every suite reports zero failures. Synthetic receipt/privacy tests must explicitly reject raw paths, filenames, source SHA values, keys, Mayo session identifiers, and patient fields; transaction fault injection must leave no published partial generation.

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
  - `bundle_file_count=1`, `sample_count=2452|768`, `source_unit_count=2452|48`, `unique_group_count=24|48`, and `upstream_cache_count=2452|48` separately;
  - feature schema/name digest;
  - canonical temporal policy digest;
  - bundle cache commitment/count;
  - upstream aggregate commitment already carried by the receipt.

  Deleting, replacing, chmod-changing, or changing one byte in the canonical key, receipt, manifest, live upstream generation, bundle, split, scaler, config, or prior checkpoint must fail before optimizer construction. A v1 artifact, self-consistent forged JSON/SHA chain, copied public cache ID, or smoke-mode HMAC must not authorize formal training.

- [ ] **Step 2: Write failing unique-frame scaler tests.**

  The private receipt must provide an ordered `source_unit_id` and original canonical 30-Hz index for every bundle slot. Recompute the train-only scaler after deduplicating `(source_unit_id, original_canonical_30hz_index)`; never use packet ID or the bundle-local `0..31` index. A Mayo source unit repeats 16 times, while each packet ID remains unique. Heldout source-unit changes must not affect the scaler. Actor/recording group isolation remains the split authority.

- [ ] **Step 3: Write failing real-runner integration tests.**

  `pretrain_dynamic_landmarks.py` must stay locked when any required file is missing or mismatched. With two exact stage directories, both upstream roots/keys, and a one-epoch smoke config, it must authorize RAVDESS, train one seed, authorize the resulting prior checkpoint, run Mayo, save both checkpoint receipts, and write one execution-only report inside `smoke/<exclusive-run-id>`. Smoke must verify finite train loss, serialization/reload, and lineage without computing or emitting heldout performance. A smoke config/receipt/path cannot be accepted as a formal checkpoint input.

- [ ] **Step 4: Write failing source-boundary parameter tests.**

  Require that RAVDESS training leaves all Mayo projections/decoder byte-identical, Mayo training leaves the RAVDESS adapter/decoder byte-identical, and only the registered shared temporal/attention/pooling modules continue across stages. Verify the downstream transfer allowlist exactly: RAVDESS-only transfers only the allowed landmark half/shared modules; RAVDESS-then-Mayo adds only registered MediaPipe projections; extra, missing, or wrong-shape state fails closed. No source scaler crosses a stage or enters PalsyNet.

- [ ] **Step 5: Run SSL tests and observe RED.**

- [ ] **Step 6: Implement manifest/stage-evidence v2.**

  Extend the immutable authorization object to retain the exact bridge receipt/key/upstream snapshots. Before optimizer construction and before checkpoint mint/load/transfer, reopen the canonical key and live upstream generation through the narrow authorizer, recompute the domain-separated receipt HMAC, and require the same keyed generation closure. Keep the five-field NPZ schema exact and require exactly one aggregate bundle file.

- [ ] **Step 7: Implement train-only unique-frame scaling.**

  Deduplicate valid observations by receipt-provided source-unit ID plus original canonical frame index before computing mean/scale. Preserve current source-specific scaler boundaries and exact artifact recomputation.

- [ ] **Step 8: Replace the unconditional runner lock with exact file authorization.**

  The CLI uses one `two-stage` subcommand and accepts stage artifact directories, bridge receipts, both live upstream roots/manifests, both canonical keys, output root, and `--mode smoke|formal`. It reconstructs split/scaler/group/sample order only from exact artifacts. `freeze-stage` first atomically commits an `inputs/` generation. The runner requires that exact committed `inputs/` generation and that `results/` does not exist; it writes checkpoints, checkpoint receipts, and the report under `.results.staging-*`, fsyncs and fully revalidates them, then atomically promotes the complete `results/` tree. Failure publishes no result and preserves the immutable inputs. Formal mode requires the frozen 30-epoch config, all three seeds, and a mode-bound formal receipt. Smoke mode requires one epoch/seed 0 and a mode-bound smoke receipt. No CLI flag may override epochs, optimizer, seeds, packet policy, or output namespace.

- [ ] **Step 9: Run focused tests and deterministic two-stage synthetic integration to GREEN.**

  ```bash
  /Users/williamqiu/opt/anaconda3/bin/python3 facial_paralysis/tests/test_dynamic_landmark_ssl.py
  /Users/williamqiu/opt/anaconda3/bin/python3 facial_paralysis/tests/test_dynamic_landmark_ssl_bridge.py
  ```

- [ ] **Step 10: Commit Task 3.**

  ```bash
  git add facial_paralysis/tests/test_dynamic_landmark_ssl.py \
    facial_paralysis/src/pretraining/dynamic_landmark_ssl.py \
    facial_paralysis/scripts/pretrain_dynamic_landmarks.py
  git commit -m "feat(ssl): authorize receipt-bound real pretraining"
  ```

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
  PLAN_COMMIT="$(git log -1 --format=%H --grep='^docs(ssl): freeze bridge execution plan$')"
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

  Expected: every command exits 0; all four suites report zero failures; worktree is clean; the approved plan commit is an ancestor and both plan files are byte-unchanged; the tracked-path privacy scan prints nothing.
- [ ] Obtain one independent specification review against this complete plan. Fix every Critical/Important and repeat the whole review.
- [ ] Only after spec approval, obtain an independent code-quality/security review covering forged bundles/receipts, key/path/FD races, recovery state, mode confusion, source/group joins, scaler leakage, time/gap semantics, transfer allowlists, and output transactions.
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

  Require RAVDESS `2452/24/299854` and Mayo `65/50/48`, plus the already-reviewed duplicate, short-clip, ARKit, frame, and gap counts.

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

  Expected: exit 0; RAVDESS `2452/24/299854`; Mayo `65 sessions`, `50 video`, `48 long unique`, `13 legacy`, `35 pending`, `221121 pending frames`, `8 ARKit trajectories`, `58054 ARKit rows`, `24 gaps`.

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

  Expected: exit 0; exact deidentified `65/50/48` aggregate; 48 MediaPipe NPZ caches; 8 separate ARKit NPZ caches; no preview video. The key is exactly 32 bytes, regular, owner-owned, non-symlink, and mode `0600`.

- [ ] **Step 4: Build the two bridge bundles.**

  ```bash
  /Users/williamqiu/opt/anaconda3/bin/python3 \
    facial_paralysis/scripts/prepare_dynamic_landmark_ssl_inputs.py build-bundles \
    --ravdess-data-root /Users/williamqiu/Desktop/Harvard/Mayo-Clinic/facial_paralysis/data/external/ravdess_facial_tracking \
    --ravdess-key /Users/williamqiu/Desktop/Harvard/Mayo-Clinic/facial_paralysis/data/external/ravdess_facial_tracking/.semantic23_private_id_key \
    --mayo-cache-root /Users/williamqiu/.config/superpowers/worktrees/Mayo-Clinic/landmark-fusion/facial_paralysis/outputs/dynamic_landmark/pretraining/mayo_ssl_cache \
    --mayo-exposure-manifest /Users/williamqiu/.config/superpowers/worktrees/Mayo-Clinic/landmark-fusion/facial_paralysis/outputs/dynamic_landmark/mayo_exposure_manifest.json \
    --mayo-key /Users/williamqiu/.config/superpowers/worktrees/Mayo-Clinic/landmark-fusion/facial_paralysis/outputs/dynamic_landmark/pretraining/.mayo_ssl_hmac.key \
    --output-root /Users/williamqiu/.config/superpowers/worktrees/Mayo-Clinic/landmark-fusion/facial_paralysis/outputs/dynamic_landmark/pretraining/bridge
  ```

  Require exact output aggregates: one RAVDESS bundle with 2,452 samples/2,452 source units/24 groups/2,452 upstream caches/0 exclusions; one Mayo bundle with 768 samples/48 source units/48 groups/48 upstream caches/0 exclusions. In every one of the four windows of every packet, the frozen mask must contain at least two non-overlapping contiguous valid spans of length four. If any packet fails, fail the entire generation and publish nothing; never silently exclude or move a window.

- [ ] **Step 5: Validate privacy, hashes, modes, disk size, and deterministic rebuild equivalence.**

  ```bash
  VERIFY_JSON="$(/Users/williamqiu/opt/anaconda3/bin/python3 \
    facial_paralysis/scripts/prepare_dynamic_landmark_ssl_inputs.py verify-determinism \
    --ravdess-data-root /Users/williamqiu/Desktop/Harvard/Mayo-Clinic/facial_paralysis/data/external/ravdess_facial_tracking \
    --ravdess-key /Users/williamqiu/Desktop/Harvard/Mayo-Clinic/facial_paralysis/data/external/ravdess_facial_tracking/.semantic23_private_id_key \
    --mayo-cache-root /Users/williamqiu/.config/superpowers/worktrees/Mayo-Clinic/landmark-fusion/facial_paralysis/outputs/dynamic_landmark/pretraining/mayo_ssl_cache \
    --mayo-exposure-manifest /Users/williamqiu/.config/superpowers/worktrees/Mayo-Clinic/landmark-fusion/facial_paralysis/outputs/dynamic_landmark/mayo_exposure_manifest.json \
    --mayo-key /Users/williamqiu/.config/superpowers/worktrees/Mayo-Clinic/landmark-fusion/facial_paralysis/outputs/dynamic_landmark/pretraining/.mayo_ssl_hmac.key \
    --bridge-root /Users/williamqiu/.config/superpowers/worktrees/Mayo-Clinic/landmark-fusion/facial_paralysis/outputs/dynamic_landmark/pretraining/bridge)"
  /Users/williamqiu/opt/anaconda3/bin/python3 -c \
    'import json,sys; r=json.loads(sys.argv[1]); assert r == {"bundle_count":2,"bundle_total_bytes":r["bundle_total_bytes"],"deterministic":True,"modes_ok":True,"non_0600_private_file_count":0,"privacy_ok":True,"size_ok":True}; assert 0 < r["bundle_total_bytes"] <= 104857600' \
    "$VERIFY_JSON"
  ```

  The CLI itself must fail nonzero unless all seven claims are true. `privacy_ok` covers the committed generation and emitted JSON and rejects raw paths, filenames, source SHA values, key bytes, patient/session identifiers, and patient fields; `modes_ok` covers every private bundle/closure file; `size_ok` enforces the same 100 MiB bound checked above. Expected: both commands exit 0, the JSON contains only those exact aggregate fields, and the implementation creates/removes its own safe sibling.

- [ ] **Step 6: Run the authorized one-epoch smoke path.**

  First freeze mode-bound smoke receipts/artifacts, then execute the isolated smoke runner:

  ```bash
  /Users/williamqiu/opt/anaconda3/bin/python3 \
    facial_paralysis/scripts/prepare_dynamic_landmark_ssl_inputs.py freeze-stage \
    --mode smoke --run-id preflight-seed0 \
    --bridge-root /Users/williamqiu/.config/superpowers/worktrees/Mayo-Clinic/landmark-fusion/facial_paralysis/outputs/dynamic_landmark/pretraining/bridge \
    --ravdess-data-root /Users/williamqiu/Desktop/Harvard/Mayo-Clinic/facial_paralysis/data/external/ravdess_facial_tracking \
    --ravdess-key /Users/williamqiu/Desktop/Harvard/Mayo-Clinic/facial_paralysis/data/external/ravdess_facial_tracking/.semantic23_private_id_key \
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
    --mayo-cache-root /Users/williamqiu/.config/superpowers/worktrees/Mayo-Clinic/landmark-fusion/facial_paralysis/outputs/dynamic_landmark/pretraining/mayo_ssl_cache \
    --mayo-exposure-manifest /Users/williamqiu/.config/superpowers/worktrees/Mayo-Clinic/landmark-fusion/facial_paralysis/outputs/dynamic_landmark/mayo_exposure_manifest.json \
    --mayo-key /Users/williamqiu/.config/superpowers/worktrees/Mayo-Clinic/landmark-fusion/facial_paralysis/outputs/dynamic_landmark/pretraining/.mayo_ssl_hmac.key
  ```

  Expected: exit 0; exactly one optimizer step per stage; finite train loss; both checkpoints serialize/reload with exact lineage; no heldout loss or model-selection metric is computed or emitted. Stop on missing mask span, lineage mismatch, or unexpected runtime/memory pressure.
  The two `.pt` files, two `.pt.receipt.json` files, and one execution-only report must appear together only after atomic `results/` promotion; a fault-injected run must leave `results/` absent.

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
    --mayo-cache-root /Users/williamqiu/.config/superpowers/worktrees/Mayo-Clinic/landmark-fusion/facial_paralysis/outputs/dynamic_landmark/pretraining/mayo_ssl_cache \
    --mayo-exposure-manifest /Users/williamqiu/.config/superpowers/worktrees/Mayo-Clinic/landmark-fusion/facial_paralysis/outputs/dynamic_landmark/mayo_exposure_manifest.json \
    --mayo-key /Users/williamqiu/.config/superpowers/worktrees/Mayo-Clinic/landmark-fusion/facial_paralysis/outputs/dynamic_landmark/pretraining/.mayo_ssl_hmac.key
  ```

  Expected: exit 0; exactly six formal checkpoints and six adjacent checkpoint receipts; three seed-matched RAVDESS-to-Mayo lineages. All thirteen result files (six checkpoints, six receipts, one report) appear only through one atomic `results/` promotion; no overwrite or partial result tree is possible.

- [ ] **Step 4: Aggregate only preregistered reconstruction evidence.**

  Report actor-held-out RAVDESS and recording-held-out Mayo masked SmoothL1 versus untrained and train-mean baselines, separately by seed and as mean/SD. Do not choose a seed or epoch by the heldout result.

- [ ] **Step 5: Verify every checkpoint and receipt from disk.**

  Reload with exact authorization, confirm state schemas and lineage, and confirm all validation actions leave caller RNG unchanged.

- [ ] **Step 6: Commit only code, tests, plan, and small deidentified aggregate metadata.**

  Write the trackable aggregate-only result to `facial_paralysis/outputs/dynamic_landmark/pretraining_summary.json`. It may contain stage/seed metrics, checkpoint fingerprints, code/schema versions, and claim boundaries, but no local path, key, receipt mapping, raw cache SHA, source name, or PHI. Never stage keys, raw data, bridge receipts, bundles, checkpoints, or ignored formal reports.

---

## Task 7: Post-training verification, documentation, and handoff to the locked PalsyNet outer experiment

**Files:**

- Create: `facial_paralysis/outputs/dynamic_landmark/pretraining_summary.json`
- Modify: `facial_paralysis/docs/superpowers/plans/2026-07-13-dynamic-landmark-pretraining.md`
- Modify: `facial_paralysis/docs/landmark_research_20260713.md`
- Modify: `facial_paralysis/docs/model_design.md`
- Modify: `facial_paralysis/docs/training_runs.md`
- Modify: `facial_paralysis/autoresearch_fp/FINDINGS.md`
- Modify or create only after checkpoint freeze: `facial_paralysis/configs/dynamic_landmark_experiment_registry.json`

- [ ] Generate the deidentified summary, then update `2026-07-13-dynamic-landmark-pretraining.md`, research documentation, findings, and the experiment registry with exact checkpoint hashes and claim boundaries. These are evidence-only updates: do not alter this frozen scientific contract or producer/trainer behavior.
- [ ] Run Task 1–3 focused tests plus all previously completed source-builder/SSL tests on that exact final diff.
- [ ] Run `py_compile`, static checks, `git diff --check`, `git status --short`, and tracked-file privacy/size scans on that exact final diff.
- [ ] Obtain a fresh plan-document review of the evidence-only upper-plan amendment, then independent final specification review and independent final code-quality/security review of the exact complete diff. Fix every Critical/Important and repeat all affected verification and reviews after every change, including a pure documentation correction.
- [ ] If any review requires a producer, trainer, bridge, or scientific-contract change, declare every formal result made by the prior producer invalid, move it to an explicitly non-citable quarantine or remove it, and return to Task 4 for both gate reviews followed by a new `freeze-stage` and complete three-seed training. Never retain the old checkpoint or summary as current evidence.
- [ ] Only after the exact final diff passes all reviews, register the two pretrained candidates for the one-shot PalsyNet outer evaluation. Do not view outer predictions during bridge/pretraining work.
- [ ] Stage only the exact files listed above after a tracked-file privacy/size audit, then commit the verified implementation and deidentified summary.

  ```bash
  git add facial_paralysis/outputs/dynamic_landmark/pretraining_summary.json \
    facial_paralysis/docs/superpowers/plans/2026-07-13-dynamic-landmark-pretraining.md \
    facial_paralysis/docs/landmark_research_20260713.md \
    facial_paralysis/docs/model_design.md \
    facial_paralysis/docs/training_runs.md \
    facial_paralysis/autoresearch_fp/FINDINGS.md \
    facial_paralysis/configs/dynamic_landmark_experiment_registry.json
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
