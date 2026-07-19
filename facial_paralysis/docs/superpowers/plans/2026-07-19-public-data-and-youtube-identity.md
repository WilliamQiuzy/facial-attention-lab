# Public Data and YouTube Identity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement code tasks. Data acquisition remains fail-closed and provenance-first.

**Goal:** Expand dynamic facial-geometry pretraining data with no-application public sources while preventing identity leakage, license drift, or unauthorized YouTube material from entering training.

**Architecture:** Every source receives a canonical top-level directory under `data/external`, immutable upstream references, archive hashes, a local inventory, and an explicit training-eligibility decision. Public datasets are adapted to canonical 30-Hz landmark/geometry packets using identity-held-out splits. YouTube stays quarantine-only: local duplicate/identity proposals support human review, but rights, consent, and provenance gates remain independent and mandatory.

**Tech Stack:** OSF/Git official downloads, Git LFS, SHA-256, safe archive extraction, ffprobe/OpenCV/MediaPipe, NumPy, local-only append-only identity review artifacts.

---

## Source policy

- Download only from official project, institutional, or author-maintained endpoints; record URL, retrieval date, file size, SHA-256, license, citation, and exact Git commit where relevant.
- No dataset becomes training eligible solely because it is publicly downloadable. Require an explicit compatible license, safe extraction, identity inventory, no cross-split duplicate leakage, and a reviewed feature adapter.
- AST-Face public data remain `training_eligible=false` until its OSF record gains an explicit reuse license or the authors clarify it.
- SZU-EmoDage is synthetic and CC BY 4.0; it can become SSL eligible after integrity, identity grouping, duplicate, and rendering-artifact audits.
- Facing Asymmetry is synthetic and CC BY 4.0; it is a high-priority static asymmetry auxiliary after base-identity grouping and optimization-artifact audit, but it is not clinical palsy ground truth.
- Facing Asymmetry remains `dynamic_ssl_eligible=false`; it needs a separately reviewed static objective and cannot enter the frozen dynamic experiment merely because it is high priority.
- CREMA-D is ODbL/DCL and directly downloadable; preserve attribution/share-alike obligations and keep it SSL-only until the exact use and export terms are reviewed.
- Safe extraction means rejection of absolute/`..` paths, symlinks, special files, duplicate normalized members, case-collisions, and zip bombs before extraction, followed by owner-only staging and no-replace publication. Frozen per-archive limits are: AST-Face 6,000 entries/5 GiB, each SZU archive 2,500 entries/2 GiB, Facing Asymmetry 310,000 entries/5 GiB, Cafca pilot 8,000 entries/2 GiB, and Microsoft FaceSynthetics pilot 3,500 entries/1 GiB; every archive also has a maximum compression ratio of 100:1. Raw pixels and real-actor media remain untracked, mode-`0600` files beneath mode-`0700` directories.
- YouTube public visibility or a YouTube license is not research consent. Written permission, subject/person-cluster consent/IRB scope, current availability, and takedown status remain separate gates.

## Task 1: Finish canonical public dataset acquisition

**Files:**

- Create: `tests/test_safe_archive.py`
- Create: `src/data_acquisition/safe_archive.py`

- [ ] Write RED tests for every path/member/type/case/duplicate/budget/ratio failure above, staged residue, and no-replace publication; observe the missing helper before implementation, then use the helper for every future archive instead of ordinary `unzip`.
- [ ] Preserve AST-Face official archive, verified SHA-256, safe extraction, 98-subject inventory, and owner-only storage at `data/external/ast_face/`.
- [ ] Preserve all three SZU-EmoDage official OSF archives, hashes, safe extraction, license record, 120 synthetic identities, and file-count audit at `data/external/szu_emodage/`.
- [ ] Preserve the official Facing Asymmetry Figshare archive, supplied MD5 plus local SHA-256, safe extraction, CC BY 4.0 record, and observed 201-versus-declared-200 identity discrepancy at `data/external/facing_asymmetry/`; keep `dynamic_ssl_eligible=false`.
- [ ] Clone the official CREMA-D mirror at an exact commit with Git LFS, verify `git lfs fsck`, reject pointer stubs, decode all 7,442 video clips, verify 91 actors and license files, and store owner-only at `data/external/crema_d/`. Record that the official form is requested for access tracking but is not an approval gate, with the official README as evidence.
- [ ] For each dataset, add a concise README and machine-readable local inventory. Keep absolute paths, faces, and raw filenames out of tracked artifacts.

## Task 2: Audit and prioritize additional no-application sources

- [ ] Record official URL, modality, identity count, size, license, and direct-download status for S3DFM, Cafca, IMavatar, MEAD, Microsoft FaceSynthetics, FLUXSynID, and any newly discovered candidate.
- [ ] Reject or defer sources that require a form/EULA/DUA, have an ambiguous license, are medically mismatched, or would consume disproportionate storage without a clear incremental signal.
- [ ] Download at most a representative pilot from very large, noncommercial, or static-only sources until its adapter and incremental contribution are proven.

## Task 3: Build SZU-EmoDage dynamic landmark adapter with TDD

**Files:**

- Create: `tests/test_szu_emodage_adapter.py`
- Create: `src/pretraining/szu_emodage_adapter.py`
- Create: `scripts/prepare_szu_emodage_ssl.py`

- [ ] Write RED tests for exact identity parsing, case/typo normalization, 1-second/3-second pairing, duplicate detection, deterministic identity-held-out split, invalid-video rejection, and path-free output.
- [ ] Use the exact Mayo runtime (`Python 3.10.2`, MediaPipe `0.10.35`, NumPy `1.26.4`, OpenCV `4.11.0.86`), FaceLandmarker VIDEO mode, model SHA-256 `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`, confidence thresholds `0.5`, one face, and the same canonical 30-Hz resampling/detector-mask representation. Bind runtime/model/producer digests into private receipts.
- [ ] Freeze short-video packet policy `valid_quantile_32x4_span4_context_v1`: after canonical 30-Hz resampling, form the sorted start set `F` of 32-frame windows whose detector mask contains two non-overlapping valid spans of length four and at least one additional valid unmasked context frame. A clip contributes exactly four windows at `F[floor(i*(|F|-1)/3)]` for `i=0..3` only when `|F|>=4`; otherwise it enters the exact exclusion manifest. Nominal 1-second clips are normally too short and audit-only, while a well-detected 3-second clip remains usable. Tests must cover gaps and the no-context failure, and a real-inventory smoke must produce nonzero packets before promotion.
- [ ] Extract landmarks/blendshapes without changing source pixels, convert to `clinical23_v2`/95-dimensional canonical packets, and persist detector masks and exact provenance receipts.
- [ ] Audit the four subjects with extra neutral clips and every malformed/missing sequence rather than silently balancing counts.
- [ ] Keep a source-level group ID so frames/clips from one synthetic identity cannot cross splits.

## Task 4: Build CREMA-D adapter only after acquisition audit

**Files:**

- Create: `tests/test_crema_d_adapter.py`
- Create: `src/pretraining/crema_d_adapter.py`
- Create: `scripts/prepare_crema_d_ssl.py`

- [ ] Write RED tests for exact actor/label grammar, actor-held-out split, pointer stub/corrupt video, duplicate media, path-free receipt, and runtime/model mismatch; observe the missing adapter failure before implementation.
- [ ] Detect exact/near duplicate media before extraction and keep duplicate groups within one actor-held-out partition.
- [ ] Reuse the exact frozen SZU/Mayo runtime/model/VIDEO-mode/30-Hz/mask representation and `valid_quantile_32x4_span4_context_v1` short-video policy; bind repository commit plus media, runtime, model, and producer digests.
- [ ] Publish an exact exclusion manifest with one row for every corrupt, zero-duration, wrong-frame, anomalously long, or shorter-than-32-frame clip and its structured reason. Input count must reconcile exactly to packetized plus excluded; errors never silently reduce the 7,442-clip inventory.
- [ ] Treat emotion labels as auxiliary evaluation only, not HB labels.
- [ ] Run the CREMA test to GREEN and independently decode all media before promotion.

## Task 5: Bootstrap YouTube inputs without blessing dirty provenance

- [ ] Accept explicit read-only `--legacy-root` and `--v2-root`; never infer the main workspace or copy its files silently.
- [ ] Inventory and SHA-256 every candidate manifest/event/code input before parsing. Treat the modified legacy manifest and all untracked v2 files in the main workspace as untrusted observations, never as a canonical baseline.
- [ ] Review any v2 implementation file before porting it to the branch with `apply_patch`; record the source digest and require focused tests. Do not overwrite, stage, commit, or normalize the user's current main-workspace manifest.
- [ ] Until the v2 CLI is reviewed and ported, run its read-only status evidence only by the explicit main-workspace absolute path; the legacy worktree CLI is not expected to implement `status`.

## Task 6: Add a local YouTube identity-review layer with RED tests

**Files:**

- Create: `src/data_acquisition/youtube_identity.py`
- Create: `data_acquisition/youtube_identity.py`
- Create: `tests/test_youtube_identity_review.py`
- Modify: `data_acquisition/README.md`
- Modify: `docs/leakage_policy.md`
- Create after bootstrap review: `src/data_acquisition/youtube_catalog.py`
- Modify after bootstrap review: `data_acquisition/youtube_curate.py`

- [ ] Build a deterministic, read-only inventory that reconciles legacy manifest/media/crops and v2 candidates without blessing the currently modified legacy manifest.
- [ ] Split each local video's crop timeline into eight deterministic equal-count bins; select the crop with maximum variance-of-Laplacian sharpness in each bin, breaking ties by earliest frame then byte SHA-256. Resize grayscale to `9x8`, compare adjacent columns to form a 64-bit dHash, and let Hamming distance `<=6` create a proposal only. Exact video/media identity may link content; no perceptual match auto-merges people.
- [ ] Store path-free owner-only local artifacts under `data_acquisition/youtube_v2/identity/`. `decision_events.jsonl` has exact fields `schema_version,event_id,previous_event_sha256,reviewer_id,reviewed_at,decision,left_video_id,right_video_id,evidence_digest`; each event hashes the canonical previous event and appends under a persistent mode-`0600` lock. A deterministic reducer rejects duplicate IDs, broken chains, and same/same/different transitive contradictions by marking the entire involved cluster unresolved.
- [ ] Require explicit human same/different-person decisions before assigning person clusters. Conflicts, multi-face clips, zero-crop clips, and v2 candidates without local media remain unresolved.
- [ ] Person-held-out eligibility requires an exhaustive reviewed pair matrix over every candidate admitted to that evaluation pool, a reviewer-signed coverage-set digest, and zero unresolved/conflicting pairs; otherwise the only permitted claim is `known_duplicate_group_heldout_not_person_heldout`, and unresolved candidates cannot be separated across an evaluation split.
- [ ] Wire confirmed assignments into `validate_rights_gate`, `validate_metadata_gate`, group split validation, downloader dry-run, and SSL export construction. Prove unresolved/conflicted identity, absent permission/consent, stale availability, takedown, or duplicate groups crossing splits remain ineligible end to end.

## Task 7: Verify and choose the next pretraining increment

```bash
/Users/williamqiu/opt/anaconda3/bin/python3 tests/test_szu_emodage_adapter.py
/Users/williamqiu/opt/anaconda3/bin/python3 tests/test_crema_d_adapter.py
/Users/williamqiu/opt/anaconda3/bin/python3 tests/test_youtube_identity_review.py
/Users/williamqiu/opt/anaconda3/bin/python3 /Users/williamqiu/Desktop/Harvard/Mayo-Clinic/facial_paralysis/data_acquisition/youtube_collect.py status --as-of 2026-07-19T00:00:00-04:00
git diff --check
```

- [ ] Compare detector coverage, identity count, motion diversity, duplicate rate, and feature distribution against the existing RAVDESS/Mayo bridge.
- [ ] Add only the highest-value eligible dataset to a new preregistered pretraining experiment; do not change the frozen formal fusion run after viewing results.
- [ ] Obtain spec and code-quality review before any new dataset becomes training eligible.
