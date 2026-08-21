# Universal Clinical Router v4 Canonicalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Universal Clinical Router v4 the only default model surface, clearly separate historical research, and remove obsolete or sensitive generated artifacts that should not live in the public source tree.

**Architecture:** Add one machine-readable model registry and one Python `current` facade that bind the v4 runtime and artifact. Keep historical source modules importable only by explicit legacy paths, while active documentation points exclusively to v4. Preserve aggregate scientific evidence and the frozen 110D dependency; delete scratch search products, old checkpoints, patient-level derived outputs, and packaged training bundles.

**Tech Stack:** Python 3.10, NumPy, JSON, direct Python tests, Git.

---

### Task 1: Freeze the canonical model contract

**Files:**
- Create: `docs/model_registry.json`
- Create: `src/models/current.py`
- Modify: `src/models/__init__.py`
- Create: `tests/test_current_model_contract_v4.py`

- [x] Write a failing contract test requiring v4 as the sole default package export, exact artifact SHA-256, closed registry schema, and no old champion names in the active registry.
- [x] Run the test and confirm RED because the registry and facade do not exist and `src.models` still exports the pre-v4 network.
- [x] Implement the minimal registry/facade and make direct legacy modules available only through explicit imports.
- [x] Run the contract and existing v4 runtime/release tests to GREEN.

### Task 2: Separate current documentation from history

**Files:**
- Rewrite: `docs/CURRENT_MODEL.md`
- Modify: `docs/PIPELINE.md`
- Create: `docs/archive/README.md`
- Create: `docs/archive/models/pre_universal_v4_model_history.md`
- Modify: `docs/PROGRESS_REPORT.md`
- Modify: `docs/SUMMARY.md`
- Modify: `docs/model_design.md`
- Modify: `docs/模型与训练_中文说明.md`
- Modify: `docs/results/mayo_failure_analysis_robust_inference_v1.md`

- [x] Move the old long-form `CURRENT_MODEL.md` into the archive and prepend an explicit historical warning.
- [x] Write a short current-only `CURRENT_MODEL.md` covering v4 architecture, three participant-disjoint results, Mayo boundary, artifact path, and promotion policy.
- [x] Update the pipeline/read order and place an archived-status banner on every known document that still calls 110D the current champion.
- [x] Extend the contract test so active docs cannot regress to a pre-v4 champion.

### Task 3: Remove obsolete generated artifacts

**Files:**
- Modify: `.gitignore`
- Delete: ignored `.firecrawl/` scratch tree
- Delete: `outputs/checkpoints/`, `outputs/embeddings/`, `outputs/palsynet_bundles_norm/`, `outputs/marlin_probe/`, `outputs/mayo_*`, `outputs/depth*/`, `outputs/viz/`, old run/train manifests and obsolete model outputs
- Delete: `autoresearch_fp/experiments/`, search TSVs, and logs while retaining the minimal archived harness and findings

- [x] Add ignore rules preventing model checkpoints, participant-level arrays/images, scratch search reports, and patient-level cards from being recommitted.
- [x] Remove tracked generated artifacts except the v4 aggregate artifact, its frozen 110D dependency/evidence, and deidentified current benchmark reports.
- [x] Remove ignored local autoresearch scratch after confirming its promoted aggregate hashes are already recorded.
- [x] Test that no tracked path matches the retired artifact registry and that the current artifact still resolves its frozen 110D dependency.

### Task 4: Final verification and commit

- [x] Run current-model contract, v4 runtime/release tests, historical direct-import smoke tests, JSON/SHA validation, secret/identifier scan, and `git diff --check`.
- [x] Inspect the deletion list and verify no raw data, v4 model artifact, frozen 110D artifact, or current aggregate report was removed.
- [x] Stage only canonicalization files and explicit deletions, commit on `codex/universal-phenotype-v3`, and keep the worktree without pushing.
