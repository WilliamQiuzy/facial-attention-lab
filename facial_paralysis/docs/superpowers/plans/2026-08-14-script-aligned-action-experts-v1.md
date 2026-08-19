# Script-Conditioned Action Capacity v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an authoritative script-segmentation contract and an exploratory action-specific bilateral-capacity branch without relabeling unscripted PalsyNet or changing the frozen Bell's-palsy model.

**Architecture:** The current 110D model remains the free-video asymmetry branch. Scripted recordings are cut only by an exogenous event/audio/manual timeline. Three separate fixed L2 Logistic experts learn absolute mouth capacity from NeuroFace's task-identified pucker, opening, and smile videos; they are evaluated participant-disjoint and are not fused into the Bell's-palsy model.

**Tech Stack:** Python 3.9/3.10, NumPy, scikit-learn, existing clinical23_v2/trajectory utilities, direct Python test harness, Nebius H200.

---

### Task 1: Authoritative action timeline and segmentation

**Files:**
- Create: `src/preprocessing/script_action_segmentation_v1.py`
- Test: `tests/test_script_action_segmentation_v1.py`

- [ ] Write RED tests for exact schema/order, SHA-256, source enum, integer-ms
  bounds, optional reanimated smile, 32 uniform positions, tracking threshold,
  NeuroFace `recording_task_label` whole-recording scope, and deterministic
  output. Include a prompted flat-response action and assert `eligible=true`,
  `observed_motion=false`. Reject unanchored script mode.
- [ ] Implement immutable timeline/segment dataclasses and bounded validation.
  Keep `prompted`, `observed_motion`, `tracking_adequate`, and `eligible`
  separate. Visual evidence may not alter prompt timing or eligibility.
- [ ] Run the new direct test plus existing trajectory/action regressions.
- [ ] Commit only timeline code and tests.

### Task 2: Frozen 18D bilateral-capacity features

**Files:**
- Create: `src/preprocessing/action_capacity_features_v1.py`
- Test: `tests/test_action_capacity_features_v1.py`

- [ ] Write RED tests for the exact 18 names, exclusion of all asymmetry/static
  names, flat-response zeros only when all sampled landmarks are valid, masked
  invalid rows never becoming zeros, the 26-of-32 support floor, mirror
  involution, exact four-window-to-one-110D-to-one-18D aggregation, finite
  validation, and refusal to summarize an ineligible interval.
- [ ] Implement name-bound extraction from Landmark 110D vectors and an exact
  side-block mirror transform. Do not expose selectable feature lists.
- [ ] Run new tests plus `tests/test_trajectory_features.py` and
  `tests/test_110d_generalization_features.py`.
- [ ] Commit feature code and tests.

### Task 3: Participant-disjoint task experts

**Files:**
- Create: `src/evaluation/neuroface_action_capacity_v1.py`
- Test: `tests/test_neuroface_action_capacity_v1.py`
- Create: `scripts/run_neuroface_action_capacity_v1.py`
- Test: `tests/test_run_neuroface_action_capacity_v1.py`

- [ ] Write RED tests for exact tasks/order, frozen six folds, no participant
  overlap, one expert per task, fixed `C=0.01`, per-task participant-equal and
  mirror-equal weights, held-out original/mirror probability averaging,
  unweighted three-task participant aggregation, full-mask rule,
  exact `healthy_control=0`/`als+post_stroke=1` target, three-cohort bootstrap
  sizes `11/11/14`, identifier-free report, and zero PalsyNet access counters.
- [ ] Implement task OOF fitting and participant aggregation. Fail closed if a
  train task lacks both classes or any held-out participant lacks a primary
  task. Report mask-only diagnostic and 5,000 fixed-seed cohort-stratified
  participant bootstraps.
- [ ] Implement a no-overwrite runner bound to the frozen NeuroFace private
  manifest, cache collection, folds, feature names, and code hashes. There are
  no tuning CLI arguments.
- [ ] Run all new tests and commit.

### Task 4: H200 exploratory capacity experiment

- [ ] Create a new no-overwrite H200 release and run a container that mounts
  only the frozen NeuroFace input root and new output root. Record the exact
  mount list plus host process-file audit; verify no PalsyNet path is mounted or
  accessed and its access count remains exactly zero.
- [ ] Run once. Record runtime, task coverage, participant metrics, bootstrap
  intervals, exact dependencies, and immutable artifact hashes.
- [ ] Recompute the public aggregate report from private OOF arrays in a
  separate process with numerical tolerance `1e-14`; keep row outputs private.
- [ ] Decide only whether bilateral action capacity is technically supported.
  Never promote, fuse, or claim Bell's-palsy/Mayo accuracy from this result.

### Task 5: Historical Mayo timing feasibility

**Files:**
- Create: `scripts/audit_mayo_action_anchor_feasibility_v1.py`
- Test: `tests/test_mayo_action_anchor_feasibility_v1.py`

- [ ] Write RED tests for read-only media inventory, duplicate-content
  collapse, audio stream/duration accounting, event-sidecar discovery,
  aggregate-only output, and fail-closed eligibility when no audited timeline
  exists.
- [ ] Implement the inventory audit without extracting identity, transcript,
  audio, video, or row-level paths into the public report.
- [ ] Report the current event/audio/manual anchor counts. Do not score Mayo
  action experts unless the pre-transcription private registry selects the 12
  smallest audio-bearing deduplicated source SHAs and its 72 required events
  pass pooled precision/recall 0.95, median all-event temporal IoU 0.80, and the
  fixed set contains at least two manually verified prompted-flat attempts.
- [ ] Commit code, tests, and aggregate-safe feasibility report.

### Task 6: Final decision and verification

**Files:**
- Create: `docs/results/script_conditioned_action_capacity_v1.md`
- Update: `docs/results/current_development_model.json`

- [ ] Document why visual-peak segmentation erases severe bilateral weakness,
  why PalsyNet cannot supply attempt labels, the NeuroFace capacity result, and
  the exact remaining requirement for labeled scripted Mayo patients/controls.
- [ ] Keep the current 110D artifact and development metrics unchanged; add the
  capacity branch as research-only if its feasibility criterion is met.
- [ ] Run every new/affected direct test, Python compilation, independent metric
  recomputation, secret/path/identifier scan, `git diff --check`, and exact diff
  review. Delete temporary PDF renders and never stage private artifacts.
- [ ] Commit protocol, implementation, aggregate reports, and decision. Do not
  merge or push automatically.
