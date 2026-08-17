# Selective Universal Clinical Router v5 candidate implementation plan

> Execute in the isolated `codex/universal-router-v5-candidate` worktree.  Keep
> Universal Clinical Router v4 immutable and fail closed on missing private
> evidence.  Follow red-green-refactor for every behavior change.

## Task 1: Freeze and test the selective evaluator

**Files**

- Create: `src/evaluation/selective_router_v5.py`
- Create: `tests/test_selective_router_v5.py`

1. Write failing tests for the four exact score formulas, closed input schema,
   deterministic top-coverage selection, class-preserving metrics and the 0.70
   promotion gate.
2. Run the direct test and verify that it fails because the module is absent.
3. Implement the minimum immutable NumPy evaluator.
4. Run direct and UCR4 contract tests; refactor only while green.

## Task 2: Build an authenticated private H200 evidence runner

**Files**

- Create: `scripts/run_selective_universal_router_v5.py`
- Create: `tests/test_run_selective_universal_router_v5.py`

1. Write failing synthetic tests for exact private-NPZ schema, anonymous group
   uniqueness, aggregate-only output, v4 SHA binding and rejection of protected
   or Mayo inputs.
2. Implement a runner that consumes exact-byte private profile NPZs, recomputes
   every score/metric, emits one canonical aggregate JSON and never serializes
   row-level values.
3. Bind every profile payload SHA, v4 model/report SHA and implementation SHA.
4. Verify red-green evidence, malformed NPZ rejection and no-overwrite output.

## Task 3: Reconstruct v4 development evidence on H200

**Private-only output**

- PalsyNet development: exact four-fold group-disjoint 110D original/mirror OOF.
- NeuroFace: exact six-fold clinical and fixed 18-head MARLIN OOF.
- MEEI: exact six-fold two-head cue-sequence OOF.

1. Transfer only committed runner/evaluator code to an owner-private H200 work
   directory and record exact helper/data commitments.
2. Confirm `NVIDIA H200` and confirm protected PalsyNet and Mayo roots are not
   mounted/read by the experiment.
3. Generate one anonymous private NPZ per evidence profile; validate its public
   aggregate against the frozen v4 report before selective evaluation.
4. Run the four-candidate experiment once and export only canonical aggregate
   JSON plus hashes.

## Task 4: Audit and decide

1. Independently recompute all reported metrics from the private NPZs.
2. Test label permutation cannot change retained indices and component-column
   permutation cannot change symmetric candidate scores.
3. Compare the public baseline metrics and v4 hashes byte-for-byte with HEAD.
4. Apply the frozen 0.70 gate.  Do not change the gate after seeing results.

## Task 5: Maintain model/version surfaces

**Files**

- Create: `docs/model_candidates.json`
- Create: `docs/results/universal_clinical_router_v5_candidate.md`
- Create: `docs/results/artifacts/universal_clinical_router_v5_candidate/report.json`
- Modify: `docs/results/README.md`
- Modify: `docs/PIPELINE.md`
- Test: `tests/test_selective_universal_router_v5_release.py`

1. If the gate passes, register the selective layer as a non-current development
   candidate with exact artifact/report hashes.  If it fails, register the
   rejected experiment and its reason instead.
2. Keep the current registry/current import/current-model document byte-identical
   to pre-experiment values.
3. Run targeted tests, all facial-paralysis direct tests, compile, diff/secret
   scans and a clean-clone artifact/hash check.
4. Commit the isolated branch.  Do not push without an explicit user request.
