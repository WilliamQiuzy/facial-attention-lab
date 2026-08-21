# Shared V9 Research Pipeline

This is the active reproducibility index. Shared V9 / BLV9-009 is the current
research model; Shared V8 remains the separate deployment record and earlier
pipelines are explicit benchmark or archive paths.

## Canonical model surface

- `docs/model_registry.json` — machine-readable V9 research, V8 deployment,
  v4 benchmark, and archived registry.
- `docs/model_candidates.json` — non-current candidate and rejection registry.
- `src/deployment/shared_v9_research_release.py` — checksum-bound public loader
  and three-member probability ensemble.
- `releases/shared-v9-research-v1/` — complete public scaler and fitted weights.
- `docs/results/artifacts/broad_literature_shared_v9/report.json` — aggregate
  participant-disjoint selection evidence and claim boundary.

## Data flow

1. Authenticate the recording, participant grouping, task identity, timing
   authority, MediaPipe schema, and modality availability.
2. Produce the authenticated 110D clinical geometry for every action and the
   478-point dynamic trajectory when that protocol supports it.
3. Apply the one public scaler embedded identically in all three weight files.
4. Run all three BLV9-009 members through the same shared patient encoder and
   protocol task head, then average their probabilities.
5. Emit only aggregate research metrics unless a separately authorized private
   workflow permits row-level predictions.

## Reproducibility checks

Run from `facial_paralysis/` with the project Anaconda interpreter:

```bash
PYTHONPATH=. python3 tests/test_shared_v9_research_release.py
PYTHONPATH=. python3 tests/test_export_shared_v9_research_release_h200.py
PYTHONPATH=. python3 tests/test_current_model_contract_v4.py
```

## Development rules

- Start every candidate from V9 and assign a new version; never silently mutate
  the public V9 weights or reuse an archived checkpoint as the default.
- Preserve participant-disjoint splits and protected-data boundaries.
- Keep raw media, participant-level arrays, private manifests, and exploratory
  search products outside Git. Only explicitly released aggregate artifacts and
  checksum-bound model tensors may enter the public repository.
- Update `docs/model_registry.json`, `docs/CURRENT_MODEL.md`, the aggregate
  report, tests, and artifact hash together when a successor is promoted.
- Record failed or awaiting-confirmation work in `docs/model_candidates.json`;
  never point `docs/model_registry.json` at it.
- The v6 dense-action candidate is evaluated by
  `scripts/run_dense_action_router_v6.py`; its private full-mesh caches and
  participant-level arrays remain off Git, while only the aggregate report and
  fixed profile registry are public.
- All clinician-facing interfaces, summaries, reports, and presentation copy
  must follow `docs/CLINICIAN_LANGUAGE_POLICY.md`.

Historical architecture narratives and experiment logs are indexed in
`docs/archive/`. Their scripts remain explicit research tools, not current
inference entrypoints.
