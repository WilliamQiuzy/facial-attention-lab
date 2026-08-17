# Universal Clinical Router v4 Pipeline

This is the active reproducibility index. Universal Clinical Router v4 is the
only default model; historical pipelines are explicit archive/research paths.

## Canonical model surface

- `docs/model_registry.json` — machine-readable current/archived registry.
- `docs/model_candidates.json` — non-current candidate and rejection registry.
- `src/models/current.py` — only supported default Python import surface.
- `src/models/universal_clinical_router_v4.py` — evidence routing and frozen
  head execution.
- `docs/results/artifacts/universal_clinical_router_v4/model.json` — executable
  parameters.
- `docs/results/artifacts/universal_clinical_router_v4/report.json` — aggregate
  evidence and claim boundary.

## Data flow

1. Authenticate the recording, participant grouping, task identity, timing
   authority, MediaPipe schema, and modality availability.
2. Build the evidence profile without dataset or institution identity.
3. Produce the required feature representation:
   - free recording: four 32-frame `clinical23_v2` windows to 110D;
   - scripted three-task recording: Landmark, Py-Feat AU, and frozen MARLIN;
   - externally timed seven-action recording: cue-aligned Landmark summaries
     and DCT features.
4. Call the matching v4 expert and fail closed if required evidence is absent.
5. Emit only aggregate research metrics unless a separately authorized private
   workflow permits row-level predictions.

## Reproducibility checks

Run from `facial_paralysis/` with the project Anaconda interpreter:

```bash
/Users/williamqiu/opt/anaconda3/bin/python3 tests/test_current_model_contract_v4.py
/Users/williamqiu/opt/anaconda3/bin/python3 tests/test_universal_clinical_router_v4.py
/Users/williamqiu/opt/anaconda3/bin/python3 tests/test_universal_clinical_router_release_v4.py
/Users/williamqiu/opt/anaconda3/bin/python3 tests/test_selective_universal_router_v5_release.py
```

## Development rules

- Start every candidate from v4 and assign a new version; never silently mutate
  the v4 artifact or reuse an archived checkpoint as the default.
- Preserve participant-disjoint splits and protected-data boundaries.
- Keep raw media, participant-level arrays, checkpoints, and exploratory search
  products outside Git.
- Update `docs/model_registry.json`, `docs/CURRENT_MODEL.md`, the aggregate
  report, tests, and artifact hash together when a successor is promoted.
- Record failed or awaiting-confirmation work in `docs/model_candidates.json`;
  never point `src/models/current.py` at it.

Historical architecture narratives and experiment logs are indexed in
`docs/archive/`. Their scripts remain explicit research tools, not current
inference entrypoints.
