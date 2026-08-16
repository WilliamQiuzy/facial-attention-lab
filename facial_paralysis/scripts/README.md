# Script support boundary

Universal Clinical Router v4 is the sole current model. The supported model
surface is `src.models.current`; the frozen artifact and exact evaluation
entrypoints are registered in `docs/model_registry.json` and
`docs/CURRENT_MODEL.md`.

There is **No raw-video production CLI** in this research repository. Current
v4 inference accepts already authenticated evidence that satisfies one of the
three frozen profiles. A future end-to-end video command must implement the
same fail-closed identity, task, timing, feature-schema, and artifact checks
before it can be documented as supported.

## Current release/evidence scripts

- `run_110d_generalization_v1.py` and `run_110d_outer_release_v1.py`: frozen
  free-recording expert evidence.
- `run_neuroface_action_capacity_v1.py`: scripted three-action expert evidence.
- `run_meei_external_v1.py`: cue-aligned action expert evidence.
- `audit_mayo_action_anchor_feasibility_v1.py`: Mayo timing feasibility audit;
  it does not produce Mayo classification accuracy.

All other scripts are research utilities or historical reproduction tools.
They are not supported current training or inference entrypoints merely because
they remain importable. Ambiguous pre-v4 generic commands such as
`predict.py`, `train_palsynet.py`, `train_v*_pod.py`, and `run2` through `run10`
have been removed; Git history retains them when an old experiment must be
audited.
