# Script support boundary

Shared V9 / BLV9-009 is the current research model. Its supported release
surface is `src.deployment.shared_v9_research_release`; the complete public
weights and exact identity are registered in `docs/model_registry.json` and
`releases/shared-v9-research-v1/`.

There is **No raw-video production CLI** in this research repository. Current
V9 inference accepts already authenticated evidence that satisfies one of the
three frozen profiles. A future end-to-end video command must implement the
same fail-closed identity, task, timing, feature-schema, and artifact checks
before it can be documented as supported.

## Current release/evidence scripts

- `export_shared_v9_research_release_h200.py`: owner-only full-data training and
  tensor-only export for the locked three-seed V9 ensemble.
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
