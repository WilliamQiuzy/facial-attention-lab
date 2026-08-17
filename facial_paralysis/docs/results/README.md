# Result status

Universal Clinical Router v4 is the sole current model. Its authoritative
human-readable result is `universal_clinical_router_v4.md`; its exact model and
aggregate report are under `artifacts/universal_clinical_router_v4/` and are
bound by `../model_registry.json`.

New candidates are recorded separately in `../model_candidates.json` so a
failed experiment cannot silently change the default. The latest bounded
experiment, `universal_clinical_router_v5_candidate.md`, tested a selective
confidence layer and was rejected; its aggregate machine report remains under
`artifacts/universal_clinical_router_v5_candidate/` as negative evidence.

Other files in this directory are frozen point-in-time experiments, component
evidence, external-cohort audits, or failed candidate studies. Terms such as
“current,” “champion,” or “promoted” inside those files describe the decision at
the date of that experiment and do not override the v4 registry. They may be
used to audit how a v4 expert was selected, but not as alternative current model
specifications.

`current_development_model.json` is retained as the frozen 110D component record
referenced by v4; it is not a separate current model.
