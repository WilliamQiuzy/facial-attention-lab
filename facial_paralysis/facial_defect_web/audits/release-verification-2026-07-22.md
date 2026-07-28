# Release verification — 2026-07-22

This record covers the local clinician UX build in `facial_paralysis/facial_defect_web/`.
It is not evidence of model validation, clinical authorization, patient-data readiness,
or public deployment.

## Fresh gates

| Gate | Fresh result |
| --- | --- |
| `CI=1 pnpm exec vitest run --maxWorkers=4` | 37 files, 538 tests passed, 0 failed |
| Focused derivation, gateway, reducer, exact-review, and patient-export contract | 10 files, 202 tests passed, 0 failed |
| `pnpm typecheck` | Passed |
| `pnpm build` | Passed; 1,845 modules transformed |
| `git diff --check` | Passed |
| Playwright clinician journey | 63/63 baseline checks plus focused revised-result desktop, tablet, and mobile checks passed |

## Production asset audit

- Built PNG files: **10**
- Unique built PNG hashes: **10**
- Hash-set difference from `audits/approved-synthetic-provenance.json`: **0**
- Forbidden build-string hits: **0**

The forbidden scan covers real-output paths, facial-paralysis research assets, retired
demo fixtures, and retired inference endpoints. The Vite build also verifies the source
bytes against the approved synthetic allowlist before emitting the bundle.

## Safety and interaction result

- Summary is now the default result view. It presents a fail-closed 3 × 3
  image-relative regional summary with explicit text bands; no interpretation depends
  on color alone.
- Separate source/signal presentation is available, while the original overlay remains
  available as an option.
- Cases, ROI, single-run inference, recent simulations, batch jobs, model comparison,
  result review, and gated patient explanation remain interactive.
- The daily shell exposes only Cases, Reviews, and Help. Runs, Jobs, Models, Methods,
  and Integration remain reachable from Help under Research tools.
- Analysis shows Result and Next step first; technical evidence is retained in a closed
  disclosure. The patient explanation similarly shows the result before optional
  discussion and export details.
- A case switch cannot inherit another case's output. Changed settings keep the previous
  map clearly labeled and require rerun before review.
- Batch submission is atomic. Malformed manifests, excluded cases, and damaged session
  state fail closed without creating partial runs or exposing result evidence.
- Connected outputs remain research-unvalidated and cannot unlock patient preview.
- No upload, browser persistence, PHI, real-patient image, or human-gaze data path exists.
- All mock output remains labeled `SIMULATED — NOT HUMAN GAZE`; clinical use remains blocked.

Detailed browser evidence is in `audits/browser-verification.md`.

## Scope boundary

The final exact-file-filter fingerprints match the amended pre-change protected-scope
baseline byte for byte:

- Protected porcelain entries: **144**
- Status SHA-256: `2d6790eb3f69e8bd48ca4c5da0849a8f29f0c09c42ab745466281917bd7fcef9`
- Tracked binary-diff SHA-256: `d3d9e37029dac7257156cc475fe4d1efb0f1611992968eeda6f2775efb44e8e0`
- Recursive untracked-content SHA-256: `a118bdd2a8dfcc81900708adfefee392915309c44eddf077159e5cd0541fcd03`

No unrelated dirty-tree change is attributed to this implementation. Independent
final specification, code-quality, and visual reviews reported no remaining Critical
or Important findings.
