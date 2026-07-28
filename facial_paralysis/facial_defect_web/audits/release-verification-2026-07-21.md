# Release verification — 2026-07-21

This document records fresh evidence from the relocated canonical application at
`facial_paralysis/facial_defect_web/`. It does not represent clinical validation,
model validation, public deployment, or authorization to use patient data.

## Automated verification

| Gate | Command | Fresh result |
| --- | --- | --- |
| Full unit/integration suite | `pnpm test:run` | 35 files, 355 tests passed, 0 failed |
| TypeScript | `pnpm typecheck` | passed, exit 0 |
| Production build | `pnpm build` | passed; Vite transformed 1,843 modules |
| Patch integrity | `git diff --check` | passed, no output |
| Local preview availability | `curl http://127.0.0.1:5173/` | HTTP 200 |

The production build emitted one HTML document, one CSS bundle, one JavaScript
bundle, and exactly ten PNG files.

## Production asset-byte audit

The freshly built `dist/` directory contained:

- PNG files: **10**
- Unique PNG SHA-256 values: **10**
- Exact SHA-256 set match against the canonical allowlist: **yes**
- Missing or extra image hashes: **0**
- Forbidden build strings (`output/real`, `facial_paralysis`, retired `demo-001`
  / `demo-002`, or retired dual endpoints): **0**

The build itself also executed the Vite pre-build verifier, which re-read every
allowlisted source file and rejected any path or byte digest outside the immutable
synthetic boundary.

## Browser verification

The single-session desktop/mobile journey passed with no external request, storage
write, console error, warning, page exception, failed request, or responsive overflow.
It covered all operational lifecycles, strict unknown-ID handling, strict patient
query handling, safe export, stale-result invalidation, and mobile canvas-first
inference layout. Detailed evidence and artifact paths are recorded in
`audits/browser-verification.md`.

## Scope verification

The migration-aware repository-wide and `facial_paralysis/`-subtree outside-scope
fingerprints were computed before and after final verification and matched exactly.
The five pre-existing dirty entries inside the containing research subtree are listed
without attributing ownership in `audits/prework-baseline.md`.

## Release boundary

- Synthetic-only, ten exact hash-pinned standalone identities
- Session memory only; refresh clears operational state
- Default mock gateway performs no inference network request
- Connected gateway requires explicit configuration and never falls back to mock
- No upload, browser persistence, PHI, real-patient asset, or human-gaze dataset
- Patient explanation/export requires one exact approved current mock result and
  automatically locks after stale, malformed, revoked, connected, or replaced state
- All mock output remains permanently labeled simulated and not human gaze
- Clinical use and model promotion remain blocked

## Independent final review

The independent final code and safety re-review returned **READY** with no Critical
or Important findings. The reviewer independently confirmed the 355-test suite,
TypeScript, dist hash set, forbidden-string scan, scope fingerprints, strict query
and malformed-state gates, current Methods/Integration contract, and 390 px
canvas-first inference layout.
