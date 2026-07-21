# Workspace location migration

On 2026-07-21 the user explicitly requested that the complete frontend project move under the facial-paralysis workspace.

- Previous repository-relative application path: `facial_defect_web/`
- Current repository-relative application path: `facial_paralysis/facial_defect_web/`
- Plans moved to: `facial_paralysis/docs/superpowers/plans/`

No existing destination was overwritten. The Vite imports and build verifier now resolve the immutable synthetic allowlist from the sibling repository at `../../facial_defect_synthesis` relative to the application root. The canonical runtime source paths and SHA-256 values did not change.

The earlier `prework-baseline.md` remains a historical record of the original root-level implementation boundary. All continued development and verification now use the current paths above.
