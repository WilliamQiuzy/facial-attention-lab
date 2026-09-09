# Public Release Verification — 2026-07-30

## Scope

This release synchronizes the three public Attention Lab projects:

- the synthetic-image generation tools and 10 approved assets;
- the session-only facial reconstruction research prototype;
- the English Vitestro live-device evaluation.

No participant data, patient data, internal generated-image collection,
generation metadata log, trained clinical model, or observer-attention model
is included.

## Web application verification

The synchronized web source passed:

- `pnpm typecheck`;
- `pnpm exec vitest run --maxWorkers=1`: 56 files and 860 tests;
- `pnpm build`: 1,874 modules transformed and a successful production build.

Higher-concurrency runs exposed one or two intermittent asynchronous
state/focus assertion failures after 858–859 successful tests. The affected
test passed in isolation, and the complete single-worker run passed all 860
tests. This remains a test-scheduling stability risk rather than evidence of a
production build failure. The build reported a non-blocking JavaScript
chunk-size warning.

The production build emitted exactly the 10 approved synthetic PNG assets. It
also emitted the pinned MediaPipe Face Landmarker bundle used for same-origin,
on-device facial registration:

`64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`

The model source, version, size, SHA-256, and Apache-2.0 license are documented
in `src/assets/mediapipe/README.md`.

## Public-data boundary

- The 10 approved public PNG files are unchanged from the prior allowlist.
- The larger local synthetic-image collection and metadata log are excluded.
- Environment files, dependency directories, build output, coverage output,
  Firecrawl research cache, private keys, and logs are ignored.
- A repository scan found no committed API key, private key, access token, or
  patient file in the release scope.

## Use boundary

This is a research and interface-rehearsal release. It is not a diagnostic
system, clinical decision aid, validated presyncope detector, or approved
patient workflow.
