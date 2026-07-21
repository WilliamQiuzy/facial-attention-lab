# Pre-work repository baseline

Captured 2026-07-20 before application implementation. The repository was already dirty; this work is limited to `facial_defect_web/**` plus `docs/superpowers/plans/2026-07-20-facial-defect-attention-web.md`.

Commands are run from the repository root and exclude those two allowed paths:

```sh
git status --porcelain=v1 --untracked-files=all -- . \
  ':(exclude)facial_defect_web/**' \
  ':(exclude)docs/superpowers/plans/2026-07-20-facial-defect-attention-web.md' | shasum -a 256

git diff --binary -- . \
  ':(exclude)facial_defect_web/**' \
  ':(exclude)docs/superpowers/plans/2026-07-20-facial-defect-attention-web.md' | shasum -a 256
```

- Outside-scope status fingerprint: `fb635d1a4bec1d36b635104e3f6e053a2f73a8e6573bed5ebce7707c1c5e0a1e`
- Outside-scope tracked diff fingerprint: `d1663a367a8362217ee0d6a64916a07edf1de6e64a0e9bbf63c7514a7b008094`
- Outside-scope status entries at capture: `248`

The original two-asset prototype allowlist was superseded before release. The delivered exact-ten allowlist and sanitized generation evidence are versioned in `approved-synthetic-provenance.json`; retired prototype assets and the complete local generation log remain outside the release tree.

They are distinct AI-generated identities and are approved only for a simulated, unpaired interface demonstration.

## Handoff verification

Recomputed on 2026-07-20 with the exact commands above:

- Outside-scope status fingerprint: `fb635d1a4bec1d36b635104e3f6e053a2f73a8e6573bed5ebce7707c1c5e0a1e` — matches capture
- Outside-scope tracked diff fingerprint: `d1663a367a8362217ee0d6a64916a07edf1de6e64a0e9bbf63c7514a7b008094` — matches capture
- Outside-scope status entries: `248` — matches capture

No outside-scope change was introduced by this implementation.
