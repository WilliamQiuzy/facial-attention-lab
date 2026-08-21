# Shared V9 Public Weight Release Plan

> **Objective:** Publish a self-contained, public-safe Shared V9 research release that another machine can clone, verify, load, and run without access to H200 or private clinical data.

## Release boundary

- Publish: selected BLV9-009 code, three-seed full-training weights, frozen scaler, loader/inference code, checksums, model card, and reproducibility metadata.
- Do not publish: raw videos/images, patient identifiers, per-patient features or predictions, labels, private manifests, host paths, credentials, or protected-test artifacts.
- Preserve the frozen V8 deployment release as historical production evidence; designate V9 clearly as the current research model, not a clinically validated product.

## Tasks

1. Add failing tests for a checksum-bound three-seed V9 bundle and deterministic ensemble inference.
2. Implement the minimal V9 release loader/export contract and pass targeted tests.
3. Train the exact selected V9 candidate on all eligible PalsyNet, NeuroFace, and MEEI development participants for seeds 0/1/2 on H200; export only model/scaler tensors and aggregate metadata.
4. Pull the release to the Mac, verify all hashes and inference parity, and add the small weight files to Git with exact allowlist exceptions.
5. Resolve documentation ambiguity: make V9 the current research entry, convert stale current-model documentation into compatibility/archive pointers, and keep useful historical evidence in an archive.
6. Run focused and repository release tests, compile/diff/secret/private-path scans, and verify the public branch plus an anonymously downloaded weight by SHA-256.
7. Push `codex/shared-v9-public-release` and open a draft pull request against `main` for collaborator review.

