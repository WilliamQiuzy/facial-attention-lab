# 110D-Generalization v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fail-closed, identity-reviewed development comparison of the three frozen 110D-Generalization v1 representations, establish an auditable YFP regional-ordinal manifest, and prepare MEEI/AFLFP acquisition without opening the protected PalsyNet outer test.

**Architecture:** A small preprocessing module owns the exact 110D/168D/204D feature contracts; a separate aggregate runner owns the fixed logistic protocol and refuses unreviewed identity or unfrozen splits. Identity evidence and YFP provenance are validated by dataset-specific modules, never by the model runner. Outer scoring remains a separate absent/disabled capability until a locked registry is authorized.

**Tech Stack:** `/Users/williamqiu/opt/anaconda3/bin/python3` (Python 3.9.12) for tests, manifests, analysis, SciPy, scikit-learn, and OpenCV; `/Users/williamqiu/.cache/facial-paralysis/mediapipe-py310/bin/python` only for MediaPipe extraction; NumPy, JSON/CSV manifests, SHA-256 provenance, grouped nested cross-validation, paired PalsyNet group bootstrap, YFP subject-cluster bootstrap, and stratified MEEI participant bootstrap.  Bare `python3` is forbidden because it resolves to an incompatible Python 3.14 environment on this Mac.

---

## Task 1: Freeze protocol and regression boundaries

**Files:**
- Add: `docs/superpowers/specs/2026-08-05-110d-generalization-v1.md`
- Add: `docs/superpowers/plans/2026-08-05-110d-generalization-v1.md`
- Modify: `docs/CURRENT_MODEL.md`

- [ ] Record the three exact candidate names, dimensions, shared classifier, candidate-lock rule, claim boundary, and one-shot outer rule.
- [ ] State explicitly that PalsyNet Action/Phase are proxies, YFP is static ordinal auxiliary evidence, and MEEI is the first intended standardized-action/HB external cohort; MEEI does not itself provide frame-level Phase truth.
- [ ] Run `git diff --check` and commit only documentation after the plan review passes.

## Task 2: Strengthen the identity-reviewed gate

**Files:**
- Add: `tests/test_palsynet_identity_review_ledger.py`
- Add: `src/datasets/palsynet_identity_review.py`
- Add: `scripts/finalize_palsynet_identity_review.py`
- Modify: `scripts/audit_palsynet_identity.py`
- Modify: `tests/test_palsynet_identity_audit.py`

- [ ] RED: write tests showing that an arbitrary nonempty reviewer-evidence file, missing recording coverage, any missing decision among all 1,176 unordered recording pairs, unresolved uncertainty, label-informed grouping, duplicate/reversed pair decisions, invalid opaque IDs, digest mismatch, and a cross-label group without adjudication all fail.  Reject non-transitive `same` decisions, `same` across final groups, and `different` within one final group.
- [ ] Run `/Users/williamqiu/opt/anaconda3/bin/python3 tests/test_palsynet_identity_review_ledger.py` and capture the expected import/contract failure.
- [ ] Implement a closed-schema ledger validator bound to the source collection, contact-sheet inventory, complete recording-to-group assignment, pair decisions, and evidence SHA-256.
- [ ] Remove the rule that silently prevents a real same-person group from crossing labels.  Require a separate closed-schema adjudication artifact bound to the source and ledger digests; its only outcomes are whole-group exclusion or a documented correction of a proven source-label error.  Test that a bare flag cannot change eligibility.
- [ ] Add a label-blinded overview sheet plus a review-ledger template covering all 1,176 pairs without publishing face images or identifiers.
- [ ] GREEN: run the two identity test files; run `git diff --check`; commit the identity gate.

## Task 3: Implement the three frozen geometry representations

**Files:**
- Add: `tests/test_110d_generalization_features.py`
- Add: `src/preprocessing/generalization_110d.py`

- [ ] RED: test exact candidate order `(110, 168, 204)`, exact feature names, finite/schema checks, no cross-window derivatives, and exact invariance of the 58D/36D additions to capture-side swaps.
- [ ] RED: test the 36D order as four windows × eye/brow/mouth × excursion/asymmetry/velocity and verify window-position information is preserved.
- [ ] RED: prove the new 110D rows are `np.array_equal` to the frozen champion extractor and, on a fixed synthetic split fixture, its model settings, mirror pairing, and OOF probabilities are bit-identical to the current frozen reference implementation.
- [ ] Run `/Users/williamqiu/opt/anaconda3/bin/python3 tests/test_110d_generalization_features.py` and capture the expected import failure.
- [ ] Implement the 36D coarse-window phase proxy by reusing validated trajectory primitives; concatenate only the frozen 110D and 58D blocks.
- [ ] GREEN: run feature, clinical-dynamics, trajectory-feature, and mirror tests; run `git diff --check`; commit the representation layer.

## Task 4: Build the fail-closed development runner

**Files:**
- Add: `tests/test_110d_generalization_v1.py`
- Add: `scripts/run_110d_generalization_v1.py`

- [ ] RED: require a reviewed identity ledger and frozen patient/group split registry before any candidate extraction.  Verify unreviewed, legacy-video, mismatched-digest, mixed-label-unadjudicated, or missing-registry inputs produce zero feature/fit/prediction audit events.
- [ ] RED: verify all three candidates use the same `C=0.01` L2 logistic settings, grouped weights, mirror augmentation/inference, threshold, inner folds, and aligned validation rows; expose no tuning CLI.
- [ ] RED: permute mirrored rows for each of the 110D, 168D, and 204D candidates and require the pairing audit to fail; inject an outer row into extraction, scaling, fitting, or prediction and require failure.
- [ ] RED: reject report leaf type/range/enum mutations, incoherent counts, invalid metrics, and any identifiers/probabilities; independently recompute all metrics, gates, `passed`, and champion selection before serialization.
- [ ] Implement four-fold group-disjoint inner OOF, fixed metrics, record-to-group probability averaging, and 5,000 paired class-stratified group resamples at seed `20260805`; each repeat resamples affected/unaffected development groups separately, reuses the identical draw for all candidates, and reports 95% percentile intervals for candidate metrics and all pairwise metric deltas.  Lock hierarchically 110D→168D→204D from unrounded point estimates only, with strict AUROC improvement and exact-tie retention of the simpler model, and emit a closed aggregate report.
- [ ] Do not implement real outer scoring; report `outer_evaluation_authorized=false`.
- [ ] GREEN: run the runner tests plus existing mirror/action runner tests; run `git diff --check`; commit the development runner.

## Task 5: Generate and manually complete the PalsyNet identity audit

**Files (local ignored artifacts only):**
- Generate: `outputs/palsynet_identity_audit/generation/identity_manifest.json`
- Generate: `outputs/palsynet_identity_audit/generation/contact_sheet_inventory.json`
- Generate: `outputs/palsynet_identity_audit/generation/contact_sheets/`
- Generate: `outputs/palsynet_identity_audit/generation/review_ledger_template.json`
- Complete: `outputs/palsynet_identity_audit/review/review_ledger.json`
- Complete: `outputs/palsynet_identity_audit/review/reviewer_evidence.json`
- Complete: `outputs/palsynet_identity_audit/review/cross_label_adjudication.json`
- Generate: `outputs/palsynet_identity_audit/reviewed/identity_manifest.json`
- Generate: `outputs/palsynet_identity_audit/person_split_registry.json`

- [ ] Reuse or create exactly one owner-only audit salt; record its commitment but never use salted IDs to assign folds.  From the project root run `/Users/williamqiu/opt/anaconda3/bin/python3 scripts/audit_palsynet_identity.py --video-root /Users/williamqiu/Desktop/Harvard/Mayo-Clinic/facial_paralysis/data/external/palsynet/data --bundle-root /Users/williamqiu/Desktop/Harvard/Mayo-Clinic/facial_paralysis/data/external/palsynet/derived/identity_marlin_v1 --bundle-provenance /Users/williamqiu/Desktop/Harvard/Mayo-Clinic/facial_paralysis/data/external/palsynet/derived/identity_marlin_v1/provenance.json --output-root outputs/palsynet_identity_audit/generation --top-pairs 1176` exactly once for the unreviewed generation; the output root is no-overwrite and can never become the reviewed root.
- [ ] Review every recording overview and all 1,176 unordered recording pairs without labels; record `same`/`different`, resolve every uncertainty, and assign final opaque reviewer-inferred groups.
- [ ] Resolve every uncertain/conflicting decision or keep the gate closed; never infer `patient` from one-video-per-file.
- [ ] Complete the required closed-schema adjudication file after labels are rejoined; use an authenticated empty `decisions` list if no reviewed group crosses labels.  It must bind the source and final ledger digests, and cannot split a group.
- [ ] Validate the structured ledger and create a separate reviewed manifest exactly once with `/Users/williamqiu/opt/anaconda3/bin/python3 scripts/finalize_palsynet_identity_review.py --generated-manifest outputs/palsynet_identity_audit/generation/identity_manifest.json --contact-inventory outputs/palsynet_identity_audit/generation/contact_sheet_inventory.json --review-ledger outputs/palsynet_identity_audit/review/review_ledger.json --reviewer-evidence outputs/palsynet_identity_audit/review/reviewer_evidence.json --cross-label-adjudication outputs/palsynet_identity_audit/review/cross_label_adjudication.json --output outputs/palsynet_identity_audit/reviewed/identity_manifest.json`; the output is no-overwrite and authenticates every listed input digest.
- [ ] Add `tests/test_palsynet_person_split_registry.py` and `scripts/freeze_palsynet_person_split_registry.py`. RED-test fixed semantic group keys from sorted member source digests, fixed domain separator, outer fold `0`, exact five-outer/four-inner coverage, no salt/group-ID dependence, no overwrite, and rejection of any alternate registry.
- [ ] Run `/Users/williamqiu/opt/anaconda3/bin/python3 scripts/freeze_palsynet_person_split_registry.py --reviewed-manifest outputs/palsynet_identity_audit/reviewed/identity_manifest.json --review-ledger outputs/palsynet_identity_audit/review/review_ledger.json --output outputs/palsynet_identity_audit/person_split_registry.json` exactly once before candidate feature extraction and record its SHA-256.
- [ ] Keep all face sheets, salts, mappings, and identifiers ignored and owner-only.

## Task 6: Repair YFP into an audit-first regional manifest

**Files:**
- Add: `tests/test_yfp_region_manifest.py`
- Add: `tests/test_yfp_region_ordinal.py`
- Add: `src/datasets/yfp_region_manifest.py`
- Add: `src/models/l2_cumulative_logit.py`
- Add: `scripts/build_yfp_region_manifest.py`
- Add: `scripts/finalize_yfp_region_manifest.py`
- Add: `scripts/extract_yfp_clinical23.py`
- Add: `scripts/run_yfp_region_ordinal.py`

- [ ] RED: test native valid XML, the single allowed EOF repair, mismatched-tag quarantine, DTD/entity rejection, truncated BMP rejection, image/XML dimension mismatch, unknown/conflicting labels, duplicate digest/key rejection, and symlink/path escape.
- [ ] RED: verify eye/mouth targets are separate 0/1/2 ordinal labels, brow/action/phase are absent, one anchor shares one source/group commitment, and missing license or reviewed group map forces `training_eligible=false`.
- [ ] RED: freeze eye order as fissure-height mean/absdiff, fissure-width mean/absdiff, eye-area mean/absdiff; freeze mouth order as commissure-height mean/absdiff, commissure-radius mean/absdiff, mouth-width, mouth-open.  Test exact capture-swap invariance.
- [ ] RED: test train-fold-only standardization; `P(y<=k)=sigmoid(theta_k-x@beta)`; gap `softplus(raw_gap)+1e-6`; no intercept; group-weighted summed NLL plus `||beta||^2/(2C)` at `C=0.01` with unpenalized cut-points and total weight one per group; zero-beta, `theta0=-0.5`, gap-1 initialization; L-BFGS-B `maxiter=2000, ftol=1e-12, gtol=1e-8, maxls=50`; `1e-12` probability floor; hard failure on non-convergence/nonfinite objective; deterministic group OOF; and no tuning surface.
- [ ] RED: test anchor-level OOF scoring with total metric weight one per subject, exact weighted QWK/balanced-accuracy/macro-grade-MAE recomputation, no subject-label collapsing, and the fixed 5,000-repeat subject-cluster bootstrap contract including the 100,000-attempt failure bound.
- [ ] RED: require the ordinal runner to stop before MediaPipe extraction, fitting, or prediction when license or subject-map eligibility is false; reject static-frame tiling into dynamic 110D.
- [ ] Implement in-memory repair only; never modify the source XML/BMP tree and never use a generic recovery parser.
- [ ] Generate an aggregate audit report and provenance manifest.  Once eligibility passes, extract clinical23 with the MediaPipe Python, then run eye and mouth separately with group-disjoint anchor-level OOF.  Give each subject total metric weight one; report weighted QWK, weighted balanced accuracy, and the mean of the three true-grade-specific weighted argmax-grade MAEs.  For each target, use 5,000 subject-cluster resamples with replacement at seed `20260805`, carry all anchors and subject multiplicity, accept only draws containing all three grades, and fail unless 5,000 valid draws are obtained within 100,000 attempts.  Until then, verify the fail-closed runner without training.
- [ ] Build the audit-first manifest with `/Users/williamqiu/opt/anaconda3/bin/python3 scripts/build_yfp_region_manifest.py --yfp-root /Users/williamqiu/Desktop/Harvard/Mayo-Clinic/facial_paralysis/data/external/YFP --output outputs/yfp_region_manifest_v1/audit/manifest.json`; it is no-overwrite, permanently records `training_eligible=false`, and is never mutated or promoted in place.
- [ ] After the researcher supplies `outputs/yfp_region_manifest_v1/review/license_artifact`, `outputs/yfp_region_manifest_v1/review/reviewed_subject_map.json`, and `outputs/yfp_region_manifest_v1/review/eligibility_authorization.json`, create a separate eligible manifest exactly once with `/Users/williamqiu/opt/anaconda3/bin/python3 scripts/finalize_yfp_region_manifest.py --audit-manifest outputs/yfp_region_manifest_v1/audit/manifest.json --license-artifact outputs/yfp_region_manifest_v1/review/license_artifact --reviewed-subject-map outputs/yfp_region_manifest_v1/review/reviewed_subject_map.json --eligibility-authorization outputs/yfp_region_manifest_v1/review/eligibility_authorization.json --output outputs/yfp_region_manifest_v1/eligible/manifest.json`.  The finalizer revalidates all audit rows and evidence digests; it is no-overwrite and no single flag can authorize training.
- [ ] Only after the eligible successor exists, extract the frozen static clinical23 cache with `/Users/williamqiu/.cache/facial-paralysis/mediapipe-py310/bin/python scripts/extract_yfp_clinical23.py --manifest outputs/yfp_region_manifest_v1/eligible/manifest.json --output-root outputs/yfp_region_manifest_v1/clinical23_v2`, then run `/Users/williamqiu/opt/anaconda3/bin/python3 scripts/run_yfp_region_ordinal.py --manifest outputs/yfp_region_manifest_v1/eligible/manifest.json --feature-cache-root outputs/yfp_region_manifest_v1/clinical23_v2`; both outputs are no-overwrite.
- [ ] GREEN: run the YFP tests, inspect aggregate counts against the read-only inventory, run `git diff --check`, and commit.

## Task 7: Run the reviewer-inferred person/group-disjoint development comparison

**Files (local ignored artifact):**
- Generate: `outputs/dynamic_landmark/benchmarks/development/110d-generalization-v1/report.json`
- Modify after verification: `docs/CURRENT_MODEL.md`
- Modify after verification: `docs/results/current_development_model.json`

- [ ] Authenticate all input, implementation, identity, and split-registry digests.
- [ ] From the project root run `/Users/williamqiu/opt/anaconda3/bin/python3 scripts/run_110d_generalization_v1.py --palsynet-cache-root /Users/williamqiu/Desktop/Harvard/Mayo-Clinic/facial_paralysis/data/external/palsynet/derived/clinical23_v2_windows --reviewed-identity-manifest outputs/palsynet_identity_audit/reviewed/identity_manifest.json --review-ledger outputs/palsynet_identity_audit/review/review_ledger.json --split-registry outputs/palsynet_identity_audit/person_split_registry.json`.  The CLI exposes only authenticated input locations, writes the default aggregate report once with no-overwrite, and exposes no model, candidate, fold, threshold, bootstrap, or output override.
- [ ] Run the exact three candidates once on the reviewed outer-development groups; do not access candidate features or predictions for outer groups.
- [ ] Independently verify aligned OOF coverage, group disjointness, candidate dimensions, fixed model settings, aggregate schema, and zero protected-use counters.
- [ ] Lock the selected candidate by the prespecified rule and publish only aggregate development metrics and digests.
- [ ] Run focused and full relevant test suites; commit the locked development result.

## Task 8: Run the independent MEEI/AFLFP acquisition lane from the start

**Files:**
- Add: `docs/data_access/110d-generalization-v1/MEEI.md`
- Add: `docs/data_access/110d-generalization-v1/AFLFP_APPLICATION.md`
- Add: `docs/data_access/110d-generalization-v1/external_dataset_status.json`
- Add before final PalsyNet fitting: `tests/test_freeze_110d_generalization_v1_artifact.py`
- Add before final PalsyNet fitting: `scripts/freeze_110d_generalization_v1_artifact.py`
- Add before MEEI extraction: `tests/test_meei_dynamic_cache.py`
- Add before MEEI extraction: `scripts/build_meei_participant_manifest.py`
- Add before MEEI extraction: `scripts/extract_meei_clinical23_v2_windows.py`
- Add before MEEI outcomes are exposed: `tests/test_meei_external_v1.py`
- Add before MEEI outcomes are exposed: `scripts/run_meei_external_v1.py`
- Generate locally after final lock: `outputs/dynamic_landmark/artifacts/110d-generalization-v1/final_palsynet_artifact.json`
- Generate locally after MEEI acquisition: `outputs/meei_external_v1/participant_manifest.json`
- Generate locally after MEEI extraction: `/Users/williamqiu/Desktop/Harvard/Mayo-Clinic/facial_paralysis/data/external/MEEI/derived/clinical23_v2_windows/collection_manifest.json`
- Generate locally after explicit authorization: `outputs/meei_external_v1/report.json`

- [ ] Start this non-model-selection lane immediately after plan approval, independently of Tasks 2–7.  Record the current official MEEI open-source link, paper DOI, target labels/actions, access terms, and a hash-first inventory.  Do not claim an application is required if the publisher link is directly accessible.
- [ ] Inventory MEEI file count and declared bytes before download; with only about 33 GiB currently free, refuse acquisition if the staged and final copies cannot fit safely.  If safe, download once into a quarantined ignored directory, hash every member, and keep it sealed from PalsyNet candidate selection.
- [ ] Download the current official AFLFP EULA and prepare the exact official application email for a valid institutional account.
- [ ] Reconcile the tracked `applied 2026-06-18` statement against sent-mail/receipt evidence.  If evidence exists, record `application_sent`; if absent, record `historical_claim_unverified`.  Do not overwrite history with `awaiting_researcher_signature` without this check.
- [ ] Record that the current EULA limits recipients to a full-time faculty researcher or organization employee.  Leave eligibility confirmation, signature, institutional-email sending, and acceptance of terms to that eligible researcher.
- [ ] Define the MEEI external binary contract now: apply the frozen final PalsyNet artifact unchanged, with no MEEI refit, recalibration, threshold change, or model selection, to normal versus facial-palsy participants; HB/eFACE/Sunnybrook are secondary descriptive strata.  A true Phase claim remains blocked until a separately frozen frame-level phase annotation protocol exists.
- [ ] Replace “representation only” with one exact artifact contract: after the one-shot PalsyNet outer result is sealed, fit the locked representation, train-fold-equivalent scaler, and fixed logistic classifier once on all eligible reviewed PalsyNet groups only; freeze its training manifest, scaler, coefficients, threshold, feature registry, and implementation digests before MEEI outcomes are exposed.  Apply that artifact unchanged to MEEI.
- [ ] RED-test the final-artifact freezer: it must reject an unsealed/mismatched outer result, unlocked candidate, unreviewed identity, altered cache/split/report/protocol/implementation digest, missing or duplicate eligible group, model-setting drift, and overwrite.  After the outer result is sealed, it fits the locked representation, train-fold-equivalent scaler, mirror augmentation, and fixed logistic classifier once on all eligible reviewed PalsyNet groups with group-balanced weights and emits only the authenticated no-overwrite model artifact.
- [ ] Generate that artifact exactly once with `/Users/williamqiu/opt/anaconda3/bin/python3 scripts/freeze_110d_generalization_v1_artifact.py --palsynet-cache-root /Users/williamqiu/Desktop/Harvard/Mayo-Clinic/facial_paralysis/data/external/palsynet/derived/clinical23_v2_windows --reviewed-identity-manifest outputs/palsynet_identity_audit/reviewed/identity_manifest.json --review-ledger outputs/palsynet_identity_audit/review/review_ledger.json --split-registry outputs/palsynet_identity_audit/person_split_registry.json --locked-development-report outputs/dynamic_landmark/benchmarks/development/110d-generalization-v1/report.json --sealed-outer-report outputs/dynamic_landmark/benchmarks/protected/110d-generalization-v1/report.json --output outputs/dynamic_landmark/artifacts/110d-generalization-v1/final_palsynet_artifact.json`; this command is forbidden until the protected outer report is sealed.
- [ ] Build and freeze an MEEI participant manifest that inventories and joins every photo/video to exactly one participant and one binary normal/facial-palsy label, with closed enums `media_type={photo,video}` and label-blind `dynamic_binary_eligible`.  Photographs are always ineligible for this dynamic endpoint and remain static-descriptive only; forbid tiling, repetition, interpolation, label-informed exclusion, and ambiguous joins.  Keep HB/eFACE/Sunnybrook as secondary descriptive strata.
- [ ] Build that inventory exactly once with `/Users/williamqiu/opt/anaconda3/bin/python3 scripts/build_meei_participant_manifest.py --data-root /Users/williamqiu/Desktop/Harvard/Mayo-Clinic/facial_paralysis/data/external/MEEI --output outputs/meei_external_v1/participant_manifest.json`; the output is no-overwrite and hashes every media asset before labels are joined.
- [ ] RED-test a provenance-locked MEEI dynamic cache extractor that consumes every manifest video, requires the exact MediaPipe model digest registered by the PalsyNet cache, reuses the same deterministic four-by-32 window/schema/timestamp/mask/source-index/QC contract, fails on raw/static input or selective omission, and writes a no-overwrite cache manifest with participant/media coverage plus all label-blinded exclusion reasons.  No photograph may enter its decode or feature path.
- [ ] After acquisition and manifest freeze, run `/Users/williamqiu/.cache/facial-paralysis/mediapipe-py310/bin/python scripts/extract_meei_clinical23_v2_windows.py --data-root /Users/williamqiu/Desktop/Harvard/Mayo-Clinic/facial_paralysis/data/external/MEEI --participant-manifest outputs/meei_external_v1/participant_manifest.json --palsynet-cache-manifest /Users/williamqiu/Desktop/Harvard/Mayo-Clinic/facial_paralysis/data/external/palsynet/derived/clinical23_v2_windows/collection_manifest.json --model-path /Users/williamqiu/Desktop/Harvard/Mayo-Clinic/facial_paralysis/data/mediapipe_out/_models/face_landmarker.task --output-root /Users/williamqiu/Desktop/Harvard/Mayo-Clinic/facial_paralysis/data/external/MEEI/derived/clinical23_v2_windows`; the output is no-overwrite, and a missing exact model byte artifact keeps MEEI scoring blocked rather than silently substituting a model.
- [ ] For the external endpoint, average all eligible authenticated video probabilities once per participant before every metric.  Never score photographs.  Participants without a compatible video remain in the coverage denominator but not the metric denominator, with exclusions produced solely by frozen label-blinded QC.
- [ ] Preregister participant-level AUROC as the primary endpoint and average precision, Brier score, balanced accuracy, sensitivity, and specificity at threshold `0.5` as secondary endpoints.  Freeze 95% percentile intervals from 5,000 affected/unaffected-stratified participant bootstrap draws at seed `20260805`.
- [ ] RED-test that the external runner rejects any MEEI fit/calibration path, raw/static/incompatible media, incomplete cache coverage, ambiguous participant joins, changed artifact/participant-manifest/cache-manifest/protocol/implementation digests, repeat scoring, malformed aggregate reports, or identifier/label/probability leakage.  It must aggregate authenticated video probabilities once per participant, independently recompute all metrics and coverage counts, expose no tuning/output override, and write one no-overwrite closed-schema report.
- [ ] Only after the PalsyNet one-shot outer result and final artifact are sealed, create a separate explicit one-shot MEEI authorization bound to the final artifact, participant/source manifest, dynamic-cache manifest, implementation, and protocol digests.  Then run `/Users/williamqiu/opt/anaconda3/bin/python3 scripts/run_meei_external_v1.py --final-artifact outputs/dynamic_landmark/artifacts/110d-generalization-v1/final_palsynet_artifact.json --participant-manifest outputs/meei_external_v1/participant_manifest.json --feature-cache-root /Users/williamqiu/Desktop/Harvard/Mayo-Clinic/facial_paralysis/data/external/MEEI/derived/clinical23_v2_windows --authorization docs/registries/110d-generalization-v1-meei-authorization.json`; the default report path is `outputs/meei_external_v1/report.json` and is no-overwrite.  Do not create the authorization or run this command during candidate selection.
- [ ] Store no credentials, tokens, face data, or signed agreements in Git; run `git diff --check`; commit the acquisition packet.

## Task 9: Outer authorization remains sealed

**Files:**
- Add only after the candidate and all digests are frozen: `docs/registries/110d-generalization-v1-outer-template.json`

- [ ] Verify the template contains no result, prediction, or callable bypass and cannot authorize a legacy video-level partition.
- [ ] Require an exact match to the reviewed person mapping, person split registry, locked candidate, source bytes, implementation digest, and protocol digest.
- [ ] Stop before outer scoring and request explicit authorization for the one protected evaluation.
