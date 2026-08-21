# 110D-Generalization v1

## Goal

Test whether two prespecified, clinically interpretable geometry additions
improve the current mirror-invariant Landmark 110D representation while the
classifier, regularization, split logic, threshold, and evaluation budget stay
fixed.  Candidate selection happens only inside a newly frozen, identity-
reviewed PalsyNet development partition.  The corresponding outer partition
is used once only after the candidate registry is locked.

## Claim boundary

- PalsyNet currently has 49 recordings and 48 provisional groups.  Its present
  identity state is `video_held_out / unreviewed`; it is not yet a patient-
  disjoint cohort.
- PalsyNet has neither standardized action labels nor onset/apex/recovery
  labels.  The v1 additions below are therefore registered as **Action proxy**
  and **Phase proxy** geometry expansions.  They satisfy the requested
  three-model engineering comparison but do not constitute a true Action or
  clinical Phase experiment and cannot establish an action-specific disease
  mechanism.
- YFP is a static, region-severity auxiliary task.  It must never be repeated
  or tiled into a fake `(4, 32, 95)` trajectory and does not test the full
  dynamic 110D representation.
- No result in this protocol is Mayo performance, House-Brackmann accuracy,
  clinical validation, or deployment evidence.  True cross-institutional
  validation starts only after the MEEI standard set is acquired.

## Frozen candidate registry

Every candidate uses the same mirror augmentation and symmetric validation
inference as the current champion, one `StandardScaler` fitted on each training
fold, L2 logistic regression with `C=0.01`, `liblinear`, `max_iter=2000`, random
state `0`, group-balanced sample weights, and threshold `0.5`.  There is no
hyperparameter search, feature selection, interaction expansion, or candidate-
specific calibration.

1. `landmark_mi_110d` — the current frozen 110D trajectory representation.
2. `landmark_mi_110d_action_proxy_168d` — the same 110D plus the frozen 58D
   direction-free clinical-dynamics block.  The block summarizes bilateral
   synchrony, invariant amplitude ratio, absolute lag, excursion, velocity,
   and explicit eye/brow/mouth geometry.
3. `landmark_mi_110d_action_phase_proxy_204d` — the 168D representation plus a
   fixed 36D coarse phase block.  For each of the four deterministic temporal
   windows and each region (eye, brow, mouth), it records three direction-free
   summaries across the region's bilateral landmark pairs: mean bilateral
   excursion, absolute excursion asymmetry, and mean bilateral peak velocity.

The four temporal slots are recording-position phases, not clinically labeled
activation/apex/recovery phases.  This restriction is deliberate: inventing
clinical phases from unlabeled YouTube content would create a false label.  A
future true Phase experiment requires a separate frame-level annotation
protocol; standardized MEEI movements and clinical scores alone are not phase
ground truth.

## Identity-reviewed development gate

Before any three-candidate PalsyNet result is produced:

1. Generate opaque, label-blinded contact sheets for every recording and all
   1,176 unordered recording pairs.
2. Require a structured review ledger bound to the source-collection digest.
   It must cover every recording, provide one final opaque reviewer-inferred
   identity group per recording, record label-blinded `same` or `different`
   decisions for all `49 choose 2 = 1,176` recording pairs, resolve every
   uncertainty, and bind the reviewer evidence digest.  An arbitrary nonempty
   file or nearest-neighbor-only review is insufficient.
   Pair decisions and final groups must agree exactly: `same` if and only if
   two recordings share a group and `different` if and only if they do not.
   The implied `same` relation must be reflexive, symmetric, and transitive.
3. Identity grouping must be performed without using the binary label.  A
   reviewed group that later proves to cross binary labels is not split apart;
   it is ineligible unless a separate closed-schema adjudication artifact,
   bound to both source and review-ledger digests, records exactly one allowed
   outcome: exclude the whole group or correct a proven source-label error with
   documented evidence.  An arbitrary boolean cannot change eligibility.
4. Create the five-outer/four-inner group-disjoint registry exactly once before
   feature extraction.  Split ordering uses a protocol-fixed semantic key made
   from the sorted source SHA-256 members of each reviewed group plus the public
   domain separator `110d-generalization-v1-person-split`; it never uses a
   mutable audit salt or opaque group ID.  Outer fold `0` is fixed, the registry
   is no-overwrite, and alternate regenerations are rejected.  The current
   video-level outer fold becomes a sealed legacy partition and is never
   reinterpreted as patient-held-out.

The audit has two immutable stages.  Initial generation writes only an
`unreviewed` manifest, contact-sheet inventory, and blank ledger template under
`outputs/palsynet_identity_audit/generation/`; that directory is no-overwrite.
The completed ledger, reviewer evidence, and a required closed-schema
cross-label adjudication file (empty decisions are valid when no group crosses
labels) live under `outputs/palsynet_identity_audit/review/`.  A separate
finalizer authenticates the source, generated-manifest, contact inventory,
ledger, reviewer-evidence, and adjudication digests and writes exactly one
no-overwrite `outputs/palsynet_identity_audit/reviewed/identity_manifest.json`.
It never mutates the generated manifest.  Only this finalized reviewed manifest
may feed the person split freezer or model runner.

If any condition fails, the runner exits before candidate feature extraction,
fitting, or prediction.

## Development comparison and candidate lock

The three candidates receive one aligned inner-OOF probability per eligible
recording.  When a reviewed group has multiple recordings, average its aligned
recording probabilities once before every metric and bootstrap; labels inside
one included group must agree.  Report group-level AUROC, average precision,
Brier score, balanced accuracy, sensitivity, and specificity, plus 5,000
paired, class-stratified group bootstraps at seed `20260805`.  In every repeat,
resample affected and unaffected eligible development groups separately with
replacement and apply the identical group draw to all three candidates after
recording-to-group probability averaging.  Report 95% percentile intervals for
each candidate metric and for every pairwise candidate metric delta.  Candidate
locking uses only the prespecified unrounded point estimates, never bootstrap
intervals.

Candidate locking is hierarchical.  The 168D candidate advances only if its
unrounded AUROC is strictly higher than 110D, its balanced accuracy is no
lower, and its Brier score is no higher.  The 204D candidate advances only if
its AUROC is strictly higher and the other two metrics are non-inferior against
both 110D and 168D.  Exact ties retain the simpler model.  If 204D advances, it
is locked; else if 168D advances, it is locked; otherwise 110D remains locked.
This prevents a
nominal AUROC increment from being mislabeled as a Phase gain when the 204D
model is worse than the Action proxy model.

The development report is aggregate and closed-schema.  Every leaf has an
exact type, enum, and valid range; counts and decisions are independently
recomputed before writing.  It contains no record
IDs, group IDs, labels, per-record probabilities, file names, or paths.  The
candidate registry, feature-name lists, reviewed identity manifest digest,
split-registry digest, implementation digest, and report digest are frozen
before outer authorization can be created.

## Protected outer test

No outer prediction API is enabled by this development runner.  A separate
authorization artifact must match the locked candidate, exact source bytes,
reviewed identity manifest, split registry, implementation, and model protocol.
After authorization, the selected candidate is fitted on all outer-development
groups and scored exactly once on the protected person groups.  No method or
threshold changes are permitted after the outer result is visible.

## YFP regional ordinal transfer

Build a new provenance-locked manifest without modifying source XML or BMP
files.  Only combined XML files that are valid as-is, or that fail solely
because one terminal `</annotation>` is absent and parse after appending that
single tag in memory, are eligible.  Mismatched-tag regional XML, truncated
images, dimension conflicts, unknown labels, duplicate-key conflicts, and
unlicensed data fail closed.

Eye and mouth are separate three-level ordinal targets (`Normal`, `Slight`,
`Strong`); brow is missing.  Action and phase remain null, and regional
`Normal` must not be converted into a patient-level unaffected label.  The
static feature contract is six capture-swap-invariant MediaPipe clinical23
features per target.  Eye uses, in order: fissure-height bilateral mean,
fissure-height absolute difference, fissure-width bilateral mean,
fissure-width absolute difference, eye-area bilateral mean, and eye-area
absolute difference.  Mouth uses, in order: commissure-height bilateral mean,
commissure-height absolute difference, commissure-radius bilateral mean,
commissure-radius absolute difference, mouth width, and mouth opening.

Within each fold, `StandardScaler` is fitted on training groups only.  The
ordinal head is a three-class proportional-odds cumulative-logit linear model.
For standardized input `x`, severity is `x @ beta`,
`P(y <= k) = sigmoid(theta_k - x @ beta)`, and
`theta_1 = theta_0 + softplus(raw_gap) + 1e-6`; there is no separate intercept.
Minimize group-weighted summed class negative log-likelihood plus
`||beta||^2 / (2 * C)` with `C=0.01`; cut-points are not penalized and each
subject group has total weight one.  Use deterministic L-BFGS-B with zero beta,
`theta_0=-0.5`, initial ordered gap `1.0`, `maxiter=2000`, `ftol=1e-12`,
`gtol=1e-8`, `maxls=50`, and a class-probability floor of `1e-12`.  Any failed
convergence or nonfinite Hessian/objective fails the fold.  There is no feature
or hyperparameter selection.  Evaluate eye and mouth separately with
group-disjoint OOF predictions at the eligible anchor/image level; do not
collapse action-dependent regional labels to one subject label.  Give every
subject total metric weight one by assigning each of its eligible target
anchors weight `1 / n_subject_target_anchors`.  Report weighted quadratic
kappa, weighted balanced accuracy, and macro grade MAE: compute weighted
absolute error of the argmax grade separately within each true grade and then
average the three grade means.  For each target, generate 5,000 percentile
intervals by resampling subject groups with replacement using seed `20260805`,
carrying all of a sampled subject's anchors and its multiplicity; accept only
draws containing all three target grades and fail if 5,000 valid draws are not
obtained within 100,000 attempts.  This is a subject-cluster bootstrap, not a
paired comparison between eye and mouth.  Until a license
artifact and reviewed subject map are present, the manifest is audit-only with
`training_eligible=false` and the runner must stop before extraction/fitting.
Subject-folder grouping is disclosed as unreviewed and cannot be called
patient-disjoint until independently reconciled.

YFP eligibility also has two immutable stages.  The inventory builder writes
only `audit/manifest.json`, permanently records `training_eligible=false`, and
never upgrades it in place.  If the researcher later supplies an authenticated
license/access artifact, a complete independently reviewed subject map, and a
closed-schema eligibility authorization bound to both digests and the audit
manifest digest, a separate finalizer may write one no-overwrite
`eligible/manifest.json` with `training_eligible=true`.  The finalizer must
revalidate every audit row and all three evidence digests; no flag in the audit
manifest or authorization alone can enable extraction or training.  Only the
eligible successor manifest may feed MediaPipe extraction and ordinal fitting.

## External acquisition

- MEEI: use the publisher-linked open-source Facial Palsy Photo and Video
  Standard Set.  Inventory before downloading, enforce the local storage
  budget, record exact bytes/terms/hashes, and quarantine it from candidate
  selection.  Its standardized expressions and HB/eFACE/Sunnybrook scores can
  support an external binary test using the frozen final artifact without
  MEEI refit and later ordinal work; they do not supply frame-level Phase truth
  without a separate annotation protocol.
  After the one-shot PalsyNet outer result is sealed, fit one final scaler and
  the locked logistic classifier on all eligible reviewed PalsyNet groups
  only.  Freeze its training manifest, scaler, coefficients, threshold, and
  implementation digests before exposing any MEEI outcome; then apply that
  artifact without MEEI refit or calibration at participant level.
  The final artifact is generated once by a tested freezer that authenticates
  the reviewed PalsyNet manifest, dynamic cache, split registry, locked
  development result, sealed one-shot outer-result commitment, and protocol
  and implementation digests.  It refits the locked representation, scaler,
  and fixed logistic classifier on all eligible reviewed PalsyNet groups and
  writes a no-overwrite artifact; it cannot run before the outer result is
  sealed.

  The MEEI participant manifest inventories and joins every asset but freezes
  `media_type` as `photo` or `video` and `dynamic_binary_eligible` as a derived,
  label-blinded field.  Photographs are always false for this dynamic binary
  endpoint and may be used only in separately registered static descriptive
  work.  Never tile, repeat, or interpolate a photograph into a trajectory.
  A provenance-locked extractor must process every manifest video with the
  exact PalsyNet MediaPipe model bytes, deterministic four-by-32 window logic,
  95-column schema, timestamps, masks, source indices, and quality gates.  It
  writes a no-overwrite cache manifest bound to every source digest and reports
  participant/media coverage and all label-blinded exclusion reasons.  The
  external runner accepts only this authenticated dynamic cache and rejects raw
  media, static assets, schema drift, selective omission, or a model/cache hash
  mismatch.

  Average all eligible authenticated video probabilities once per participant.
  A participant with no schema-compatible video is not scored and remains in
  the coverage denominator; exclusion never depends on diagnosis or outcome.
  The
  preregistered primary endpoint is participant-level AUROC; secondary metrics
  are average precision, Brier score, balanced accuracy, sensitivity, and
  specificity at threshold `0.5`.  Report 95% percentile intervals from 5,000
  affected/unaffected-stratified participant bootstrap draws with seed
  `20260805`.  A separate one-shot authorization must bind the final artifact,
  participant/source manifest, implementation, and protocol digests.  Scoring
  writes one no-overwrite aggregate closed-schema report without identifiers,
  labels, per-media data, or probabilities.
- AFLFP: prepare the official non-commercial academic application email and
  current End User License Agreement from `Yifan313/AFLFP`.  Reconcile the
  repository's historical `applied 2026-06-18` note against sent-mail or receipt
  evidence before assigning a current state.  The agreement requires a
  full-time faculty researcher or organization employee, a valid institutional
  email, and the recipient's signature; the software must not sign or accept
  the EULA on the researcher's behalf.
