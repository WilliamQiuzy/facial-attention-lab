# 110D-Generalization v1: sealed outer release

Recorded: 2026-08-11

## Outcome

The locked `landmark_mi_110d` candidate was evaluated exactly once on the
untouched PalsyNet outer partition. The fit used only the 39 development
recordings / 38 reviewed person groups. Scoring used 10 protected recordings /
10 different reviewed person groups, evenly divided between affected and
unaffected labels.

| Protected group-level metric | Point estimate | Descriptive bootstrap 95% interval |
|---|---:|---:|
| AUROC | 1.000 | 1.000–1.000 |
| Average precision | 1.000 | 1.000–1.000 |
| Balanced accuracy | 0.900 | 0.700–1.000 |
| Sensitivity | 0.800 | 0.400–1.000 |
| Specificity | 1.000 | 1.000–1.000 |
| Brier score | 0.118 | 0.070–0.171 |

The fixed threshold was `0.5`. AUROC shows perfect ranking within these ten
groups, while the thresholded result missed one of five affected groups. The
bootstrap intervals only resample this small observed outer set and are
descriptive; they do not remove the large uncertainty caused by having ten
people.

## Frozen protocol

- Input: MediaPipe FaceMesh 478 landmarks reduced to `clinical23_v2`, then the
  locked 92 trajectory summaries plus 18 bilateral-dynamics features.
- Training: original and horizontally mirrored 110D rows, equal total weight
  per reviewed group, train-only `StandardScaler`.
- Classifier: L2 Logistic Regression, `C=0.01`, `liblinear`,
  `max_iter=2000`, random state `0`.
- Inference: mean of original and horizontal-mirror probabilities, fixed
  threshold `0.5`.
- Statistics: one probability per reviewed person group; 5,000
  class-stratified group bootstrap draws at seed `20260805`.
- Post-result changes: none to candidate, representation, threshold, split,
  calibration, model, or seed.

## Final inference artifact

Only after the outer report was sealed and pinned, the same protocol was fit
once on all 49 eligible PalsyNet recordings / 48 reviewed groups: 26 affected
and 22 unaffected groups. The resulting JSON contains 110 feature names,
scaler mean/scale, Logistic coefficients/intercept, mirror-inference rule,
threshold, counts, and authenticated provenance. It contains no recording or
group identifiers, filenames, paths, labels, or per-record probabilities.

The artifact's serialized parameters reproduced a deterministic synthetic
mirror-averaged probability of `0.9585654903146288`; all parameter arrays were
finite and had exactly 110 entries.

## Execution and integrity evidence

- Source branch: `codex/110d-generalization-v1`
- Frozen implementation commit: `ad5b670`
- Host: Nebius H200, NVIDIA H200 with 143,771 MiB, driver `580.159.04`
- H200 release: `/home/ssh-ziyue/facial-paralysis-h200/releases/110d-outer-release-v1-ad5b670`
- Preflight: 49 cache filenames, all identity/split/source/report hashes
  matched; authorization passed with zero NPZ arrays loaded and zero fits.
- H200 regression: 29/29 focused release and existing 110D tests passed.
- Outer invocation: exactly one, exit zero, 17.68 seconds, one scaler fit, one
  model fit, ten protected predictions.
- Artifact invocation: exactly one, exit zero, 2.36 seconds, one all-eligible
  scaler fit and one all-eligible model fit.
- Reports are mode `0600` on creation and use atomic no-overwrite writers.
- After verification, the complete H200 release directory was changed to
  read-only; a scan found no writable regular file in that release.

SHA-256 trust chain:

- Reviewed identity manifest:
  `fa756b79f0e1bc9053527de4632216281d9011a1f75e2bf652371dab38d2da9f`
- Review ledger:
  `865fe78137d3d97b11da3bf37c6db105e387174b9f115c01824afea6a5368afd`
- Frozen person split:
  `738980264a698cb8a2d45a12fdc1ff95f349bbb4ac76787296e5314e40981ba0`
- Source collection:
  `30263bc3784a1ed9eeec196f1448b3b36234036e25282d2bed17dcee97aab3ae`
- Locked development report:
  `e3f7eb6b9c91fbad74a514be8ba6f0c51418d7953155518033fedb1e228a1f43`
- Outer authorization:
  `7aca8ddffeca5479a53ee28103b091acf1f1d64badd46d62ce8a601654b0881b`
- Release implementation aggregate:
  `b5ac56315b161648662d7b3dcf0991b30413b1dffada11b697547ed7db2aff9d`
- Protected outer report:
  `0e44cfaf2fe5bb3e8e9d9ea8629f8ef873a5af2d94a56025b3f6f32f35227df3`
- Final PalsyNet artifact:
  `cbc49d0aa54b504915bebd00fdbe005458378e5675b57461ce83d3385f9b60f9`

Machine-readable files:

- Protected report:
  `outputs/dynamic_landmark/benchmarks/protected/110d-generalization-v1/report.json`
- Final inference artifact:
  `outputs/dynamic_landmark/artifacts/110d-generalization-v1/final_palsynet_artifact.json`
- Current public summary: `docs/results/current_development_model.json`

## Claim boundary and next gate

This result supports an identity-reviewed, person-disjoint **internal PalsyNet
outer evaluation**. It is not Mayo two-class accuracy, HB-grade accuracy,
cross-institutional validation, clinical validation, or deployment evidence.
The next scientific step is to audit MEEI license/labels/identity groups and
preprocessing eligibility, then score this exact artifact once without MEEI
tuning. Mayo remains a positive-only consistency challenge.
