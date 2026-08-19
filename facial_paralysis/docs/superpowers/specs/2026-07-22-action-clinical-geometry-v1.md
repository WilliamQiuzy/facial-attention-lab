# Action-Aligned Clinical Geometry v1

## Decision

Implement one small, development-only successor to the current BiGRU/Fusion smoke. It must test whether fixed, clinically structured landmark dynamics add discrimination beyond the nine recorded acquisition-nuisance covariates, without tuning a deeper network and without generating any protected outer-fold prediction.

This protocol is informed by earlier exploratory development analysis. It is therefore a successor direction screen, not independent confirmation and not clinical validation.

## Frozen feature contract

The candidate uses only the 23 `clinical23_v2` MediaPipe-derived geometry columns. It is direction-free because capture mirroring and patient laterality are unknown. Because PalsyNet has no action labels, this is a clinical region/action proxy, not true action alignment.

- Six bilateral pairs (three eye, one brow, two mouth) each contribute pooled correlation, invariant amplitude ratio, absolute best lag, mean bilateral range, absolute range difference, mean bilateral maximum absolute velocity, and absolute maximum-velocity difference.
- Six explicit bilateral absolute-difference channels plus mouth width and mouth opening each contribute range and maximum absolute velocity.
- Derivatives remain inside each 32-frame window and never cross detector gaps. Lag uses at most five frames, never crosses a window, and requires valid endpoints; it does not enforce validity of every intermediate frame.

The ordered contract contains 58 features: 42 paired features plus 16 explicit asymmetry/global dynamics. Feature names and dimension are code constants and are not selected from labels.

## Fixed development protocol

- Dataset: validated PalsyNet `clinical23_v2_windows` cache only.
- Split: outer fold 0 is treated as protected and untouched; only its outer-train rows enter four-fold grouped inner OOF evaluation.
- Candidates, in fixed order: `nuisance`, `landmark`, `clinical_dynamics`, `clinical_dynamics_plus_nuisance`. `landmark` is a reference-only comparator to the existing classical baseline and does not select the direction.
- Model: standardized L2 logistic regression with fixed `C=0.01`, `liblinear`, `random_state=0`, `max_iter=2000`, group-balanced sample weights, no hyperparameter search.
- Metrics: group-collapsed AUROC, average precision, Brier score, balanced accuracy, sensitivity, and specificity at 0.5.
- Primary comparison: deterministic 5,000-repeat paired class-stratified group bootstrap of `clinical_dynamics_plus_nuisance` minus `nuisance` AUROC. This tests incremental information conditional on the nine recorded covariates; it cannot exclude unrecorded nuisance.

## Gate and claim boundary

The direction passes only if all are true:

1. clinical-dynamics-plus-nuisance AUROC minus nuisance AUROC is at least 0.10;
2. the descriptive paired-bootstrap 95% lower bound is greater than zero;
3. clinical-dynamics sensitivity and specificity are each at least 0.70;
4. the report confirms zero protected outer-fold feature extractions, fits, and predictions.

A pass authorizes only identity/action review and a separately frozen successor protocol. It does not authorize outer evaluation, HB grading claims, Mayo clinical accuracy claims, or deployment.

## Output safety

The validated cache and labels are loaded only to reconstruct the existing frozen split. Candidate feature extraction is restricted to outer-fold-0 train indices; all scaler/model fits use inner-train indices, and all predictions use inner-validation indices. The runner writes one private (`0600`), atomic, no-overwrite JSON report containing aggregate counts, protocol, metrics, bootstrap comparison, index-audit summaries, and the decision. It must contain no recording IDs, group IDs, source paths, raw features, or per-record probabilities.
