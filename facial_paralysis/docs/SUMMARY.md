# Facial-palsy project — technical summary (as of 2026-08-05)

> **ARCHIVED SNAPSHOT:** This pre-v4 summary is retained for research history.
> Universal Clinical Router v4 is the sole current model; see
> `CURRENT_MODEL.md` and `model_registry.json`.

One historical narrative over the scattered docs (`model_design.md`,
`archive/experiments/training_runs_pre_v4.md`, `autoresearch_fp/FINDINGS.md`,
`mayo_faces_analysis.md`, `generalization.md`, and
`archive/experiments/mayo_loop_findings_pre_v4.md`).

> **Current-model authority:** `CURRENT_MODEL.md`, `model_registry.json`, and
> `results/current_development_model.json` override older uses of “champion” or
> “deployable” in historical experiment logs.

## Canonical result at the time of this snapshot

- The development champion at that time was the **mirror-invariant 110D Landmark
  trajectory model**: AUROC **0.980**, balanced accuracy **0.952**, sensitivity
  **0.905**, and specificity **1.000** on the identity-reviewed PalsyNet
  patient/group-disjoint development partition.
- The frozen 168D Action-proxy and 204D Action+Phase-proxy candidates both
  reduced AUROC and balanced accuracy, so neither passed the preregistered
  promotion gate; the 110D representation remains locked.
- It is a fixed standardized L2 logistic regression over interpretable MediaPipe
  landmark dynamics, not a deeper neural-network result.
- This is affected-vs-unaffected classification on 38 development groups. It is
  not HB grading, Mayo-65 accuracy, cross-institutional validation, or deployment
  evidence.
- All 1,176 recording pairs were reviewed label-blind, yielding 48 identity
  groups. Ten protected groups had zero cache loads, feature construction,
  scaler/model fitting, and prediction; outer evaluation remains unauthorized.

## Bottom line
- We now have a strong, interpretable Landmark development classifier on
  PalsyNet. Earlier web-QWK, MARLIN, Fusion, and SSL models are historical
  baselines for present-tense reporting.
- Historical experiments did **not observe Mayo transfer** from the web-trained
  appearance models; their outputs tracked camera/appearance more than the
  left-right deficit out of domain.
- Relative geometric left-right asymmetry is the leading cross-domain candidate
  because it shifted least in the domain diagnostics and can be measured
  directly from scripted FACES actions. Its clinical transfer is still
  unestablished.
- But at **n=13 patients with no HB labels, nothing about Mayo is statistically
  validated** — not the model's transfer, not even the per-patient reliability of the
  label-free measures. The bottleneck is data, quantified: **~35–50 patients to establish
  transfer, ~40–60 HB labels for a trustworthy accuracy number.**

## 1. Historical web model (in-domain baseline)
- Dual-stream: frozen MARLIN video encoder (768-d appearance) ⊕ MediaPipe geometry
  (52 blendshapes + L/R asymmetry). Only small heads are trained (MARLIN always frozen).
- **autoresearch-fp** (`autoresearch_fp/`, ~100 model versions, faithful karpathy-style
  loop with a fixed 3-seed QWK metric): historical web champion **0.668**
  (eyes 0.49, mouth 0.83), up
  from a 0.53 baseline. Key wins: engineered nonlinear asymmetry features, static MLP over
  the dead GRU (metric data is single-frame), per-region MARLIN treatment (eyes width 256,
  mouth full 768) + deeper trunk, CORN ordinal loss. Generalizes cross-dataset (FNP↔YFP)
  better than baseline.
- Verdict: after ~100 configs the web plateau is ~0.67; the ceiling is data, not model.

## 2. The FACES protocol unlocked dynamic, label-free clinical measures
- The 8-action battery (repose, brow, gentle+forced eye closure, smile, pucker,
  lower-teeth, reanimated smile) maps onto eFACE/Sunnybrook. From the per-action clips we
  compute, with NO labels: regional asymmetry, 60fps EAR eye-closure (lagophthalmos +
  closure asymmetry), synkinesis, and gentle-vs-forced recruitment.
- These are DYNAMIC measures a still cannot contain, and they are statistically
  independent of the static-peak asymmetry — genuinely new signal.
- Deliverable: `scripts/mayo_scorecard.py` → a per-patient clinical scorecard (13 cards +
  cohort table). Unsupervised phenotype map separates flaccid / synkinetic / mild.

## 3. Generalization — what transfers (rigorous)
- **Web→Mayo severity transfer ≈ 0** for appearance models (champion, baseline). The
  MARLIN detector saturates on Mayo (flags 13/13 but no spread — no in-domain negatives).
- **Geometry-only (drop MARLIN)** is the only thing that moves transfer positive, BUT the
  bootstrap correction is essential: eyes +0.48 was against a NON-independent blendshape
  target; against an independent landmark-EAR target it is **+0.05, 95% CI [-0.55,+0.63]**.
  Combined eyes+mouth +0.13 [-0.36,+0.56]. **Transfer is not established at n=13.**
- MARLIN helps within-modality (web) generalization but blocks cross-modality (Mayo)
  transfer — opposite optima; the historical geometry-only candidate is
  `deploy_config.json` (web QWK 0.577; mouth 0.86 strong, eyes 0.29 weak
  without MARLIN/dynamics).
- **Mechanism (permutation importance):** mouth severity is driven almost entirely by the
  geometric L/R asymmetry deltas (QWK drop 0.55) and MARLIN is useless-to-harmful for it
  (-0.05); eyes genuinely relies on MARLIN (0.13). This is *why* mouth transfers/is
  MARLIN-free and eyes doesn't — confirmed from web QWK, transfer, AND feature attribution.
- Even the label-free measures have limited per-patient reliability (side-consistency
  0.77; blendshape-vs-EAR weaker-eye agreement 8/14) — robust only for clear/extreme cases.

## 4. No-label levers tried (all negative or unverifiable at this n)
CORAL domain adaptation (no effect), feature-noise augmentation (worst-case +45% but a
peak/robustness tradeoff), generalization-objective search (champion already near-optimal),
geometry-only search (plateau 0.577), AU-pretraining pipeline (built + synthetic-verified,
awaiting DISFA/BP4D). Nothing closes the modality gap without data.

## 5. What to do next (ranked, all data-side)
1. **Record in-domain healthy controls** (same FACES protocol) → unlocks a trustworthy
   binary detector (the one place MARLIN may transfer) and a calibration baseline.
2. **HB labels on ~40–60 takes** (start with the highest-disagreement ones,
   `outputs/mayo_active_learning.json`) → the only path to a supervised, validated
   clinical severity model, and enough to get a usable accuracy CI.
3. **Recruit toward ~35–50 patients** → enough power to establish (or refute) transfer.
4. **AU-dynamics corpora (DISFA/BP4D)** → pretrain the geometric/temporal stream on real
   movement (pipeline ready).

Until then the honest Mayo output is the **research-only label-free scorecard**
for triage/ranking of clear cases, explicitly not a validated per-patient
clinical grade or deployment.
