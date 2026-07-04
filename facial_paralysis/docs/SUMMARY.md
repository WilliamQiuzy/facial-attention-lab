# Facial-palsy severity project — technical summary (as of 2026-07-04)

One authoritative narrative over the scattered docs (`model_design.md`,
`training_runs.md`, `autoresearch_fp/FINDINGS.md`, `mayo_faces_analysis.md`,
`generalization.md`, `loop_findings.md`). Bottom line up front, evidence below.

## Bottom line
- We built a strong facial-palsy severity model **on public web images** (region QWK
  ~0.65) and an autonomous search that squeezed it to its ceiling.
- **None of that transfers to the Mayo clinical videos.** Web-trained appearance models
  read camera/appearance, not the left-right deficit, out of domain.
- The **only signal that is domain-robust is the geometric left-right asymmetry** of the
  scripted FACES actions (eye closure / smile / etc.), which we measure directly and
  label-free (EAR closure, synkinesis, forced recruitment).
- But at **n=13 patients with no HB labels, nothing about Mayo is statistically
  validated** — not the model's transfer, not even the per-patient reliability of the
  label-free measures. The bottleneck is data, quantified: **~35–50 patients to establish
  transfer, ~40–60 HB labels for a trustworthy accuracy number.**

## 1. Web model (the part that works, in-domain)
- Dual-stream: frozen MARLIN video encoder (768-d appearance) ⊕ MediaPipe geometry
  (52 blendshapes + L/R asymmetry). Only small heads are trained (MARLIN always frozen).
- **autoresearch-fp** (`autoresearch_fp/`, ~100 model versions, faithful karpathy-style
  loop with a fixed 3-seed QWK metric): champion **0.668** (eyes 0.49, mouth 0.83), up
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
  transfer — opposite optima; the deployment model is geometry-only (`deploy_config.json`,
  web QWK 0.577; mouth 0.86 strong, eyes 0.29 weak without MARLIN/dynamics).
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

Until then the honest, deployable Mayo output is the **label-free scorecard** for
triage/ranking of clear cases, explicitly not a validated per-patient clinical grade.
