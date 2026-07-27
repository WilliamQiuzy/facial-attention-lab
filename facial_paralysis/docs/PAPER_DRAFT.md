# Why a strong web-trained facial-palsy model does not transfer to clinical iPhone video: an appearance confound, a geometric-asymmetry signal, and a sample-size wall

> **Historical draft:** This manuscript analyzes the earlier web-QWK model and
> must not be used for current-model reporting. The current 110D Landmark
> development result and its non-clinical limits are recorded in
> `CURRENT_MODEL.md`.

*Draft — Menapace lab, Mayo Clinic × Harvard. All numbers reproducible from `facial_paralysis/`.*

## Abstract
Automated facial-palsy grading is bottlenecked by data: there is no large public dataset,
clinical cohorts are small and consent-bound, and House-Brackmann (HB) labels are scarce.
We built a dual-stream severity model (frozen MARLIN video encoder ⊕ MediaPipe geometry)
and, using an autonomous architecture search (~100 configurations under a fixed leak-safe
3-seed metric), pushed region-severity agreement to QWK **0.668** on public web images — a
strong in-domain result. We then tested whether this model transfers to an in-domain target:
**13 facial-palsy patients** recorded on iPhone under a standardized 8-action protocol
(FACES; IRB 24-004956). It does not. The learned severity is uncorrelated with the true
left-right clinical asymmetry on the target domain, and — critically — we show this
non-transfer **cannot even be statistically resolved at n=13** (every bootstrap 95% CI
crosses zero). Permutation feature-importance localizes the cause: mouth severity is driven
almost entirely by geometric L/R asymmetry (importance 0.55) and the MARLIN appearance
stream is useless-to-harmful for it (−0.05), whereas eye severity leans on MARLIN appearance
— a domain-confounded feature that fails out of distribution. From the FACES protocol we
derive label-free, clinically-grounded dynamic measures (EAR eye-closure/lagophthalmos,
synkinesis, gentle-vs-forced recruitment) that are domain-invariant by construction, and we
quantify exactly how much data would close the gap: ~35–50 patients to power a transfer
test, ~40–60 HB labels for a usable accuracy estimate. The contribution is methodological
and cautionary: a high in-domain benchmark number is not evidence of clinical utility, the
diagnostic signal is geometric asymmetry rather than holistic appearance, and honest
uncertainty quantification is essential in the small-n clinical regime.

## 1. Introduction
Facial paralysis is graded clinically on scales (House-Brackmann I–VI; eFACE; Sunnybrook)
that score **voluntary movement and left-right symmetry** of specific facial actions.
Machine-learning approaches are hampered by data scarcity: clinical papers train on private
institutional cohorts (~100–400 patients) that are not shareable, and method papers reuse a
small set of tiny, mostly email/EULA-gated public benchmarks. We ask a question that the
field's benchmark-chasing tends to skip: *does a model that scores well on public data
actually work on the target clinical distribution?* We answer it rigorously for an in-domain
Mayo cohort and find the honest answer is "unknowable at current n" — and we turn that into a
concrete data plan.

## 2. Data
- **Public (web):** PalsyNet (49 YouTube subjects, binary), FNP and YFP (per-region eye/mouth
  severity, 3 ordinal levels), used with perceptual-hash dedup and subject-level leak-safe
  splits (web sets train-only; region metric on FNP-valid + held YFP subjects).
- **Mayo (target):** 13 unique facial-palsy patients (14 takes; one duplicate) recorded on
  iPhone under the FACES protocol — 8 held actions (repose, brow raise, gentle + forced eye
  closure, relaxed smile, lip pucker, lower-teeth show, reanimated smile). No HB labels; no
  in-domain healthy controls.

## 3. Methods
**Model.** Per action: a frozen MARLIN (ViT-B, 768-d, video-pretrained) appearance vector ⊕
MediaPipe geometry (52 ARKit blendshapes + 20 L/R asymmetry deltas). Only small heads train.
A per-region-decoupled trunk maps to region severities via CORN ordinal cut-point heads.

**Architecture search.** We adapted karpathy/autoresearch to our task: an agent edits one
model file; a fixed harness (`prepare_fp.py`) owns the data, leak-safe split, and a 3-seed
mean-QWK metric; keep-if-better over ~100 configurations. (`autoresearch_fp/`)

**FACES per-action measures (label-free).** Each per-action clip yields a left-right
asymmetry index |L−R|/(L+R) on the action's signature blendshapes, a 60 fps eye-aspect-ratio
(EAR) closure trajectory (depth, residual = lagophthalmos, closure asymmetry), synkinesis
(cross-region co-activation), and gentle-vs-forced recruitment. (`scripts/mayo_eface.py`,
`ear_clips.py`, `ear_analyze.py`)

**Generalization evaluation.** We score the Mayo per-action clips with the web-trained model
and correlate its region severity with the domain-invariant clinical asymmetry (Spearman),
using **independent** targets (landmark EAR / mouth-corner) to avoid circularity, with
bootstrap 95% CIs. We also measure cross-dataset (FNP↔YFP) generalization and run a
simulation-based power analysis. (`mayo_generalization.py`, `mayo_transfer_robust.py`,
`cross_dataset.py`, `power_analysis.py`)

## 4. Results
**4.1 Strong in-domain model.** The search improved region-QWK from a 0.530 baseline to
**0.668** (5-seed ±0.009; eyes 0.49, mouth 0.83). Ablation ranks the design: per-region
decoupling (−0.058 if removed) and engineered nonlinear asymmetry features (−0.043) are the
substantive wins; dropping the temporal GRU for a static MLP is a free simplification
(single-frame data). Calibration: mouth is well-calibrated (ECE 0.09); eyes is poorly
calibrated (ECE 0.29, → 0.21 after temperature scaling — still poor).

**4.2 No transfer to Mayo, and it is unresolvable at n=13.** Against independent geometric
targets, web→Mayo severity correlation is +0.05 (eyes, 95% CI [−0.55,+0.63]) and +0.26
(mouth, CI crossing zero); combined +0.13 [−0.36,+0.56]. The MARLIN-based binary detector
saturates on Mayo (flags 13/13 at ≈0.95 with no spread — the no-in-domain-negatives trap).
A previously-encouraging +0.48 eyes correlation was shown to be an artifact of a target that
shares the model's own inputs.

**4.3 Mechanism.** Permutation importance: scrambling the L/R asymmetry deltas collapses
mouth QWK (drop 0.55) while scrambling MARLIN slightly *improves* it (−0.05); eyes depends on
MARLIN (0.13). So mouth rides a domain-invariant geometric signal and needs no appearance;
eyes leans on domain-confounded appearance — exactly why mouth is MARLIN-independent and eyes
fails to transfer. The appearance-vs-asymmetry account is confirmed from in-domain QWK,
cross-modality transfer, and direct feature attribution.

**4.4 Reliability of the label-free measures.** Cross-action side-consistency averages 0.79
(5/13 patients fully consistent); two independent eye methods agree on the weaker side in
only 7/13 (≈chance) — partly because muscle *activation* (blendshape) and geometric *closure*
(EAR) are different constructs. The measures are robust for clear/extreme cases (isolated
brow deficit; a strongly synkinetic patient) but noisy for borderline ones.

**4.5 How much data closes the gap.** To detect a true transfer at 80% power: ~34 patients
(ρ=0.5), ~52 (ρ=0.4), ~94 (ρ=0.3); at n=13 power is 15–37%. A trustworthy HB-accuracy (QWK
±0.10) needs ~40–60 labeled takes.

## 5. Discussion
Three lessons generalize beyond this cohort. (i) **A high benchmark number is not clinical
utility** — our best web model transfers to nothing on the real target. (ii) **The diagnostic
signal is geometric left-right asymmetry, not holistic appearance**; appearance encoders,
however powerful, encode domain factors (camera, lighting, identity) that dominate their
variance and fail out of distribution. Relative within-face geometric measures cancel these
and are the more portable, interpretable research signal — and they are what clinical
scales actually score. (iii) **In the small-n clinical regime, uncertainty quantification is
not optional**; a point estimate at n=13 is indistinguishable from noise, and reporting it as
a finding would mislead.

## 6. Limitations
n=13, no HB labels, no in-domain healthy controls. The label-free measures are objective
physical-symmetry quantities, not validated clinical grades. 60 fps EAR resolves the closure
transient but the cohort is small; MediaPipe blendshapes are pose-robust but not pose-immune.

## 7. Conclusion & the data that would settle it
The historical pipeline (per-action segmentation, label-free dynamic scorecard,
geometry-only candidate, AU-dynamics pretraining) is built and waiting. The remaining lever is data, quantified:
(1) in-domain healthy controls → a trustworthy detector; (2) ~40–60 HB labels → a supervised,
validated severity model with a usable accuracy CI; (3) ~35–50 patients → power to establish
transfer; (4) AU corpora (DISFA/BP4D) → temporal-stream pretraining. Until then the honest
clinical output is a label-free triage scorecard for clear cases, explicitly not a per-patient
grade.

---
*Reproducibility: see `docs/PIPELINE.md`. Figures: `outputs/summary_figure.png`,
`outputs/mayo_eface/`, `outputs/mayo_scorecard/`. All experiments run locally on cached
features except 60 fps EAR (RunPod GPU).*
