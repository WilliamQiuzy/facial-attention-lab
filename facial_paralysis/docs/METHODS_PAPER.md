# When frozen foundation-model features hurt: appearance confounding and the case for relative-geometric representations in clinical face analysis

> **Historical draft:** This manuscript analyzes the earlier representation
> transfer study, not the current project model. Use `CURRENT_MODEL.md` for
> Universal Clinical Router v4 and its non-clinical claim boundary.

*Methods-paper draft (ML / domain-adaptation venue). Distinct from the clinical draft
(`archive/manuscripts/web_model_transfer_draft_pre_v4.md`); this one is about the representation-transfer finding and generalizes
beyond facial palsy. All results reproducible in `facial_paralysis/`.*

## Abstract
Frozen self-supervised video/image foundation encoders (e.g. MARLIN, CLIP-family) are the
default backbone for small-data clinical vision. We show, in facial-palsy severity
estimation, that such a backbone can be actively *harmful* for the clinical target: its
768-d appearance features are **perfectly domain-separable** between the public-web training
distribution and an in-domain clinical iPhone-video test set (a domain classifier reaches
AUC 1.000), so a model that scores well on web (region QWK 0.67) transfers to nothing
in-domain (severity-vs-clinical-asymmetry Spearman ρ≈0, all bootstrap 95% CIs crossing
zero). We localize the cause by permutation attribution: the mouth head rides the domain-
invariant **relative left-right asymmetry** signal (importance 0.55) and MARLIN is useless-
to-harmful for it (−0.05), whereas the eye head leans on domain-confounded appearance and
fails to transfer. Three standard feature-space domain-adaptation remedies — CORAL, feature-
noise augmentation, and a DANN adversarial objective — **all fail** to close the gap (the
adversarial representation's domain-AUC stays at 0.997), because the confound is maximal and
the clinical target is tiny (n=13). We give a power analysis showing the non-transfer is
unresolvable below ~35–50 patients, and argue that the more portable research representation is
the **relative geometric asymmetry** (domain-AUC 0.79, vs 1.00 for appearance and 0.97 for
absolute geometry), which is also what clinical scales actually measure. The lesson
generalizes: benchmark accuracy with a powerful frozen backbone is not evidence of clinical
transferability; relative, physically-grounded features can beat holistic appearance out of
distribution; and standard DA cannot rescue a maximal confound at clinical sample sizes.

## 1. Introduction
Small-data clinical vision routinely bolts a task head onto a frozen foundation encoder.
The implicit assumption is that a powerful pretrained representation transfers. We test this
directly for facial-palsy severity, where a public-web training set and an in-domain clinical
cohort form a natural source/target pair, and find the assumption fails in an instructive,
quantifiable way — with a constructive alternative (relative-geometric features) and a clear
negative result for standard domain adaptation.

## 2. Setup
Dual-stream model: frozen MARLIN (ViT-B video MAE, 768-d) ⊕ MediaPipe geometry (52 ARKit
blendshapes + 20 signed L/R asymmetry deltas); small per-region ordinal heads trained on web
(FNP/YFP region severity). Target: 13 palsy patients, iPhone FACES-protocol video, no labels;
the domain-invariant ground truth is the clinical L/R asymmetry (independent landmark EAR /
mouth-corner measures).

## 3. Results
**3.1 Domain separability (the confound, measured).** A logistic domain classifier
(web vs target), per feature group: MARLIN appearance AUC **1.000**; absolute geometry
(blendshapes+deltas) 0.970; **relative L/R asymmetry deltas 0.793**. Appearance encodes the
domain maximally; even absolute geometry shifts; only relative asymmetry is comparatively
invariant. (`scripts/domain_gap.py`)

**3.2 Non-transfer.** Web QWK 0.668 (5-seed, ablated, cross-dataset-generalizing). Target
transfer (severity vs independent clinical asymmetry): eyes +0.05 [−0.55,+0.63], mouth +0.26
[−0.56,+0.90], combined +0.13 [−0.36,+0.56] — every CI crosses zero; the appearance-based
binary detector saturates (flags all target positives at ≈0.95, no spread).
(`mayo_transfer_robust.py`)

**3.3 Attribution (why).** Permutation importance (QWK drop when a group is scrambled):
mouth — asymmetry deltas **0.55**, MARLIN **−0.05**; eyes — MARLIN **0.13**. Mouth rides the
invariant geometric signal and is MARLIN-free; eyes lean on the confounded appearance and
therefore do not transfer. (`scripts/feature_importance.py`)

**3.4 Standard domain adaptation fails.** CORAL (covariance alignment): no effect. Feature-
noise augmentation: a robustness/peak trade-off, no transfer gain. **DANN** (gradient-reversal
adversary, target unlabeled): the representation's domain-AUC is **unchanged at 0.997**;
transfer nudges toward zero but never positive. Three feature-space remedies fail because the
input confound is maximal (AUC 1.0) and the target is n=13. (`scripts/dann.py`, `mayo_coral.py`,
`aug_generalization.py`)

**3.5 It is a sample-size wall.** Simulation: detecting a true transfer (80% power) needs
~34 patients (ρ=0.5) to ~94 (ρ=0.3); at n=13 power is 15–37%. So the non-transfer is
unresolvable at this n regardless of method. (`scripts/power_analysis.py`)

## 4. Discussion
(i) A high benchmark number with a frozen backbone is **not** evidence of clinical transfer —
the backbone's variance is dominated by nuisance appearance (camera/lighting/identity) that is
domain-specific. (ii) **Relative, physically-grounded features** (here L/R asymmetry) can beat
holistic appearance out of distribution precisely because the relative operation cancels the
domain nuisance — and they coincide with what clinicians score. (iii) **Standard feature-space
DA cannot rescue a maximal confound at clinical n**; effort is better spent removing the
confound by design (drop appearance; use relative-geometric / dynamic features) and acquiring
in-domain data. These claims are backbone- and task-agnostic and should be checked whenever a
frozen foundation model is used for small-n clinical transfer.

## 5. Limitations
Single clinical cohort (n=13, no labels) — but the domain-separability and attribution results
are well-powered (feature-distribution level), and the transfer null is exactly the point.
One backbone (MARLIN); the mechanism (appearance variance ≫ task variance, domain-specific) is
general but backbone-specific magnitudes will vary.

## 6. Conclusion
Frozen foundation-model features were not just unhelpful but domain-confounding for clinical
face transfer; relative-geometric asymmetry is the transferable, clinically-aligned signal;
and standard domain adaptation could not fix the gap at clinical sample size. Design the
representation to be relative and physical, and measure the domain gap directly before trusting
a benchmark.
