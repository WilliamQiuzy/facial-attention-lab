# Overnight loop findings (2026-07-04) — the honest state of Mayo

A rigorous autonomous pass over the mouth-transfer question and the reliability of
everything we've built. The headline is sobering but important: **at n=13 and with no
labels, nothing about Mayo is statistically validated** — neither the learned model's
transfer nor, at the per-patient level, the label-free measurements themselves.

## What the rigor overturned
1. **Learned transfer is NOT established.** The earlier "geometry-only eyes transfers
   +0.48" was measured against the blendshape `eye_asym`, which shares the model's own
   blendshape inputs (partial circularity). Against an INDEPENDENT target (landmark EAR
   closure), it collapses to +0.05 with bootstrap 95% CI [-0.55, +0.63]. Combined
   eyes+mouth +0.13 [-0.36, +0.56]. Every CI crosses zero. (`mayo_transfer_robust.py`)
2. **Mouth "0.00" was partly a noisy target.** The blendshape `oral_asym` and the direct
   60fps mouth-corner asymmetry disagree (rho -0.21). Against the corner target,
   geometry-only mouth is +0.26 — but also with a CI crossing zero. (`mouth_corner.py`,
   `mouth_transfer.py`)
3. **The label-free measurements have limited per-patient reliability.**
   (`reliability.py`) Cross-action side-consistency mean 0.79 (only 5/13 patients fully
   consistent on which side is weaker); blendshape-vs-EAR agreement on the weaker eye is
   7/13 (54%, ~chance). Extreme cases (MySlate_6, synkinetic MySlate_23) are robust; borderline
   cases are noisy / method-dependent. Part of the blendshape-vs-EAR disagreement is real
   (muscle *activation* vs geometric *closure* are different constructs), but it means a
   single "weaker side / severity" summary per patient is not yet reliable.

## What still holds
- The WEB model results (autoresearch 0.649, cross-dataset generalization champion>baseline).
- The web→Mayo modality gap for appearance models (severity doesn't transfer; MARLIN
  detector saturates) — those are consistent across every test.
- MARLIN helps within-modality generalization but blocks cross-modality transfer
  (opposite optima; `xsearch.py`).
- The direct measurements are still the most defensible thing on Mayo FOR EXTREME/CLEAR
  cases and for cohort-level description; they just aren't validated per-patient.

## The unavoidable conclusion
The Mayo effort is bottlenecked by **n=13 + no labels**, hard. We cannot validate a
transfer claim, a severity model, OR the reliability of the label-free tool without
either (a) HB labels (even ~10-14) to anchor accuracy, or (b) more patients + in-domain
healthy controls to get the sample size where these CIs close. Every no-label modeling
lever we tried (CORAL, augmentation, geometry-only, generalization-objective search) is
either negative or unverifiable at this n. **Data is the only remaining lever, full stop.**

## Power analysis — how much data closes the gap (`scripts/power_analysis.py`)
Turns "we need more data" into numbers:
- **Detect a real severity→clinical transfer (80% power):** rho=0.5 → 34 patients,
  rho=0.4 → 52, rho=0.3 → 94. At n=13, power to detect even rho=0.5 is 37%.
- **Trustworthy HB-accuracy (QWK ±0.10):** ~40–60 HB-labeled takes (n=14 gives ±0.24, too wide).

So the concrete asks are: ~35–50 patients to establish transfer, and ~40–60 HB labels for a
usable severity-accuracy number. n=13 is far below both — the inconclusiveness is a sample-size
problem, definitively not a modeling one.

## Champion (0.668) error analysis + calibration (scripts/champion_analysis.py, calibrate.py)
- **Mouth is reliable:** acc 0.76, Normal/Strong recall ~0.80, well-calibrated (ECE 0.09).
  Rare Slight class (n=41) is the weak spot (recall 0.39).
- **Eyes is the hard region:** acc 0.61, and BADLY calibrated (ECE 0.29 — under-confident;
  a bin predicting P=0.67 has empirical 0.99). The majority Slight class (recall 0.56) is
  confused with Normal/Strong — the static-still ambiguity again.
- **Temperature scaling** (fit on disjoint cal split, tested on held-out): eyes T≈0.30 →
  ECE 0.29→0.21; mouth T≈0.60 → 0.10→0.08. Both improve; mouth becomes good, **eyes stays
  poorly calibrated (0.21) even after** — recalibration helps but eyes needs better
  features/data, not just scaling. Apply these T's before any probabilistic use.

## Feature importance — the mechanism, at the feature level (scripts/feature_importance.py)
Permutation importance (region-QWK drop when an input group is scrambled):

| group | eyes | mouth |
|---|---|---|
| MARLIN appearance (768) | +0.134 | **-0.052** |
| raw blendshapes [:52] | +0.156 | +0.217 |
| L/R asymmetry deltas [52:72] | +0.096 | **+0.552** |

This mechanistically explains the whole generalization story at the feature level:
- **Mouth severity is driven almost entirely by the geometric L/R asymmetry deltas
  (0.552 drop); MARLIN is useless-to-harmful (-0.052).** → mouth is strong (0.83),
  MARLIN-independent, and rides a domain-INVARIANT signal.
- **Eyes genuinely relies on MARLIN appearance (0.134 drop).** → eyes is weak
  geometry-only and does NOT transfer, because it leans on the domain-CONFOUNDED
  appearance stream.

So "drop MARLIN for Mayo" hurts eyes (which needs it on web) but not mouth (which never
used it) — exactly matching the geometry-only results and the transfer pattern. The
appearance-vs-asymmetry thesis is now confirmed from web QWK, cross-modality transfer,
AND direct feature attribution.

## Champion per-source breakdown (scripts/per_source.py)
Region QWK by web source (ensemble 3-seed):

| | FNP | YFP |
|---|---|---|
| eyes | 0.257 (n=91) | 0.521 (n=394) |
| mouth | 0.721 (n=98) | 0.763 (n=394) |

The weakest cell is **FNP eyes (0.26)** — web-scraped still eye crops with noisier labels;
YFP (per-subject) is handled better everywhere, mouth is good on both. Matches the
cross-dataset result (YFP→FNP eyes was the hardest transfer direction).

## MARLIN problem: domain-gap diagnostic (scripts/domain_gap.py)
Domain separability web-vs-Mayo (classifier AUC; 1.0 = totally different domains, 0.5 = same):

| feature group | domain-AUC |
|---|---|
| MARLIN appearance (768) | **1.000** (maximal confound) |
| geometry all (72, blendshapes+deltas) | 0.970 |
| L/R asymmetry deltas (20) | **0.793** (most invariant) |

Hard (distribution-distance) evidence that MARLIN is maximally domain-confounded. Refinement:
even raw blendshape VALUES shift across domains (0.97); only the relative L-R asymmetry deltas
are comparatively invariant (0.79). Actionable: a Mayo-deployment model should rely on the
asymmetry deltas, not raw appearance OR raw blendshape values. Augmentation could only help by
pulling the 1.0/0.97 gaps down — a tall order for MARLIN, and it cannot create the missing eye
closure-dynamics.
