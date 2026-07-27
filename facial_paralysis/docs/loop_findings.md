# Overnight loop findings (2026-07-04) — the honest state of Mayo

> **Historical status:** This document predates the current 110D Landmark
> development champion. Use `CURRENT_MODEL.md` for present-tense model claims;
> all 0.668 “champion” references below describe the earlier web-QWK study.

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

## Historical web champion (0.668) error analysis + calibration (scripts/champion_analysis.py, calibrate.py)
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

## MARLIN problem #2: DANN (domain-adversarial) — FAILS (scripts/dann.py)
| model | rep domain-AUC | transfer eyes | transfer mouth |
|---|---|---|---|
| baseline (no DANN) | 0.997 | -0.27 | -0.66 |
| DANN lambda=1.0 | 0.997 | -0.01 | -0.38 |

DANN cannot make the representation domain-invariant (AUC unchanged at 0.997) — the frozen-
MARLIN input is perfectly domain-separable and the Mayo target is n=13, so adversarial
alignment has nothing to grip. Transfer nudges toward zero but never positive. **Third
representation-space domain-adaptation approach to fail (after CORAL and feature-noise), which
is itself the answer:** you cannot fix the MARLIN confound in feature space at this scale.

### Verdict on "how to solve the MARLIN problem"
1. **Feature-space domain adaptation (CORAL, DANN, augmentation-of-appearance) will NOT work** —
   the diagnostic explains why (MARLIN AUC 1.0, irreducible at n=13). Stop spending effort here.
2. **Pragmatic solution (now):** lean on the L/R asymmetry only (AUC 0.79) — drop MARLIN AND
   raw blendshape values. This is the geometry/asym deploy model; eyes stay weak by construction.
3. **Real solution (needs data):** replace MARLIN's eye crutch with eye-closure DYNAMICS (60fps
   EAR), which is domain-invariant and captures what appearance was faking on stills — requires
   in-domain video (have) + HB labels (don't). Pipeline built.
4. **Augmentation-to-look-like-Mayo** could only pull the appearance gap down (and DANN shows even
   aggressive alignment can't budge it much) and cannot create the missing dynamics — partial at best.

## Direction #1: rigorous label-free synkinesis quantification (scripts/mayo_synkinesis.py)
Synkinesis = involuntary co-contraction; a core FP problem, hard to grade, and needs NO
severity label (it is a physical co-activation measurement). We require the involuntary
movement to be TIME-LOCKED to the voluntary action (Pearson corr of the two per-frame
traces), distinguishing true synkinesis from coincidental activation.

Results (13 pts): **6/13 have detectable ocular→oral synkinesis**; patients separate by
pattern — FACES021 strong ocular→oral (0.52), MySlate_23 strong BOTH ways (reanimation
patient), FACES014/FACES018 oral→ocular. Validation: determinism IDENTICAL on the
duplicate take; gentle-vs-forced eye provocations consistent (std 0.066); time-locked
measure agrees with the crude eface synk (rho +0.67) but adds specificity.
→ outputs/mayo_eface/synkinesis.json. This is a candidate standalone paper (objective
synkinesis quantification), label-free by construction.

## Direction #4: DEPTH decompression BLOCKER SOLVED (scripts/depth_decode.py)
The Oodle-Kraken depth (`depth_data.bin`), previously documented as needing the proprietary
Oodle SDK, is now **decompressable with the open-source ooz** (pre-built `ooz/liboodle.dylib`,
called via ctypes from **arm64** /usr/bin/python3 — anaconda python is x86_64 and can't load it).
- Container cracked: record 0x02 = file header (intrinsics JSON, PixelSize 0.001, iPhone15,4),
  records 0x05 = per-frame depth; each = [0x05][15-char ts][16-B sub-header: compressed size at
  [12:16]][Kraken block]. ~1000 frames/take, decompress to exactly **460800 B = 640×360×2**
  (true size confirmed: Kraken returns 460800, fails on larger buffers).
- **PIXEL FORMAT SOLVED**: Kraken -> 16-bit horizontal Sub de-filter (per-row cumsum mod 65536)
  -> little-endian fp16 (meters). Verified by frame-to-frame temporal corr **0.93** and a clean
  face surface (~46k face px, face ~0.19m / background ~1.6m). Image is Orientation-4 (rotate to
  upright). Minor residual horizontal streaks (delta-chain resets at invalid runs) — cosmetic.
- (old note) earlier REMAINING: the 460800 bytes are byte0-
  dominant and spatially incoherent → a pre-compression filter/format (not plain fp16/uint16/
  planar/row-delta — all tested). Next: invert with the LiveLinkFace/Apple AVDepthData encoding
  (disparity? Paeth-like predictor?) — a format-reference task, not a compute one.
- PAYOFF once finished: appearance/camera-INVARIANT 3D facial asymmetry, which sidesteps the
  entire MARLIN domain-confound (domain-AUC 1.0) — the highest-upside direction.

## Direction #4 status: depth SOLVED; clinical 3D asymmetry needs landmarks (pod)
- **Decoding: fully SOLVED + committed** (Kraken + 16-bit horizontal Sub + le-fp16; temporal
  corr 0.93; recognizable face depth). The weeks-long "needs proprietary Oodle SDK" blocker is
  broken with open-source ooz. All 18 depth takes decode.
- **3D asymmetry pipeline built** (`scripts/mayo_depth3d.py`): decode → de-streak → head
  extraction → global reflective-symmetry (min mirror residual = irreducible asymmetry).
  It is DETERMINISTIC (duplicate take FACES018≡MySlate_14 → identical 0.0078) and differentiates
  patients 5× (5.6–28mm). `outputs/depth/asym3d.json`.
- **BUT it is not clinically meaningful yet:** 3D residual vs 2D clinical asymmetry rho=+0.08
  (n.s.), vs EAR closure -0.36 (n.s.). The landmark-free residual on the head+neck+shoulder blob
  captures head POSE/shape, not palsy facial asymmetry — this noisy compressed depth can't be
  cleanly registered without facial landmarks.
- **Completion route (scoped):** run MediaPipe on the RGB (pod) → face landmarks → map to depth
  (RGB 1280×720 is exactly 2× the 640×360 depth) → isolate the FACE, anchor the midsagittal plane
  on the nose bridge, and measure per-region (mouth-corner / eye) 3D asymmetry during each action.
  That is the clinically-valid, appearance-invariant 3D signal. The decoder + pipeline are ready
  to consume landmarks the moment they exist.

## Direction #4 COMPLETED: landmark-anchored 3D asymmetry pipeline (pod)
Built and ran the full landmark-anchored pipeline. Three sub-problems SOLVED:
- **Depth↔RGB registration (exact).** `frame_log.csv` gives every video/depth frame a ns
  timestamp on one shared clock; depth records embed a matching timecode key. Depth is 30fps,
  video 60fps, ~148 depth frames dropped → must match by timecode, not ordinal. Result: each
  resting depth frame matches its RGB frame to **0.0 ms**. Spatially, upright RGB 1280×720 =
  exactly 2× upright depth 640×360; rotation is `rot90(native, k=1)`, **confirmed by landmark
  overlay** (mesh lands exactly on the depth face). `scripts/depth_rgb_prep.py`.
- **MediaPipe landmarks (pod).** RTX 3090 pod, 478 landmarks/take on the matched resting RGB
  frames (14 takes). `outputs/depth3d/landmarks.json` (gitignored — biometric).
- **Pose-corrected per-region 3D asymmetry** (`scripts/asym3d_landmark.py`): sample metric
  depth Z at each landmark (nose-anchored ±7cm face-band gate + adaptive window + temporal
  median over ±7 frames to beat the ~10mm/px streak noise & IR holes), remove rigid head pose
  (LS plane Z~au+bv+c), then per anatomically-symmetric pair asymmetry = |r_L−r_R|. No camera
  intrinsics needed. `outputs/depth3d/asym3d_landmark.json`, figure `asym3d_summary.png`.

**Results (n=9 usable of 14; 5 dropped: face too far/small or depth too holey):**
- DETERMINISTIC: duplicate take FACES018≡MySlate_14 → identical (9.42 mm). Magnitudes now
  physically sane (6–25 mm; the landmark-free version gave impossible 500 mm).
- **But still no significant clinical correlation at this n:** 3D-overall vs 2D-blendshape
  asymmetry Spearman +0.21 (p=0.64, n=7); per-region mouth/eye ~0; brow +0.49 (p=0.27, best).
- **Honest read:** (1) same **n-wall** as everything else — 7–9 patients can't power a
  correlation; (2) 3D-depth asymmetry measures the **Z/protrusion axis**, orthogonal by
  construction to 2D blendshapes' XY-motion axis, so low correlation is expected, not proof of
  noise — it may be a *complementary* signal, but confirming that needs a 3D ground truth or
  more patients. The decode + registration + pipeline are the durable deliverables; clinical
  validation is gated on n, exactly like the transfer result. Depth decode reconfirmed correct
  (subH horizontal-Sub is smoothest at 9.58 mm/px vs 22 vertical / 61 2D).

## Direction #4 — what more is possible with NO new patients / NO labels (4 tracks)
Question: can #4 advance without the two things we can't get? Ran 4 label-free, no-new-patient
tracks (`asym3d_simulate.py`, `harvest_frames.py`→pod mediapipe→`asym3d_analyze.py`/`asym3d_final.py`;
`outputs/depth3d/asym3d_tracks.json`, figure `asym3d_tracks_summary.png`). Harvested 164 frames
(85 rest + 79 action peaks, matched to depth) across 15 takes; 102 usable.

- **Track 4 (simulation / measurement error):** symmetrize a real face depth (known asym=0),
  inject a known Δ on one side, recover it. **Noise floor ≈ 2.8 mm**; recovery **near-unbiased
  (slope ~0.95, linear)** where landmark coverage is dense, but insensitive on holey takes
  (FACES014's mouth). So sub-3mm asymmetry is indistinguishable from noise.
- **Track 3 (reliability):** per-frame resting 3D asymmetry is **NOT reliable — ICC 0.10, CV 32%**.
  This is the ROOT CAUSE of the null 2D correlations (the ~3mm floor is a big fraction of the
  6–25mm signal). **Constructive fix that works:** pooling frames at the landmark level (median Z
  per landmark across frames) lifts reliability to **0.77** (split-half r=0.62, Spearman-Brown) —
  matches the k≈27-frame prediction. So the measure CAN be made reliable without patients/labels.
- **Track 1 (dynamic 'asymmetry on demand'):** is region asym larger at the targeting action than
  at rest? mouth 9.7→15.3 (p=0.30, hints), eye/brow flat/down — **no significant effect** at n=7.
- **Track 2 (depth-unique):** nasolabial-fold relief asymmetry (5.5–48 mm across 9) and a
  lagophthalmos eye-asym-at-closure proxy are computable (2D can't), but inherit the reliability
  limits and show no significant closure-vs-rest effect (p=0.81).

**Bottom line (the honest ceiling without n/labels):** we CAN characterize the measurement
(floor ~3mm) and make it RELIABLE (0.77 via pooling) — a real methods result. But even the
reliable pooled 3D asymmetry stays **orthogonal to 2D** (Spearman +0.03, p=0.96) — it reliably
measures a *structural* axis (Z/protrusion) that is genuinely different from 2D motion asymmetry.
Deciding whether that is a **complementary clinical signal** vs **non-clinical face shape** is
exactly what requires a 3D ground truth, HB/eFACE labels, or more patients. That is the hard wall;
everything upstream of it is now done.

## Direction #5 DONE: measurement reliability of the label-free measures (no labels/patients)
A measure must be reliable before it can be valid. Established reliability with two label-free
sources (no test-retest recording exists): split-half within an action hold (pure measurement
error) + cross-provocation agreement (robustness lower bound). `scripts/reliability_suite.py`,
`outputs/mayo_eface/reliability.json`, figure `reliability_suite.png`.

- **2D blendshape L/R asymmetry is highly reliable AS A MEASUREMENT** (split-half ICC / SB-full):
  eye 0.97/0.98, brow 0.91/0.95, smile 0.74/0.86 — all GOOD. Concrete **MDC95** (smallest
  trustworthy change, asymmetry-index units): eye **0.034**, smile **0.082**, brow **0.150**.
- **Cross-provocation agreement is only moderate** (eye gentle-vs-forced ICC 0.58; EAR 0.54;
  smile relaxed-vs-reanimated 0.40) — but this is REAL physiology (forced recruits differently,
  reanimated ≠ relaxed), not measurement noise. Practical rule: **fix the provocation for
  longitudinal tracking** (always gentle-to-gentle), don't mix probes.
- **3D depth asymmetry (#4): per-frame ICC 0.10, reliable only after pooling (0.77).**

Takeaway: our 2D dynamic-asymmetry measures are trustworthy enough for clinical/home use; the
MDC95 values are the change thresholds feeding #6. 3D needs frame-pooling to be usable.

## Direction #6 DONE: home iPhone self-monitoring tool (spec + prototype)
Payoff of #5: because the label-free asymmetry measures are reliable with known MDC95, a patient
can self-record the FACES protocol at home and we can flag REAL change (|Δ| > MDC95) with no
clinician and no labels. `scripts/home_monitor.py`, `docs/HOME_MONITORING.md`,
`outputs/mayo_eface/home_monitor.png` + `home_scores.json` (12 patients).
- Per-session report card: per-region L/R asymmetry with the ±MDC/2 noise band.
- Change detection: a 50% recovery in eye-closure asymmetry is detectable unless the baseline is
  already near-symmetric (Δ < MDC95=0.034); averaging k sessions sharpens MDC ~1/√k.
- Fixed-probe rule (from #5): always compare gentle-to-gentle, never mix provocations.
- Honest boundary: it is a CHANGE/TREND tool, not a severity grader (HB/eFACE mapping needs labels);
  3D depth stays a research channel (per-frame unreliable). Real longitudinal validation needs
  follow-up recordings (not labels) — the one remaining step.
