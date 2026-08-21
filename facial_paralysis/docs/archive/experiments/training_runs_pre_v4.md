# Training Runs

Log of real (non-synthetic) training runs of the facial-paralysis pipeline.
Architecture/design: see `docs/model_design.md`. Synthetic wiring validation is
in `tests/test_pipeline_e2e.py`; this file records runs on real datasets.

> **Archived historical run log:** Run-level winners and words such as
> “deployable” or “default” below describe their original experiment only. The
> sole current model is Universal Clinical Router v4; use
> `../../CURRENT_MODEL.md`.
> Removed generated files and generic entrypoints remain available in Git
> history when an old run must be audited.

---

## Run #1 — PalsyNet binary (public-data warm-start, Stage 1)

**Date:** 2026-06-10
**Goal:** first real end-to-end exercise of the full pipeline and a usable
public-data baseline, per the "don't wait for clinical data" strategy — train a
strong palsy detector / severity proxy on public data now, fine-tune to HB when
Mayo/MEEI labels arrive.

**Data:** PalsyNet — 49 YouTube subjects (27 palsy / 22 healthy), one video each,
binary label. Each video treated as a single pseudo-action (`n_actions=1`); no
per-action structure exists in PalsyNet.

**Pipeline exercised (end to end):**
`video → action_bundle (frozen MARLIN × 4 windows + MediaPipe 72-d feature seq)
→ MultiStreamPatientDataset → FacialPalsyModel (binary ordinal head)`.
Only the temporal GRU + severity trunk + binary head are trained; MARLIN frozen.

**Protocol:** subject-level **stratified 5-fold CV** (each subject in exactly one
fold; no clip leakage). Train cfg: lr 5e-4, weight_decay 3e-2, 60 epochs, early
stop on val (patience 12), dropout 0.1. Metric: ROC AUC + accuracy, scored from
the binary head's P(palsy) = σ(s − θ).

**Reference point:** the feasibility-gate frozen-MARLIN *linear probe* scored
subject-level AUC **0.872** (see `project_marlin_feasibility`). This run asks
whether the full trainable model (GRU dynamics + severity head) matches/beats it.

### Results

Raw JSON: `outputs/palsynet_bundles/results.json`. Per-fold AUC:
`[0.833, 0.750, 1.000, 1.000, 0.900]`.

| Metric | Mean ± std (5-fold) | Pooled |
|---|---|---|
| ROC AUC | **0.897 ± 0.097** | 0.860 |
| Accuracy | **0.878** | 0.878 |

**Interpretation:** the full trainable pipeline (frozen MARLIN ⊕ trainable GRU →
severity trunk → binary ordinal head) **matches/slightly beats the frozen-MARLIN
linear probe (0.872)** under the same subject-level CV. So the end-to-end model
is wired correctly and learns a strong palsy signal on real data — not just on
the synthetic planted-signal test.

The gain over the linear probe is small, which is expected: binary palsy is
already (near-)linearly separable in frozen MARLIN features, so the temporal/GRU
stream adds little *here*. The temporal stream is expected to matter much more
for (a) ordinal HB severity and (b) per-action clinical data, where the
asymmetry *trajectory* — not just appearance — is diagnostic. PalsyNet has no
per-action structure, so this run under-exercises that stream by design.

### Notes / caveats

- Small N (49) → high per-fold variance (folds 3–4 hit AUC 1.000 on val sets of
  only 10). The **pooled AUC 0.860** is the more conservative single number than
  the fold-mean 0.897.
- Binary only — this calibrates the low end of the severity axis and gives a
  palsy detector; precise HB I–VI still needs MEEI/Mayo HB labels (Stage 2).
- PalsyNet is MARLIN's home domain (YouTube). Generalization to the Mayo iPhone
  domain is a separate question (the feasibility gate showed non-collapse there,
  centered identity margin +0.806, but palsy discrimination on Mayo is untested
  for lack of labels).
- Reproduce: `KMP_DUPLICATE_LIB_OK=TRUE <dev-python> scripts/train_palsynet.py`.

---

## Run #2 — Roboflow facial-paralysis COCO set

**Date:** 2026-06-10. Script: `scripts/run2_roboflow.py`. JSON:
`outputs/roboflow_bundles/results.json`.

**What the dataset actually is (verified, important):** 118 still **images**
(82/24/12 train/valid/test), every annotation labeled "facial-paralysis". It is a
**single-class palsy detection set — no healthy/negative class, and images not
video.** 117/118 had a detectable face. Consequence: it **cannot by itself raise a
palsy-vs-healthy AUC**, because a balanced classifier needs negatives. Images run
through the model as the degenerate clip case (MARLIN sees the image tiled to 16
frames; MediaPipe sequence length 1 → zero dynamics). So both the temporal stream
and the "extra data helps discrimination" premise are under-exercised by design.

### Experiment B — external generalization (the meaningful test)

Train binary model on **all** PalsyNet, then score the 117 independent Roboflow
palsy images.

| metric | value |
|---|---|
| sensitivity (flagged palsy) | 1.000 |
| mean P(palsy) | 0.996 |

**Honest interpretation:** sensitivity 1.0 on a **positives-only** set is **weak
evidence on its own** — a degenerate "always palsy" model scores 1.0 too (this is
exactly the Oo failure mode). It is only meaningful *combined with* Run #1, where
the same-style model achieved AUC 0.86 on PalsyNet (so it is **not** an
always-palsy model). Taken together: the detector assigns high palsy probability
to a completely different image source without collapsing — mildly encouraging,
but **not proof** of generalization (needs negatives from the same source).

### Experiment A — augmented CV (the literal "higher AUC?" request)

Pool the 117 Roboflow palsy images in as extra positives; subject-level
stratified 5-fold CV. Pool = 144 palsy / 22 healthy.

| Metric | Run #1 (PalsyNet only) | Run #2 (+ Roboflow) |
|---|---|---|
| pooled ROC AUC | 0.860 | **0.988** |
| pooled accuracy | 0.878 | 0.976 |
| fold AUC | [.83,.75,1,1,.90] | [1,.974,1,.991,.964] |

**Honest interpretation — the higher AUC is real arithmetic but largely an
artifact, not a genuinely better detector:**
- We added **117 easy positives** (clear web photos of palsy) to an **unchanged,
  tiny negative set** (still the same 22 PalsyNet healthy). Ranking metrics inflate
  when you add easy-to-rank positives; it does **not** show better separation of
  hard cases. So 0.988 is **not apples-to-apples** with Run #1's 0.860.
- The negative class (22) is now badly outnumbered (144:22); AUC is dominated by
  how the few negatives rank against many easy positives.
- Possible cross-source **leakage**: PalsyNet and Roboflow are both web-scraped;
  the same individuals could appear in both, inflating CV. Not dedupable here.

**Bottom line:** yes, the AUC number went up (0.860 → 0.988), but it reflects an
easier positive distribution, not a stronger model. The honest read of Run #2 is:
the pipeline ingests image data and a new source cleanly, and the detector does
not collapse on out-of-source palsy — but this data **does not meaningfully
improve palsy-vs-healthy discrimination**, because what we actually lack is more
**healthy/negative** examples and **severity/HB** labels. Roboflow has neither.

**What would actually move the needle:** balanced negatives (healthy faces), and
HB/severity-labeled data (MEEI, Mayo) for the ordinal task. Single-class positive
images are best reserved as *augmentation* for a future severity/encoder stage,
not as binary-AUC fuel.

---

## Run #3 — FNP region-ORDINAL (first test of the ordinal cut-point heads)

Script: `scripts/run3_fnp_region.py`. Results: `outputs/fnp_bundles/results.json`.

Every prior run was binary; this is the **first time the ordinal cut-point design
is exercised on real graded data**. FNP Detection is the only collected set with
graded labels: per image, eye and mouth regions annotated normal / paralyzed
{Weak,Mid,Severe} → 4-level ordinal severity per region (max severity if multiple
boxes). Attached as two **uncoupled** region tasks (own severity projection from
the shared trunk, per model_design §4.1) with ordinal heads. Images run as the
degenerate clip (MARLIN tiled to 16 frames, length-1 MediaPipe). **Quality
normalizer ON** (mode=normalize, work_size=112) — FNP is exactly the blurry
public data it targets. FNP's own train/valid/test split (367/103/49 images with a
detected face → 658/189/86 region records).

### Results (quadratic-weighted kappa / MAE-in-levels / accuracy)

| Region | split | QWK | MAE | acc |
|---|---|---|---|---|
| **mouth** | valid | **0.718** | 0.561 | 0.582 |
| **mouth** | test  | **0.626** | 0.591 | 0.568 |
| **eyes**  | valid | 0.503 | 0.692 | 0.440 |
| **eyes**  | test  | 0.459 | 0.762 | 0.476 |

### Interpretation
- The ordinal machinery **works**: mouth QWK 0.63 (test) is in the "good" band,
  eyes 0.46 "moderate". Confusion matrices concentrate near the diagonal (errors
  mostly ±1 level) → predictions are rank-consistent, **not collapsed**.
- **Eyes < mouth**, as expected: eye-closure severity is subtle in a *still* image
  (most error is weak↔mid, classes 1↔2); mouth asymmetry is grosser and easier.
  Also an argument for the temporal stream — a static frame can't see incomplete
  *blink dynamics*.

### Caveats
- Web-scraped images; **subject overlap across the FNP split is unknown** → numbers
  may be optimistic. A method check (does the ordinal head learn a real ranking?),
  not a clinical figure.
- Single-frame only (no real dynamics); GRU sees a length-1 sequence.

---

## Mayo iPhone domain — end-to-end face-validity / non-collapse check

Script: `scripts/validate_mayo_domain.py`. Results: `outputs/mayo_bundles/validation.json`.

The feasibility gate showed the frozen **encoder** doesn't collapse on Mayo
(centered identity margin +0.806). This checks the **full trained model**
(encoder→trunk→severity) on the 15 local iPhone takes. No HB labels → not a metric
run. Binary palsy model trained on all PalsyNet, then scored on the takes. Mayo
encoded **un-normalized** to match the (un-normalized) PalsyNet bundles.

### Finding — two parts (the split matters)
| Quantity | Across 15 takes | Read |
|---|---|---|
| **latent severity `s`** | std **1.01**, range **3.44** (6.73 → 10.17) | ✅ **non-collapsed** — real spread/ordering on the iPhone domain |
| **P(palsy)** | mean **1.0**, std **0.0** (0.999–1.000) | ⚠️ **saturated** — every take ≈1.0 |

### Interpretation
- The **representation is healthy**: the severity axis varies meaningfully across
  takes (std 1.0), end-to-end, consistent with the encoder-level gate — **not** the
  Oo "everything identical" failure mode at the representation level.
- The **binary probability saturates** because the PalsyNet-trained threshold isn't
  calibrated to the Mayo domain: `s` lands at 6.7–10.2, far above the learned palsy
  threshold, so sigmoid→1 for all. Two non-exclusive causes: (a) domain shift
  inflates the absolute `s` scale; (b) the takes may genuinely be mostly patients
  (it's a palsy study) — no labels to separate the two.
- **So:** absolute palsy/healthy calls do **not** transfer to Mayo without in-domain
  calibration — exactly what Stage-2 HB calibration on Mayo labels is for. Ranking
  by `s` is the usable signal until then.
- Note: the script's `non_collapsed` flag keyed on P(palsy) spread (the saturated
  quantity) → reported False. The correct measure is the spread of `s`, which is
  healthy; flag logic should switch to `s` next iteration.

### Caveat
- Take folder names (*FACES* vs *MySlate*) are **not** reliable palsy/healthy labels
  (see memory `project_data_duplicates`), so the ranking can't be scored, only
  eyeballed. The takeaway is the non-collapse of `s`, not the ordering itself.

---

## Run #4 — CFD healthy controls (a confound-aware NEGATIVE result)

**Date:** 2026-06-15. Script: `scripts/run4_cfd_controls.py`. JSON:
`outputs/cfd_bundles/results.json`.

**Goal:** Run #2 said we lack healthy negatives. Tried to fill the gap with the
**CFD (Chicago Face Database)** healthy faces found in the sibling `facial_defect`
project. CFD is *studio* photos; our palsy data is *YouTube/iPhone* — so this was
run **confound-aware**: both PalsyNet and CFD re-extracted through the **same**
quality normalizer (work_size=112), plus two honesty checks. Pool: 880 subjects
(27 palsy / 853 healthy = 22 PalsyNet-web + 831 CFD-studio neutral faces).

### Results

| Check | AUC | What it means |
|---|---|---|
| **Headline** palsy vs healthy (MARLIN feats) | **0.989** | looks great… |
| **Honesty A** — asymmetry-only (domain-invariant) | **0.534** | …but real palsy signal is ~chance |
| **Honesty B** — CFD-studio vs PalsyNet-web, *both healthy* | **1.000** | encoder reads the camera perfectly |

### Interpretation — the headline 0.989 is domain confound, NOT palsy signal

- **B = 1.000**: MARLIN perfectly separates CFD-studio from web *healthy* faces.
  Since 100% of palsy are web and 97% of healthy are CFD-studio, a classifier can
  score 0.99 just by detecting **"studio vs web"** — i.e. reading the camera, not
  the face. This is the Run #2 trap in its purest form.
- **A = 0.534**: strip appearance/domain and use only domain-invariant L/R
  asymmetry features → palsy vs healthy collapses to ~chance.
- **The quality normalizer did NOT fix it.** Capping resolution at 112px still
  leaves the domains perfectly separable to MARLIN (B=1.0). Out-of-domain healthy
  controls are simply not usable here.

### Conclusions (these matter for strategy)

1. **CFD — and out-of-domain healthy faces in general — cannot serve as controls**
   for our web/iPhone palsy data. Any metric built on them is a domain detector.
   Don't use the `facial_defect` CFD images as our negative class.
2. **We need IN-DOMAIN healthy controls** — healthy faces captured the *same way*
   as patients (Mayo iPhone/LiveLinkFace, or at least web video like PalsyNet's own
   healthy). This is the real missing-negatives fix.
3. A≈chance on **resting** faces re-confirms the project's core bet: the palsy
   signal lives in **movement / per-action dynamics**, not a static neutral frame.
   Reinforces the prompted per-action capture protocol.
4. **The confound-aware design earned its keep**: without checks A/B we'd have
   reported 0.989 and been badly misled. Keep these checks standard whenever a new
   negative source is from a different domain.

**Net:** a useful negative result. The sibling project's data does not give us
controls; the path forward is in-domain healthy captures + HB/severity labels
(MEEI/Mayo). Bundles cached at `outputs/cfd_bundles/`, `outputs/palsynet_bundles_norm/`.

---

## Run #5 — YFP region-ordinal, SUBJECT-LEVEL CV

**Date:** 2026-06-15. Script: `scripts/run5_yfp_region.py`. JSON:
`outputs/yfp_bundles/results.json`.

YFP (now complete: 5 zips, `data/external/YFP/`) is the second graded set after FNP
(Run #3). Labels = Pascal-VOC XML, object name `{Normal,SlightPalsy,StrongPalsy}_{Eyes,Mouth}`
→ 3-level ordinal per region. Two **uncoupled** region heads, images as degenerate
clip, quality normalizer ON. Key upgrade vs Run #3: **honest subject-level GroupKFold**
(Run #3's FNP split may have leaked). Frames subsampled ≤200/subject, rare (non-Slight)
frames kept preferentially.

### Results (4-fold subject-level CV, pooled out-of-fold)

| Region | n | QWK | MAE | acc | usable levels |
|---|---|---|---|---|---|
| **eyes** | 937 | **0.472** | 0.495 | 0.525 | 3 (363/500/74 N/Sl/St) |
| **mouth** | 937 | **0.754** | 0.228 | 0.870 | **2** (644/0/293 — no Slight) |

### Interpretation (with the caveats that matter)
- **Eyes = the real 3-level test, and it holds up honestly: QWK 0.472** under
  leak-free subject CV — essentially identical to Run #3's leaky-split 0.46/0.50.
  So that moderate eye-severity signal is **real, not leakage**. The middle class
  (Slight, the majority) is the hard one (confusion smears Slight↔Normal/Strong) —
  expected: eye-closure severity is subtle in a **still** frame with no blink
  dynamics. Direct argument for the temporal stream + in-domain video.
- **Mouth QWK 0.754 is NOT a 3-level result** — the image-having subjects contain
  **zero Slight-mouth** frames, so this is effectively a binary normal-vs-strong
  mouth classifier (good: 88%/84% per-class) dressed as 3-level. Do not cite it as
  ordinal severity.

### Caveats (significant)
- Only **11 subjects** with image+label survived matching+subsampling, and very
  uneven (subjects 1/14/17/11/10 dominate; others contribute 2–11 frames). 4-fold
  CV is honest but thin → high variance; some folds are ~2 subjects.
- The rare-preferring subsample distorts the natural label mix (inflates Normal/
  Strong) — kappa is on a re-balanced, not natural, distribution.
- Web stills, no dynamics; positives only (not a control source).
- Most of YFP's labels (32 XML subjects, ~26.7k) lack matching images here (only 16
  image subjects, 11 usable) — getting the remaining image subjects would help.

**Net:** the ordinal region machinery generalizes under honest subject-level CV
(eyes 0.47 robust across two datasets). Ceiling on a still frame is moderate; the
clear next unlocks are **dynamics** (per-action video) and **more subjects /
balanced Slight** — i.e. in-domain Mayo captures, not more web stills.

---

## Run #6 — Unified public-data warm-start (v1 deployable checkpoint)

**Date:** 2026-06-15. Script: `scripts/run6_unified.py`. JSON: `outputs/run6_results.json`.
Checkpoint: `outputs/checkpoints/warmstart_v1.pt` (616 KB — trainable head only;
frozen MARLIN loaded separately at inference).

First run to train **one** `FacialPalsyModel` jointly on all usable sources via the
multi-task design — PalsyNet→`binary` (coupled), FNP+YFP→shared `eyes`/`mouth`
(3-level ordinal, uncoupled). FNP's native 4 levels mapped to the common 3-level
{Normal, Slight, Strong} (`0→0, {Weak,Mid}→1, Severe→2`) so both sources train the
same heads. Honest per-source holdout (PalsyNet stratified 20%; FNP uses its valid
split; YFP holds out subjects 14/17/26). Train 1869 recs / val 987.

### Results (held-out val)

| task | n | metric | overall | FNP | YFP |
|---|---|---|---|---|---|
| binary (palsy) | 10 | AUC | 1.00\* | — | — |
| eyes severity | 485 | QWK | 0.38 | 0.41 | 0.38 |
| mouth severity | 492 | QWK | **0.82** | 0.61 | 0.86 |

\* binary val n=10 (the held-out PalsyNet subjects) — AUC 1.0 here is **not a
meaningful number**, just "didn't break"; trust Run #1's pooled 0.86 for binary.

### Interpretation (honest)

- The unified multi-task routing **works**: one shared trunk learns from binary +
  two ordinal region tasks at once, first time exercised together.
- **mouth severity is strong (QWK 0.82)** and **eyes modest (0.38)** — consistent
  with every prior run. Eyes severity is capped on still frames (eye-closure
  asymmetry is a *dynamic* event); mouth asymmetry is more visible in a static pose.
- **Cross-source generalization is real, not source-confounded**: eyes holds across
  FNP (0.41) and YFP (0.38); mouth above chance on both. Combining sources did not
  collapse to a per-source label prior.
- Saved as **v1 warm-start checkpoint**, and `scripts/predict.py` scores any
  video/image through the identical pipeline. Demo sanity (held-out PalsyNet):
  palsy clip → P(palsy) 0.995, eyes/mouth "Strong"; healthy clip → 0.052,
  "Normal". The deployable artifact behaves sensibly.

### Caveats / what v1 is and isn't

- v1 = a **palsy detector + mouth-severity estimator** on web-still-quality faces.
  Eyes-severity is weak; there is **no HB I–VI head yet** (needs HB labels).
- All training data is web stills (no dynamics), FNP subject overlap across its
  splits is unknown, val sets are small. These are **method checks, not clinical
  numbers**.
- Next unlocks unchanged: in-domain Mayo captures (dynamics + healthy controls) and
  HB/MEEI labels. v1 is the fine-tuning starting point for when they arrive
  (`predict.py` already runs on Mayo `.mov`).

---

## Run #7 — YFP TEMPORAL CLIPS (testing whether real motion helps) — NEGATIVE/INCONCLUSIVE

**Date:** 2026-06-15. Script: `scripts/run7_yfp_clips.py`. JSON: `outputs/yfp_clip_bundles/results.json`.

Hypothesis: feed YFP as real 16-frame motion clips (~2.7 s, 6fps) instead of single
tiled frames, so MARLIN sees motion and the GRU gets a real L–R asymmetry trajectory
— finally training the temporal stream. Added a `side` (affected L/R) head from XML `<pose>`.
536 clips / 11 subjects, subject-level GroupKFold.

| task | n | QWK | vs Run #5 single-frame |
|---|---|---|---|
| eyes | 536 | **0.26** | ↓ (was 0.47) |
| mouth | 536 | 0.61 | ≈ |
| side | 339 | 1.00 | **degenerate** |

**Honest read — motion did NOT help here, and the comparison is confounded:**
- **`side` head is useless**: all 339 palsy clips are labeled `Left` (single class) →
  QWK 1.0 is meaningless. Either our sampled subjects are all left-affected or YFP's
  pose is overwhelmingly Left. Dropped.
- **Not a clean A/B**: Run #7 used 90 anchors/subject (vs Run #5's 200) → ~half the
  eyes labels, and the rare-preferring subsample left **mouth with no level-1 at all**
  ({0:363, 2:173}). Different N and label distribution, so eyes 0.26-vs-0.47 is not a
  fair head-to-head.
- **Plausible mechanism even so**: MARLIN **mean-pools** the 16-frame window, which
  *blurs a transient event* — eye closure is brief, so averaging over 2.7 s dilutes
  exactly the signal the per-frame label refers to. Long-window pooling is the wrong
  way to exploit motion for a per-frame label.
- **High fold variance** (best-kappa 0.29 / 0.53 / 0.62 / 0.89 across 4 folds) — 11
  subjects is too few to read a small QWK difference.

**Conclusion / lesson:** YFP's labels are properties of *held-expression frames*;
surrounding motion adds noise, not signal. **The temporal stream's real payoff needs
per-ACTION in-domain video** (Mayo prompted captures), where motion *is* the
diagnostic event and the label is per-action — not per-frame web video. We stop
trying to squeeze dynamics out of YFP. **Run #6 (single-frame) remains the best
public warm-start (v1).**

---

## Run #9 — Temporal-POOLING ablation (was Run #7 a pooling artifact?)

**Date:** 2026-06-17. Script: `scripts/run9_temporal_pool.py`. JSON:
`outputs/yfp_clip_bundles/pooling_ablation.json`. Log: `outputs/run9_pooling.log`.

Run #7 concluded "motion didn't help" (eyes QWK 0.26) but flagged its own
hypothesis: the GRU's masked-**mean** over the ~3 s clip *dilutes* a transient
event (incomplete eye closure during a blink). Until now the temporal encoder
*only* mean-pooled — so "motion is useless" and "our pooling washed out the
motion" were confounded. This run separates them. New code adds a first-class
`temporal_pool` knob (`mean | max | attention`) to `TemporalLandmarkEncoder`
(+ a `marlin_window_pool` knob); 10 unit tests in `tests/test_temporal_pool.py`,
incl. the mechanism check that a transient spike survives `max` but is diluted by
`mean`. **Pure cache replay** of Run #7's 536 clips / 11 subjects (no MARLIN /
MediaPipe / raw frames): SAME data, SAME subject-level 4-fold splits, SAME model
size — the *only* variable is the GRU pool.

### Results (OOF-pooled QWK / accuracy, subject-level CV)

| pool | eyes QWK | eyes acc | mouth QWK | mouth acc |
|---|---|---|---|---|
| **mean** (≈Run #7) | 0.287 | 0.573 | 0.630 | 0.789 |
| **max** (peak)     | 0.255 | 0.517 | **0.723** | 0.830 |
| **attention**      | **0.409** | 0.655 | 0.523 | 0.726 |
| *Run #7 mean (ref)* | *0.26* | — | *0.61* | — |

(mean-pool reproduces Run #7's 0.26/0.61 — small delta because Run #9 drops the
degenerate `side` head, leaving 2 tasks vs Run #7's 3. Confirms a faithful replay.)

### Interpretation — hypothesis PARTLY confirmed, with a real nuance

- **Pooling matters, and Run #7 undersold the clip approach for eyes.** Switching
  mean→**attention** lifts eyes **0.287 → 0.409 (+0.12 QWK, +58% rel)** on
  *identical* data. So part of Run #7's negative eyes result *was* a mean-pool
  artifact, exactly as that run hypothesized — not purely "motion is noise".
- **No single pool wins both — the optimum is region-dependent:**
  - **Eyes → attention.** Eye-closure is a brief *sub-sequence* (a blink), not one
    peak frame; attention learns to weight that window. `max` (single peak frame)
    actually *hurt* eyes (0.255) — too brittle on noisy GRU outputs.
  - **Mouth → max.** Mouth asymmetry is a *sustained* most-asymmetric pose; peak
    pooling captures it (0.63 → 0.72). Attention *hurt* mouth (0.523), likely
    overfitting on thin data (mouth is effectively binary here, {0:363, 2:173}).
- **BUT clips still lose to single frames on this web data.** Best clip result
  (eyes attention 0.409) is below single-frame Run #5 (0.47) and far below Run #8's
  single-frame + EAR/geometry/stream-balance (eyes **0.716**). On web stills the
  explicit geometric features remain the bigger lever than motion.

### Caveats (significant — do not oversell the ±0.12)

- **11 subjects, 4 folds, huge variance:** per-fold best-kappa ranged 0.06–0.92.
  The pool ordering (attention>mean for eyes, max>mean for mouth) is *directional*
  and mechanistically sensible, but **not statistically conclusive** at this N.
- Still web-stills-as-clips, not in-domain per-action video; mouth has no Slight
  class (binary in disguise).

**Net:** Run #9 *refines* Run #7 rather than overturning it. The pooling knob is now
real, tested, and matters; "motion is useless" was too strong for eyes. The
strategic conclusion is unchanged — **the motion payoff needs in-domain per-action
video, not web clips** — but we now have (a) a tested `temporal_pool` knob, and
(b) evidence that pooling should be **region-aware** (attention for the dynamic
eye head, max for the sustained mouth head). Carry both into the Mayo-video stage.

---

## Run #10 — Mayo severity / non-collapse, CORRECTED (v1 on cached bundles)

**Date:** 2026-06-17. Script: `scripts/run10_mayo_severity.py`. JSON:
`outputs/mayo_bundles/run10_severity.json`.

Supersedes the ad-hoc `validate_mayo_domain.py` check, which had a **verdict bug**:
it keyed "non-collapse" on the spread of **P(palsy)** — saturated (~1.0 for all
takes) because the public-trained threshold isn't calibrated to the Mayo domain —
and so wrongly printed "COLLAPSED-LIKE". The correct signal is the spread of the
latent severity `s`. Run #10 loads the **actual deployed v1 checkpoint**
(binary + eyes + mouth heads, `warmstart_v1.pt`) and scores the 15 cached
two-stream Mayo bundles (no re-encode, no mediapipe needed).

### Results (14 unique takes; no HB labels → ranking, not metrics)

| quantity | mean | std | min | max | spread? |
|---|---|---|---|---|---|
| latent severity `s` | 3.52 | **1.50** | 1.68 | 6.21 | ✅ real |
| P(palsy) | 0.94 | 0.06 | 0.84 | 1.00 | mild (not saturated) |
| eyes severity (E[grade]) | 0.84 | 0.31 | 0.28 | 1.30 | ✅ |
| mouth severity (E[grade]) | 0.76 | 0.57 | 0.08 | 1.66 | ✅ |

eyes grade histogram `[6, 7, 1]` (N/Sl/St); mouth `[7, 3, 4]`.

### Interpretation
- **NON-COLLAPSED by `s`** (std 1.50, range 4.5) — the corrected verdict. The
  representation produces a real, ordered severity spread on the iPhone domain,
  matching the encoder-level feasibility gate. The old "collapsed" verdict was a
  flag-logic artifact.
- **P(palsy) is NOT saturated with v1** (0.84–1.00), unlike the binary-only model
  in validate_mayo (all ≈1.0). The unified model's `s` sits lower (1.7–6.2 vs
  6.7–10.2), so the sigmoid isn't pinned. All takes still read >0.84 palsy —
  consistent with a palsy-study cohort and/or residual domain miscalibration.
- **Region severities differentiate takes** (mouth especially, std 0.57) — the
  model isn't emitting one constant grade.
- **Determinism check passed:** the duplicate `20260305_FACES018` /
  `20260305_MySlate_14` (same recording, see manifest) produced **identical**
  scores (s=4.30, eyes=0.82, mouth=0.96) — sanity-confirms the pipeline.

### Caveats (honest)
- **No HB labels** → ranking/non-collapse only, not accuracy.
- **Preprocessing mismatch:** v1 trained on quality-**normalized** public crops
  (work_size 112); the cached Mayo bundles are **un-normalized**. So absolute `s`
  is not domain-calibrated — only the ranking is informative. The fix is a
  normalized re-encode; a mediapipe-free cv2 face-crop encoder
  (`src/preprocessing/face_crop_cv2.py`) was built and validated for this
  (`scripts/mayo_cv2crop_validate.py`).
- Folder names (FACES* vs MySlate*) are NOT palsy/healthy labels; the corrupt
  1-second take `20260313_MySlate_22` (68 frames) should be excluded going forward.

**Net:** corrects the Mayo non-collapse story (it was never collapsed — the metric
was wrong) and adds per-region severity readouts. The blockers are unchanged: HB
labels + per-action segmentation + domain-calibrated preprocessing.

---

## Run #11 — Mayo re-extracted with REAL MediaPipe on GPU (RunPod A100)

**Date:** 2026-06-17. Pod: RunPod A100-SXM4-80GB. Scripts: `scripts/mayo_extract_pod.py`
(extract), `scripts/run10_mayo_severity.py mayo_bundles_norm` (re-score). JSON:
`outputs/mayo_bundles_norm/run10_severity.json`.

Run #10 carried a caveat: v1 was trained on quality-**normalized** crops but the
cached Mayo bundles were **un-normalized**, and the geometric stream had never been
extracted with real MediaPipe locally (no wheel). On a RunPod A100 (isolated venv:
mediapipe 0.10.35 + transformers + torch-cuda) we re-extracted **all 15 Mayo takes**
with the REAL MediaPipe FaceLandmarker (52 blendshapes + L/R asymmetry, 72-d) + MARLIN,
applying v1's normalizer (work_size 112) — preprocessing-consistent this time.

(Engineering note: naive cv2 decode of the 60fps ~2-min takes over the pod's network
FS ran at ~10 fps → 17 min/take. Copying each mov to /dev/shm + **seeking** only the
sampled frames cut it to ~2 min/take.)

### Result — the caveat was immaterial; the ranking is robust

| quantity | Run #10 (un-normalized) | Run #11 (real mediapipe, normalized) |
|---|---|---|
| `s` std / range | 1.50 / 4.5 | **1.505 / 4.66** |
| P(palsy) range | 0.84–1.00 | 0.83–1.00 |
| eyes sev (mean/std) | 0.84 / 0.31 | 0.89 / 0.29 |
| mouth sev (mean/std) | 0.76 / 0.57 | 0.79 / 0.57 |
| non-collapsed by `s` | ✅ | ✅ |

Re-extracting with the correct preprocessing **barely moved anything** — `s` spread,
the take ranking, and the non-collapse verdict are all preserved. So:
- The Run #10 normalization caveat is **closed**: it did not materially affect the
  Mayo severity ranking. The earlier un-normalized scoring was fine.
- For the first time, **real MediaPipe geometric features exist for the Mayo data**
  (not the local cv2-crop approximation), extracted at full quality on GPU.
- Blockers unchanged and now isolated to the two we cannot synthesize: **HB labels**
  and the **protocol transcript** (to label per-action windows).

---

## Run #12 — Unified warm-start v2 on GPU (mean vs attention pooling)

**Date:** 2026-06-17. Pod: RunPod A100. Script: `scripts/train_v2_pod.py` (reads
`outputs/train_manifest.json` — generated locally by `make_train_manifest.py` —
so the pod trains from cached bundles with no raw data). Same unified data as
Run #6 (2856 recs: PalsyNet binary + FNP/YFP eyes/mouth 3-level; train 1869 / val 987).
Trains the SAME model on GPU with two temporal-pooling settings, to carry Run #9's
finding into the full historical candidate model.

### Results (held-out val)

| pool | binary AUC* | eyes QWK | mouth QWK | YFP eyes | YFP mouth |
|---|---|---|---|---|---|
| **mean** (reproduces v1) | 1.0 | 0.407 | 0.835 | 0.461 | 0.878 |
| **attention** (Run #9) | 1.0 | **0.426** | **0.859** | **0.485** | **0.902** |
| *v1 (Run #6, CPU)* | *1.0* | *0.38* | *0.82* | *0.38* | *0.86* |

\* binary val n=10 — not meaningful (trust Run #1's 0.86).

### Interpretation
- **mean-pool v2 reproduces v1** (eyes 0.41 vs 0.38, mouth 0.84 vs 0.82) — confirms
  the GPU pipeline matches the CPU one; small deltas are seed/precision.
- **Attention pooling gives a modest but consistent lift** on the full unified set:
  eyes 0.407→0.426, mouth 0.835→0.859 (YFP eyes 0.461→0.485, mouth 0.878→0.902) —
  consistent with Run #9 (attention helps the dynamic eye head). FNP eyes is
  unchanged (~0.24); the gain is YFP-driven.
- **Honest caveats:** the gains are small and within the noise of an N≈987 val with
  few subjects; eyes *accuracy* actually drops (0.64→0.58) as kappa rises (attention
  spreads predictions → better ranking, more ±1 errors). Still single-frame web data
  (the temporal stream's real payoff still needs in-domain per-action video).

**Net:** `warmstart_v2_attention.pt` is a small, real improvement over v1 and the new
default warm-start. The pod now has a working GPU training path (mediapipe +
torch-cuda) for when HB labels / per-action Mayo video arrive — at which point the
HB head and the temporal stream can finally be trained for real.

---

## Run #13 — PER-ACTION segmentation of Mayo + model in its intended n_actions mode

**Date:** 2026-06-17. Pod: RunPod A100. Scripts: `mayo_blendshape_segment.py`
(segment), `mayo_build_per_action.py` (bundle), `mayo_score_per_action.py` (score).
JSON: `outputs/mayo_blendshapes/segments.json`, `outputs/mayo_action_bundles/per_action_scores.json`.

This resolves the **second of the two project blockers** — per-action segmentation
(model_design §2) — for the Mayo recordings. The clinician confirmed there is **no
precise timing** (home self-recordings) but a **fixed action ORDER** of 7-8 actions
(memory: mayo-action-protocol). Each action has a distinct MediaPipe blendshape
signature (browInnerUp / eyeBlink / eyeSquint / mouthSmile / mouthPucker /
mouthLowerDown), so we:
1. Extract a dense per-frame blendshape time-series (ffmpeg 6fps + MediaPipe).
2. Detect each action by its **signature peak** (baseline-subtracted, scipy peaks);
   label by signature (NOT timeline position) so order deviations are handled;
   split the two eye-closures / two smiles by occurrence order.
3. Build one MARLIN+MediaPipe bundle per detected action, in its canonical slot.

### Segmentation result (all 15 takes)
- **13 usable takes segmented into 5-7 labeled actions each** (mean 5.8); the corrupt
  1-second `MySlate_22` correctly yields 0. Validated visually
  (`outputs/mayo_blendshapes/*.png`): signatures are clean and separable.
- **Home recordings DO deviate from the canonical order** (e.g. `MySlate_28` starts
  with eye-closure, does eyebrow 5th; `MySlate_23` puckers after the reanimated
  smile). Signature-driven labeling handles this; strict order-matching would not.

### Per-action scoring (v2-attention, n_actions=7) — first time in the designed mode

| quantity | whole-take (Run #11) | **per-action (Run #13)** |
|---|---|---|
| latent `s` std / range | 1.50 / 4.5 | **0.92 / 3.3** |
| non-collapsed by `s` | ✅ | ✅ |
| duplicate FACES018≡MySlate_14 | identical | **identical (s=2.48)** ✅ |
| mean actions/take | 1 (whole clip) | 5.8 |

- The model runs cleanly in its **intended per-action mode** for the first time on
  real data: non-collapsed, deterministic (the duplicate scores identically), with
  sensible per-region readouts.
- `s` spread is tighter per-action (0.92 vs 1.50) — averaging 6 action embeddings
  smooths the whole-take variance; expected, not a problem.
- Still **no HB labels** → this is non-collapse + ranking, not accuracy. But the
  per-action *structure* the architecture was built around now exists end-to-end on
  Mayo data.

**Net:** both design blockers are now technically cleared on the Mayo set —
per-action segmentation works (this run) and the full mediapipe pipeline runs on GPU
(Runs #11-12). The ONLY remaining gate is **clinical HB labels**; once they arrive,
`train_v2_pod.py` + these per-action bundles can train the real HB head and the
temporal stream (where per-action *motion* is finally the diagnostic signal).

---

## Run #14 — Label-free L/R asymmetry severity (and it DISAGREES with the model)

**Date:** 2026-06-17. Script: `scripts/mayo_asymmetry_severity.py`. JSON:
`outputs/mayo_asymmetry/asymmetry_severity.json` + `regions.png`. Runs locally on the
cached blendshapes + segmentation — no labels, no mediapipe, no GPU.

Facial palsy IS movement asymmetry, so per action we measure the left-vs-right
imbalance of its signature blendshapes at the held-expression peak:
`AI = |L−R|/(L+R+eps)` (0 = symmetric/healthy → 1 = one side dead), averaged over
the action's L/R pairs and aggregated into eFACE-like regions (brow/eye/mouth). This
is clinically grounded, interpretable, and — per Run #4 — the one signal that is
**domain-invariant**, so unlike the learned `s` it should transfer to Mayo.

### Results (13 unique takes)
- Asymmetry overall: mean 0.14, range 0.07–0.26. Determinism check passes (duplicate
  FACES018≡MySlate_14 both 0.162).
- **Most patients have a CONSISTENT weaker side across actions** (side-consistency
  1.0 for 7/13) — a real, coherent asymmetry, not noise.
- Per-region pattern varies by patient (some brow-dominant, some eye/mouth) —
  clinically realistic; palsy hits regions unevenly.

### The important finding — asymmetry ANTI-correlates with the learned `s`

| comparison | Spearman |
|---|---|
| asymmetry-overall vs model `s` | **−0.52** |
| eye-asymmetry vs model eyes-sev | −0.01 |
| mouth-asymmetry vs model mouth-sev | −0.50 |

The label-free clinical-asymmetry severity and the warm-start model's learned
severity **disagree (even anti-correlate) on the Mayo domain.** This operationalizes
Run #4's lesson on the actual target data: the frozen-MARLIN-driven `s` is largely an
**appearance/domain reading** that does NOT capture clinical movement asymmetry
out-of-domain, whereas the asymmetry score reads the real left-right deficit and is
domain-invariant. So **on Mayo, the asymmetry score is the more trustworthy severity
proxy right now** — the learned `s` should not be taken as clinical severity here.

### Caveats
- 13 patients (small); no HB labels to adjudicate which ranking is correct.
- Head pose can inflate AI (MediaPipe blendshapes are pose-robust but not immune).
- A few takes lack smile/teeth actions → mouth-region AI is NaN there.

### Implications (strategy)
1. We now have a **deployable, interpretable, label-free severity estimator** for
   Mayo ("your left smile is 40% weaker than right"), usable for ranking/triage today
   and as the sanity baseline the future learned HB head must beat.
2. When HB labels arrive, expect the **geometric/asymmetry + temporal streams**, not
   the frozen-MARLIN appearance, to carry the Mayo severity signal — weight them
   heavily, and consider feeding the asymmetry indices in as explicit features.
3. The asymmetry score is the natural **anchor for active learning**: have clinicians
   HB-label the takes where asymmetry and the model disagree most.

---

## Run #16 — Expanded, LEAK-SAFE training (anisa + kaggle + stroke) → mixed/neutral

**Date:** 2026-06-18. Pod: RunPod A100. Scripts: `build_expanded_plan.py`,
`extract_image_bundles_pod.py`, `make_train_manifest_v4.py`, `train_v4_pod.py`.
Leakage guard: `dedup_check.py` / `docs/leakage_policy.md`.

Added 3 newly-acquired web sets as **TRAIN-ONLY, deduped, group-safe** (no val/test):
anisa → `coarse3` (Normal/Medium/Strong), kaggle-droop → `binary` positives, stroke →
`coarse3` (binned). Dedup dropped 265 anisa internal dups + 246 kaggle↔FNP + 120 kaggle
within (so kaggle added only 658 of 1024). Eval UNCHANGED on the clean holdouts.

**Two findings:**
1. **`stroke` is unusable in a face pipeline:** all 709 of its "eye-stroke-regression"
   images are *eye-region crops* — MediaPipe finds no full face and rejects them (709
   skipped). Good that it failed loudly rather than feeding garbage. So v4 effectively
   adds **anisa 740 (coarse3) + kaggle 658 (binary)**.
2. **Expanded data did not decisively help** (held-out, clean):

| metric | v2-attention | **v4-expanded** |
|---|---|---|
| eyes QWK | 0.426 | **0.458** ↑ |
| mouth QWK | 0.859 | 0.803 ↓ |
| binary AUC | (n=10) | (n=10) |

The weak **eyes head nudged up** (+0.03 — extra global-severity data via coarse3
regularizes the shared trunk) while the already-strong **mouth nudged down** (−0.06).
Both are within the noise of an N≈987 / few-subject val. Net ≈ neutral.

**Interpretation (honest):** more *web-still* palsy images give diminishing returns —
they're the same modality/distribution as FNP/YFP. The leak-safe pipeline + coarse3
head + `warmstart_v4_expanded.pt` are kept, but the decisive levers remain unchanged:
**in-domain Mayo video, HB labels, AU-dynamics data** — not more web stills. v2-attention
and v4 are both reasonable warm-starts; pick v4 for the extra coarse3 head, v2 for the
slightly better mouth.

---

## Run #17 — autoresearch-FP: 40-experiment autonomous config search → plateau

**Date:** 2026-06-18. Full write-up: `autoresearch_fp_pre_v4.md`. Harness:
`scripts/fp_research.py` + `fp_batch{1,2,3}.json`; log `outputs/autoresearch_fp/fp_results.tsv`.

Adapted karpathy/autoresearch (edit→train→measure→keep-if-better loop) to our task:
each experiment = a model/data/optimizer CONFIG, metric = mean(eyes,mouth) QWK on the
clean holdouts. Ran **40 experiments** over 3 adaptive batches (search budget 40ep/256,
~2 min each), then verified the winners at the full budget (80ep/64) over 3 seeds.

**Result:** the short-budget "winner" (region-only + reweight + lr1e-3) **did not hold**
at the proper budget — it was a budget artifact (lr1e-3 overshoots at 80ep, mouth→0.77).
At the full budget, **v2ref (0.635±0.015) ≈ region-only (0.634±0.016)** are tied at the
top; no config beats the existing v2 recipe beyond noise.

**Takeaways:**
- **We are at a config plateau (~0.635); the ceiling is DATA, not architecture** — eyes
  QWK is stuck ~0.42 because eye-closure severity needs *video dynamics*, absent in still
  web images (cf. Runs #7/#9). Autoresearch confirmed this rather than breaking it.
- **Simplification win:** `region-only` (FNP+YFP, eyes+mouth; drop binary+coarse3+PalsyNet)
  matches v2 with a simpler model — prefer it if only region severity is needed.
- The harness is reusable: re-point the metric/holdout at Mayo HB labels when they arrive
  and the same loop will tune the real HB model.

---

## Run #15 — v3: reweight the geometric stream (NEGATIVE on public val) + B-2/3/4

**Date:** 2026-06-17. Scripts: `train_v3_pod.py` (B-1), `mayo_active_learning.py`
(B-3), `src/datasets/au_intensity_adapter.py` (B-2). Docs: `docs/data_acquisition.md`.

### B-1 — reweight geometric stream (Run #14 hypothesis): does NOT help the measurable metric
Trained v3 = v2 + `marlin_proj_dim=128` (MARLIN 768→128) + `stream_layernorm` +
attention pooling, to down-weight the domain-confounded appearance stream relative
to the geometric one (motivated by Run #14).

| model | eyes QWK | mouth QWK |
|---|---|---|
| v2-attention | **0.426** | **0.859** |
| **v3 reweighted** | 0.341 | 0.811 |

**v3 is WORSE on the public-stills validation.** Honest read: on **web stills**, the
MARLIN appearance stream genuinely *is* informative for the eyes/mouth region labels,
so shrinking it hurts. Run #14's "appearance is confounded" finding was about the
**Mayo** domain — and Mayo improvement is **unmeasurable without HB labels**. So
reweighting trades measurable public-fit for hypothetical Mayo-robustness; we cannot
confirm the trade is worth it yet. **v2-attention remains the deployable default.**
(`warmstart_v3_reweighted.pt` kept for when Mayo labels let us test the hypothesis.)

### B-3 — active-learning HB-labeling priority (done)
`outputs/mayo_active_learning.json`: ranks Mayo takes by disagreement between the
label-free asymmetry severity and the model `s`. Top conflicts (label these first):
`FACES020` (model says severe s=4.92, asymmetry says symmetric 0.08), `MySlate_29`,
`FACES024`. Labeling the highest-disagreement takes maximally resolves which signal
is right and best calibrates the HB head.

### B-2 — AU-intensity pretraining infra (done; data gated)
`src/datasets/au_intensity_adapter.py` maps FACS AU intensities (DISFA/BP4D/FEAFA+) →
the model's (T,F) blendshape-style feature sequence (with L/R asymmetry), so AU data
can pretrain the temporal/geometric encoder on real dynamics with no palsy label.
Datasets are application-gated (EULA/request) — access paths in `docs/data_acquisition.md`.

### B-4 — depth 3D asymmetry: BLOCKED
`depth_data.bin` is real depth (640×360, 30fps) but **Oodle/Kraken compressed**;
decoding needs the proprietary Oodle SDK. Documented in `docs/data_acquisition.md`;
unusable until the SDK or an uncompressed re-export is available.

**Net of B:** the one model change trainable without new data (stream reweighting)
does not help the only metric we can measure; the real levers are **data we must
acquire** (in-domain healthy controls, FPara HB-video, AU corpora) and **HB labels**.
The active-learning list tells clinicians which takes to label first.
