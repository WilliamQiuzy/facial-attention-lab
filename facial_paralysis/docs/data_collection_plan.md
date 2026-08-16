# Mayo data-collection plan (operationalizing the power analysis)

> **Model-status note (2026-07-27):** This collection plan remains useful, but
> its older model names are historical. Universal Clinical Router v4 and its
> evaluation boundary are defined only in `CURRENT_MODEL.md` and
> `model_registry.json`.

The current development screen is complete; the clinical bottleneck is data. This turns the power analysis
(`scripts/power_analysis.py`) into a concrete, prioritized collection protocol so the
next N months of data collection are spent on exactly what closes the gaps.

## The three asks, in priority order

### 1. In-domain healthy controls (fastest, unblocks the most)
- **What:** ~15–25 healthy volunteers (Mayo staff/volunteers), recorded with the SAME
  FACES protocol (IRB 24-004956), same iPhone/arm's-length setup, same 8 actions.
- **Why:** we currently have ZERO in-domain negatives. Without them the binary detector
  saturates (flags everyone; unverifiable) and we have no symmetry baseline. This is the
  single cheapest unlock — no clinical labeling needed, just recordings.
- **Unlocks:** a trustworthy palsy-vs-healthy detector (the one place MARLIN may transfer),
  a healthy asymmetry baseline to calibrate the scorecard's "normal" range.

### 2. HB labels on existing + new takes (the accuracy unlock)
- **What:** House-Brackmann grade (I–VI) per take by the treating clinician. Target
  **≥40–60 labeled takes** (power analysis: n=14 gives a QWK CI of ±0.24 = useless;
  n≈40–60 gives ±0.10–0.14 = usable).
- **Order:** start with the 14 existing takes, then label as new patients enroll.
  Prioritize the highest model-vs-asymmetry disagreement takes first
  (`outputs/mayo_active_learning.json`: FACES020, MySlate_29, FACES024).
- **How to keep it cheap:** the scorecard (`scripts/mayo_scorecard.py`) already surfaces
  per-region asymmetry + lagophthalmos + synkinesis per patient — hand the clinician the
  card + video and collect a single HB grade (+ optional eFACE if easy). Minutes per take.
- **Unlocks:** the first SUPERVISED, validated severity model; anchors accuracy.

### 3. Recruit toward ~35–50 palsy patients (the transfer unlock)
- **What:** grow the palsy cohort from 13 → ~35–50 with usable video.
- **Why:** power analysis — to establish (or refute) that any model/measure tracks
  clinical severity on Mayo with 80% power you need ~34 patients (if the true effect is
  strong, rho 0.5) up to ~52 (rho 0.4). At n=13 we have 15–37% power — inconclusive by
  construction.
- **Unlocks:** the ability to make ANY validated per-patient claim on the Mayo domain.

## Recording-quality checklist (reduce noise at the source)
Current takes lose signal to avoidable issues (`docs/mayo_data.md`): 2 takes had no audio,
1 corrupt, variable framing. For new recordings enforce:
- full face + neck in frame, eye-level, arm's length, plain background, even frontal light;
- all 8 actions performed and held ~3 s each; verbally confirm each action was done;
- keep audio on (helps automatic per-action segmentation);
- 60 fps (already default) — needed for the EAR closure-dynamics.

## What is READY to consume the data the moment it arrives
- Per-action segmentation + FACES-labeled clips (`mayo_blendshapes/segments.json`).
- Label-free scorecard + phenotype + 60fps EAR/synkinesis measures.
- Historical geometry-only candidate (`autoresearch_fp/deploy_config.json`) and
  the frozen 110D free-recording expert inside Universal Clinical Router v4;
  neither is deployment-authorized.
- AU-dynamics pretraining pipeline (`scripts/au_pretrain.py`) for DISFA/BP4D.
- Evaluation + power tooling to re-check CIs as n grows (`mayo_transfer_robust.py`,
  `power_analysis.py`).

**Net:** every piece of the pipeline is built and waiting; the only missing input is
labeled/in-domain data, and we now know exactly how much of each is needed.
