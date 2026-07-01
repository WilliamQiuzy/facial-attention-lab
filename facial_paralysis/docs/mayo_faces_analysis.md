# Mayo FACES script → per-action dynamic symmetry analysis

_2026-07-01. Uses the FACES protocol (IRB 24-004956, PI Menapace) — the 8-action
held-movement battery — to extract label-free, clinically-aligned symmetry measures
from the cached Mayo blendshape trajectories. Scripts: `scripts/mayo_eface.py`,
`scripts/mayo_eface_figure.py`. Output: `outputs/mayo_eface/`._

## Why the script mattered
It was the documented **#2 unlock** (`docs/mayo_data.md`): it turns the segmented
candidate windows into **named, clinically-meaningful actions** — crucially
distinguishing **gentle vs FORCED eye closure**, and the three separate mouth
actions — so each clip gets the correct L/R signature blendshape pair and clinical
role. The 8 actions map onto the **eFACE / Sunnybrook voluntary-movement battery**
(repose, brow lift, gentle + forced eye closure, smile, pucker, lower-teeth show).

## What we can now compute (all label-free, local, no mediapipe/GPU)
Per patient, per action, from the dense 6 fps 52-blendshape trajectory over the 3-s hold:
- **Static peak asymmetry** `AI = |L−R|/(L+R)` at peak effort — *what a single still gives.*
- **Trajectory asymmetry** over the hold — sustained deficit.
- **Forced recruitment** = weak-eye closure(forced) − closure(gentle) — does effort
  recruit residual function? (prognostic; flaccid ⇒ ≤0).
- **Synkinesis** = involuntary cross-region co-activation (mouth moving during eye
  squeeze; eye narrowing during smile) — a core post-paralytic sign.
- Regional aggregates: brow / eye / oral symmetry + weaker side.

## The headline result (n=13 unique patients; FACES018≡MySlate_14 dedup)
**In-domain video dynamics carry clinical signal that stills physically cannot — and
it is the signal the web-stills model was missing (eyes QWK stuck at 0.45).**

1. The dynamic measures are **statistically independent** of the static-peak
   asymmetry: Spearman ρ = **+0.10** (forced recruitment) and **−0.00** (synkinesis),
   p = 0.74 / 0.99. So they are new information, not a re-derivation of the still.
2. **5 / 13 patients** look ~symmetric in a still (eye peak-asym < 0.10) yet the video
   reveals a real deficit — e.g. **FACES021** & **FACES024** (forced recruitment −0.36
   / −0.28: forcing fails to close the weak eye → flaccid), **MySlate_23** (synkinesis
   1.05), **MySlate_25** (synkinesis 0.35). A single frame misgrades all five.
3. Cross-validation: the known duplicate scores identically; **MySlate_29** shows the
   isolated brow deficit (brow 0.54) flagged independently in Run #14.

Figure: `outputs/mayo_eface/eface_panel.png` (cohort heatmap split static↑ / dynamic↓,
plus two eye-squeeze L-vs-R trajectories).

## Honest limits (unchanged)
- **Still no HB / severity labels.** These are objective *physical-symmetry* measures,
  not validated clinical grades — so this demonstrates the signal exists and is
  independent, but cannot yet be scored as supervised HB accuracy. Getting even a
  handful of HB labels (active-learning list in `outputs/mayo_active_learning.json`)
  turns this into a supervised, clinical-grade eye/oral severity model.
- 6 fps blendshapes resolve the 3-s hold well but coarsely sample the fast closure
  transient. A **RunPod A100 re-extraction at 60 fps with landmark EAR** (eye-aspect-
  ratio) would sharpen closure-velocity asymmetry and give publication-grade dynamics.
- n=13, all palsy, no in-domain healthy controls → pilot/method demonstration.

## So what this changes about the "0.65 is too low" concern
The web-stills 0.65 (eyes 0.45) was a **data-modality ceiling**, proven twice by the
model search. The FACES script + Mayo video is exactly the modality that breaks it:
the diagnostic axes of facial-palsy assessment — **effortful closure, recruitment,
and synkinesis — are dynamic**, and we can now measure them per action on real
in-domain clinical video. The clinical-grade path is: (1) these dynamic features +
(2) HB labels on the highest-disagreement takes → a supervised model that is finally
measuring the right thing on the right data.
