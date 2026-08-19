# Baseline-to-HB Strategy (locked 2026-05-26)

Every baseline in our shortlist must produce a prediction that we can compare
against ground-truth HB grades. The strategy is **3 fine-tune + 1 binary**:
three baselines get HB heads trained on our data once HB labels arrive; one
baseline (Shagdar) is evaluated on its native binary task using the released
weights.

## Strategy table

| Baseline | Adapter | Backbone treatment | Output target | Trainable params (head only) |
|---|---|---|---|---|
| **Gaber ASI** (A1) | `GaberASIForHB` | Deterministic, no weights | HB I..VI | **126** |
| **Oo MLP-Mixer** | `OoMixerForHB` | Frozen 768-d encoder (validated) | HB I..VI | **4,614** |
| **Baumann ResNet-18 ×5** (A8) | `BaumannForHB` | **Fine-tune** 5 sub-models on our data | HB I..VI | full backbone (~110M params) |
| **Shagdar GCNMid** | `ShagdarBinaryEval` | Frozen — use released weights as-is | **binary stroke/nonstroke** | **0** (inference only) |

## Per-baseline plan

### Gaber ASI — `GaberASIForHB`

- **What's frozen**: everything. ASI is `(1 - |L - R|) * 100`, no weights.
- **What's learned**: one Linear layer mapping `(n_actions, asi_dim)` → 6 HB
  classes. Default config = `Linear(5*4, 6) = 126 params`.
- **Trainable on**: cached ASI features from
  `python -m src.baselines.gaber_fau.adapt_to_ours --mediapipe-root data/mediapipe_out`.
- **Gating**: HB labels + per-pose segmentation (Task 3).

### Oo MLP-Mixer — `OoMixerForHB`

- **What's frozen**: MLP-Mixer 768-d backbone (validated as FP-relevant +
  identity-stable, see [[project-video-embeddings]]). We use cached embeddings
  from `outputs/embeddings/<take>.npz`.
- **What's learned**: `Linear(768, 6) = 4,614 params`.
- **Optional**: full fine-tune of the Mixer. Not recommended on ~30 patients
  (86M params would overfit hard) but feasible if needed.
- **Trainable on**: cached embeddings (preprocessing already done via
  `scripts/preprocess.py`).
- **Gating**: HB labels only. Per-pose segmentation is optional; per-take
  mean-pool also works at lower fidelity.

### Baumann ResNet-18 ×5 — `BaumannForHB`

- **What's frozen**: nothing — we fine-tune.
- **What's learned**: all 5 sub-models (symmetry, eye, mouth, forehead,
  hb_direct) starting from the released OTH weights. Baumann's own
  `source/hbmedicalprocessing/train.py` provides the pipeline; the
  `AUDIT-FIX` patches we already applied to `detect.py` apply to training too
  (face_alignment API rename, `torch.load(weights_only=False)`,
  pre-cached face-detector weights + new patch to
  `utils/database_utils.py:56`).
- **Output**: HB I..VI direct (3 fusion outputs available; default
  `grade_direct` per thesis-reported F1).
- **Trainable on**: 9 jpgs per patient at Baumann's pose schema (rest, brow,
  smile-closed, smile-open, lip-pucker, eye-easy, eye-forced, nose-wrinkle,
  lip-depress).
- **Gating**: HB labels + per-pose segmentation (Task 3, biggest dependency).

### Shagdar GCNMid — `ShagdarBinaryEval`

- **What's frozen**: everything — we keep the released `.pt` weights verbatim.
- **What's learned**: nothing.
- **Output**: per-graph probability of `stroke` (class 1) ∈ [0, 1].
- **How we compare to HB**: derive a binary collapse of HB labels
  (e.g., `{HB=I} → nonstroke; {HB≥II} → stroke`) and report binary metrics
  (accuracy, F1, AUROC) on that collapsed task.
- **Why not retrain to HB**: head surgery + retraining is a bigger lift than
  just using what was published, and there is no strong prior that the
  binary-stroke backbone features transfer to HB severity. If the binary
  evaluation does well, we will revisit and consider retraining; if it does
  poorly, the architecture transfer was unlikely to help anyway.
- **Trainable on**: not applicable.
- **Gating**: per-frame graph construction pipeline (use
  `src/baselines/shagdar_gnn/graphizer478.py`).

## Which models need retraining? (user's direct question)

| Baseline | Retrain backbone? | Retrain head? | Why |
|---|---|---|---|
| Gaber ASI | No (no weights to train) | **Yes, from scratch** | Head is all there is |
| Oo MLP-Mixer | No (use frozen encoder) | **Yes, from scratch** | Head probes the validated 768-d embedding |
| Baumann ResNet-18 | **Yes, fine-tune** from released weights | (fused into backbone) | OTH n=86 likely overfit; output stays HB I..VI |
| Shagdar GCNMid | **No** | **No** | Use released weights as-is for native binary task |

## Common gating dependencies

1. **HB labels** — none yet. Coming from clinical team.
2. **Per-pose segmentation** — Task 3, pending clinician timestamps.

When HB labels arrive:
- **Oo MLP-Mixer head** trains immediately (embeddings cached; per-take
  mean-pool works without per-pose segmentation).
- **Gaber ASI head** trains after per-pose segmentation (ASI features are
  cached per-frame already).
- **Baumann fine-tune** waits for per-pose segmentation (needs 9 jpgs per
  patient at the right poses).
- **Shagdar binary eval** can run any time once we build per-frame graphs
  from our MediaPipe landmarks (`graphizer478.py`).

Related: [[project-baseline-shortlist]], [[project-hb-head-framework]],
[[project-gaber-baseline]], [[project-oo-baseline-reconstruction]],
[[project-shagdar-baseline]], [[project-baumann-baseline]],
[[project-hb-adapter-status]], [[project-video-embeddings]].
