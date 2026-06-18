# Facial-Paralysis Severity Model — Design

_Last updated: 2026-06-10 (v2 — MARLIN video encoder + MediaPipe dual-stream
front-end). Supersedes the single-task HB head in
`src/models/baselines_hb/HB_ADAPTERS.md` and the rationale block in
`src/models/hb_head.py`._

This document is the source of truth for **what the model is and is not
responsible for**. Read it before changing anything under `src/models/`.

---

## 1. Problem statement

Predict a clinical facial-paralysis severity grade from a recording of a
patient performing facial expressions. The **primary target is the
House-Brackmann (HB) scale, grades I–VI** (ordinal). Inputs may be a **single
image** or a **video**; video is strongly preferred because facial paralysis is
fundamentally a *dynamic* deficit (synkinesis, delayed/asymmetric movement
onset, incomplete eye closure during a blink) that a resting photo can hide.

Two hard constraints shape the whole design:

1. **Tiny in-domain cohort.** Mayo has ~14–21 patients and HB labels are not in
   yet. Training a deep encoder from scratch is impossible. Transfer learning is
   not optional — it is the foundation (see §6).
2. **Heterogeneous labels across the data we *do* have.** The public datasets we
   can use are each labeled on a *different* scale. We refuse to waste them, and
   we refuse to fabricate labels by force-mapping one scale onto another (§4).

---

## 2. The preprocessing / model boundary (READ THIS FIRST)

**Decision (2026-06-10, locked):** all temporal segmentation and long-video
handling lives in **preprocessing, never in the model.**

The capture app guides the patient with audio prompts to perform the fixed HB
action sequence (rest → brow raise → light eye closure → forced eye closure →
smile, plus any extra poses). Because the protocol is known and prompted, we
segment the raw recording into **one short clip per action (~3 s each)** as a
preprocessing step — *before* anything reaches the model. A 5-minute recording
becomes 6–7 short, named, per-action clips on disk.

Consequences:

- The model **always** receives a clean, fixed set of per-action clips. It never
  sees a 5-minute stream, never does event detection, never does temporal
  localization. Those are data-pipeline concerns.
- This holds at **training time too**: any long training video is segmented by
  the same preprocessing path. The model's input contract is identical for
  train and inference.
- An **image** is the degenerate case: a clip with no motion. The appearance
  encoder (MARLIN) needs a fixed 16-frame clip, so a still image is tiled to 16
  identical frames; the temporal streams then simply carry zero motion. Images
  and videos share one code path (§5.1).

```
┌──────────────────────────── PREPROCESSING (not the model) ─────────────────────────────┐
 raw .mov (≤5 min)
   → audio-prompted capture → segment by action → per-action clips (~3 s)
   For EACH per-action clip:
     ├─ face-align + sample 16 frames → MARLIN (frozen) → clip embedding (768-d)   [cached]
     └─ MediaPipe per frame → 52 blendshapes + L/R landmark-asymmetry feats
                            → per-frame feature SEQUENCE (T × F)                    [cached]
└────────────────────────────────────────────────┬───────────────────────────────────────┘
                                                  ▼
   per-action bundle:  MARLIN clip vec (768)  +  MediaPipe feature sequence (T × F)
┌──────────────────────────────────── MODEL (this doc) ──────────────────────────────────┐
   MediaPipe sequence → trainable temporal encoder (GRU) → dynamics vec
   concat[ MARLIN vec , dynamics vec ] → per-action embedding (D)
   → shared per-action MLP → action pooling → SEVERITY trunk (s) → multi-task heads
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

**Why the split.** MARLIN is *frozen*, so its clip embedding is computed once and
**cached** in preprocessing. The MediaPipe streams feed a *trainable* temporal
encoder, so their raw per-frame **sequences** are cached (small) and consumed by
the model at train time — the model learns the asymmetry dynamics. Temporal
modeling therefore lives in two frozen/trained places, never in segmentation:
appearance dynamics inside frozen MARLIN, geometric dynamics inside the trainable
GRU.

The on-disk contract: `PatientVideoDataset` consumes
`<data_root>/<patient_id>/<action>.mov` and per-action caches at
`<cache>/<patient_id>/<action>.npz`. The `.npz` schema is extended to hold both
the MARLIN clip embedding and the MediaPipe feature sequence (see §8).

---

## 3. Input representation (two streams, concatenated per action)

**Decision (2026-06-10):** the per-action embedding is the **concatenation** of an
appearance-temporal stream (MARLIN) and a geometric-temporal stream (MediaPipe).
Both carry time; they are complementary, not redundant.

| Stream | Source | Temporal? | Why it matters for FP |
|---|---|---|---|
| **Appearance + motion** | **MARLIN** (frozen, video-native MAE), 16-frame clip → 768-d | ✓ inside MARLIN (tubelet attention) | Holistic facial motion, texture, coarse asymmetry — learned from 700k face videos |
| **Geometry + dynamics** | **MediaPipe**: 52 blendshapes + left/right landmark-asymmetry features, per-frame sequence → trainable temporal encoder | ✓ inside the GRU | Explicit, interpretable left-vs-right asymmetry trajectory: synkinesis, asymmetric onset/peak, incomplete eye closure. Cheap and highly FP-specific |

Per action: `embed = concat(MARLIN_vec[768], dynamics_vec[H_t])`. The
`SeverityTrunk` (§5) is indifferent to the provenance; only `embed_dim` changes.

### 3.1 MARLIN (the appearance-temporal encoder) — verified spec

On disk at `data/external/marlin_vit_base_ytf/`. ViT-Base masked autoencoder for
facial **video**.

- **Architecture:** encoder = 12 transformer layers, 12 heads, **embed_dim 768**.
  Video is patchified into spatiotemporal **tubelets** (`tubelet_size=2` frames ×
  16×16 px). A 16-frame 224² clip → 14×14 spatial × 8 temporal = **1568 tokens**.
  Motion is captured because attention runs over space *and* time.
- **Input:** `(B, 3, T=16, 224, 224)` — 16 RGB frames of an **aligned face crop**.
- **Output:** `keep_seq=True` → `(B, 1568, 768)` token sequence; `keep_seq=False`
  → `(B, 768)` mean-pooled clip embedding (what we use by default).
  *(Note: the HF README example prints `...384`; that is the **decoder** dim. The
  encoder/feature dim is 768 — verified against `config.json` and `encoder.py`.)*
- **Pretraining task:** self-supervised masked reconstruction (facial-region
  tube masking) on ~700k YouTube-Faces clips. No labels. Transfers to facial
  attribute/expression recognition, deepfake detection, lip-sync.
- **License:** **CC BY-NC 4.0 — non-commercial.** Fine for research/publication;
  a commercial Mayo product would need re-training or a separate license. Flagged
  here so it is not a surprise downstream.

We **freeze** MARLIN and drop its decoder; only the encoder runs, as a feature
extractor. Alternatives kept for ablation: Oo MLP-Mixer (per-frame, no time),
FaRL (per-frame). V-JEPA 2 / VideoMAE v2 are larger and *generic* (not
face-specific) — optional comparison only.

### 3.2 MediaPipe streams (the geometric-temporal encoder)

Per frame, MediaPipe gives 478 landmarks + 52 blendshape coefficients. We derive
a compact **per-frame feature vector** and keep the **sequence** over the clip:

- 52 blendshape coefficients (raw activation of each facial action).
- Left/right **asymmetry features**: signed differences of mirror-paired
  blendshapes and of key landmark distances (eye-closure gap, mouth-corner pull,
  brow height) between the two hemifaces.

This per-frame feature sequence `(T × F)` is **not** pre-encoded; it is cached raw
and fed to a small **trainable temporal encoder** (1-layer GRU or temporal
transformer) inside the model (§5.0), which emits one `dynamics_vec` per action.
Putting this stream in a trainable module (not frozen MARLIN) is deliberate: the
clinically diagnostic signal is the *asymmetry trajectory*, and we want the model
to learn it directly from our data.

---

## 4. The heterogeneous-label solution (the core idea)

We have labels on incompatible scales:

| Dataset | Label form | Native granularity |
|---|---|---|
| Mayo / MEEI | HB I–VI | ordinal, 6 levels |
| MEEI extra | SFGS / eFACE | continuous composite |
| PalsyNet | palsy / normal | binary |
| YFP | region intensity 0.5 / 1.0 (eyes, mouth) | per-region, 2 levels |
| Parra-Dominguez | healthy / slight / strong | ordinal, 3 levels |
| CK+ | emotion (used as healthy class) | n/a |

**Do not unify the labels. Unify the latent variable.** All of these are
different-resolution *views* of one underlying quantity: facial-nerve
dysfunction severity. So the model learns a single **latent severity** and a
**per-dataset link function (ordered cut-points)** that maps that severity onto
each dataset's label space.

This is the classic ordinal latent-variable / cumulative-threshold model, and it
is exactly what lets binary data calibrate the low end of the severity axis while
HB data calibrates the fine-grained middle — **no label is destroyed, none is
fabricated.**

### 4.1 Coupled multi-task structure

- A shared trunk maps a patient's per-action embeddings to a scalar **global
  severity** `s` (and a representation vector `h`).
- Each *global* task is a set of **monotonically ordered thresholds** on the
  same `s`:
  - HB → 5 thresholds (6 classes)
  - binary palsy → 1 threshold
  - 3-class coarse → 2 thresholds
- **Region-specific** tasks (YFP eyes / mouth intensity) measure *local*
  severity, not the global face grade, so each gets its **own** severity
  projection `s_region = w_region · h` with its own thresholds. They still share
  the trunk `h`, so they shape the representation without contaminating `s`.

Because every global head reads the *same* `s` through ordered thresholds,
predictions are **rank-consistent by construction** and the scales are forced
into a common ordering — a binary "palsy" example pushes `s` above the
palsy threshold, which is *below* the HB III/IV thresholds, so it cannot
silently corrupt the fine HB calibration.

### 4.2 Why not the simpler alternatives

- **Force-map everything to HB.** Injects label noise (a binary "palsy" could be
  HB II or VI). Rejected.
- **Independent heads, no shared severity (plain multi-task).** Works, wastes the
  cross-scale ordering information. We keep it available as
  `coupled=False` per task for ablation, but the default is coupled.

---

## 5. Architecture

```
PER ACTION (built in §5.0):
   MARLIN clip vec (B, n_actions, 768)          MediaPipe seq (B, n_actions, T, F)
                    │                                        │ trainable temporal
                    │                                        ▼ encoder (GRU)
                    │                            dynamics vec (B, n_actions, H_t)
                    └───────────────── concat ───────────────┘
                                       ▼
per-action embeddings        (B, n_actions, D)             [+ action_mask]
        │  shared per-action MLP  (D → H)
        ▼
per-action reps              (B, n_actions, H)
        │  action pooling  {mean | max | attention}
        ▼
patient representation  h    (B, H)
        ├──────────────► global severity  s = w·h  (B,)
        │                       │
        │         ┌─────────────┼───────────────────────────┐
        │         ▼             ▼                            ▼
        │   HB thresholds  binary threshold          3-class thresholds   (all read s)
        │   (B,5) cum-logits   (B,1)                      (B,2)
        │
        └──► region projections  s_eyes, s_mouth = w_r·h
                    ▼
              region thresholds (own severity)            (B, K_region-1)
```

### 5.0 Per-action feature assembly (the new front-end)

For each action slot the model builds one `embed` of width `D`:

1. **Appearance-temporal:** the cached MARLIN clip vector (768). Frozen; nothing
   to train. (If a clip yields several 16-frame windows, their MARLIN vectors are
   mean-pooled — this is the only place a "frame/window" axis survives, and it is
   pooled, not modeled, since MARLIN already handled intra-window time.)
2. **Geometric-temporal:** the MediaPipe per-frame feature sequence `(T × F)` →
   `TemporalLandmarkEncoder` (1-layer GRU / temporal transformer, **trainable**) →
   `dynamics_vec (H_t)`. A padding mask handles variable `T`.
3. `embed = concat(marlin_vec, dynamics_vec)`, width `D = 768 + H_t`.

This is the *only* structural addition versus the already-built model: a small
trainable temporal encoder in front of the existing `SeverityTrunk`. Everything
from `per-action embeddings` downward (trunk, severity, ordinal heads, routing,
loss) is unchanged and already tested.

### 5.1 Image vs. video

Both share one path. A **video** clip gives MARLIN real motion and the GRU a real
asymmetry trajectory. An **image** is tiled to 16 identical frames for MARLIN
(static appearance, valid embedding) and gives the GRU a length-1 sequence (zero
dynamics). No separate image branch, no flag — the difference is just how much
motion the two temporal encoders see.

### 5.2 Ordinal head (the cut-point model)

For `K` classes we keep `K−1` thresholds `θ₁ ≤ θ₂ ≤ … ≤ θ_{K−1}`, made monotone
**by construction**: `θ = θ_first + cumsum(softplus(gaps))`, so no ordering
constraint can be violated during training. Given severity `s`:

```
cumulative logit_k = s − θ_k                         k = 0 … K−2
P(y > k)           = sigmoid(s − θ_k)                (non-increasing in k)
P(y = 0)           = 1 − P(y>0)
P(y = j)           = P(y>j−1) − P(y>j)               0 < j < K−1
P(y = K−1)         = P(y>K−2)
predicted grade    = Σ_k 1[P(y>k) > 0.5]             (rank-consistent)
expected grade     = Σ_j j · P(y=j)                  (continuous readout)
```

**Loss:** extended binary cross-entropy (CORAL-style). For label `y`, the
`K−1` binary targets are `t_k = 1[y > k]`; loss is `BCEWithLogits` over the
cumulative logits, averaged over thresholds and batch. This is the standard
consistent-rank ordinal loss and — per our literature survey — **no published HB
model uses ordinal regression, so this is also a novelty axis** for the paper.

---

## 6. Training stages (transfer learning is the spine)

1. **Stage 0 — pretrained frozen encoder.** No Mayo data needed. Appearance
   encoder is **MARLIN** (self-supervised on 700k face videos), frozen; MediaPipe
   landmark/blendshape extractor is also pretrained/frozen. The only trainable
   parts anywhere are the small `TemporalLandmarkEncoder`, the `SeverityTrunk`,
   and the ordinal cut-point heads (a few hundred K params total) — which is what
   makes training feasible at N≈20–60.
2. **Stage 1 — multi-task supervised fine-tune on all public data.** PalsyNet +
   YFP + MEEI + Parra co-train the trunk and the per-task cut-points. This is
   where the heterogeneous data does the heavy lifting of shaping `s`.
3. **Stage 2 — HB calibration.** Train/calibrate the HB thresholds on MEEI's
   real HB labels, then on Mayo HB labels when they arrive.
4. **Stage 3 — domain adaptation (optional).** Adapt to the Mayo iPhone
   distribution (the few HB labels mostly *calibrate* the head rather than train
   the encoder).

**Fallback label scheme:** if 6-class HB is too noisy at small N, collapse to the
Parra 3-class scheme (healthy / slight / strong). This is a *config change*
(swap the HB task's `n_classes` and thresholds), not a code change, because every
task is just a cut-point set on the shared `s`.

---

## 7. Multi-task batch routing & loss

Each training sample carries `(embedding, task_name, label)`. A batch may mix
datasets. The model computes every head for every sample (cheap), but the loss
**supervises only the head matching each sample's `task_name`**. Gradients from a
PalsyNet sample therefore flow through `s` via the binary head; gradients from a
MEEI sample flow through `s` via the HB head. Per-task loss weights balance
dataset sizes. See `multitask_loss` in `src/models/multitask.py`.

---

## 8. File map

| File | Role | Status |
|---|---|---|
| `src/models/ordinal.py` | `OrderedThresholds`, `OrdinalThresholdHead`, `ordinal_loss`, decoders. Self-contained. | ✓ built + tested |
| `src/models/multitask.py` | `SeverityTrunk`, `TaskSpec`, `MultiTaskSeverityModel`, `multitask_loss`. §4–5. | ✓ built + tested |
| `src/models/backbones/marlin_video.py` | `MarlinVideoEncoder`: frozen MARLIN from `data/external/`, `encode_clip_bgr`/`encode_video_path`→768. Bypasses HF dynamic loader. | ✓ built + probed |
| `src/models/temporal.py` | `TemporalLandmarkEncoder` (packed bidirectional GRU over MediaPipe `T×F` → `dynamics_vec`). §5.0. | ✓ built + tested |
| `src/models/facial_palsy_model.py` | `FacialPalsyModel`: assembles MARLIN-window-pool ⊕ temporal → `SeverityTrunk` → multi-task heads. §5. | ✓ built + tested |
| `src/datasets/patient_multistream.py` | `MultiStreamPatientDataset` (+ `from_disk`, `collate_multistream`) over the bundle `.npz`. | ✓ built + tested |
| `src/preprocessing/action_bundle.py` | Per-action clip → MARLIN windows + MediaPipe feature sequence → bundle `.npz`. `MediaPipeFeatureExtractor` (blendshapes + L/R asymmetry). | ✓ built + real-data verified |
| `src/training/train_multitask.py` | Multi-task trainer for `FacialPalsyModel` (per-sample task routing, HB-kappa monitor, early stop). §6–7. | ✓ built + e2e verified |
| `src/models/backbones/oo_mlp_mixer.py` | Frozen Oo MLP-Mixer (per-frame). Appearance-only ablation. | ✓ exists |
| `src/training/train_hb.py` | Legacy single-task trainer + k-fold + metrics (works with `HBHead`). | ✓ legacy |
| `src/models/hb_head.py` | Legacy single-task HB head (6-way CE). Kept for the existing smoke test. | ✓ legacy |
| `src/evaluation/hb_metrics.py` | Quadratic-weighted kappa, MAE-in-grades, confusion. Primary metric = kappa. | ✓ exists |

**`.npz` schema.** Per `<cache>/<patient_id>/<action>.npz`:
`marlin` `(W, 768)` float32 · `mp_seq` `(T, F)` float32 (52 blendshapes + 20 L/R
asymmetry deltas → **F = 72** for the MediaPipe FaceLandmarker) · `mp_mask` `(T,)`
bool · `mp_feat_dim` scalar. `from_disk` falls back to a legacy `embeddings` key
(appearance-only) when the new keys are absent. Set `FacialPalsyConfig.mp_feat_dim`
to the stored `mp_feat_dim`.

**Status: the full pipeline is implemented and tested end-to-end** (47 assert-based
tests across `tests/test_{ordinal,multitask,pipeline,pipeline_e2e}.py`, all green;
real-data extract→dataset→model round-trip verified). Remaining before real
training: clinical HB labels + per-action segmentation (external dependency).

## 9. Validation strategy

Model code ships with runnable, assert-based tests (no pytest dependency) under
`tests/`, run with the `dev` conda env. They cover, at minimum: tensor-shape
contracts, mask handling (missing actions, single-frame images), **threshold
monotonicity**, **rank-consistency of predictions**, **class-probability
simplex** (non-negative, sums to 1), gradient flow into the shared severity from
every head, multi-task routing (a task's loss only touches its own head +
the shared trunk), determinism under fixed seed, and a planted-signal training
loop that must drive loss down and correlate `s` with the true grade.

## 10. Decided since v1 / still open

**Decided:**
- Appearance encoder = **MARLIN** (video-native, face-specific, on disk). Oo
  Mixer / FaRL demoted to ablation baselines.
- Streams are **concatenated** per action (MARLIN ⊕ GRU dynamics), single
  `embed_dim`. (Was an open question in v1.)
- Geometric dynamics live in a **trainable** temporal encoder; MARLIN stays
  frozen and cached.

**Explored:**
- **Temporal pooling of the GRU outputs is a tuned knob** (`FacialPalsyConfig.
  temporal_pool` ∈ `mean|max|attention`, + `marlin_window_pool` ∈ `mean|max`),
  not a hard-coded mean. Run #9 (training_runs.md) found the optimum is
  **region-dependent**: attention for the dynamic eye head, max for the sustained
  mouth head. Mean-pooling diluted the eye signal (the Run #7 artifact).

**Still open:**
- `H_t` and the temporal-encoder type (GRU vs small temporal transformer);
  exact MediaPipe asymmetry-feature set `F`. Region-aware (per-task) pooling is a
  natural follow-up now that pooling is a knob.
- Single 16-frame window per 3 s clip vs several overlapping windows (mean-pooled).
- SFGS/eFACE as auxiliary *regression* heads on `s` — deferred until MEEI is in.
- MARLIN's **non-commercial license** for any future product deployment (§3.1).
