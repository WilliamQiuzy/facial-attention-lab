# Data acquisition plan — getting more data before HB labels

> **Model-status note (2026-07-27):** Metrics below describe historical
> baselines. Use `CURRENT_MODEL.md` for Universal Clinical Router v4, the sole
> current model;
> no HB, Mayo, outer, or clinical-validation claim is authorized.

_2026-06-17. What more data to collect to improve the model, and exactly how to get
it. Ordered by value-per-effort. Status reflects what is autonomously doable vs
human/gated._

## The honest framing
We have **no HB accuracy** (no HB labels). Historical method-check numbers: palsy detection
**AUC 0.86** (PalsyNet, subject-CV); region severity QWK ~**0.86 mouth / 0.43 eyes**
on public web stills (method checks, not clinical). Run #14 showed the learned `s`
is appearance/domain-driven on Mayo and the **geometric/asymmetry stream carries the
real signal** — so the most valuable new data trains *that* stream and/or supplies
HB labels.

## Application status (sent / pending)
| Dataset | Status | Date | Notes |
|---|---|---|---|
| **AFLFP** | ✉️ **applied** | 2026-06-18 | emailed authors (U. Portsmouth) |
| **UPFP-SG** | ✉️ **applied** | 2026-06-18 | emailed drtzhang@126.com |
| **Toronto NeuroFace** | ✉️ **applied** | 2026-06-18 | REDCap request submitted |
| **FPara** | ✉️ **applied** | 2026-06-18 | emailed authors (89 HB videos) |
| Roboflow stroke-8f6sb/eye-stroke-regression | ✅ downloaded | 2026-06-18 | 708 imgs, graded severity 0.0–9.0 → `data/external/roboflow_stroke_eye` |
| Roboflow anisa/paralysis-face (1012, 3-level) | ⚠️ manual | 2026-06-18 | API export stuck (404); download via universe.roboflow.com/anisa/paralysis-face → "Download Dataset" |
| Roboflow grboguz/face-paralysis-1 (1054) | ⚠️ blocked | 2026-06-18 | no published/generated version → not exportable |
| Kaggle facial-droop | ✅ downloaded | 2026-06-17 | 1024 droop images → `data/external/kaggle_facial_droop` |
| in-domain healthy controls | ⬜ to record (Mayo) | — | highest value |

_When any applied dataset is granted, drop the files in `data/external/` and tell me — I'll wire it into the pipeline._

## 1. In-domain healthy controls  ⬅ highest value-per-effort
Record healthy volunteers (Mayo staff/volunteers) with the **same iPhone /
LiveLinkFace protocol and the same 7-8 action order**. Fixes the #1 gap (Run #4: out-of-domain
controls are unusable; we have zero in-domain negatives → P(palsy) saturates).
Unlocks a real palsy-vs-healthy metric on the Mayo domain and a healthy baseline for
the asymmetry score. **Human action; cheap.**

## 2. FPara (NHS) — the only public HB + video set
- Paper: **https://arxiv.org/abs/2203.01800** ("Towards Automatic Facial Palsy
  Grading", 89 videos, **HB I–VI**).
- Access: **NHS ethics approval + Data Use Agreement / contact the authors** — not a
  direct download (high friction). **Worth starting now in parallel:** it would let
  us pretrain the real HB head before Mayo labels arrive.

## 3. AU-intensity corpora — train the temporal stream on real dynamics (no palsy label)
Palsy = reduced/asymmetric AU activation, so AU-intensity data pretrains the
trainable geometric/temporal encoder on real movement + anchors the healthy end.
Adapter is built: `src/datasets/au_intensity_adapter.py` (AU→blendshape mapping →
the model's (T,F) feature seq, with L/R asymmetry for free). **Application-gated:**
- **DISFA** (27 subj, AU 0–5/frame): http://mohammadmahoor.com/disfa/ — EULA.
- **DISFA+** (posed+spontaneous): same group — EULA.
- **BP4D-Spontaneous** (41 subj): http://www.cs.binghamton.edu/~lijun/Research/3DFE/3DFE_Analysis.html — request form.
- **FEAFA+** (230k frames, continuous AU): request from authors.
Once on disk, fill `load_disfa`/`load_bp4d` and pretrain the temporal encoder.

## 4. Other gated palsy sets (data/public_datasets.md)
- **AFLFP** — 5,632 images, 88 subj × 16 expr × 4 states (semi-dynamic, large) —
  email U. Portsmouth authors.
- **UPFP-SG** — 59 patients video, regional severity — email drtzhang@126.com.
- **YFP remainder** — we have 16 image subjects of 32; getting the rest helps.

## 5. Mayo depth data — BLOCKED on Oodle  (B-4 finding)
Every take's `depth_data.bin` is real depth (640×360, 30fps) and would give a novel,
appearance-invariant **3D L/R asymmetry** signal — but it is **Oodle/Kraken
compressed** (`depth_metadata.mhaical: "Compression":"Oodle"`). Decoding needs the
proprietary **Oodle SDK** (Epic/RAD), not available here. Paths: obtain the Oodle SDK,
re-export depth uncompressed from the capture app, or use Apple's decompressor. Until
then depth is unusable. (8 video-less takes are depth-only → currently unusable.)

## 6. Synthetic augmentation (last resort)
CCFExp / CFCPalsy generator (https://github.com/GaoVix/CFCPalsy) — controllable-
severity synthetic palsy video to pretrain the temporal stream. Synthetic-real gap;
use as augmentation only.

---
**Autonomous-doable now:** #3 adapter (built), #5 documented blocker. **Needs humans:**
#1 (record controls), #2/#4 (apply), #5 (Oodle SDK). The model improvement that does
NOT need new data is the **stream reweighting (v3, Run #15)**, motivated by Run #14.
