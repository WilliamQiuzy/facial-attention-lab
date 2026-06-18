# Public Datasets for Facial-Paralysis Training

Survey for Stage-1 pretraining (multi-task latent-severity model, frozen MARLIN +
trainable head). Any label scale is usable: HB / SFGS / eFACE / binary / region
intensity / coarse 3-level. Last updated 2026-06-10.

## Decisions
- **Focus: paralysis-specific datasets only.** Auxiliary (AU / expression) parked
  at the bottom for later — not pure-paralysis.
- **MEEI: dropped.** Not pursuing.
- Volume is the bottleneck: real palsy-labeled data is scarce, so grab every
  accessible paralysis set.

## ⚠️ Manual actions needed (you)
Things I cannot do without a human in the loop:

| Dataset | Why manual | Action |
|---|---|---|
| **YFP** | email request (already applied) | chase Prof. Hsu |
| **AFLFP** | email request to U. Portsmouth authors | send request |
| **UPFP-SG** | email drtzhang@126.com; full release was ~2025-12 | email + confirm full version is live |
| **FPara (NHS)** | NHS ethics + DUA (high friction) | only if we need HB-native video |
| **CCFExp / CFCPalsy** | weights on Google Drive; it's a *synthetic generator*, not a dataset | optional — only if we later do synthetic augmentation |

## ✅ Downloaded & verified (in `data/external/`)
Roboflow key stored in `.env` as `ROBOFLOW_API_KEY` (downloads scripted via `roboflow` SDK; needs `truststore` for the corp-network SSL trust store).

| Dataset | Location | Size | Verified contents | Quality |
|---|---|---|---|---|
| **FNP Detection** | `roboflow_fnp/` (train 371 / valid 104 / test 50) | 19 MB | 525 images, 945 region boxes, 8 classes (eye/mouth × normal/weak/mid/severe), well-balanced | Real frontal palsy faces, ~640px but **low-res / old web images**. Maps directly to eye/mouth region-intensity heads. Best of the open sets. |
| **facial-paralysis** (sumin) | `roboflow_facial_paralysis/` (train 82 / valid 24 / test 12) | 3.8 MB | 118 images, single class `facial-paralysis` (+1 stray `A` mislabel) | Positives only (no normal class), mixed/low quality (hospital snapshots). **Low value** — extra palsy face volume at best. |

**Dropped — palda:** landmark CSV only, **no images** (source photos web-sourced,
not redistributed). Can't train an appearance model on it → deleted from
`data/external/`.

**⚠️ Provenance caveat (FNP + facial-paralysis):** images are web-scraped (stock
photo libraries, journal figures, internet). No documented patient consent. Fine
for research/method dev; **vet with Mayo legal/IRB before any clinical or
published-image use.**

## 🔧 Resolution normalization — BUILT (2026-06-10)
The open image sets are **old, blurry, low-res**; our iPhone/LiveLinkFace captures
are **much sharper** → train/test domain gap. Handled in preprocessing, applied
identically to public frames and iPhone frames on the 224×224 aligned face crop
(the one chokepoint both paths share, in `MarlinVideoEncoder.encode_clip_bgr`).

**Decision: (b) normalize-down by default + (a) SR as an ablation knob.** Goal is
distribution match, not max sharpness; a frozen encoder can't adapt to invented
SR detail.

- **Module:** `src/preprocessing/image_quality.py` — `QualityConfig` /
  `QualityNormalizer` with 3 modes:
  - `"normalize"` (default, **b**): cap effective resolution at `work_size` (default
    112 px) via downscale→upscale, so sharp and blurry crops share one ceiling.
    `augment=True` (train only) jitters cap/blur/JPEG for robustness.
  - `"sr"` (**a**, ablation): super-resolve upward; pluggable model, bicubic fallback.
  - `"off"`: legacy passthrough.
- **Wired into:** `MarlinVideoEncoder.encode_clip_bgr` / `encode_video_path`
  (opt-in `normalizer=`, default None = unchanged) and `action_bundle.process_dataset`
  (default `quality_mode="normalize"`, `--quality-mode` / `--quality-work-size` CLI).
- **Tests:** `tests/test_image_quality.py` — 11 pass (incl. the key check that
  normalization shrinks the sharp-vs-blurry sharpness gap; inference path stays
  deterministic).
- **Open:** tune `work_size` (112 is a starting guess) once we can measure
  embedding-distribution overlap on real iPhone vs public crops; real SR model for
  the (a) ablation not committed yet (bicubic placeholder).

## A. Paralysis — open download (no application needed)

| Dataset | Size | Label | Link |
|---|---|---|---|
| **PalsyNet** ✅ have | 49 videos (27 palsy / 22 healthy), 247 MB | binary | HF `jasir/palsynet-data` |
| **FNP Detection** (Roboflow) | 525 images | region intensity, 8-class (eye/mouth) | https://universe.roboflow.com/austre/fnp-detection-bell-s-palsy |
| **facial-paralysis** (Roboflow) | 118 images | binary | https://universe.roboflow.com/sumin/facial-paralysis |
| ~~palda~~ (dropped — no images) | 203 landmark-only samples | 3-class, 68-pt CSV | https://github.com/nsourlos/palda_azure |
| **CCFExp / CFCPalsy** | synthetic generator (no fixed set) | — (augmentation only) | https://github.com/GaoVix/CFCPalsy |

## B. Paralysis — email/request to apply (open but human in loop)

| Dataset | Size | Label | Link / contact |
|---|---|---|---|
| **YFP** ⏳ applied | 32 videos / 21 patients / ~2,246 frames | region intensity (HB-derived) | https://github.com/AvLab-CV/YouTube-Facial-Palsy-Database |
| **AFLFP** | **5,632 images** (88 subj × 16 expr × 4 states) | 68-pt landmarks | paper https://ieeexplore.ieee.org/document/9831121/ · apply https://researchportal.port.ac.uk/en/publications/aflfp-a-database-with-annotated-facial-landmarks-for-facial-palsy/ |
| **UPFP-SG** | 59 patients (video) | regional severity | https://www.iiplab.net/upfp-sg/ · email drtzhang@126.com (full release ~2025-12, confirm if live) |

## C. Paralysis — high friction (only if HB-native video is needed)

| Dataset | Size | Label | Link |
|---|---|---|---|
| **FPara** | 89 videos | **HB I–VI** | NO public download. From a prior clinical study (patient consent); **contact authors to request**. arXiv 2203.01800 ("Automatic Facial Paralysis Estimation with Facial Action Units", ALGRNet) is the paper that *uses* FPara — NOT a download link. That paper also trains on public AU sets **BP4D + DISFA**. See docs/literature_datasets.md. |

Ruled out: IEEE DataPort "Facial Paralysis" (features only, paywalled), MND-MFHC
(MRI not face), UCSD Facial Nerve DB (IRB-gated).

## Pursuit order
1. **FNP Detection** ✅ have — region-intensity labels; best open set.
2. **AFLFP** — email; largest new palsy image set (5,632 imgs).
3. **UPFP-SG** — email; recent video + regional severity.
4. YFP — already applied; chase.

---

## Parked — auxiliary (not pure paralysis; revisit later)

Facial palsy = reduced/asymmetric AU activation, so AU-intensity data could anchor
the severity axis; expression sets could anchor the healthy end. Deferred for now.

- **AU intensity:** DISFA (27 subj, ~130k frames, AU 0–5), DISFA+ (posed+spont.),
  BP4D-Spontaneous (41 subj, ~150k frames), FEAFA+ (230k frames, continuous AU).
- **Expression / healthy anchor:** MEAD (open), AffectNet (~1M imgs), RAF-DB (29,672 imgs), Aff-Wild2.
- **Skip:** FER2013, AFEW/SFEW, Oulu-CASIA, CASME II, SAMM, VoxCeleb/YTF/CelebV-HQ.
