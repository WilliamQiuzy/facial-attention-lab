# What datasets do other facial-palsy ML papers train on? (2026 literature scan)

_2026-06-17. Surveyed the FP / facial-paralysis ML literature (incl. Nature-family
papers) to answer: what data do others train on, and can we get it? PDFs downloaded
to `papers/literature_2026/`. Sources at bottom._

## The blunt finding
**There is NO large public facial-palsy dataset.** The field is data-starved, and it
splits cleanly:
- **Clinical papers (incl. all the Nature-family ones) train on PRIVATE institutional
  cohorts** of ~100–400 patients that are NOT shareable (IRB/consent bound). Examples:
  npj Digital Medicine 2025 (componentized FP, neurosurgeon consensus grades); Sci
  Rep 2025 (facial symmetry scoring, **405 datasets / 198 patients**, private); Sci
  Rep 2025 (synkinesis diagnosis); Sci Rep 2024 (central FP in ER). None release data.
- **Method/CV papers reuse a SMALL set of public benchmarks** — the same handful we
  already know, all tiny, most gated by email/EULA.

So we are **not missing some big public dataset** that others have. Everyone publishing
clinical results uses private data — which is exactly why **Mayo's cohort + HB labels
are the asset**, and why the scalable *public* lever is **AU-dynamics data**, not FP data.

## Datasets actually used, with obtainability

| Dataset | Content | Label | Obtainable? | We have? |
|---|---|---|---|---|
| **PalsyNet** | 49 YouTube videos (27 palsy/22 healthy) | binary | ✅ HF download | ✅ |
| **YFP** (YouTube Facial Palsy, Hsu 2018) | ~32 videos / 21 subj | region intensity | ⏳ email AvLab-CV (applied) | partial (16 subj) |
| **MEEI standard set** | photo+video standard set | HB + eFACE | ⚠️ contact MEEI; "standard set" is a published reference set, not an open download | ✗ (dropped) |
| **AFLFP** | 5,632 images, 88 subj × 16 expr × 4 states | 68 landmarks | ⏳ email U. Portsmouth | ✗ |
| **UPFP-SG** | 59 patients, video | regional severity | ⏳ email drtzhang@126.com | ✗ |
| **FPID** (Facial Paralysis Image DB) | 480 images / 60 subj (10 HC, 50 patients) | 3-level (light/mid/severe) | ⚠️ "public" but via corresponding author | ✗ |
| **FNP Detection** (Roboflow) | 525 images | eye/mouth 8-class | ✅ Roboflow | ✅ |
| **Toronto NeuroFace** ⬅ NEW | video, 11 HC + 11 ALS + 14 stroke; >3300 landmark-annotated frames + clinical scores | clinical perceptual scores | ✅ **request form** (REDCap, 3–5 days) | ✗ — worth getting |
| **CK+** (Cohn-Kanade) | expression video | emotion (healthy anchor) | ✅ request | optional |
| **FPara** | **89 videos, HB I–VI**, patients doing the SAME exercises as Mayo (brow raise / gentle+tight eye closure / smile…) | **HB 1–6** | ⚠️ from a prior clinical study; **contact authors** (used by ALGRNet, arXiv 2203.01800 — that arXiv is the *using* paper, NOT a download) | ✗ |
| **BP4D-Spontaneous** ⬅ AU | 41 subj, ~150k frames | per-frame AU intensity | ✅ request form (Binghamton) | ✗ — get it |
| **DISFA / DISFA+** ⬅ AU | 27 subj, ~130k frames | per-frame AU 0–5 (incl L/R) | ✅ EULA (mohammadmahoor.com) | ✗ — get it |
| **CFCPalsy / CCFExp** | synthetic FP image generator | — | ✅ GitHub weights | optional (augmentation) |

## The most important takeaway for us
**A published FP paper (ALGRNet, arXiv 2203.01800) trains on BP4D + DISFA (the AU
datasets) plus a private FP set** — i.e. the AU-pretraining route (our B-2) is a
validated, peer-reviewed approach, and BP4D/DISFA are the **largest accessible facial-
dynamics data in the field**. Combined with the recurring use of MediaPipe/AU and
landmark-heatmap methods (QAFE-Net, ViTAU, FaraPy, the npj componentized model), the
literature points the same way our Run #14 did: **the movement / AU / asymmetry signal
is the substance; appearance alone is not.**

## Concrete acquisition shortlist (ranked)
1. **DISFA + BP4D** (AU dynamics) — largest accessible, validated for FP, trains our
   temporal stream. EULA/request — start now. Adapter ready (`au_intensity_adapter.py`).
2. **Toronto NeuroFace** — public video (stroke=central FP + healthy controls + clinical
   scores). REDCap request, ~days. Gives in-domain-ish video + healthy controls.
3. **AFLFP / UPFP-SG / YFP-remainder** — email the authors (already drafted in
   `data_acquisition.md`).
4. **FPara** — contact the ALGRNet / original-study authors for the 89 HB videos.
5. **Mayo** — in-domain healthy controls + HB labels remain the decisive data.

## Sources
- Review: BMC BioMed Eng OnLine 2022 (s12938-022-01036-0); MDPI Axioms 2023 (2075-1680/12/12/1091).
- Nature-family: npj Digit. Med. 2025 (s41746-025-02063-6); Sci Rep 2025 symmetry (s41598-025-17172-1); Sci Rep 2025 synkinesis (s41598-025-08548-4); Sci Rep 2024 central FP (s41598-024-53815-5); Sci Rep 2025 Bell's recovery (s41598-025-34934-z).
- Methods/datasets: ALGRNet (arXiv 2203.01800, uses BP4D/DISFA/FPara); Hsu 2018 hierarchical/YFP (CVPR-W); QAFE-Net (arXiv 2312.00856); CFCPalsy (arXiv 2409.07271); Toronto NeuroFace (Bandini 2021, slp.utoronto.ca); UPFP-SG benchmark.
