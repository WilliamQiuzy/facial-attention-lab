# (c) CFCPalsy synthetic generator — setup status & turnkey run

**Status: set up as far as this environment allows; blocked on two external walls.**
CFCPalsy is a *diffusion* model (IJCNN 2025) that synthesizes controllable-severity facial-
palsy faces (id image + palsy style → synthetic palsy image). Repo:
github.com/GaoVix/CFCPalsy.

## What is done ✓
- Repo cloned → `data_acquisition/CFCPalsy/` (gitignored: external + large).
- `src/synthesis.py` **patched** from hardcoded `.to('cuda')` → device-agnostic (CPU-ready).
- Aux identity weight `adaface_ir50_casia.ckpt` (196 MB) downloaded ✓.
- Sample id + palsy-style images present (`sample_images/`, 9 imgs) ✓.

## Two blockers (why it can't generate here)
1. **Main checkpoint download stalls.** `CFCPalsy.ckpt` (>2.4 GB) lives in a Google Drive
   folder; `gdown --folder` stalls at 2.4 GB on Google's large-file confirmation/quota (a
   known gdown issue). It needs a **manual browser download** from
   `https://drive.google.com/drive/folders/1yZz42XhsDvYnNYpS8IAru74TniPHjQD0` →
   place `CFCPalsy.ckpt` in `CFCPalsy/pretrained_models/`.
2. **No local GPU.** This box is torch-CPU-only; CPU diffusion is impractically slow.
   Run on the RunPod GPU pod (or any CUDA box).

## Turnkey run (once checkpoint + GPU available)
```bash
cd CFCPalsy
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt        # PL 1.7.1 + hydra; use an isolated venv (do NOT
                                        # install into the main anaconda env — breaks torch 2.2.1)
cd src && bash syn.sh                   # uses sample_images + pretrained_models/CFCPalsy.ckpt
# outputs synthetic palsy faces; scale via synthesis.py --style_images_root <dir>
```

## Honest value note
Per this project's findings, synthetic web-style stills are the **lowest-value** data
lever: they're the same saturated modality that (i) the web model is already at ceiling on
and (ii) does not transfer to the Mayo iPhone domain, plus a synthetic-real gap. Use only as
augmentation/pretraining if at all. The **YouTube video pipeline (b)** and Mayo-generated
data are the better free/near-free levers.
