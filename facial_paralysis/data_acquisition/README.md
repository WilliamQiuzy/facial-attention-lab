# Data-acquisition sweep (a)+(b)+(c) — results & QC (2026-07-06)

Ran the three free/no-application data routes end-to-end, with quality checks. **Honest
headline: (b) is the only genuinely useful new lever, and even it needs curation; (a) is
near-exhausted, (c) is set up but blocked and low-value.** Consistent with the whole
project's finding that more web-modality data won't move the Mayo problem.

## (a) Free-dataset inventory — DONE
See `roboflow_kaggle_inventory.md`. We already hold the main free sets (PalsyNet, FNP,
kaggle-droop). Only genuinely-new free candidate: **`sumin/facial-paralysis`, 118 images,
CC BY 4.0** (Roboflow; download needs an account/API key). Everything else is already-have
or gated (IEEE DataPort password, YFP-remainder email). **Free web pool ≈ exhausted.**

## (b) YouTube collection pipeline — DONE + WORKING (needs curation)
`youtube_collect.py`. Fixed the 2025 YouTube 403 block via `player_client=android` (web
client is SABR-forced; no cookies/PO-token needed). First run: **16 videos → 730 face crops
(224×224)** from real FP content (Bell's-palsy exercises, synkinesis, HB grading, eyelid
weights). Free, and — unlike web stills — VIDEO with real movement.

**QC (important):** the raw crops are **noisy and unlabeled** — a mix of (i) real patients
performing movements [useful], (ii) clinicians/presenters talking [not patients], and (iii)
**cartoon/animated faces** from explainer videos [garbage; Haar detects them]. Before use:
filter presenters/cartoons (e.g., face-quality + a "is-this-an-illustration" check),
keep patient movement segments, and label. It is a *collection tool + raw starting set*, not
a ready dataset. Media gitignored (patient privacy/consent — provenance URLs in `manifest.json`).

## (c) CFCPalsy synthetic generator — SET UP, BLOCKED
See `CFCPALSY_SETUP.md`. Cloned, patched CPU-ready, aux weight + sample images present. Two
external blockers: the >2.4 GB checkpoint stalls on Google Drive's large-file wall (needs
manual browser download), and generation needs a GPU (this box is CPU-only). Turnkey run
documented. **Lowest value** of the three (synthetic web stills: saturated modality +
synthetic-real gap + doesn't transfer to Mayo).

## Bottom line
The free-download routes confirm what the modeling proved: **you can't download your way out
of this problem.** (a) is exhausted, (c) is low-value synthetic. (b) genuinely adds free
VIDEO with dynamics but needs a curation pass. The real levers remain Mayo-generated data
(healthy controls + HB labels) and, for the eye region, the AU corpora (DISFA/BP4D).
