# Data-acquisition results (a)+(b)+(c) — 2026-07-07

Goal: get more data WITHOUT applications. Honest headline: the free routes work mechanically
but **none of them supplies the data that actually helps** (in-domain Mayo video + labels).
Detail per track below.

## (a) Free-dataset inventory — see roboflow_kaggle_inventory.md
The free web pool is exhausted: we already hold PalsyNet / FNP / kaggle-droop / anisa. The
only genuinely-new free candidate is `sumin/facial-paralysis` (118 imgs, Roboflow, CC BY 4.0);
everything else is gated (IEEE DataPort pw, YFP-rest email) or non-image. And more web stills
won't move the model (at ceiling; don't transfer). NET: ~118 new images, low value.

## (b) YouTube collection pipeline — WORKS (youtube_collect.py)
Fixed YouTube's 403/SABR block via the android player_client. Collected **16 videos → 730
face crops** across 6 palsy queries; QC: **99% of crops re-detect a face, 0 blank**, titles all
on-topic (Bell's palsy exercises, synkinesis management, patient videos). Scales by raising
N_PER_QUERY/SECONDS. Media gitignored (privacy/consent — public patient/education videos;
review consent + IRB scope before any use beyond method dev; manifest keeps source URLs).
VALUE: unlike web stills this is VIDEO with real movement dynamics — the one free source that
has the modality the eye problem needs — but it has NO labels and NO standardized protocol.

## (c) CFCPalsy synthetic generator — BLOCKED (needs GPU; weights stalled)
Cloned + aux weight (adaface, 196MB) downloaded, but: (1) the 2.4GB main checkpoint download
from Google Drive stalled (large-file quota) — needs manual download or gdown+cookies; (2)
`src/synthesis.py` hardcodes `.to('cuda')` — no local GPU; (3) deps are old (pytorch-lightning
1.7.1 vs our torch 2.2.1). To run: on the RunPod GPU pod, `pip install -r requirements.txt`,
manually place `CCFExp.ckpt`, patch cuda, run `src/syn.sh`. LOWEST value: synthetic web-modality
faces won't transfer to Mayo (synthetic-real gap on top of the web→Mayo gap).

## Overall
Free data ≠ useful data here. (a) is a dead end, (c) is blocked+low-value, (b) is the one
worth keeping (free video with dynamics) IF we decide to expand the detector / temporal
pretraining. The data that solves the problem is still Mayo-generated: in-domain healthy
controls + HB labels.
