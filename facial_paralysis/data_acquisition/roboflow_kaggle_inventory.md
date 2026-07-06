# (a) Free (no-application) facial-palsy dataset inventory — 2026-07-06

Scanned Roboflow Universe, Kaggle, Hugging Face, IEEE DataPort. Cross-referenced against
what we already have. **Conclusion: the free web pool is essentially exhausted — we already
hold the main ones, and the only genuinely-new free candidate is tiny (118 images).** This
matches the literature finding (no large public FP dataset; the field is data-starved).

## Already have (the main free ones)
| dataset | source | status |
|---|---|---|
| PalsyNet | HF `jasir/palsynet-data` | ✅ have (49 subj, binary) |
| FNP Detection (Bell's) | Roboflow `austre/fnp-detection` | ✅ have (our FNP) |
| Facial droop | Kaggle `kaitavmehta/facial-droop-and-facial-paralysis-image` | ✅ have (kaggle-droop) |
| anisa/paralysis-face, stroke-eye | Roboflow | ✅ have (v4; anisa useful, stroke unusable) |

## New free candidate (worth grabbing)
| dataset | source | note |
|---|---|---|
| **sumin/facial-paralysis** | Roboflow Universe, **118 images, CC BY 4.0** | small, likely NOT in our set. Download needs Roboflow API key/account (universe download is account-gated). Manual: universe.roboflow.com/sumin/facial-paralysis → "Download Dataset". |

## Free but partial / gated (not truly no-application)
| dataset | source | why not "free" |
|---|---|---|
| YFP (32 vids/21 subj) | sites.google.com/view/yfp-database | we have 16/21; the rest needs emailing AvLab-CV |
| IEEE DataPort "Facial Paralysis Dataset" | ieee-dataport.org | zip is **password-protected → email author** (gated) |
| Kaggle "Bell's palsy clinical trial" | Kaggle | tabular clinical trial data, **not images** — not useful for the vision model |

## Honest takeaway
Only ~118 new free images exist (sumin), and — per this whole project's finding — **more
web stills won't move the model** (web is at ceiling and doesn't transfer to Mayo). So (a)
confirms the free-download route is a dead end for real progress; value is in VIDEO
(the YouTube pipeline, (b)) and in Mayo-generated data (controls + HB labels). Downloads
that need credentials (Roboflow API, Kaggle API) are noted but not auto-run here.
