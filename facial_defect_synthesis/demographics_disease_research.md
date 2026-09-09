# Generation Plan — Disease Mix & Per-Disease Race Ratios (Option B)

Synthetic-dataset weighting based on US demographics and disease-by-race
epidemiology. Implemented in `weighting.py`; run with `python generate.py weighted --n N`.

## US racial/ethnic distribution (reference baseline)

White (non-Hispanic) 56.4% · Hispanic 20.1% · Black 11.6% · Asian 6.2%
(≈ East Asian 3.3 / South Asian 2.9) · Two+ 5.1% · AIAN 0.5% · NHOPI 0.2% (KFF, ACS 2024).
"Middle Eastern" is counted as White by the Census (MENA ≈ 1%).

## Disease mix (share of dataset)

| Disease | Share | | Disease | Share |
|---|---|---|---|---|
| Skin cancer / Mohs | 28% | | Burns | 7% |
| Head & neck cancer | 20% | | Vascular (hemangioma/PWS) | 5% |
| Trauma | 12% | | Rhinophyma | 4% |
| Facial paralysis | 10% | | Giant melanocytic nevus | 3% |
| Cleft lip | 8% | | Craniofacial (microsomia/microtia) | 3% |

Skin cancer + H&N = 48% (kept dominant, per the White-predominance rationale).

## Per-disease race ratio (Option B — %)

| Disease | White | Hispanic | Black | E.Asian | S.Asian | M.East | Basis |
|---|---|---|---|---|---|---|---|
| Skin cancer / Mohs | 90 | 5 | 1 | 1 | 1 | 2 | melanoma ~20–30× higher in NHW |
| Head & neck cancer | 68 | 9 | 14 | 4 | 2 | 3 | oral/HPV+ White-leaning; larynx Black |
| Trauma | 50 | 22 | 18 | 3 | 3 | 4 | ~population; Black ↑ (assault) |
| Facial paralysis | 54 | 22 | 14 | 4 | 3 | 3 | ~population; slight Black/Hispanic ↑ |
| Cleft lip | 50 | 28 | 6 | 6 | 4 | 6 | White & Hispanic high; Black/Asian low |
| Burns | 50 | 22 | 18 | 3 | 3 | 4 | ~population |
| Vascular | 60 | 18 | 10 | 5 | 4 | 3 | infantile hemangioma White-predominant |
| Rhinophyma | 88 | 5 | 1 | 2 | 2 | 2 | rosacea / fair skin, older White men |
| Giant nevus | 55 | 22 | 12 | 5 | 3 | 3 | ~population |
| Craniofacial | 45 | 28 | 8 | 9 | 5 | 5 | microtia ↑ in Hispanic/Asian |

**Sex skew** (`MALE_FRACTION`): trauma 65% male, rhinophyma 80%, H&N cancer 70%,
Mohs 55%; all others balanced. Set to 0.5 to disable.

**Implied overall dataset mix:** White ≈ 67% · Hispanic ≈ 15% · Black ≈ 10% ·
East Asian ≈ 3% · South Asian ≈ 2% · Middle Eastern ≈ 3%. (Above US's 56% White
because White-predominant cancers are weighted up; deliberately not forced to
Mayo's ~95%, since other races genuinely present.)

---

## References

**1. Facial reconstruction — scope & indications**
- Complex facial trauma reconstruction — [PMC7175762](https://pmc.ncbi.nlm.nih.gov/articles/PMC7175762/)
- Mohs defect repair / facial reconstruction (StatPearls) — [NBK553099](https://www.ncbi.nlm.nih.gov/books/NBK553099/)
- Face transplant indications (ballistic, burns, bites) — [PMC9571096](https://pmc.ncbi.nlm.nih.gov/articles/PMC9571096/)
- Craniofacial anomalies overview — [Nationwide Children's](https://www.nationwidechildrens.org/conditions/health-library/overview-of-craniofacial-anomalies)
- Cervicofacial vascular anomalies — [PMC11205235](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11205235/)
- Hemifacial microsomia — [PMC11587098](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11587098/)
- Giant congenital melanocytic nevus epidemiology — [PMC4527643](https://pmc.ncbi.nlm.nih.gov/articles/PMC4527643/)
- Facial-nerve reconstruction for flaccid paralysis (systematic review) — [PMC11298393](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11298393/)

**2. Demographics & disease-by-race epidemiology**
- US race/ethnicity distribution (KFF, ACS 2024) — [KFF](https://www.kff.org/other/state-indicator/distribution-by-raceethnicity/)
- US race distribution cross-check (Census 2023) — [Neilsberg](https://www.neilsberg.com/insights/united-states-population-by-race/)
- Skin cancer racial disparities review — [PMC9345197](https://pmc.ncbi.nlm.nih.gov/articles/PMC9345197/)
- Skin cancer facts & rates — [SkinCancer.org](https://www.skincancer.org/skin-cancer-information/skin-cancer-facts/)
- Orofacial clefts by race, 12-yr US trends (Taritsa 2024) — [PubMed 38291621](https://pubmed.ncbi.nlm.nih.gov/38291621/)
- HPV-associated head & neck cancer by race — [PMC3308956](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3308956/)
- Bell's palsy epidemiology — [News-Medical](https://www.news-medical.net/health/Bells-Palsy-Epidemiology.aspx)
- Facial fracture epidemiology (US trends) — [PubMed 40582102](https://pubmed.ncbi.nlm.nih.gov/40582102/)
- Women with facial fractures (93% Caucasian) — [PMC11562983](https://pmc.ncbi.nlm.nih.gov/articles/PMC11562983/)
