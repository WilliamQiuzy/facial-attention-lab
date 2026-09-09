# Facial Defect Synthesis

Generate **synthetic** clinical-style photographs of faces with facial
differences for research prototyping. Images are produced from text prompts
with OpenAI's GPT Image model (`gpt-image-2`); no patient photograph or patient
record is used as model input. Each generated image represents either a
**preop** state (long-standing or congenital difference) or a **healed** state
(a well-healed faint scar).

## Public release boundary

This public repository contains the generation code and exactly 10
hash-approved synthetic demonstration images used by the web application. It
does not contain the larger internal generated-image collection or the
generation metadata log. Synthetic generation does not itself establish
clinical validity, IRB status, or fitness for model training.

## Defect categories

| key | description | white-predominant? |
|-----|-------------|--------------------|
| `mohs` | Mohs / facial skin cancer & reconstruction | strongly |
| `hn_cancer` | Head & neck cancer resection & reconstruction | moderately |
| `facial_paralysis` | Facial paralysis / paresis / synkinesis | no |
| `cleft` | Cleft lip (congenital) & repair | intermediate |
| `trauma` | Long-standing post-traumatic scars / deformity | no |
| `burns` | Facial burn scars & contractures | no |
| `vascular` | Vascular anomalies (hemangioma / port-wine stain) | leaning |
| `rhinophyma` | Rhinophyma & repair | strongly |
| `nevus` | Giant congenital melanocytic nevus & reconstruction | no |
| `craniofacial` | Craniofacial microsomia / microtia & repair | no |

## Layout

```
config.py           model / size / quality / moderation; reads the API key
prompts.py          clinical prompt templates + diversity (age/sex/ethnicity/view/bg)
weighting.py        Option B: disease mix + per-disease race ratio + sex skew
openai_client.py    OpenAI Images API wrapper (retry, moderation handling)
test_connection.py  connectivity check + image-model discovery
generate.py         CLI: catalog / category / batch / weighted
output/images/<disease>/   locally generated PNGs; ignored by Git
metadata/           local generations.jsonl audit log; ignored by Git
output/synthetic/   10 public, hash-approved demonstration assets
demographics_disease_research.md   weighting rationale + citations
IRB_data_generation_statement.md   IRB method statement
```

## Setup

Set `OPENAI_API_KEY` in the environment or in the repository-root `.env` file.
Environment files and locally generated outputs are ignored by Git.

```bash
pip install -r requirements.txt
```

## Usage

```bash
# 1. Verify connectivity + see which image models the key can use
python test_connection.py
python test_connection.py --probe        # also does 1 real (cheap) generation

# 2. Catalog: one image for every sub-classification of every disease
python generate.py catalog

# 3. N diverse images of one category
python generate.py category --category mohs --n 10

# 4. Weighted mode — N images sampled from the research weighting plan:
#    disease mix + per-disease racial distribution + sex skew
python generate.py weighted --n 1000 --workers 6 --seed 42

# 5. Uniform batch (equal per category, ignores weighting)
python generate.py batch --per-category 50
```

Images are organised one folder per disease, with standardized names, e.g.
`output/images/mohs/mohs_preop_bcc-nasal-ala_elderly-white-male_a1b2c3.png`.
All views are frontal (project requirement).

## Configuration (env overrides)

| var | default | notes |
|-----|---------|-------|
| `IMAGE_MODEL` | `gpt-image-2` | the surgeon's "GPT Image 2"; `test_connection.py` lists alternatives |
| `IMAGE_SIZE` | `1024x1024` | `1024x1024` / `1536x1024` / `1024x1536` |
| `IMAGE_QUALITY` | `high` | `low` / `medium` / `high` |
| `IMAGE_MODERATION` | `low` | least restrictive for clinical imagery |

## Notes

- **Weighting** (disease %, per-disease race %, sex skew) lives in `weighting.py`;
  rationale + citations in `demographics_disease_research.md`. ~67% White overall.
- Minors + medical content trip moderation stochastically; `generate_image` retries
  blocked prompts (`MODERATION_RETRIES`), and surviving blocks are logged and skipped.
- Local `metadata/generations.jsonl` records the prompt, demographic
  descriptors, parameters, and token usage for generated images. It is not
  included in the public repository.
- ~7,200 image tokens per high-quality 1024×1024 image.
