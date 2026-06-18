# Facial Defect Synthesis

Generate **synthetic** clinical photographs of faces with facial defects, for
research on facial reconstruction (Mayo Clinic / Harvard). Images are produced by
OpenAI's GPT-Image model and depict **no real patients**, so the dataset avoids
patient-privacy / HIPAA concerns while remaining useful for model training.

## Defect categories (from the surgeon's brief)

| key | description |
|-----|-------------|
| `facial_paralysis` | Facial paralysis / paresis / synkinesis |
| `mohs` | Mohs surgical resection and reconstruction |
| `hn_cancer` | Head & neck cancer resection and reconstruction |
| `cleft` | Cleft lip / palate repair |
| `trauma` | Repair after trauma (lacerations, mandible/maxilla fractures) |

## Layout

```
config.py           model / size / quality / moderation; reads the API key
prompts.py          clinical prompt templates + diversity (age/sex/ethnicity/view/bg)
openai_client.py    OpenAI Images API wrapper (retry, moderation handling)
test_connection.py  connectivity check + image-model discovery
generate.py         CLI: samples / single-category / full batch
output/             generated PNGs (samples/ and batch/)
metadata/           generations.jsonl — one auditable record per image
```

## Setup

The OpenAI API key is read automatically from the shared `../.env`
(it accepts `OPENAI_API_KEY=...` or a bare `sk-...` line). No extra config needed.

```bash
pip install -r requirements.txt
```

## Usage

```bash
# 1. Verify connectivity + see which image models the key can use
python test_connection.py
python test_connection.py --probe        # also does 1 real (cheap) generation

# 2. Generate one representative sample per category (5 images)
python generate.py samples

# 3. Generate N diverse images of one category
python generate.py category --category facial_paralysis --n 10

# 4. Large-scale: N images for every category
python generate.py batch --per-category 50 --workers 6 --seed 42
```

## Configuration (env overrides)

| var | default | notes |
|-----|---------|-------|
| `IMAGE_MODEL` | `gpt-image-1` | set to whatever `test_connection.py` reports |
| `IMAGE_SIZE` | `1024x1024` | `1024x1024` / `1536x1024` / `1024x1536` |
| `IMAGE_QUALITY` | `high` | `low` / `medium` / `high` |
| `IMAGE_MODERATION` | `low` | `low` is least restrictive for clinical imagery |

## Notes

- Clinical/surgical imagery can trip content moderation. Prompts are written in
  clinical-documentation language and `moderation=low` is used; blocked prompts
  are logged (status `moderation_blocked`) and the batch continues.
- `metadata/generations.jsonl` records the exact prompt, demographics, params, and
  token usage for every image — the dataset is reproducible and auditable.
