"""Central configuration for the facial-defect image synthesis project.

All paths are relative to this file so the project is portable. The OpenAI API
key is read from the shared parent `.env` without modifying that file: it accepts
both a standard `OPENAI_API_KEY=...` line and a bare `sk-...` line.
"""
from __future__ import annotations

import os
from pathlib import Path

# --- Paths --------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent                 # .../Mayo-Clinic
ENV_FILE = REPO_ROOT / ".env"
OUTPUT_DIR = PROJECT_ROOT / "output"
# All generated images live in ONE flat folder (no per-category subfolders).
IMAGES_DIR = OUTPUT_DIR / "images"
METADATA_DIR = PROJECT_ROOT / "metadata"
METADATA_LOG = METADATA_DIR / "generations.jsonl"

# --- Model / API defaults -----------------------------------------------------
# The doctor's "GPT Image 2" maps to the API model id `gpt-image-2` (confirmed
# available via test_connection.py; pinned id is `gpt-image-2-2026-04-21`).
# Other options visible to this key: gpt-image-1, gpt-image-1.5, gpt-image-1-mini.
# Override any of these via environment vars.
IMAGE_MODEL = os.environ.get("IMAGE_MODEL", "gpt-image-2")
IMAGE_SIZE = os.environ.get("IMAGE_SIZE", "1024x1024")     # 1024x1024 | 1536x1024 | 1024x1536 | auto
IMAGE_QUALITY = os.environ.get("IMAGE_QUALITY", "high")    # low | medium | high | auto
# Clinical/medical imagery can trip safety filters; "low" is the least
# restrictive setting that gpt-image-1 exposes (still blocks disallowed content).
MODERATION = os.environ.get("IMAGE_MODERATION", "low")     # low | auto
OUTPUT_FORMAT = os.environ.get("IMAGE_FORMAT", "png")      # png | jpeg | webp

# --- Generation behaviour -----------------------------------------------------
DEFAULT_WORKERS = int(os.environ.get("GEN_WORKERS", "4"))
MAX_RETRIES = int(os.environ.get("GEN_MAX_RETRIES", "4"))
# Moderation blocks on minors + medical content are stochastic; retrying the same
# prompt often succeeds. A blocked (400) request does not bill image tokens, so
# these retries are cheap.
MODERATION_RETRIES = int(os.environ.get("GEN_MODERATION_RETRIES", "3"))


def load_api_key() -> str:
    """Resolve the OpenAI API key from the environment or the shared .env file.

    Order of precedence:
      1. OPENAI_API_KEY environment variable
      2. `OPENAI_API_KEY=...` line in the parent .env
      3. A bare `sk-...` token line in the parent .env
    """
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key.strip()

    if ENV_FILE.exists():
        for raw in ENV_FILE.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("OPENAI_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
            # bare key line (the OpenAI key starts with sk-; skip other vendors)
            if line.startswith("sk-"):
                return line

    raise RuntimeError(
        f"No OpenAI API key found. Set OPENAI_API_KEY or add it to {ENV_FILE}."
    )
