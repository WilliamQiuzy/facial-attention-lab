"""Generate synthetic facial-defect clinical images with the OpenAI image model.

All images are written to ONE flat folder (config.IMAGES_DIR) with standardized
names. One JSON line per image is appended to metadata/generations.jsonl.

Modes:
    python generate.py catalog
        One image for EVERY sub-classification of EVERY disease (detailed coverage).
        Filenames: <NNN>_<category>_<state>_<slug>.png

    python generate.py category --category facial_paralysis --n 10
        N diverse images of a single category.
        Filenames: <category>_<state>_<slug>_<id>.png

    python generate.py batch --per-category 20
        N diverse images for every category (uniform across categories).

    python generate.py weighted --n 1000
        N images sampled by epidemiology (weighting.py): disease mix + per-disease
        racial distribution + sex skew. This is the mode for the real dataset.

`state` is one of: preop (long-standing / congenital, pre-reconstruction) or
healed (well-healed faint mature scar).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import re
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
import prompts
import weighting
from openai_client import make_client, generate_image, GenResult


def _timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _ensure_dirs() -> None:
    config.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    config.METADATA_DIR.mkdir(parents=True, exist_ok=True)


def _log_metadata(record: dict) -> None:
    with config.METADATA_LOG.open("a") as f:
        f.write(json.dumps(record) + "\n")


def _run_one(spec: prompts.PromptSpec, fname: str, args) -> dict:
    """Generate, save to the flat images dir, and return a metadata record."""
    client = make_client()
    gid = uuid.uuid4().hex[:8]
    res: GenResult = generate_image(
        client, spec.prompt, model=args.model, size=args.size,
        quality=args.quality, moderation=args.moderation,
    )
    record = {
        "id": gid,
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "size": args.size,
        "quality": args.quality,
        "moderation": args.moderation,
        "status": "ok" if res.ok else res.error_kind,
        "file": None,
        "usage": res.usage,
        "moderation_attempts": res.moderation_attempts,
        "prompt": spec.prompt,
        **spec.meta(),
    }
    if res.ok:
        fpath = config.IMAGES_DIR / spec.category / fname   # one folder per disease
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fpath.write_bytes(res.image_bytes)
        record["file"] = f"{spec.category}/{fname}"
        print(f"  [ok]   {spec.category}/{fname}  (retries={res.moderation_attempts}, "
              f"tokens={res.usage.get('total_tokens')})")
    else:
        record["error_detail"] = res.error_detail
        print(f"  [FAIL] {fname}  {res.error_kind}: {(res.error_detail or '')[:100]}")
    _log_metadata(record)
    return record


def _catalog_jobs() -> list:
    """(spec, filename) for one image per (category, subtype)."""
    jobs = []
    for idx, spec in enumerate(prompts.catalog_specs(), start=1):
        fname = f"{idx:03d}_{spec.category}_{spec.state}_{spec.slug}.{config.OUTPUT_FORMAT}"
        jobs.append((spec, fname))
    return jobs


def _random_jobs(specs: list) -> list:
    """(spec, filename) for diverse batch/category specs (id keeps names unique)."""
    jobs = []
    for spec in specs:
        demo = _slugify(f"{spec.age}-{spec.ethnicity}-{spec.sex}")
        gid = uuid.uuid4().hex[:6]
        fname = f"{spec.category}_{spec.state}_{spec.slug}_{demo}_{gid}.{config.OUTPUT_FORMAT}"
        jobs.append((spec, fname))
    return jobs


def _run_jobs(jobs: list, args) -> list:
    print(f"Generating {len(jobs)} image(s) -> {config.IMAGES_DIR}  "
          f"(model={args.model}, size={args.size}, quality={args.quality}, "
          f"workers={args.workers})\n")
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_one, spec, fname, args): fname for spec, fname in jobs}
        for fut in as_completed(futures):
            results.append(fut.result())
    return results


def _summary(results: list) -> None:
    ok = [r for r in results if r["status"] == "ok"]
    fail = [r for r in results if r["status"] != "ok"]
    total_tokens = sum((r["usage"] or {}).get("total_tokens") or 0 for r in ok)
    print(f"\n=== Summary: {len(ok)} ok, {len(fail)} failed, "
          f"{total_tokens} image tokens used ===")
    if fail:
        kinds = {}
        for r in fail:
            kinds[r["status"]] = kinds.get(r["status"], 0) + 1
        print("    failures by kind:", kinds)
        for r in fail:
            print(f"      - {r['category']}/{r['state']}/{r['slug']}: {r['status']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Synthetic facial-defect image generator")
    parser.add_argument("mode", choices=["catalog", "category", "batch", "weighted"])
    parser.add_argument("--category", choices=prompts.CATEGORY_KEYS)
    parser.add_argument("--n", type=int, default=5, help="images for 'category'/'weighted' mode")
    parser.add_argument("--per-category", type=int, default=10, help="images/category for 'batch'")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--model", default=config.IMAGE_MODEL)
    parser.add_argument("--size", default=config.IMAGE_SIZE)
    parser.add_argument("--quality", default=config.IMAGE_QUALITY)
    parser.add_argument("--moderation", default=config.MODERATION)
    parser.add_argument("--workers", type=int, default=config.DEFAULT_WORKERS)
    args = parser.parse_args()

    _ensure_dirs()
    rng = random.Random(args.seed)

    if args.mode == "catalog":
        jobs = _catalog_jobs()
    elif args.mode == "category":
        if not args.category:
            parser.error("--category is required for 'category' mode")
        specs = [prompts.random_prompt(args.category, rng) for _ in range(args.n)]
        jobs = _random_jobs(specs)
    elif args.mode == "weighted":
        specs = weighting.weighted_specs(args.n, rng)
        jobs = _random_jobs(specs)
    else:  # batch
        specs = [prompts.random_prompt(cat, rng)
                 for cat in prompts.CATEGORY_KEYS for _ in range(args.per_category)]
        jobs = _random_jobs(specs)

    results = _run_jobs(jobs, args)
    _summary(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
