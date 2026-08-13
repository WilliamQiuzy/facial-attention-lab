"""Verify OpenAI connectivity and discover which image model id is available.

Usage:
    python test_connection.py            # list models + image-capable ids
    python test_connection.py --probe    # also do ONE tiny real generation (costs $)
"""
from __future__ import annotations

import argparse
import sys

import config
from openai_client import make_client, generate_image


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true",
                        help="Do one tiny real image generation to confirm end-to-end (costs money).")
    args = parser.parse_args()

    try:
        key = config.load_api_key()
    except RuntimeError as e:
        print(f"[FAIL] {e}")
        return 1
    print(f"[OK]   API key loaded (…{key[-6:]}), length={len(key)}")

    client = make_client()

    # 1. List models -> confirms the key works and shows available image models.
    try:
        models = client.models.list()
    except Exception as e:  # noqa: BLE001 - surface any auth/network error clearly
        print(f"[FAIL] Could not list models: {type(e).__name__}: {e}")
        return 1

    ids = sorted(m.id for m in models.data)
    image_like = [m for m in ids if any(t in m for t in ("image", "dall"))]
    print(f"[OK]   Connected. {len(ids)} models visible.")
    print("\nImage-capable models available to this key:")
    if image_like:
        for m in image_like:
            marker = "  <- configured default" if m == config.IMAGE_MODEL else ""
            print(f"    - {m}{marker}")
    else:
        print("    (none matched 'image'/'dall' — check account access)")

    if config.IMAGE_MODEL not in ids:
        print(f"\n[WARN] Configured IMAGE_MODEL='{config.IMAGE_MODEL}' is not in the visible "
              f"list. It may still work (generation endpoint), or set IMAGE_MODEL to one above.")

    # 2. Optional end-to-end probe.
    if args.probe:
        print(f"\nProbing one generation with model='{config.IMAGE_MODEL}', size=1024x1024 …")
        res = generate_image(
            client,
            "A clinical medical photograph: frontal portrait of a person against a plain "
            "medical-blue backdrop, neutral expression, photorealistic, no text.",
            size="1024x1024",
            quality="low",
        )
        if res.ok:
            print(f"[OK]   Generation succeeded: {len(res.image_bytes)} bytes, usage={res.usage}")
        else:
            print(f"[FAIL] Generation failed: {res.error_kind}: {res.error_detail}")
            return 1

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
