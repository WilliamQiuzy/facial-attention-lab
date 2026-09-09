"""Thin wrapper around the OpenAI Images API with retry + moderation handling."""
from __future__ import annotations

import base64
import time
from dataclasses import dataclass, field

from openai import OpenAI
from openai import APIConnectionError, APIStatusError, BadRequestError, RateLimitError

import config


def make_client() -> OpenAI:
    return OpenAI(api_key=config.load_api_key())


@dataclass
class GenResult:
    ok: bool
    image_bytes: bytes | None = None
    error_kind: str | None = None        # "moderation_blocked" | "bad_request" | "api_error"
    error_detail: str | None = None
    usage: dict = field(default_factory=dict)
    revised_prompt: str | None = None
    moderation_attempts: int = 0         # how many times a moderation block was retried


def _supports_moderation(model: str) -> bool:
    return model.startswith("gpt-image")


def generate_image(
    client: OpenAI,
    prompt: str,
    *,
    model: str = config.IMAGE_MODEL,
    size: str = config.IMAGE_SIZE,
    quality: str = config.IMAGE_QUALITY,
    moderation: str = config.MODERATION,
    output_format: str = config.OUTPUT_FORMAT,
    max_retries: int = config.MAX_RETRIES,
    moderation_retries: int = config.MODERATION_RETRIES,
) -> GenResult:
    """Generate one image. Returns a GenResult instead of raising for the common
    recoverable failures (moderation block, bad request) so callers can log and
    continue a batch.

    Moderation blocks are retried up to `moderation_retries` times because the
    safety system is stochastic for borderline (e.g. pediatric medical) prompts —
    the identical prompt frequently succeeds on a later attempt. Blocked requests
    do not bill image tokens, so these retries are cheap.
    """
    kwargs = {"model": model, "prompt": prompt, "size": size, "n": 1}
    if quality and quality != "auto":
        kwargs["quality"] = quality
    if _supports_moderation(model):
        kwargs["moderation"] = moderation
        kwargs["output_format"] = output_format

    api_attempt = 0
    mod_attempt = 0
    while True:
        try:
            resp = client.images.generate(**kwargs)
            datum = resp.data[0]
            image_bytes = base64.b64decode(datum.b64_json)
            usage = {}
            if getattr(resp, "usage", None) is not None:
                u = resp.usage
                usage = {
                    "total_tokens": getattr(u, "total_tokens", None),
                    "input_tokens": getattr(u, "input_tokens", None),
                    "output_tokens": getattr(u, "output_tokens", None),
                }
            return GenResult(
                ok=True,
                image_bytes=image_bytes,
                usage=usage,
                revised_prompt=getattr(datum, "revised_prompt", None),
                moderation_attempts=mod_attempt,
            )
        except BadRequestError as e:
            detail = str(e)
            is_moderation = "moderation" in detail.lower() or "safety" in detail.lower()
            if is_moderation and mod_attempt < moderation_retries:
                mod_attempt += 1
                time.sleep(1.0)  # brief pause before re-rolling the stochastic filter
                continue
            kind = "moderation_blocked" if is_moderation else "bad_request"
            return GenResult(ok=False, error_kind=kind, error_detail=detail,
                             moderation_attempts=mod_attempt)
        except (RateLimitError, APIConnectionError, APIStatusError) as e:
            api_attempt += 1
            if api_attempt > max_retries:
                return GenResult(ok=False, error_kind="api_error", error_detail=str(e),
                                 moderation_attempts=mod_attempt)
            time.sleep(min(2 ** api_attempt, 30))
