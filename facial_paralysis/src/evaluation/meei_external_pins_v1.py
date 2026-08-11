"""Out-of-band activation pin for the one-shot MEEI external evaluation."""
from __future__ import annotations

# Exact canonical digest of the result-free registry created after the code,
# cache bytes, artifact, population, protocol, and output path were frozen.
PINNED_MEEI_AUTHORIZATION_SHA256: str | None = (
    "8bf70dd1b04381a35af2de99b530e744425384a060b90a48c56156ec6b3ae6af"
)

__all__ = ["PINNED_MEEI_AUTHORIZATION_SHA256"]
