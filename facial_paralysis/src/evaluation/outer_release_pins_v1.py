"""Out-of-band one-shot trust anchors for the 110D release.

This tiny module is deliberately excluded from the release implementation
aggregate: each pin is filled only after the artifact it commits is frozen.
All executable behavior lives in separately hashed modules and scripts.
"""
from __future__ import annotations

PINNED_OUTER_AUTHORIZATION_SHA256: str | None = (
    "7aca8ddffeca5479a53ee28103b091acf1f1d64badd46d62ce8a601654b0881b"
)
PINNED_PROTECTED_OUTER_REPORT_SHA256: str | None = None

__all__ = [
    "PINNED_OUTER_AUTHORIZATION_SHA256",
    "PINNED_PROTECTED_OUTER_REPORT_SHA256",
]
