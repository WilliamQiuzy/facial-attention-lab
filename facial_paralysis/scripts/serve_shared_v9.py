#!/usr/bin/env python3
"""Serve the public Shared V9 ensemble; configuration is environment-only."""
from __future__ import annotations

import os
from pathlib import Path
import sys

import uvicorn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.deployment.shared_v9_service import create_app  # noqa: E402


def main() -> None:
    release = Path(os.environ.get("SHARED_V9_RELEASE", "/model"))
    device = os.environ.get("SHARED_V9_DEVICE", "cpu")
    host = os.environ.get("SHARED_V9_HOST", "0.0.0.0")
    port_text = os.environ.get("SHARED_V9_PORT", "8080")
    if not port_text.isascii() or not port_text.isdigit():
        raise ValueError("SHARED_V9_PORT must be an integer")
    port = int(port_text)
    if port < 1024 or port > 65535:
        raise ValueError("SHARED_V9_PORT is outside the non-privileged range")
    uvicorn.run(
        create_app(release, device=device),
        host=host,
        port=port,
        access_log=False,
        server_header=False,
        date_header=False,
        workers=1,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
