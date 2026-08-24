#!/usr/bin/env python3
"""Serve the raw-video FACES preprocessing gateway."""
from __future__ import annotations

import os
from pathlib import Path
import sys

import uvicorn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.deployment.faces_shared_v9_gateway import create_runtime_app  # noqa: E402


def main() -> None:
    model = Path(os.environ.get("FACE_LANDMARKER_MODEL", "/models/face_landmarker.task"))
    shared_v9_url = os.environ.get("SHARED_V9_URL", "http://shared-v9:8080")
    port_text = os.environ.get("FACES_GATEWAY_PORT", "8081")
    if not port_text.isascii() or not port_text.isdigit():
        raise ValueError("FACES_GATEWAY_PORT must be an integer")
    port = int(port_text)
    if port < 1024 or port > 65535:
        raise ValueError("FACES_GATEWAY_PORT is outside the non-privileged range")
    uvicorn.run(
        create_runtime_app(
            shared_v9_url=shared_v9_url,
            face_landmarker_path=model,
        ),
        host="0.0.0.0",
        port=port,
        access_log=False,
        server_header=False,
        date_header=False,
        workers=1,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
