#!/usr/bin/env python3
"""Launch the locked shared V8 container with a minimal host boundary."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess


_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}\Z")
_NAME = re.compile(r"[a-z0-9][a-z0-9_.-]{0,62}\Z")


def build_run_command(
    *, image_id: str, release_root: Path, port: int, name: str,
) -> tuple[str, ...]:
    if (
        type(image_id) is not str
        or _IMAGE_ID.fullmatch(image_id) is None
        or not isinstance(release_root, Path)
        or release_root.is_symlink()
        or not release_root.is_dir()
        or isinstance(port, bool)
        or not isinstance(port, int)
        or port < 1024
        or port > 65535
        or type(name) is not str
        or _NAME.fullmatch(name) is None
    ):
        raise ValueError("container launch configuration is invalid")
    canonical = release_root.resolve(strict=True)
    return (
        "sudo", "-n", "docker", "run", "--detach",
        "--name", name,
        "--restart", "unless-stopped",
        "--gpus", "device=0",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges:true",
        "--pids-limit=256",
        "--user=1001:1001",
        "--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=67108864,mode=1777",
        "--mount", f"type=bind,src={canonical},dst=/model,readonly",
        "--publish", f"127.0.0.1:{port}:8080",
        "--env", "SHARED_V8_RELEASE=/model",
        "--env", "SHARED_V8_DEVICE=cuda",
        image_id,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--name", default="shared-v8-deployment-v1")
    return parser


def main() -> None:
    args = _parser().parse_args()
    command = build_run_command(
        image_id=args.image_id,
        release_root=args.release_root,
        port=args.port,
        name=args.name,
    )
    inspected = subprocess.run(
        ("sudo", "-n", "docker", "image", "inspect", args.image_id,
         "--format", "{{.Id}} {{.Config.User}}"),
        check=True, capture_output=True, text=True, timeout=10,
    ).stdout.strip()
    if inspected != f"{args.image_id} 1001:1001":
        raise RuntimeError("runtime image ID or non-root user differs from the pin")
    completed = subprocess.run(
        command, check=True, capture_output=True, text=True, timeout=30,
    )
    container_id = completed.stdout.strip()
    if re.fullmatch(r"[0-9a-f]{64}", container_id) is None:
        raise RuntimeError("Docker returned an invalid container identifier")
    print(container_id)


if __name__ == "__main__":
    main()
