"""Fail-closed entry point for the not-yet-authorized neural outer benchmark."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.dynamic_landmark_benchmark import (  # noqa: E402
    require_frozen_outer_registry,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry-hash",
        help="reserved for Task 7; currently cannot unlock outer evaluation",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    require_frozen_outer_registry(args.registry_hash)


if __name__ == "__main__":
    main()
