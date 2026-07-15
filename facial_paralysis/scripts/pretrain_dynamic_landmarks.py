"""Fail-closed entry point for dynamic landmark masked-span pretraining."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pretraining.dynamic_landmark_ssl import (  # noqa: E402
    require_frozen_pretraining_inputs,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ravdess-manifest", type=Path)
    parser.add_argument("--mayo-manifest", type=Path)
    parser.add_argument("--config", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    require_frozen_pretraining_inputs(
        args.ravdess_manifest, args.mayo_manifest, args.config
    )


if __name__ == "__main__":
    main()
