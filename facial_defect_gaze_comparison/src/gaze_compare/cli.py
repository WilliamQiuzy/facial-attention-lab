from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from .analysis import run_analysis
from .cohort_analysis import run_independent_cohort_analysis
from .cohort_report import write_independent_outputs
from .cohort_simulate import load_cohort_config, write_independent_inputs
from .report import write_analysis_outputs
from .simulate import load_mock_config, write_mock_inputs


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gaze-compare",
        description="Research-only webcam vs reference-instrument gaze comparison.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    simulate = subparsers.add_parser("simulate", help="Generate deterministic synthetic inputs.")
    simulate.add_argument("--config", type=Path, default=Path("config/mock_study.json"))
    simulate.add_argument("--output", type=Path, default=Path("data/mock"))

    analyze = subparsers.add_parser("analyze", help="Validate and analyze paired inputs.")
    analyze.add_argument("--samples", type=Path, default=Path("data/mock/gaze_samples.csv"))
    analyze.add_argument("--aois", type=Path, default=Path("data/mock/aoi_definitions.csv"))
    analyze.add_argument("--config", type=Path, default=Path("config/mock_study.json"))
    analyze.add_argument("--output", type=Path, default=Path("outputs"))
    analyze.add_argument(
        "--mode",
        choices=("mock-paired", "real-paired"),
        default="mock-paired",
        help="Real mode requires a non-synthetic config with preregistered_margins.",
    )

    cohort_simulate = subparsers.add_parser(
        "cohort-simulate",
        help="Generate the primary 500+500 independent-cohort mock inputs.",
    )
    cohort_simulate.add_argument(
        "--config", type=Path, default=Path("config/mock_independent_study.json")
    )
    cohort_simulate.add_argument("--output", type=Path, default=Path("data/mock_independent"))

    cohort_analyze = subparsers.add_parser(
        "cohort-analyze",
        help="Analyze different participants as two independent cohorts.",
    )
    cohort_analyze.add_argument(
        "--participants",
        type=Path,
        default=Path("data/mock_independent/participant_summary.csv"),
    )
    cohort_analyze.add_argument(
        "--fixations", type=Path, default=Path("data/mock_independent/fixation_events.csv")
    )
    cohort_analyze.add_argument(
        "--aois", type=Path, default=Path("data/mock_independent/aoi_definitions.csv")
    )
    cohort_analyze.add_argument(
        "--config", type=Path, default=Path("config/mock_independent_study.json")
    )
    cohort_analyze.add_argument(
        "--output", type=Path, default=Path("outputs/independent_cohort_demo")
    )
    return parser


def _read_config(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "cohort-simulate":
        config = load_cohort_config(args.config)
        manifest = write_independent_inputs(config, args.output)
        print(f"Independent-cohort synthetic inputs written: {manifest}")
        return 0

    if args.command == "cohort-analyze":
        config = load_cohort_config(args.config)
        result = run_independent_cohort_analysis(
            pd.read_csv(args.participants),
            pd.read_csv(args.fixations),
            pd.read_csv(args.aois),
            config=config,
        )
        manifest = write_independent_outputs(result, args.output, config=config)
        print(f"Independent-cohort analysis written: {manifest}")
        return 0

    if args.command == "simulate":
        config = load_mock_config(args.config)
        manifest = write_mock_inputs(config, args.output)
        print(f"Synthetic inputs written: {manifest}")
        return 0

    config = _read_config(args.config)
    mock_mode = args.mode == "mock-paired"
    if not mock_mode and config.get("synthetic") is True:
        raise ValueError("real-paired mode cannot use a synthetic config")
    if mock_mode and config.get("synthetic") is not True:
        raise ValueError("mock-paired mode requires synthetic=true")
    margin_key = "illustrative_margins" if mock_mode else "preregistered_margins"
    margins = config.get(margin_key)
    if margins is not None and not isinstance(margins, dict):
        raise ValueError(f"{margin_key} must be an object of endpoint margins")

    samples = pd.read_csv(args.samples)
    aois = pd.read_csv(args.aois)
    result = run_analysis(
        samples,
        aois,
        require_paired=True,
        grid_size=int(config.get("map_grid_size", 48)),
        margins=margins,
        mock_mode=mock_mode,
    )
    manifest = write_analysis_outputs(result, args.output, config=config)
    print(f"Analysis written: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
