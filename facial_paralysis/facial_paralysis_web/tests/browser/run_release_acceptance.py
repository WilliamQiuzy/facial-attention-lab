"""Run the machine-readable Facial Process Web release gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
WEB_ROOT = HERE.parent.parent
MANIFEST = HERE / "acceptance_manifest.json"


def _run(command: list[str], env: dict[str, str]) -> dict[str, object]:
    started = time.monotonic()
    print(f"RUN {' '.join(command)}", flush=True)
    completed = subprocess.run(
        command,
        cwd=WEB_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(completed.stdout, end="", flush=True)
    return {
        "command": command,
        "exit_code": completed.returncode,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "output_tail": completed.stdout[-4_000:],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8081")
    parser.add_argument("--camera-file", type=Path, required=True)
    parser.add_argument("--skip-real-backend", action="store_true")
    parser.add_argument("--skip-unit-build", action="store_true")
    parser.add_argument(
        "--report", type=Path, default=Path("/tmp/faces-release-acceptance.json")
    )
    args = parser.parse_args()
    if not args.camera_file.is_file() or args.camera_file.suffix.casefold() != ".y4m":
        raise SystemExit("--camera-file must be one existing non-clinical Y4M fixture")

    manifest = json.loads(MANIFEST.read_text())
    env = dict(os.environ)
    env["ACCEPTANCE_BASE_URL"] = args.base_url.rstrip("/")
    report: dict[str, object] = {
        "schema_version": manifest["schema_version"],
        "base_url": args.base_url.rstrip("/"),
        "status": "running",
        "suites": [],
    }

    for suite in manifest["required_suites"]:
        if suite.get("real_backend") and args.skip_real_backend:
            continue
        if suite["id"] == "unit-build" and args.skip_unit_build:
            continue
        commands: list[list[str]] = []
        if "commands" in suite:
            commands.extend(suite["commands"])
        else:
            command = [sys.executable, str(HERE / suite["script"])]
            if suite["script"] not in {
                "wizard_acceptance.py",
                "capture_source_recovery_acceptance.py",
                "responsive_capture_acceptance.py",
            }:
                command.extend(["--base-url", args.base_url.rstrip("/")])
            if suite.get("camera_required"):
                command.extend(["--camera-file", str(args.camera_file)])
            command.extend(suite.get("arguments", []))
            commands.append(command)

        suite_result: dict[str, object] = {
            "id": suite["id"],
            "level": suite["level"],
            "cases": suite["cases"],
            "commands": [],
        }
        report["suites"].append(suite_result)
        for command in commands:
            command_result = _run(command, env)
            suite_result["commands"].append(command_result)
            if command_result["exit_code"] != 0:
                report["status"] = "failed"
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_text(json.dumps(report, indent=2) + "\n")
                raise SystemExit(
                    f"FAIL release gate at {suite['id']}; report={args.report}"
                )

    covered = sorted(
        {case for suite in report["suites"] for case in suite["cases"]}
    )
    report["covered_case_ids"] = covered
    report["status"] = "passed"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS release acceptance: {len(report['suites'])} suites, {len(covered)} case IDs")
    print(f"REPORT {args.report}")


if __name__ == "__main__":
    main()
