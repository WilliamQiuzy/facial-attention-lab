"""Run fixed-width 95-d blendshape/landmark/fusion ablations.

All arms use the same cache and model width. Feature blocks are zero-masked by
``autoresearch_fp.runner.apply_landmark_ablation`` so parameter tensor shapes,
split, seed, and optimization budget remain identical.  ``feat=regasym`` adds
blendshape-specific engineered columns, so use ``--feature-engineering raw``
when a modality-symmetric sensitivity analysis is required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ["FP_CLINICAL"] = "1"

ROOT = Path(__file__).resolve().parents[1]
AR = ROOT / "autoresearch_fp"
if str(AR) not in sys.path:
    sys.path.insert(0, str(AR))

import prepare_fp as P  # noqa: E402
import runner as R  # noqa: E402


AUDITED_CACHE_SHA256 = "fbacd1b15234168ab2afa34ec7a4c40bc26c5a2bf33b279a48d6a892841b4b5c"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_cache_sha256(path: Path, expected: str) -> str:
    if len(expected) != 64 or any(c not in "0123456789abcdefABCDEF" for c in expected):
        raise ValueError("expected cache SHA-256 must be exactly 64 hexadecimal characters")
    observed = _sha256(path)
    if observed.lower() != expected.lower():
        raise ValueError(
            f"cache SHA-256 mismatch: expected {expected.lower()}, observed {observed}; "
            "audit the rebuilt cache and pass its approved hash explicitly"
        )
    return observed


def _score_one_seed(predictions: dict) -> dict[str, float]:
    scores: dict[str, float] = {}
    for task in P.REGION_TASKS:
        truth = [r["label"] for r in P.val_records(task)]
        scores[task] = P.quadratic_kappa(
            truth, predictions[task], P.N_CLASSES[task])
    scores["metric"] = 0.5 * (scores["eyes"] + scores["mouth"])
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True,
                        choices=("blendshape_only", "landmark_only", "fusion"))
    parser.add_argument("--cache", type=Path, default=AR / "fp_ar_cache_clinical.pt")
    parser.add_argument("--config", type=Path, default=AR / "deploy_config.json")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument(
        "--expected-cache-sha256", default=AUDITED_CACHE_SHA256,
        help="approved content hash; prevents a same-width cache with changed column semantics",
    )
    parser.add_argument(
        "--feature-engineering", choices=("inherit", "raw", "asym", "regasym"),
        default="inherit",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.cache.exists():
        raise FileNotFoundError(args.cache)

    seeds = tuple(int(x) for x in args.seeds.split(",") if x.strip())
    config = dict(R.DEFAULT)
    config.update(json.loads(args.config.read_text()))
    if args.feature_engineering != "inherit":
        config["feat"] = args.feature_engineering
    config["landmark_ablation"] = args.arm
    config["name"] = f"landmark_ablation_{args.arm}"

    # Explicitly redirect the fixed harness to the audited 95-d cache.
    cache_sha256 = _validate_cache_sha256(args.cache, args.expected_cache_sha256)
    P.CACHE = args.cache.resolve()
    P.load_data.cache_clear()
    records = P.load_data()
    observed_dims = sorted({int(r["mp_seq"].shape[-1]) for r in records})
    if observed_dims != [95]:
        raise ValueError(f"expected an all-95-d cache, observed {observed_dims}")

    started = time.time()
    predictions = [R.train_one_seed(seed, config) for seed in seeds]
    metrics = P.report_metric(predictions, extra={
        "name": config["name"],
        "train_seconds": round(time.time() - started, 1),
    })
    result = {
        "arm": args.arm,
        "cache_feature_schema": "legacy_clinical23_v1_hash_pinned",
        "cache_schema_limit": "historical_pt_has_no_ordered_name_metadata",
        "experiment_scope": "static_web_metric_not_mayo_dynamic_validation",
        "selection_protocol": "best_epoch_selected_and_scored_on_same_internal_validation",
        "seeds": seeds,
        "cache": str(args.cache.resolve()),
        "cache_sha256": cache_sha256,
        "expected_cache_sha256": args.expected_cache_sha256.lower(),
        "observed_dims": observed_dims,
        "record_count": len(records),
        "config": config,
        "per_seed_metrics": [
            {"seed": seed, **_score_one_seed(pred)}
            for seed, pred in zip(seeds, predictions)
        ],
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
