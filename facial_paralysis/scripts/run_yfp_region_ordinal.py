#!/usr/bin/env python3
"""Run the fixed YFP eye/mouth ordinal protocol after all evidence gates pass."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.extract_yfp_clinical23 import CACHE_SCHEMA
from src.datasets.yfp_region_manifest import (
    ManifestError,
    authenticate_eligible_manifest,
    clinical23_region_features,
    write_manifest_once,
)
from src.models.l2_cumulative_logit import (
    BOOTSTRAP_ATTEMPT_LIMIT,
    BOOTSTRAP_REPEATS,
    BOOTSTRAP_SEED,
    FIXED_C,
    OPTIMIZER_OPTIONS,
    group_oof_probabilities,
    subject_cluster_bootstrap,
    weighted_ordinal_metrics,
)

YFP_OOF_SPLITS = 5
DEFAULT_REPORT = ROOT / "outputs" / "yfp_region_manifest_v1" / "ordinal" / "report.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read {label}") from exc
    if not isinstance(value, dict):
        raise ManifestError(f"{label} must be a JSON object")
    return value


def run_yfp_region_ordinal(
    manifest_path: str | Path,
    feature_cache_root: str | Path,
    output: str | Path = DEFAULT_REPORT,
) -> dict:
    """Evaluate both targets; eligibility is checked before cache/model access."""
    manifest_path = Path(manifest_path)
    feature_cache_root = Path(feature_cache_root)
    manifest, manifest_digest = authenticate_eligible_manifest(
        manifest_path, return_digest=True)
    cache_manifest_path = feature_cache_root / "manifest.json"
    cache = _load_json(cache_manifest_path, "YFP feature cache manifest")
    if (cache.get("schema_version") != CACHE_SCHEMA
            or cache.get("eligible_manifest_sha256") != manifest_digest
            or cache.get("feature_schema") != "clinical23_v2_static_single_frame"
            or cache.get("feature_dimension") != 23
            or cache.get("static_only") is not True
            or cache.get("dynamic_tiling_allowed") is not False):
        raise ManifestError("feature cache is not bound to the eligible static contract")
    cache_rows = cache.get("rows")
    if not isinstance(cache_rows, list):
        raise ManifestError("feature cache rows are missing")
    by_anchor = {row.get("anchor_key"): row for row in cache_rows}
    if len(by_anchor) != len(cache_rows) or set(by_anchor) != {
            row["anchor_key"] for row in manifest["rows"]}:
        raise ManifestError("feature cache must cover every eligible anchor exactly")

    clinical: dict[str, np.ndarray] = {}
    for row in manifest["rows"]:
        cached = by_anchor[row["anchor_key"]]
        path = feature_cache_root / cached["relative_path"]
        if path.is_symlink() or _sha256_file(path) != cached["sha256"]:
            raise ManifestError("feature cache digest mismatch")
        try:
            with np.load(path, allow_pickle=False) as values:
                vector = np.asarray(values["clinical23"], dtype=np.float64)
                anchor_key = str(np.asarray(values["anchor_key"]).item())
                source_commitment = str(np.asarray(values["source_commitment"]).item())
                schema = str(np.asarray(values["schema_version"]).item())
        except (OSError, ValueError, KeyError) as exc:
            raise ManifestError("feature cache row is malformed") from exc
        if (vector.shape != (23,) or not np.isfinite(vector).all()
                or anchor_key != row["anchor_key"]
                or source_commitment != row["source_commitment"]
                or schema != "clinical23_v2_static_single_frame"):
            raise ManifestError("feature cache provenance mismatch")
        clinical[anchor_key] = vector

    target_reports: dict[str, dict] = {}
    total_fits = 0
    total_predictions = 0
    for target in ("eye", "mouth"):
        rows = [row for row in manifest["rows"] if row["targets"][target] is not None]
        x = np.stack([clinical23_region_features(clinical[row["anchor_key"]], target)
                      for row in rows])
        y = np.asarray([row["targets"][target] for row in rows], dtype=np.int64)
        groups = np.asarray([row["reviewed_group"] for row in rows])
        oof = group_oof_probabilities(
            x, y, groups, n_splits=YFP_OOF_SPLITS, return_audit=True)
        prediction = np.argmax(oof.probabilities, axis=1).astype(np.int64)
        metrics = weighted_ordinal_metrics(y, prediction, groups)
        intervals = subject_cluster_bootstrap(y, prediction, groups)
        target_reports[target] = {
            "anchor_count": len(rows),
            "reviewed_group_count": len(np.unique(groups)),
            "oof_folds": YFP_OOF_SPLITS,
            "metrics": metrics,
            "bootstrap": intervals,
        }
        total_fits += oof.fit_count
        total_predictions += oof.prediction_count
    report = {
        "schema_version": "yfp_region_ordinal_report_v1",
        "dataset": "YFP",
        "claim": "static_region_severity_auxiliary_not_dynamic_110d",
        "eligible_manifest_sha256": manifest_digest,
        "feature_cache_manifest_sha256": _sha256_file(cache_manifest_path),
        "protocol": {
            "model": "three_class_proportional_odds_cumulative_logit",
            "C": FIXED_C,
            "standardization": "training_fold_only",
            "oof_splits": YFP_OOF_SPLITS,
            "optimizer": "L-BFGS-B",
            "optimizer_options": OPTIMIZER_OPTIONS,
            "bootstrap_repeats": BOOTSTRAP_REPEATS,
            "bootstrap_attempt_limit": BOOTSTRAP_ATTEMPT_LIMIT,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "counters": {"extractions": cache["aggregate"]["extractions"],
                     "fits": total_fits, "predictions": total_predictions},
        "targets": target_reports,
    }
    write_manifest_once(report, output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-cache-root", type=Path, required=True)
    args = parser.parse_args()
    report = run_yfp_region_ordinal(args.manifest, args.feature_cache_root)
    print(json.dumps({"counters": report["counters"], "targets": report["targets"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
