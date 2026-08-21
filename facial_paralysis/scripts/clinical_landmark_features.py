"""Compatibility CLI for the production clinical-landmark transform."""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing.clinical_landmarks import (  # noqa: E402
    LEGACY_CLINICAL_LANDMARK_NAMES,
    legacy_clinical23_v1_features,
)

NAMES = list(LEGACY_CLINICAL_LANDMARK_NAMES)


def clinical_feats(xy, w, h):
    """Frozen legacy return shape: ``(feature_vector, feature_names)``."""
    return legacy_clinical23_v1_features(xy, w, h), NAMES


if __name__ == "__main__":
    d = json.loads(Path(sys.argv[1]).read_text())
    k = next(iter(d)); e = d[k]
    v, names = clinical_feats(np.array(e["xy"]), e["w"], e["h"])
    for n, val in zip(names, v):
        print(f"  {n:16s} {val:+.3f}")
