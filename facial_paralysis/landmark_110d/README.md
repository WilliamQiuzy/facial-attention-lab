# Landmark 110D facial-paralysis research model

This directory is the collaboration-ready public source package for the current
facial-paralysis **development champion**. It converts MediaPipe Face Mesh
landmarks into a fixed 110-dimensional recording vector and trains a small,
fixed L2-logistic classifier.

The performance gain comes from dynamic clinical geometry, not a deep neural
classifier:

- 23 clinical eye, brow, and mouth measurements per frame;
- four statistics per channel: median, IQR, range, and maximum absolute
  velocity per second (`23 × 4 = 92`);
- correlation, invariant amplitude ratio, and best lag for six bilateral pairs
  (`6 × 3 = 18`);
- `92 + 18 = 110` dimensions.

## Current evidence

| Development candidate | AUROC | Balanced accuracy | Sensitivity | Specificity |
| --- | ---: | ---: | ---: | ---: |
| Nuisance-only | 0.768 | 0.716 | 0.667 | 0.765 |
| 58D Clinical Dynamics | 0.922 | 0.857 | 0.714 | 1.000 |
| Clinical Dynamics + Nuisance | 0.913 | 0.881 | 0.762 | 1.000 |
| **110D Landmark** | **0.938** | **0.905** | **0.810** | **1.000** |

These are grouped inner out-of-fold results on the PalsyNet development
partition: 39 recordings / 38 provisional groups, with 21 affected and 17
unaffected groups. Ten protected recordings remain unused. Identity status is
still unreviewed, so the claim is video-held-out rather than patient-held-out.
This is not HB accuracy, Mayo clinical accuracy, external validation, or
deployment evidence. See [MODEL_CARD.md](MODEL_CARD.md) and the
[machine-readable result](results/current_development_model.json).

## Install

Python 3.9 or newer is required.

```bash
python -m pip install -e .
```

## Use the feature and model contracts

```python
import numpy as np

from landmark110d import Landmark110DEstimator, build_110d_features

# One 432-frame recording: the four frozen, evenly spaced 32-frame windows.
clinical23 = np.zeros((4, 32, 23), dtype=np.float32)
valid_mask = np.ones((4, 32), dtype=bool)
timestamps = np.stack([w * 10 + np.arange(32) / 30 for w in range(4)])
source_indices = np.stack([
    start + np.arange(32) for start in (0, 133, 266, 400)
])

row = build_110d_features(
    clinical23,
    valid_mask,
    timestamps,
    source_indices,
    source_frame_count=432,
)

# Fit only on governed data with nonempty, subject-disjoint string group IDs.
x = np.stack([row, row + 0.01, row + 0.02, row + 0.03])
y = np.asarray([0, 0, 1, 1])
groups = np.asarray(["subject-a", "subject-b", "subject-c", "subject-d"])
model = Landmark110DEstimator().fit(x, y, groups)
probability = model.predict_proba(x)
```

`clinical23_from_mediapipe(...)` converts one MediaPipe 478-point face mesh to
the frozen 23-channel geometry. Capture-side names deliberately use MediaPipe
mesh anchors; patient left/right cannot be inferred until mirror provenance is
known.

## Verify

```bash
python -m unittest discover -s tests -v
```

The test suite locks the 23D and 110D dimensions, ordered feature names,
window-boundary behavior, bilateral statistics, fixed classifier parameters,
group balancing, serialization, and the public result boundary.

## Why no trained weights are committed

The current result is a four-fold development evaluation, not a single
deployment checkpoint. A public full-development scaler/classifier artifact is
intentionally withheld until PalsyNet identity and derived-model licensing are
reviewed. This repository contains no Mayo recordings, patient-derived
features, per-record predictions, identifiers, or clinical labels.

The estimator supports transparent JSON-safe serialization for a collaborator
who fits it on an appropriately governed cohort. Do not publish a fitted
artifact without completing the relevant data and model-governance review.

## Provenance

- Canonical source branch: `codex/action-clinical-geometry`
- Canonical source commit: `632bf993a8d38a7426fc52b23923e1d8d14dd857`
- Implementation experiment: `action-clinical-geometry-v1`
- Original implementation commit: `e3b069cfef3634b8f210b4315dd574d8b9fa46a6`

No open-source license grant is included in this repository yet.
