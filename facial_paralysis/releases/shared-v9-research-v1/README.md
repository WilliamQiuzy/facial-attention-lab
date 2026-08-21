# Shared V9 research release

This directory is the complete public inference bundle for the locked research
model `BLV9-009` (Masked Clinical Reconstruction). It contains three H200-fitted
members, the common 110D scaler embedded in every member, and a checksum-bound
manifest. A normal Git clone is sufficient; Git LFS and access to H200 are not
required.

## Load

```python
from pathlib import Path
from src.deployment.shared_v9_research_release import load_release

predictor = load_release(
    Path("releases/shared-v9-research-v1"),
    device="cpu",  # or "cuda"
)
prediction = predictor.predict("scripted_three_action", request_arrays)
print(prediction.probability, prediction.member_probabilities)
```

`request_arrays` must follow the exact protocol tensor contract enforced by
`validate_request_arrays` in `src/deployment/shared_v8_release.py`. The final
probability is the arithmetic mean of the three member probabilities at the
fixed threshold 0.5.

## Files

- `manifest.json`: model identity, source-level training commitments, sizes,
  SHA-256 checksums, and claim boundaries.
- `weights-seed0.npz`, `weights-seed1.npz`, `weights-seed2.npz`: tensor-only
  NumPy archives (`allow_pickle=False`), including the identical frozen scaler.
- `acceptance_summary.json`: local CPU/H200 GPU parity, hash/load checks, and
  explicit clinical and privacy claim boundaries.

These weights were fitted on the exposed participant-disjoint development
sources only: 38 PalsyNet, 36 NeuroFace, and 56 MEEI participants. The release
contains no raw media, participant identifiers, labels, per-participant features
or predictions, Mayo data, or private manifest paths.

This is not a clinically validated diagnostic model. It has not been trained or
evaluated on Mayo labels and does not predict House-Brackmann grade.
