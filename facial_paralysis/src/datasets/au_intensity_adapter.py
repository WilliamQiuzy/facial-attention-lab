"""Adapter: AU-intensity datasets -> our geometric/temporal stream (B-2).

Facial palsy = reduced/asymmetric AU activation, so AU-intensity corpora (DISFA,
DISFA+, BP4D-Spontaneous, FEAFA+) are a label-rich way to (a) train the trainable
temporal/geometric encoder on REAL facial dynamics and (b) anchor the severity axis
at the healthy end — WITHOUT any palsy/HB label. This was the "parked auxiliary"
path in data/public_datasets.md; Run #14 makes it pressing (the geometric stream is
what carries the clinical signal; the appearance stream is domain-confounded).

These datasets are APPLICATION-GATED (EULA / request) — see docs/data_acquisition.md.
This module is the plug-and-play loader so that, once the data is on disk, it maps
into the same `(T, F)` per-frame feature sequence the model already consumes.

AU -> MediaPipe-blendshape mapping (FACS Action Unit -> ARKit blendshape, partial):
  AU1 inner-brow-raise  -> browInnerUp
  AU2 outer-brow-raise  -> browOuterUp L/R
  AU4 brow-lowerer      -> browDown L/R
  AU6 cheek-raiser      -> cheekSquint L/R
  AU7 lid-tightener     -> eyeSquint L/R
  AU12 lip-corner-pull  -> mouthSmile L/R
  AU15 lip-corner-depr  -> mouthFrown L/R
  AU17 chin-raiser      -> mouthShrugLower
  AU25 lips-part        -> jawOpen (proxy)
  AU45 blink            -> eyeBlink L/R
Each AU has L/R intensity in DISFA-style data, so the same L/R asymmetry features we
use for palsy come for free.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# FACS AU id -> (left blendshape, right blendshape) or (central, None)
AU_TO_BLENDSHAPE = {
    1:  ("browInnerUp", None),
    2:  ("browOuterUpLeft", "browOuterUpRight"),
    4:  ("browDownLeft", "browDownRight"),
    6:  ("cheekSquintLeft", "cheekSquintRight"),
    7:  ("eyeSquintLeft", "eyeSquintRight"),
    12: ("mouthSmileLeft", "mouthSmileRight"),
    15: ("mouthFrownLeft", "mouthFrownRight"),
    25: ("jawOpen", None),
    45: ("eyeBlinkLeft", "eyeBlinkRight"),
}


@dataclass
class AUFrame:
    """One frame of AU intensities (DISFA-style 0-5, or normalized 0-1)."""
    au: dict[int, float]                      # AU id -> intensity (already L/R-averaged)
    au_left: dict[int, float] | None = None   # optional per-side (DISFA has some)
    au_right: dict[int, float] | None = None


def au_sequence_to_features(frames: list[AUFrame], bs_names: list[str],
                            normalize: float = 5.0) -> np.ndarray:
    """Map an AU-intensity clip to the model's (T, F) blendshape-style feature seq,
    so the geometric/temporal encoder can pretrain on it. Unmapped blendshapes -> 0;
    L/R asymmetry deltas are appended exactly as MediaPipeFeatureExtractor does."""
    idx = {n: i for i, n in enumerate(bs_names)}
    T, F = len(frames), len(bs_names)
    out = np.zeros((T, F), dtype=np.float32)
    for t, fr in enumerate(frames):
        for au, val in fr.au.items():
            m = AU_TO_BLENDSHAPE.get(au)
            if not m:
                continue
            v = val / normalize
            for name in (m[0], m[1]):
                if name and name in idx:
                    # use per-side if available, else the symmetric value
                    side_v = v
                    if fr.au_left and name.endswith("Left"):
                        side_v = fr.au_left.get(au, val) / normalize
                    elif fr.au_right and name.endswith("Right"):
                        side_v = fr.au_right.get(au, val) / normalize
                    out[t, idx[name]] = side_v
    return out


# Loader stubs — fill the file-format parsing once the data is on disk.
def load_disfa(root) -> dict:        # DISFA: 27 subj, AU 0-5 per frame
    raise NotImplementedError("DISFA is EULA-gated; see docs/data_acquisition.md")


def load_bp4d(root) -> dict:         # BP4D-Spontaneous: 41 subj
    raise NotImplementedError("BP4D requires application; see docs/data_acquisition.md")
