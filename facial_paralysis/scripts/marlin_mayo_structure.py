"""Necessary-condition check: does frozen MARLIN produce STRUCTURED (non-collapsed)
features on our actual Mayo iPhone domain?

We have no Mayo HB labels yet, so we cannot test palsy discrimination here. But we
CAN test non-collapse + identity structure, which is exactly what failed for the
Oo encoder on this domain:
  - global spread: mean pairwise cosine over ALL clips. ~1.0 => collapsed.
  - within vs across take: clips of the same patient should be more similar to
    each other than to other patients (identity stability). within > across by a
    clear margin => the encoder captures real facial structure on iPhone data.

Run:
  KMP_DUPLICATE_LIB_OK=TRUE \
  /opt/homebrew/Caskroom/miniforge/base/envs/dev/bin/python scripts/marlin_mayo_structure.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data" / "livelinkface_data"
CACHE = ROOT / "outputs" / "marlin_probe" / "mayo_marlin.npz"


def encode(n_clips: int = 6) -> dict:
    import torch
    from src.models.backbones.marlin_video import MarlinVideoEncoder
    from utils import MediaPipeFaceLandmarker  # type: ignore

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    enc = MarlinVideoEncoder.from_default_weights().to(device).eval()
    lm = MediaPipeFaceLandmarker()

    X, take = [], []
    for vp in sorted(DATA.glob("*/*.mov")):
        tid = vp.parent.name
        embs = enc.encode_video_path(vp, n_clips=n_clips, landmarker=lm)
        if embs is None:
            print(f"  [skip] {tid}"); continue
        for e in embs:
            X.append(e); take.append(tid)
        print(f"  {tid}: {len(embs)} clips")
    X = np.stack(X); take = np.array(take)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, X=X, take=take)
    return {"X": X, "take": take}


def _within_across(X: np.ndarray, grp: np.ndarray):
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    S = Xn @ Xn.T
    n = len(Xn)
    within, across = [], []
    for i in range(n):
        for j in range(i + 1, n):
            (within if grp[i] == grp[j] else across).append(S[i, j])
    within, across = np.array(within), np.array(across)
    global_cos = float(S[np.triu_indices(n, 1)].mean())
    return global_cos, within, across


def analyze(X: np.ndarray, take: np.ndarray) -> None:
    takes = np.unique(take)
    print(f"\n{len(X)} clips from {len(takes)} Mayo takes, dim={X.shape[1]}")
    print("================= MAYO DOMAIN STRUCTURE =================")

    # CRITICAL: transformer embeddings are anisotropic — raw cosine is ~1.0 for
    # everything (the "cone effect"), which looks like collapse but is not. The
    # honest test centers the features first (same thing StandardScaler does in
    # the supervised probe), removing the dominant common direction.
    for tag, M in [("raw", X), ("centered", X - X.mean(0, keepdims=True))]:
        g, wi, ac = _within_across(M, take)
        print(f"  [{tag:8s}] global={g:+.3f}  within={wi.mean():+.3f}±{wi.std():.3f}  "
              f"across={ac.mean():+.3f}±{ac.std():.3f}  margin={wi.mean()-ac.mean():+.3f}")

    _, wi_c, ac_c = _within_across(X - X.mean(0, keepdims=True), take)
    margin = wi_c.mean() - ac_c.mean()
    var = X.var(axis=0)
    print(f"  feature variance: mean {var.mean():.4f}  median {np.median(var):.4f}  "
          f"frac<1e-6 {float((var < 1e-6).mean()):.2f}")
    verdict = ("STRUCTURED — strong patient separation after centering -> MARLIN "
               "does NOT collapse on the iPhone domain (necessary condition met)"
               if margin > 0.3 else
               "WEAK — little patient structure even after centering; investigate")
    print(f"  VERDICT (on CENTERED margin {margin:+.3f}): {verdict}")
    print("========================================================")


def main():
    if CACHE.exists():
        d = np.load(CACHE, allow_pickle=True)
        data = {"X": d["X"], "take": d["take"]}
    else:
        print("Encoding Mayo takes with frozen MARLIN...")
        data = encode()
    analyze(data["X"], data["take"])


if __name__ == "__main__":
    main()
