"""Sanity check: within-take cosine should beat across-take cosine.
If frames of the same subject are more similar to each other than to frames of
different subjects, the encoder has at least basic identity stability — a
necessary (not sufficient) property for using it as a downstream backbone.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EMB_DIR = ROOT / "outputs" / "embeddings"


def main():
    files = sorted(EMB_DIR.glob("*.npz"))
    print(f"loaded {len(files)} per-take .npz files\n")

    takes: dict[str, np.ndarray] = {}
    for f in files:
        d = np.load(f)
        emb = d["embeddings"]
        # L2 normalize for clean cosine reasoning
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        takes[f.stem] = emb

    # within-take: average pairwise cosine inside one .npz
    within = {}
    for take, E in takes.items():
        if E.shape[0] < 2:
            continue
        sim = E @ E.T  # already L2-normalized → cosine
        mask = ~np.eye(E.shape[0], dtype=bool)
        within[take] = float(sim[mask].mean())

    # across-take: average cosine between all frames of take A and all of take B
    take_list = sorted(takes)
    pair_means = []
    for i, a in enumerate(take_list):
        for b in take_list[i + 1:]:
            sim = takes[a] @ takes[b].T
            pair_means.append((a, b, float(sim.mean())))

    print(f"{'take':<26s}  {'within-cos':>10s}")
    print("-" * 40)
    for take, c in sorted(within.items(), key=lambda kv: -kv[1]):
        print(f"{take:<26s}  {c:>10.3f}")

    w = np.array(list(within.values()))
    a = np.array([m for _, _, m in pair_means])
    print(f"\nwithin-take mean cosine:  mean={w.mean():.3f}  std={w.std():.3f}")
    print(f"across-take mean cosine:  mean={a.mean():.3f}  std={a.std():.3f}")
    print(f"\nseparation (within - across) = {w.mean() - a.mean():.3f}")
    if w.mean() > a.mean():
        print("encoder IS identity-stable: same-subject frames cluster tighter than different subjects.")
    else:
        print("encoder is NOT identity-stable: different subjects look as similar as same subject. "
              "Backbone needs replacement.")


if __name__ == "__main__":
    main()
