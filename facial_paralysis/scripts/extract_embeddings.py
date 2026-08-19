"""Extract a 768-d MLP-Mixer face embedding for every take's thumbnail.jpg,
plus the Obama healthy control. Saves to outputs/oo_mlp_mixer_embeddings.npz
and prints sanity-check statistics (variance, pairwise cosine, etc.) so we
can verify the encoder produces non-degenerate signal before we build a
downstream HB head on top.

Run from project root:
    KMP_DUPLICATE_LIB_OK=TRUE conda run -n dev python scripts/extract_embeddings.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.backbones import OoMLPMixerEncoder  # noqa: E402
from src.baselines.oo_multimodal.utils import MediaPipeFaceLandmarker  # type: ignore  # noqa: E402


DATA_DIR = ROOT / "data" / "livelinkface_data"
OUT_DIR = ROOT / "outputs"
OBAMA = ROOT / "src" / "baselines" / "oo_multimodal" / "input_images" / "healthy_portrait.jpg"


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"device: {device}")

    encoder = OoMLPMixerEncoder.from_default_weights().to(device).eval()
    landmarker = MediaPipeFaceLandmarker()

    inputs: list[tuple[str, Path]] = []
    # Mayo takes — one frame per take (the auto-generated thumbnail)
    for take in sorted(p.name for p in DATA_DIR.iterdir() if p.is_dir()):
        thumb = DATA_DIR / take / "thumbnail.jpg"
        if thumb.exists():
            inputs.append((take, thumb))
    # Public healthy control
    if OBAMA.exists():
        inputs.append(("OBAMA_HEALTHY", OBAMA))

    names: list[str] = []
    embeddings: list[np.ndarray] = []
    times = []
    print(f"\n{'name':<26s}  {'group':<8s}  {'norm':>7s}  {'std':>6s}  {'ms':>5s}")
    print("-" * 60)
    for name, path in inputs:
        t0 = time.perf_counter()
        emb = encoder.encode_image_path(path, landmarker=landmarker)
        dt = (time.perf_counter() - t0) * 1000
        if emb is None:
            print(f"{name:<26s}  NO_FACE")
            continue
        group = ("OBAMA" if name == "OBAMA_HEALTHY"
                 else "FACES" if "FACES" in name
                 else "MySlate")
        names.append(name)
        embeddings.append(emb)
        times.append(dt)
        print(f"{name:<26s}  {group:<8s}  {np.linalg.norm(emb):>7.2f}  "
              f"{emb.std():>6.3f}  {dt:>5.0f}")

    if not embeddings:
        print("No embeddings extracted.")
        return

    X = np.stack(embeddings, axis=0)  # (N, 768)
    print(f"\nembedding matrix: {X.shape}")
    print(f"per-feature std:  mean={X.std(axis=0).mean():.3f}  "
          f"min={X.std(axis=0).min():.3f}  max={X.std(axis=0).max():.3f}")
    print(f"embedding norms:  mean={np.linalg.norm(X,axis=1).mean():.2f}  "
          f"min={np.linalg.norm(X,axis=1).min():.2f}  "
          f"max={np.linalg.norm(X,axis=1).max():.2f}")

    # Sanity check: cosine distances. If the encoder is non-degenerate,
    # different subjects should land at different points in 768-d space.
    n = len(embeddings)
    cos = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cos[i, j] = _cosine(X[i], X[j])
    off_diag = cos[~np.eye(n, dtype=bool)]
    print(f"\npairwise cosine (off-diagonal): "
          f"mean={off_diag.mean():.3f}  min={off_diag.min():.3f}  "
          f"max={off_diag.max():.3f}  std={off_diag.std():.3f}")
    print("  (mean ≈ 1.0 = collapsed; mean ≈ 0.0 = decorrelated; "
          "we want something in between with non-trivial std)")

    # If Obama exists, print Obama's cosine to each Mayo take — useful to see
    # whether the encoder treats him as an outlier vs the Mayo distribution.
    if "OBAMA_HEALTHY" in names:
        oi = names.index("OBAMA_HEALTHY")
        print(f"\ncosine(OBAMA, *):")
        ranked = sorted(
            ((names[j], cos[oi, j]) for j in range(n) if j != oi),
            key=lambda kv: kv[1], reverse=True,
        )
        for name, c in ranked[:5]:
            print(f"  closest #{ranked.index((name,c))+1:>2}: {name:<26s}  cos={c:.3f}")
        for name, c in ranked[-3:]:
            print(f"  farthest:        {name:<26s}  cos={c:.3f}")

    OUT_DIR.mkdir(exist_ok=True)
    npz = OUT_DIR / "oo_mlp_mixer_embeddings.npz"
    np.savez(npz, names=np.array(names), embeddings=X)
    print(f"\nsaved → {npz}")
    print(f"mean latency per image: {np.mean(times):.0f} ms "
          f"(first image excluded from typical: median={np.median(times):.0f} ms)")


if __name__ == "__main__":
    main()
