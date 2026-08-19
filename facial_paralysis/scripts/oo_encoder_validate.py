"""Encoder validation: do Oo's MLP-Mixer features place a clearly-healthy face
(Obama official portrait) and a clearly-palsy face (Wikimedia Bellspalsy.JPG,
James Heilman MD, CC BY-SA 3.0) at meaningfully different positions in 768-d
space, relative to our 14 Mayo takes?

If yes → the encoder is OK and the published classifier head is the broken
piece. We can confidently build our own HB head on top.
If no  → the encoder itself is not strong enough. We need a different backbone
(e.g. MARLIN, FaRL) for the downstream HB head.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.backbones import OoMLPMixerEncoder  # noqa: E402
from src.baselines.oo_multimodal.utils import MediaPipeFaceLandmarker  # type: ignore  # noqa: E402

OBAMA = ROOT / "src" / "baselines" / "oo_multimodal" / "input_images" / "healthy_portrait.jpg"
BELLSPALSY = ROOT / "assets" / "bellspalsy_wikimedia.jpg"
EMB_NPZ = ROOT / "outputs" / "oo_mlp_mixer_embeddings.npz"


def _cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    enc = OoMLPMixerEncoder.from_default_weights().to(device).eval()
    lm = MediaPipeFaceLandmarker()

    print("=== Encoding control faces ===")
    obama = enc.encode_image_path(OBAMA, landmarker=lm)
    palsy = enc.encode_image_path(BELLSPALSY, landmarker=lm)
    print(f"obama embed: norm={np.linalg.norm(obama):.2f}  std={obama.std():.3f}")
    print(f"palsy embed: norm={np.linalg.norm(palsy):.2f}  std={palsy.std():.3f}")
    print(f"cosine(OBAMA_healthy, BELLS_palsy) = {_cosine(obama, palsy):.3f}")

    # Compare each to our Mayo takes (already cached)
    print("\n=== Cosine of each control vs our 23 Mayo thumbnails ===")
    if not EMB_NPZ.exists():
        print(f"(no cache at {EMB_NPZ}; run scripts/extract_embeddings.py first)")
        return
    data = np.load(EMB_NPZ, allow_pickle=True)
    names = list(data["names"])
    X = data["embeddings"]
    # Drop OBAMA from the cached set if present, so we don't compare to ourselves
    drop = [i for i, n in enumerate(names) if n == "OBAMA_HEALTHY"]
    keep = [i for i in range(len(names)) if i not in drop]
    names = [names[i] for i in keep]
    X = X[keep]

    def _table(label, vec):
        print(f"\n  cosine({label}, *):")
        pairs = sorted(((names[i], _cosine(vec, X[i])) for i in range(len(names))),
                       key=lambda kv: kv[1], reverse=True)
        for n, c in pairs:
            grp = "FACES" if "FACES" in n else "MySlate"
            print(f"    {grp:<8s}  {n:<26s}  cos={c:.3f}")
        cs = np.array([c for _, c in pairs])
        print(f"    -> mean={cs.mean():.3f}  std={cs.std():.3f}  range=[{cs.min():.3f}, {cs.max():.3f}]")

    _table("OBAMA_healthy", obama)
    _table("BELLS_palsy",   palsy)

    # The key question: is Bells_palsy distinguishably closer to FACES (patient)
    # group than Obama is? Or vice versa?
    cos_obama  = np.array([_cosine(obama, X[i]) for i in range(len(names))])
    cos_palsy  = np.array([_cosine(palsy, X[i]) for i in range(len(names))])
    faces_idx   = [i for i, n in enumerate(names) if "FACES" in n]
    myslate_idx = [i for i, n in enumerate(names) if "FACES" not in n]
    print("\n=== Distinguishability summary ===")
    print(f"OBAMA  → FACES   mean cos = {cos_obama[faces_idx].mean():.3f}")
    print(f"OBAMA  → MySlate mean cos = {cos_obama[myslate_idx].mean():.3f}")
    print(f"BELLS  → FACES   mean cos = {cos_palsy[faces_idx].mean():.3f}")
    print(f"BELLS  → MySlate mean cos = {cos_palsy[myslate_idx].mean():.3f}")
    print(f"cosine(OBAMA, BELLS) = {_cosine(obama, palsy):.3f}")


if __name__ == "__main__":
    main()
