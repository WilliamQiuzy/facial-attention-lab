"""Validate the mediapipe-free cv2 face crop against the cached MediaPipe-crop
MARLIN embeddings, in the MEAN-CENTERED space.

Raw cosine between any two Mayo face clips is ~0.99 (MARLIN embeddings share a
huge common face/domain component), so it cannot tell crops apart. The
discriminative signal lives in the mean-centered embedding (cf. the feasibility
gate's "centered identity margin"). This script re-encodes each cached take with
the cv2 crop, centers both sets by the cached mean, and asks: is cv2(take_i)
nearest to cached(take_i) among all cached takes? High rank-1 retrieval ⇒ cv2
crop preserves the per-take signal and is a valid MediaPipe-free substitute.

Run:  KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/mayo_cv2crop_validate.py
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.backbones.marlin_video import MarlinVideoEncoder  # noqa: E402
from src.preprocessing.face_crop_cv2 import encode_video_cv2  # noqa: E402

LLF = ROOT / "data" / "livelinkface_data"
MAYO_BUNDLES = ROOT / "outputs" / "mayo_bundles"


def find_mov(take: str) -> str | None:
    movs = glob.glob(str(LLF / take / "*.mov"))
    return movs[0] if movs else None


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main():
    enc = MarlinVideoEncoder.from_default_weights().eval()
    takes, cached, mine = [], [], []
    for npz in sorted(MAYO_BUNDLES.glob("2026*/clip.npz")):
        take = npz.parent.name
        mov = find_mov(take)
        if mov is None:
            continue
        cv = np.load(npz)["marlin"].mean(0)            # cached mediapipe-crop, pooled
        m = encode_video_cv2(enc, mov, n_clips=4)      # streams only sampled frames
        if m is None:
            print(f"  [skip] {take}: cv2 found no face", flush=True)
            continue
        takes.append(take); cached.append(cv); mine.append(m.mean(0))
        print(f"  encoded {take}", flush=True)

    cached = np.stack(cached); mine = np.stack(mine)
    mu = cached.mean(0)                                 # center by cached mean
    c_c = cached - mu
    m_c = mine - mu

    n = len(takes)
    rank1 = 0
    same_cos, other_cos = [], []
    print(f"\n{'take':<26s} {'same-take':>9s} {'best-other':>10s} {'rank':>5s}")
    for i in range(n):
        sims = np.array([cos(m_c[i], c_c[j]) for j in range(n)])
        order = np.argsort(-sims)
        rank = int(np.where(order == i)[0][0]) + 1
        rank1 += (rank == 1)
        same = sims[i]
        best_other = max(sims[j] for j in range(n) if j != i)
        same_cos.append(same); other_cos.append(best_other)
        print(f"{takes[i]:<26s} {same:9.3f} {best_other:10.3f} {rank:5d}")

    res = {
        "n_takes": n,
        "rank1_retrieval_acc": round(rank1 / n, 3),
        "mean_centered_cos_same_take": round(float(np.mean(same_cos)), 3),
        "mean_centered_cos_best_other": round(float(np.mean(other_cos)), 3),
        "interpretation": (
            "cv2 crop preserves per-take signal if rank1 acc is high and "
            "same-take centered cosine clearly exceeds best-other."
        ),
        "takes": takes,
    }
    print(f"\nrank-1 retrieval acc: {res['rank1_retrieval_acc']}  "
          f"(same {res['mean_centered_cos_same_take']} vs best-other "
          f"{res['mean_centered_cos_best_other']})")
    out = MAYO_BUNDLES / "cv2crop_validation.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
