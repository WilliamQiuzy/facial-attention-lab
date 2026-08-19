"""#2: AU-dynamics pretraining pipeline for the geometric/temporal encoder.

Palsy = reduced/asymmetric AU activation, so FACS AU-intensity corpora (DISFA / BP4D /
FEAFA+) let us pretrain the TRAINABLE temporal encoder on REAL facial dynamics and
anchor the healthy (symmetric) end — with NO palsy/HB label. Builds on
`src/datasets/au_intensity_adapter.py`.

Pipeline:
  AUFrame sequences  --adapter-->  (T,72) feature seq [52 blendshapes + 20 L/R deltas]
                     --masked reconstruction-->  pretrained BiGRU temporal encoder
  save encoder state_dict -> warm-start FacialPalsyModel / runner.Net geometric stream.

The DISFA/BP4D loaders are EULA-gated (docs/data_acquisition.md) — filled once data lands.
This script runs end-to-end on SYNTHETIC AU data to verify the pipeline is ready.

Usage:  python au_pretrain.py            # synthetic self-test
        python au_pretrain.py --disfa <dir>   # once data is on disk
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.datasets.au_intensity_adapter import AUFrame, au_sequence_to_features, AU_TO_BLENDSHAPE

# 52 ARKit blendshape names (verified order) + their L/R mirror pairs
BS_NAMES = ["_neutral","browDownLeft","browDownRight","browInnerUp","browOuterUpLeft","browOuterUpRight",
    "cheekPuff","cheekSquintLeft","cheekSquintRight","eyeBlinkLeft","eyeBlinkRight","eyeLookDownLeft",
    "eyeLookDownRight","eyeLookInLeft","eyeLookInRight","eyeLookOutLeft","eyeLookOutRight","eyeLookUpLeft",
    "eyeLookUpRight","eyeSquintLeft","eyeSquintRight","eyeWideLeft","eyeWideRight","jawForward","jawLeft",
    "jawOpen","jawRight","mouthClose","mouthDimpleLeft","mouthDimpleRight","mouthFrownLeft","mouthFrownRight",
    "mouthFunnel","mouthLeft","mouthLowerDownLeft","mouthLowerDownRight","mouthPressLeft","mouthPressRight",
    "mouthPucker","mouthRight","mouthRollLower","mouthRollUpper","mouthShrugLower","mouthShrugUpper",
    "mouthSmileLeft","mouthSmileRight","mouthStretchLeft","mouthStretchRight","mouthUpperUpLeft",
    "mouthUpperUpRight","noseSneerLeft","noseSneerRight"]
PAIRS = [(i, BS_NAMES.index(n.replace("Left", "Right"))) for i, n in enumerate(BS_NAMES)
         if n.endswith("Left") and n.replace("Left", "Right") in BS_NAMES]
MP_FEAT_DIM = 72


def build_features(frames: list[AUFrame]) -> np.ndarray:
    """AUFrame seq -> (T,72): 52 blendshape cols + 20 L/R asymmetry deltas (padded)."""
    bs = au_sequence_to_features(frames, BS_NAMES)                 # (T,52)
    deltas = np.stack([bs[:, l] - bs[:, r] for l, r in PAIRS], 1)  # (T, n_pairs)
    d = np.zeros((bs.shape[0], MP_FEAT_DIM - 52), np.float32)
    d[:, :min(deltas.shape[1], d.shape[1])] = deltas[:, :d.shape[1]]
    return np.concatenate([bs, d], 1).astype(np.float32)          # (T,72)


class MaskedPretrainer(nn.Module):
    """BiGRU temporal encoder + linear decoder; masked-frame reconstruction (self-supervised)."""
    def __init__(self, fdim=MP_FEAT_DIM, hidden=64):
        super().__init__()
        self.gru = nn.GRU(fdim, hidden, batch_first=True, bidirectional=True)
        self.dec = nn.Linear(2 * hidden, fdim)

    def forward(self, x):                       # x (B,T,F)
        h, _ = self.gru(x)
        return self.dec(h)                      # reconstruct per-frame features


def synthetic_au(n=400, tmin=8, tmax=24, seed=0):
    """Mostly-symmetric 'healthy' AU clips (anchor the low-asymmetry end)."""
    rng = np.random.default_rng(seed)
    seqs = []
    for _ in range(n):
        T = int(rng.integers(tmin, tmax))
        frames = []
        for _ in range(T):
            au = {a: float(rng.uniform(0, 4)) for a in AU_TO_BLENDSHAPE}
            jitter = {a: v * (1 + rng.normal(0, 0.05)) for a, v in au.items()}  # small L/R diff = healthy
            frames.append(AUFrame(au=au, au_left=au, au_right=jitter))
        seqs.append(build_features(frames))
    return seqs


def pretrain(seqs, epochs=15, hidden=64, mask=0.3, seed=0):
    torch.manual_seed(seed)
    model = MaskedPretrainer(hidden=hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    Tmax = max(s.shape[0] for s in seqs)
    X = np.zeros((len(seqs), Tmax, MP_FEAT_DIM), np.float32)
    M = np.zeros((len(seqs), Tmax), bool)
    for i, s in enumerate(seqs):
        X[i, :s.shape[0]] = s; M[i, :s.shape[0]] = True
    X = torch.tensor(X); M = torch.tensor(M)
    rng = np.random.default_rng(seed)
    losses = []
    for ep in range(epochs):
        idx = rng.permutation(len(seqs)); tot = 0.0
        for st in range(0, len(idx), 64):
            b = torch.tensor(idx[st:st + 64])
            xb, mb = X[b], M[b]
            drop = (torch.rand_like(xb[..., 0]) < mask) & mb        # frames to mask
            xin = xb.clone(); xin[drop] = 0.0
            out = model(xin)
            loss = ((out - xb) ** 2)[drop].mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss)
        losses.append(tot)
    return model, losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--disfa"); ap.add_argument("--bp4d")
    a = ap.parse_args()
    if a.disfa or a.bp4d:
        from src.datasets.au_intensity_adapter import load_disfa, load_bp4d
        raw = load_disfa(a.disfa) if a.disfa else load_bp4d(a.bp4d)   # returns {clip: [AUFrame,...]}
        seqs = [build_features(v) for v in raw.values()]
        src = a.disfa or a.bp4d
    else:
        print("No AU data given -> SYNTHETIC self-test (pipeline verification).")
        seqs = synthetic_au(); src = "synthetic"
    print(f"{len(seqs)} AU clips -> features {seqs[0].shape} (T,72); {len(PAIRS)} L/R pairs")
    model, losses = pretrain(seqs)
    print(f"pretrain recon MSE: {losses[0]:.3f} -> {losses[-1]:.3f} ({'OK, decreasing' if losses[-1] < losses[0] else 'NOT learning'})")
    out = ROOT / "outputs" / "au_pretrain"; out.mkdir(parents=True, exist_ok=True)
    torch.save({"gru": model.gru.state_dict(), "source": src, "feat_dim": MP_FEAT_DIM}, out / "geo_encoder.pt")
    print(f"saved pretrained BiGRU -> {out/'geo_encoder.pt'} (warm-starts runner.GeoGRU / temporal stream)")


if __name__ == "__main__":
    main()
