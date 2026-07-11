"""#4 self-supervised pretraining on unlabeled palsy-video blendshape trajectories.

The transferable signal is relative facial GEOMETRY/DYNAMICS (not appearance, which is
domain-confounded). So we pretrain a small Transformer encoder on blendshape trajectories
from curated YouTube palsy videos via masked autoencoding (MAE): mask random frames, predict
them from context. This learns the temporal structure of palsy facial movement WITHOUT labels;
the encoder is then ready to fine-tune on the few clinical labels when they arrive.

Trajectories: outputs/ssl_traj/<vid>.npz with key 'bs' (T,52). Extract via MediaPipe on the
curated video frames (pod). Runs on CPU (small model). Usage: python ssl_pretrain.py
"""
from __future__ import annotations
import glob, math, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
TRAJ = ROOT / "outputs" / "ssl_traj"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
D = 52                       # blendshape dim
SEQ = 32                     # training window length
MASK = 0.4                   # fraction of frames masked


def load_windows():
    """All fixed-length windows from every trajectory (z-scored per channel globally)."""
    seqs = []
    for f in glob.glob(str(TRAJ / "*.npz")):
        bs = np.load(f)["bs"].astype(np.float32)
        if bs.shape[0] >= 8:
            seqs.append(bs)
    if not seqs:
        return None, None, None
    allf = np.concatenate(seqs, 0)
    mu, sd = allf.mean(0), allf.std(0) + 1e-6
    windows = []
    for bs in seqs:
        bs = (bs - mu) / sd
        for st in range(0, max(1, bs.shape[0] - SEQ + 1), SEQ // 2):
            w = bs[st:st + SEQ]
            if w.shape[0] < SEQ:                       # pad tail
                w = np.pad(w, ((0, SEQ - w.shape[0]), (0, 0)))
            windows.append(w)
    return np.stack(windows), mu, sd


class TrajMAE(nn.Module):
    def __init__(self, d=D, dim=128, layers=3, heads=4):
        super().__init__()
        self.inp = nn.Linear(d, dim)
        self.mask_tok = nn.Parameter(torch.zeros(dim))
        pe = torch.zeros(SEQ, dim)
        pos = torch.arange(SEQ).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)
        enc = nn.TransformerEncoderLayer(dim, heads, dim * 2, batch_first=True, dropout=0.1)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.out = nn.Linear(dim, d)

    def forward(self, x, mask):
        h = self.inp(x)
        h = torch.where(mask.unsqueeze(-1), self.mask_tok, h) + self.pe
        h = self.enc(h)
        return self.out(h)

    def embed(self, x):                                # for downstream use: mean-pooled code
        return self.enc(self.inp(x) + self.pe).mean(1)


def main():
    W, mu, sd = load_windows()
    if W is None:
        print(f"no trajectories in {TRAJ} — extract them first (pod MediaPipe on curated videos)")
        return
    X = torch.tensor(W, device=DEV)
    print(f"{len(X)} windows of {SEQ}x{D} from {len(glob.glob(str(TRAJ/'*.npz')))} videos")
    model = TrajMAE().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(0)
    n = len(X); bs = min(64, n)
    for ep in range(60):
        idx = rng.permutation(n); tot = 0.0
        model.train()
        for st in range(0, n, bs):
            b = X[idx[st:st + bs]]
            m = torch.tensor(rng.random((b.shape[0], SEQ)) < MASK, device=DEV)
            m[:, 0] = False                            # keep an anchor
            pred = model(b, m)
            loss = ((pred - b)[m]).pow(2).mean()       # reconstruct masked frames only
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * b.shape[0]
        if ep % 10 == 0 or ep == 59:
            print(f"  epoch {ep:2d}  masked-recon MSE {tot/n:.4f}")
    out = ROOT / "outputs" / "ssl_encoder.pt"
    torch.save({"state": model.state_dict(), "mu": mu, "sd": sd,
                "cfg": {"D": D, "SEQ": SEQ}}, out)
    print(f"saved pretrained encoder -> {out}")


if __name__ == "__main__":
    main()
