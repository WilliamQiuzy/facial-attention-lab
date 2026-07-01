"""EDITABLE model + training loop for the facial-palsy autoresearch loop.

This is the ONLY file the agent edits (the karpathy `train.py` analog). Everything
is fair game: architecture, feature engineering on the geometric stream, fusion,
the ordinal loss, optimizer, schedule, capacity. The fixed harness `prepare_fp.py`
owns the data, the leak-safe splits, and the metric — do not bypass them.

Contract: train one model per seed in prepare_fp.SEEDS, predict integer region
grades on prepare_fp.val_records("eyes"/"mouth") IN ORDER, and hand the per-seed
predictions to prepare_fp.report_metric(). It prints the `^metric:` line the loop
greps. Higher metric = better. Baseline to beat ~= 0.635.

--- v0: faithful reimplementation of the v2-attention baseline ---
Dual stream: frozen MARLIN vec (768) + BiGRU over the MediaPipe (T,72) sequence
with attention pooling -> concat -> trunk MLP -> per-task severity -> ordinal
cut-point heads. Multi-task routing: each sample's loss touches only its own head.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prepare_fp as P  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- config knobs (all editable) ----
CFG = dict(
    temporal_hidden=64,
    temporal_out=64,
    trunk_hidden=96,
    dropout=0.1,
    pool="attention",       # attention | mean | max  (GRU output pooling)
    lr=5e-4,
    weight_decay=3e-2,
    batch_size=128,
    epochs=P.MAX_EPOCHS,
    eval_every=4,
)
LW = {"binary": 0.5, "eyes": 0.3, "mouth": 0.3}   # per-task loss weights


# ---------------------------------------------------------------------------
# Ordinal cut-point head (CORAL-style), monotone thresholds by construction.
# ---------------------------------------------------------------------------
class OrderedThresholds(nn.Module):
    def __init__(self, k: int):
        super().__init__()
        self.theta0 = nn.Parameter(torch.zeros(1))
        self.gaps = nn.Parameter(torch.zeros(max(k - 2, 0)))

    def forward(self) -> torch.Tensor:
        if self.gaps.numel() == 0:
            return self.theta0
        return torch.cat([self.theta0, self.theta0 + torch.cumsum(F.softplus(self.gaps), 0)])


def ordinal_loss(s: torch.Tensor, thetas: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    cum = s.unsqueeze(1) - thetas.unsqueeze(0)                      # (B, k-1)
    kt = thetas.numel()
    tgt = (labels.unsqueeze(1) > torch.arange(kt, device=s.device).unsqueeze(0)).float()
    return F.binary_cross_entropy_with_logits(cum, tgt)


def predict_grades(s: torch.Tensor, thetas: torch.Tensor) -> torch.Tensor:
    cum = s.unsqueeze(1) - thetas.unsqueeze(0)
    return (torch.sigmoid(cum) > 0.5).sum(1)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class TemporalGRU(nn.Module):
    def __init__(self, fdim: int, hidden: int, out: int, pool: str):
        super().__init__()
        self.gru = nn.GRU(fdim, hidden, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(2 * hidden, out)
        self.att = nn.Linear(2 * hidden, 1) if pool == "attention" else None
        self.pool = pool

    def forward(self, seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(seq)                                     # (B, T, 2h)
        if self.pool == "attention":
            sc = self.att(out).squeeze(-1).masked_fill(~mask, -1e9)
            w = torch.softmax(sc, dim=1).unsqueeze(-1)
            pooled = (out * w).sum(1)
        elif self.pool == "max":
            pooled = out.masked_fill(~mask.unsqueeze(-1), -1e9).max(1).values
        else:
            m = mask.unsqueeze(-1).float()
            pooled = (out * m).sum(1) / m.sum(1).clamp(min=1)
        return self.proj(pooled)


class Net(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.temporal = TemporalGRU(P.MP_FEAT_DIM, cfg["temporal_hidden"], cfg["temporal_out"], cfg["pool"])
        d = 768 + cfg["temporal_out"]
        H = cfg["trunk_hidden"]
        self.trunk = nn.Sequential(nn.Linear(d, H), nn.ReLU(), nn.Dropout(cfg["dropout"]))
        self.sev = nn.ModuleDict({t: nn.Linear(H, 1, bias=False) for t in P.TASKS})
        self.thr = nn.ModuleDict({t: OrderedThresholds(P.N_CLASSES[t]) for t in P.TASKS})

    def rep(self, marlin, mp_seq, mp_mask):
        dyn = self.temporal(mp_seq, mp_mask)
        return self.trunk(torch.cat([marlin, dyn], dim=1))

    def severity(self, h, task):
        return self.sev[task](h).squeeze(-1)


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------
def predict_val(model: Net, val_in: dict) -> dict:
    model.eval()
    out = {}
    with torch.no_grad():
        for t in P.REGION_TASKS:
            b = P.make_batch(val_in[t], DEVICE)
            h = model.rep(b["marlin"], b["mp_seq"], b["mp_mask"])
            out[t] = predict_grades(model.severity(h, t), model.thr[t]()).cpu().numpy()
    return out


def train_one_seed(seed: int) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    train = P.train_records()
    val_in = {t: P.val_records(t) for t in P.REGION_TASKS}
    truth = {t: [r["label"] for r in val_in[t]] for t in P.REGION_TASKS}

    model = Net(CFG).to(DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG["epochs"])

    idx = np.arange(len(train))
    best, best_preds = -1.0, None
    for ep in range(CFG["epochs"]):
        model.train()
        rng.shuffle(idx)
        for st in range(0, len(idx), CFG["batch_size"]):
            recs = [train[i] for i in idx[st:st + CFG["batch_size"]]]
            b = P.make_batch(recs, DEVICE)
            h = model.rep(b["marlin"], b["mp_seq"], b["mp_mask"])
            loss = torch.zeros((), device=DEVICE)
            for t in P.TASKS:
                ti = [j for j, tt in enumerate(b["task"]) if tt == t]
                if not ti:
                    continue
                jj = torch.tensor(ti, device=DEVICE)
                s = model.severity(h.index_select(0, jj), t)
                loss = loss + LW[t] * ordinal_loss(s, model.thr[t](), b["label"].index_select(0, jj))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
        sched.step()
        if (ep + 1) % CFG["eval_every"] == 0 or ep == CFG["epochs"] - 1:
            preds = predict_val(model, val_in)
            m = 0.5 * (P.quadratic_kappa(truth["eyes"], preds["eyes"], 3)
                       + P.quadratic_kappa(truth["mouth"], preds["mouth"], 3))
            if m > best:
                best, best_preds = m, preds
    return best_preds


def main():
    t0 = time.time()
    preds_per_seed = [train_one_seed(s) for s in P.SEEDS]
    P.report_metric(preds_per_seed, extra={"train_seconds": round(time.time() - t0, 1)})


if __name__ == "__main__":
    main()
