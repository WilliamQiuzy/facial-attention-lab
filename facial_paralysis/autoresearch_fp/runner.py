"""Experiment engine for the autoresearch-FP loop (superset of train_fp.py).

Same metric contract as train_fp.py (per-seed preds -> prepare_fp.report_metric),
but exposes the STRUCTURAL levers Run #17 never explored, selectable by a JSON
config so many genuinely-different models are directly comparable:

  feat        raw | asym            engineered nonlinear asymmetry features
  geo_encoder gru | mlp             sequence GRU vs static MLP (metric data is T=1)
  fusion      concat | gate         per-region learned MARLIN/geo gate
  loss        coral | corn | *_cw   ordinal loss; _cw = inverse-freq class weighting
  ls          0.0..                 ordinal label smoothing
  decode      threshold | expected  grade readout
  marlin_proj None | int            low-rank MARLIN projection + LayerNorm (rebalance)
  trunk_hidden / trunk_layers / dropout / lr / weight_decay / warmup

Default CFG reproduces the v0 baseline (dual-stream BiGRU-attention + CORAL).
Usage: python runner.py '<json cfg overrides>'   (or a path to a .json file)
The asymmetry block is dims [52:72] of the 72-d geo vector (signed L-R deltas).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prepare_fp as P  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ASYM_LO, ASYM_HI = 52, 72          # signed L/R asymmetry delta block within the 72-d geo vector

DEFAULT = dict(
    feat="raw", geo_encoder="gru", fusion="concat", loss="coral", ls=0.0,
    decode="threshold", marlin_proj=None, marlin_ln=False, per_region_trunk=False,
    per_region_geo=False,
    temporal_hidden=64, temporal_out=64,
    trunk_hidden=96, trunk_layers=1, dropout=0.1, pool="attention",
    lr=5e-4, weight_decay=3e-2, batch_size=128, warmup=0,
    marlin_noise=0.0, geo_noise=0.0,
    epochs=P.MAX_EPOCHS, eval_every=4,
)
LW = {"binary": 0.5, "eyes": 0.3, "mouth": 0.3}


# ---------------------------------------------------------------------------
# Feature engineering on the geometric stream
# ---------------------------------------------------------------------------
# MediaPipe blendshape L/R pairs (idx0 = _neutral) grouped by clinical region.
EYE_PAIRS = [(9, 10), (19, 20), (21, 22)]                    # eyeBlink, eyeSquint, eyeWide
BROW_PAIRS = [(1, 2), (4, 5)]                                # browDown, browOuterUp
CHEEK_PAIRS = [(7, 8)]                                       # cheekSquint
MOUTH_PAIRS = [(44, 45), (30, 31), (28, 29), (34, 35), (48, 49), (46, 47)]  # smile,frown,dimple,lowerDown,upperUp,stretch
REGION_PAIRS = [EYE_PAIRS, MOUTH_PAIRS, BROW_PAIRS, CHEEK_PAIRS]


def _pair_feats(bs: torch.Tensor, pairs) -> torch.Tensor:
    """Per-pair |L-R| and scale-invariant asymmetry ratio |L-R|/(L+R), + region sum/max."""
    feats, absd = [], []
    for l, r in pairs:
        L, R = bs[..., l:l + 1], bs[..., r:r + 1]
        ad = (L - R).abs()
        feats += [ad, ad / (L + R + 1e-3)]                   # magnitude + clinical asymmetry index
        absd.append(ad)
    A = torch.cat(absd, -1)
    feats += [A.sum(-1, keepdim=True), A.amax(-1, keepdim=True)]
    return torch.cat(feats, -1)                              # 2*len(pairs)+2


def engineer(mp_seq: torch.Tensor, cfg: dict) -> torch.Tensor:
    """Append nonlinear asymmetry invariants a linear layer can't derive.
    feat='asym': global aggregates. feat='regasym': + per-region (eye/mouth/brow/cheek)
    per-pair |L-R| and scale-invariant asymmetry ratios, so the eye head gets an explicit
    eye-closure asymmetry signal (the piece that lost MARLIN)."""
    if cfg["feat"] not in ("asym", "regasym"):
        return mp_seq
    d = mp_seq[..., ASYM_LO:ASYM_HI]                          # (B,T,20) signed deltas
    ad = d.abs()
    parts = [mp_seq, ad, ad.mean(-1, keepdim=True), ad.amax(-1, keepdim=True),
             ad.sum(-1, keepdim=True), (d ** 2).mean(-1, keepdim=True)]
    if cfg["feat"] == "regasym":
        bs = mp_seq[..., :52]
        parts += [_pair_feats(bs, pairs) for pairs in REGION_PAIRS]
    return torch.cat(parts, dim=-1)


def geo_feat_dim(cfg: dict) -> int:
    d = P.MP_FEAT_DIM + (24 if cfg["feat"] in ("asym", "regasym") else 0)  # 20 |d| + 4 aggregates
    if cfg["feat"] == "regasym":
        d += sum(2 * len(p) + 2 for p in REGION_PAIRS)       # per-region pair feats
    return d


# ---------------------------------------------------------------------------
# Ordinal heads / losses
# ---------------------------------------------------------------------------
class OrderedThresholds(nn.Module):
    def __init__(self, k: int):
        super().__init__()
        self.theta0 = nn.Parameter(torch.zeros(1))
        self.gaps = nn.Parameter(torch.zeros(max(k - 2, 0)))

    def forward(self):
        if self.gaps.numel() == 0:
            return self.theta0
        return torch.cat([self.theta0, self.theta0 + torch.cumsum(F.softplus(self.gaps), 0)])


def ordinal_loss(s, thetas, labels, cfg, w=None):
    kt = thetas.numel()
    cum = s.unsqueeze(1) - thetas.unsqueeze(0)                # (B, k-1)
    tgt = (labels.unsqueeze(1) > torch.arange(kt, device=s.device).unsqueeze(0)).float()
    if cfg["ls"] > 0:                                         # ordinal label smoothing
        tgt = tgt * (1 - cfg["ls"]) + 0.5 * cfg["ls"]
    per = F.binary_cross_entropy_with_logits(cum, tgt, reduction="none")  # (B,k-1)
    if cfg["loss"] == "corn":                                # conditional: only rows where y>=k count
        cond = (labels.unsqueeze(1) >= torch.arange(kt, device=s.device).unsqueeze(0)).float()
        per = per * cond
        denom = cond.sum().clamp(min=1)
        base = per.sum() / denom
    else:
        base = per.mean(1)                                   # (B,)
        if w is not None:
            base = (base * w).sum() / w.sum().clamp(min=1)
        else:
            base = base.mean()
    return base


def predict_grades(s, thetas, cfg):
    cum = torch.sigmoid(s.unsqueeze(1) - thetas.unsqueeze(0))
    if cfg["decode"] == "expected":
        return cum.sum(1).round().clamp(0, thetas.numel())
    return (cum > 0.5).sum(1)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class GeoGRU(nn.Module):
    def __init__(self, fdim, hidden, out, pool):
        super().__init__()
        self.gru = nn.GRU(fdim, hidden, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(2 * hidden, out)
        self.att = nn.Linear(2 * hidden, 1) if pool == "attention" else None
        self.pool = pool

    def forward(self, seq, mask):
        out, _ = self.gru(seq)
        if self.pool == "attention":
            sc = self.att(out).squeeze(-1).masked_fill(~mask, -1e9)
            pooled = (out * torch.softmax(sc, 1).unsqueeze(-1)).sum(1)
        elif self.pool == "max":
            pooled = out.masked_fill(~mask.unsqueeze(-1), -1e9).max(1).values
        else:
            m = mask.unsqueeze(-1).float()
            pooled = (out * m).sum(1) / m.sum(1).clamp(min=1)
        return self.proj(pooled)


class GeoMLP(nn.Module):
    """Static geometric encoder: masked-mean over frames -> MLP. Apt for T=1 data.
    geo_layers controls depth of the hidden stack (1 = original)."""
    def __init__(self, fdim, hidden, out, dropout, layers=1):
        super().__init__()
        mods, d = [], fdim
        for _ in range(max(layers, 1)):
            mods += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden
        mods += [nn.Linear(d, out)]
        self.net = nn.Sequential(*mods)

    def forward(self, seq, mask):
        m = mask.unsqueeze(-1).float()
        static = (seq * m).sum(1) / m.sum(1).clamp(min=1)
        return self.net(static)


def mlp(din, dout, hidden, layers, dropout):
    mods, d = [], din
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
        d = hidden
    mods += [nn.Linear(d, dout)]
    return nn.Sequential(*mods)


class Net(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        gfd = geo_feat_dim(cfg)

        def _mk_geo():
            if cfg["geo_encoder"] == "mlp":
                return GeoMLP(gfd, cfg["temporal_hidden"], cfg["temporal_out"], cfg["dropout"],
                              layers=cfg.get("geo_layers", 1))
            return GeoGRU(gfd, cfg["temporal_hidden"], cfg["temporal_out"], cfg["pool"])

        self.per_region_geo = cfg.get("per_region_geo", False)
        if self.per_region_geo:
            self.geos = nn.ModuleDict({t: _mk_geo() for t in P.TASKS})
        else:
            self.geo = _mk_geo()

        H = cfg["trunk_hidden"]
        self.fusion = cfg["fusion"]
        self.per_region_trunk = cfg.get("per_region_trunk", False)
        self.pr_marlin = cfg.get("pr_marlin")            # {task: dim | "full"} per-region MARLIN width
        self.marlin_proj = None
        self.marlin_ln = None
        if self.pr_marlin:
            # each region routes MARLIN through its OWN bottleneck (eyes narrow, mouth wide)
            self.mnorm = nn.ModuleDict()
            self.trunks = nn.ModuleDict()
            for t in P.TASKS:
                dim = self.pr_marlin.get(t, 768)
                if dim in (768, "full"):
                    self.mnorm[t] = nn.LayerNorm(768)
                    md = 768
                else:
                    self.mnorm[t] = nn.Sequential(nn.Linear(768, dim), nn.LayerNorm(dim))
                    md = dim
                self.trunks[t] = mlp(md + cfg["temporal_out"], H, H, cfg["trunk_layers"], cfg["dropout"])
            trunk_out = H
        else:
            mdim = 768
            if cfg["marlin_proj"]:
                self.marlin_proj = nn.Sequential(nn.Linear(768, cfg["marlin_proj"]), nn.LayerNorm(cfg["marlin_proj"]))
                mdim = cfg["marlin_proj"]
            elif cfg.get("marlin_ln"):
                self.marlin_ln = nn.LayerNorm(768)       # normalize full MARLIN, no dim cut
            if cfg["fusion"] == "gate":
                self.mp = nn.Linear(mdim, H)
                self.gp = nn.Linear(cfg["temporal_out"], H)
                self.gate = nn.ModuleDict({t: nn.Linear(mdim + cfg["temporal_out"], 1) for t in P.TASKS})
                self.post = mlp(H, H, H, max(cfg["trunk_layers"] - 1, 0), cfg["dropout"]) if cfg["trunk_layers"] > 1 else nn.Identity()
                trunk_out = H
            elif self.per_region_trunk:
                self.trunks = nn.ModuleDict(
                    {t: mlp(mdim + cfg["temporal_out"], H, H, cfg["trunk_layers"], cfg["dropout"]) for t in P.TASKS})
                trunk_out = H
            else:
                self.trunk = mlp(mdim + cfg["temporal_out"], H, H, cfg["trunk_layers"], cfg["dropout"])
                trunk_out = H

        self.sev = nn.ModuleDict({t: nn.Linear(trunk_out, 1, bias=False) for t in P.TASKS})
        self.thr = nn.ModuleDict({t: OrderedThresholds(P.N_CLASSES[t]) for t in P.TASKS})

    def rep(self, marlin, mp_seq, mp_mask, task):
        mp_seq = engineer(mp_seq, self.cfg)
        if self.cfg.get("drop_marlin"):
            marlin = torch.zeros_like(marlin)            # geometry-only: ablate appearance (domain-confound)
        elif self.training and self.cfg.get("marlin_noise", 0.0) > 0:
            marlin = marlin + torch.randn_like(marlin) * self.cfg["marlin_noise"]  # augmentation: fight appearance-overfit
        if self.training and self.cfg.get("geo_noise", 0.0) > 0:
            mp_seq = mp_seq + torch.randn_like(mp_seq) * self.cfg["geo_noise"]
        g = self.geos[task](mp_seq, mp_mask) if self.per_region_geo else self.geo(mp_seq, mp_mask)
        if self.pr_marlin:
            return self.trunks[task](torch.cat([self.mnorm[task](marlin), g], -1))
        if self.marlin_proj is not None:
            m = self.marlin_proj(marlin)
        elif self.marlin_ln is not None:
            m = self.marlin_ln(marlin)
        else:
            m = marlin
        if self.fusion == "gate":
            a = torch.sigmoid(self.gate[task](torch.cat([m, g], -1)))   # (B,1)
            h = a * torch.relu(self.mp(m)) + (1 - a) * torch.relu(self.gp(g))
            return self.post(h)
        cat = torch.cat([m, g], -1)
        if self.per_region_trunk:
            return self.trunks[task](cat)
        return self.trunk(cat)

    def severity(self, marlin, mp_seq, mp_mask, task):
        return self.sev[task](self.rep(marlin, mp_seq, mp_mask, task)).squeeze(-1)


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------
def class_weights(train, task, k):
    labs = np.array([r["label"] for r in train if r["task"] == task])
    freq = np.bincount(labs, minlength=k) + 1
    inv = freq.sum() / freq
    return torch.tensor(inv / inv.mean(), dtype=torch.float32, device=DEVICE)


def predict_val(model, val_in, cfg):
    model.eval()
    out = {}
    with torch.no_grad():
        for t in P.REGION_TASKS:
            b = P.make_batch(val_in[t], DEVICE)
            s = model.severity(b["marlin"], b["mp_seq"], b["mp_mask"], t)
            out[t] = predict_grades(s, model.thr[t](), cfg).cpu().numpy()
    return out


def train_one_seed(seed, cfg):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    train = P.train_records()
    val_in = {t: P.val_records(t) for t in P.REGION_TASKS}
    truth = {t: [r["label"] for r in val_in[t]] for t in P.REGION_TASKS}
    cw = {t: class_weights(train, t, P.N_CLASSES[t]) for t in P.TASKS} if cfg["loss"].endswith("_cw") else None

    lw = {t: cfg.get(f"lw_{t}", LW[t]) for t in P.TASKS}   # per-task loss weight (overridable)
    model = Net(cfg).to(DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])

    idx = np.arange(len(train))
    best, best_preds = -1.0, None
    for ep in range(cfg["epochs"]):
        if ep < cfg["warmup"]:
            for g in opt.param_groups:
                g["lr"] = cfg["lr"] * (ep + 1) / max(cfg["warmup"], 1)
        model.train()
        rng.shuffle(idx)
        for st in range(0, len(idx), cfg["batch_size"]):
            recs = [train[i] for i in idx[st:st + cfg["batch_size"]]]
            b = P.make_batch(recs, DEVICE)
            tasks = np.array(b["task"])
            loss = torch.zeros((), device=DEVICE)
            for t in P.TASKS:
                ti = np.where(tasks == t)[0]
                if len(ti) == 0:
                    continue
                jj = torch.tensor(ti, device=DEVICE)
                s = model.severity(b["marlin"].index_select(0, jj), b["mp_seq"].index_select(0, jj),
                                   b["mp_mask"].index_select(0, jj), t)
                lab = b["label"].index_select(0, jj)
                w = cw[t][lab] if cw is not None else None
                loss = loss + lw[t] * ordinal_loss(s, model.thr[t](), lab, cfg, w)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
        if ep >= cfg["warmup"]:
            sched.step()
        if (ep + 1) % cfg["eval_every"] == 0 or ep == cfg["epochs"] - 1:
            preds = predict_val(model, val_in, cfg)
            m = 0.5 * (P.quadratic_kappa(truth["eyes"], preds["eyes"], 3)
                       + P.quadratic_kappa(truth["mouth"], preds["mouth"], 3))
            if m > best:
                best, best_preds = m, preds
    return best_preds


def main():
    cfg = dict(DEFAULT)
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        override = json.loads(Path(arg).read_text()) if arg.endswith(".json") and os.path.exists(arg) else json.loads(arg)
        cfg.update(override)
    t0 = time.time()
    preds = [train_one_seed(s, cfg) for s in P.SEEDS]
    P.report_metric(preds, extra={"train_seconds": round(time.time() - t0, 1),
                                  "name": cfg.get("name", "exp")})


if __name__ == "__main__":
    main()
