"""DANN (domain-adversarial) to attack the MARLIN domain-confound.

A gradient-reversal domain classifier forces the shared representation to be
web-vs-Mayo INDISCRIMINABLE while staying severity-predictive. Uses Mayo as the
UNLABELED target (n=13). We report, vs a lambda=0 baseline:
  (1) domain-AUC of the learned rep h (does the gap shrink? — well-powered)
  (2) Mayo severity vs INDEPENDENT clinical asymmetry (EAR eyes / corner mouth) — transfer
"""
from __future__ import annotations
import os, sys, json, glob
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../autoresearch_fp")
import prepare_fp as P, runner as R
from mayo_generalization import load_mayo
ROOT = Path(__file__).resolve().parent.parent
DEV = "cpu"; TASKS = ("eyes", "mouth")


class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd; return x.clone()
    @staticmethod
    def backward(ctx, g):
        return -ctx.lambd * g, None


class DANNNet(nn.Module):
    def __init__(self, H=96, md=128):
        super().__init__()
        gfd = P.MP_FEAT_DIM + 24
        self.geo = nn.Sequential(nn.Linear(gfd, 64), nn.ReLU(), nn.Linear(64, 64))
        self.mproj = nn.Sequential(nn.Linear(768, md), nn.LayerNorm(md))
        self.trunk = nn.Sequential(nn.Linear(md + 64, H), nn.ReLU(), nn.Dropout(0.2))
        self.sev = nn.ModuleDict({t: nn.Linear(H, 1, bias=False) for t in TASKS})
        self.thr = nn.ModuleDict({t: R.OrderedThresholds(3) for t in TASKS})
        self.dom = nn.Sequential(nn.Linear(H, 64), nn.ReLU(), nn.Linear(64, 2))

    def rep(self, marlin, mp_seq, mp_mask):
        mp = R.engineer(mp_seq, {"feat": "asym"})
        mmask = mp_mask.unsqueeze(-1).float()
        static = (mp * mmask).sum(1) / mmask.sum(1).clamp(min=1)
        return self.trunk(torch.cat([self.mproj(marlin), self.geo(static)], -1))

    def sev_of(self, h, t): return self.sev[t](h).squeeze(-1)


def batch(recs):
    b = P.make_batch(recs, DEV); return b


def train(lambd, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    web = P.train_records(); mayo = load_mayo()
    mayo_recs = [r for p in mayo.values() for t in TASKS for r in p[t]]
    net = DANNNet().to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=3e-2)
    idx = np.arange(len(web)); mi = np.arange(len(mayo_recs))
    for ep in range(60):
        net.train(); rng.shuffle(idx)
        lam = lambd * min(1.0, ep / 20)          # ramp
        for st in range(0, len(idx), 128):
            wr = [web[i] for i in idx[st:st + 128]]
            bw = batch(wr); h = net.rep(bw["marlin"], bw["mp_seq"], bw["mp_mask"])
            tasks = np.array(bw["task"]); loss = torch.zeros((), device=DEV)
            for t in TASKS:
                ti = np.where(tasks == t)[0]
                if len(ti) == 0: continue
                jj = torch.tensor(ti)
                loss = loss + R.ordinal_loss(net.sev_of(h.index_select(0, jj), t), net.thr[t](),
                                             bw["label"].index_select(0, jj), {"loss": "coral", "ls": 0.0})
            if lam > 0:                            # domain-adversarial
                mr = [mayo_recs[i] for i in rng.choice(mi, min(64, len(mi)), replace=False)]
                bm = batch(mr); hm = net.rep(bm["marlin"], bm["mp_seq"], bm["mp_mask"])
                hh = torch.cat([h, hm], 0)
                dy = torch.cat([torch.zeros(len(h)), torch.ones(len(hm))]).long()
                loss = loss + nn.functional.cross_entropy(net.dom(GRL.apply(hh, lam)), dy)
            opt.zero_grad(); loss.backward(); opt.step()
    return net.eval()


def reps_for(net, recs):
    b = batch(recs)
    with torch.no_grad():
        return net.rep(b["marlin"], b["mp_seq"], b["mp_mask"]).numpy()


def evaluate(nets):
    # (1) domain-AUC of h
    web_val = [r for r in P.load_data() if r["split"] == "val" and r["task"] in TASKS][:300]
    mayo = load_mayo(); mayo_recs = [r for p in mayo.values() for t in TASKS for r in p[t]]
    hw = np.mean([reps_for(n, web_val) for n in nets], 0)
    hm = np.mean([reps_for(n, mayo_recs) for n in nets], 0)
    X = StandardScaler().fit_transform(np.vstack([hw, hm]))
    y = np.r_[np.zeros(len(hw)), np.ones(len(hm))]
    dauc = cross_val_score(LogisticRegression(max_iter=2000, C=0.5), X, y, cv=5, scoring="roc_auc").mean()
    # (2) transfer vs independent targets
    ear = json.loads((ROOT / "outputs/mayo_ear/ear_dynamics.json").read_text())
    mc = json.loads((ROOT / "outputs/mayo_ear/mouth_corner_asym.json").read_text())
    eyes_t = {p: ear[p]["TightEyeSqueeze"]["closure_asym"] for p in ear if "TightEyeSqueeze" in ear[p]}
    tgt = {"eyes": eyes_t, "mouth": mc}
    rhos = {}
    for t in TASKS:
        ps, sev = [], []
        for pid, rec in mayo.items():
            if pid == "MySlate_14" or not rec[t] or pid not in tgt[t]: continue
            b = batch(rec[t])
            s = np.mean([n.sev_of(n.rep(b["marlin"], b["mp_seq"], b["mp_mask"]), t).mean().item() for n in nets])
            ps.append(tgt[t][pid]); sev.append(s)
        rhos[t] = spearmanr(sev, ps)[0] if len(ps) > 3 else float("nan")
    return dauc, rhos


def main():
    print("DANN vs baseline — domain-AUC of learned rep (lower=more invariant) + Mayo transfer\n")
    for lambd in (0.0, 1.0):
        nets = [train(lambd, s) for s in range(3)]
        dauc, rhos = evaluate(nets)
        tag = "baseline (no DANN)" if lambd == 0 else "DANN lambda=1.0"
        print(f"  {tag:20s}: rep domain-AUC={dauc:.3f}   transfer eyes={rhos['eyes']:+.2f} mouth={rhos['mouth']:+.2f}")


if __name__ == "__main__":
    main()
