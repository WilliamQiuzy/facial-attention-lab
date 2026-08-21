"""Cross-dataset generalization on the web data (non-circular, no Mayo, no labels beyond
the web region labels). Train on ONE source, test on the OTHER — measures how well the
model generalizes to an unseen dataset vs. the within-mix number (~0.65).

A model that only fits one split will crater cross-dataset; one that learns the real
severity signal will hold up. This is the honest generalization metric, and it tells us
which architecture choices GENERALIZE (not just fit).
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../autoresearch_fp")
import prepare_fp as P
import runner as R

ROOT = Path(__file__).resolve().parent.parent
DEVICE = "cpu"


def train_on(recs, cfg, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = R.Net(cfg).to(DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    lw = {t: cfg.get(f"lw_{t}", R.LW[t]) for t in P.TASKS}
    idx = np.arange(len(recs))
    for ep in range(cfg["epochs"]):
        model.train(); rng.shuffle(idx)
        for st in range(0, len(idx), cfg["batch_size"]):
            b = P.make_batch([recs[i] for i in idx[st:st + cfg["batch_size"]]], DEVICE)
            tasks = np.array(b["task"]); loss = torch.zeros((), device=DEVICE)
            for t in P.TASKS:
                ti = np.where(tasks == t)[0]
                if len(ti) == 0: continue
                jj = torch.tensor(ti, device=DEVICE)
                s = model.severity(b["marlin"].index_select(0, jj), b["mp_seq"].index_select(0, jj),
                                   b["mp_mask"].index_select(0, jj), t)
                loss = loss + lw[t] * R.ordinal_loss(s, model.thr[t](), b["label"].index_select(0, jj), cfg)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0); opt.step()
        sched.step()
    return model.eval()


def qwk_on(models, recs, task):
    idx = [i for i, r in enumerate(recs) if r["task"] == task]
    if not idx: return None
    sub = [recs[i] for i in idx]
    b = P.make_batch(sub, DEVICE)
    ps = []
    for m in models:
        with torch.no_grad():
            ps.append(R.predict_grades(m.severity(b["marlin"], b["mp_seq"], b["mp_mask"], task),
                                       m.thr[task](), {"decode": "threshold"}).cpu().numpy())
    pred = np.round(np.mean(ps, 0)).astype(int)
    true = [r["label"] for r in sub]
    return P.quadratic_kappa(true, pred, 3)


def main():
    data = P.load_data()
    by_src = {s: [r for r in data if r["source"] == s and r["task"] in ("eyes", "mouth")] for s in ("fnp", "yfp")}
    CHAMP = dict(R.DEFAULT); CHAMP.update(json.loads((ROOT / "autoresearch_fp/best_config.json").read_text()))
    BASE = dict(R.DEFAULT)
    print("Cross-dataset generalization (region QWK); within-mix ref ~0.65\n")
    for name, cfg in (("champion", CHAMP), ("baseline", BASE)):
        print(f"=== {name} ===")
        for tr, te in (("fnp", "yfp"), ("yfp", "fnp")):
            models = [train_on(by_src[tr], cfg, s) for s in (0, 1)]
            qe, qm = qwk_on(models, by_src[te], "eyes"), qwk_on(models, by_src[te], "mouth")
            print(f"  train {tr}->test {te}:  eyes QWK={qe:.3f}  mouth QWK={qm:.3f}")
        print()


if __name__ == "__main__":
    main()
