"""#1: CORAL domain adaptation using the Mayo clips as the UNLABELED target domain.

Train the champion on web (labeled) + a CORAL loss aligning the feature-covariance of
web vs Mayo clips per region. Goal: make features domain-invariant so the FULL model
(keeping MARLIN's web performance) also transfers to Mayo — i.e. move the Mayo
severity-vs-clinical-asymmetry correlation up from ~0, WITHOUT dropping appearance.

No Mayo labels used (CORAL is label-free). Test = same as mayo_generalization.
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path
import numpy as np, torch
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../autoresearch_fp")
import prepare_fp as P, runner as R
from mayo_generalization import load_mayo, score_mayo
ROOT = Path(__file__).resolve().parent.parent
DEVICE = "cpu"


def coral(A, B):
    d = A.shape[1]
    Ac, Bc = A - A.mean(0, keepdim=True), B - B.mean(0, keepdim=True)
    cs = Ac.T @ Ac / max(A.shape[0] - 1, 1)
    ct = Bc.T @ Bc / max(B.shape[0] - 1, 1)
    return ((cs - ct) ** 2).sum() / (4 * d * d)


def train_coral(cfg, mayo, lam, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    train = P.train_records()
    # target (Mayo) reps per region, as tensors
    tgt = {t: [r for p in mayo.values() for r in p[t]] for t in ("eyes", "mouth")}
    tgt_b = {t: P.make_batch(tgt[t], DEVICE) for t in tgt if tgt[t]}
    model = R.Net(cfg).to(DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    lw = {t: cfg.get(f"lw_{t}", R.LW[t]) for t in P.TASKS}
    idx = np.arange(len(train))
    for ep in range(cfg["epochs"]):
        model.train(); rng.shuffle(idx)
        for st in range(0, len(idx), cfg["batch_size"]):
            b = P.make_batch([train[i] for i in idx[st:st + cfg["batch_size"]]], DEVICE)
            tasks = np.array(b["task"]); loss = torch.zeros((), device=DEVICE)
            for t in P.TASKS:
                ti = np.where(tasks == t)[0]
                if len(ti) == 0: continue
                jj = torch.tensor(ti, device=DEVICE)
                s = model.severity(b["marlin"].index_select(0, jj), b["mp_seq"].index_select(0, jj),
                                   b["mp_mask"].index_select(0, jj), t)
                loss = loss + lw[t] * R.ordinal_loss(s, model.thr[t](), b["label"].index_select(0, jj), cfg)
                # CORAL: align this region's source reps to Mayo target reps
                if t in tgt_b and t in ("eyes", "mouth"):
                    src_rep = model.rep(b["marlin"].index_select(0, jj), b["mp_seq"].index_select(0, jj),
                                        b["mp_mask"].index_select(0, jj), t)
                    tb = tgt_b[t]
                    tgt_rep = model.rep(tb["marlin"], tb["mp_seq"], tb["mp_mask"], t)
                    loss = loss + lam * coral(src_rep, tgt_rep)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0); opt.step()
        sched.step()
    return model.eval()


def main():
    ef = json.loads((ROOT / "outputs/mayo_eface/eface_scores.json").read_text())
    ef = {k.split("_", 1)[1]: v for k, v in ef.items()}
    mayo = load_mayo()
    pids = [p for p in mayo if p != "MySlate_14" and p in ef]
    CH = dict(R.DEFAULT); CH.update(json.loads((ROOT / "autoresearch_fp/best_config.json").read_text()))
    print("CORAL domain adaptation -> Mayo transfer (Spearman severity vs clinical asymmetry)")
    print("champion WITHOUT coral was: eyes -0.06, mouth -0.19; geometry-only eyes +0.48\n")
    for lam in (0.0, 1.0, 10.0):
        models = [train_coral(CH, mayo, lam, s) for s in (0, 1, 2)]
        sev = score_mayo(models, mayo)
        out = []
        for task, ak in (("eyes", "eye_asym"), ("mouth", "oral_asym")):
            s = np.array([sev[p].get(task, np.nan) for p in pids])
            a = np.array([ef[p][ak] if ef[p].get(ak) is not None else np.nan for p in pids])
            m = ~(np.isnan(a) | np.isnan(s))
            rho, pv = spearmanr(s[m], a[m])
            out.append(f"{task} rho={rho:+.2f}(p={pv:.2f})")
        print(f"  CORAL lambda={lam:>4}:  " + "  ".join(out))


if __name__ == "__main__":
    main()
