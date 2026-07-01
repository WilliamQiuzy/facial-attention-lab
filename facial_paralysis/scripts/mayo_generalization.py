"""Generalization test: does the autoresearch champion transfer to the Mayo domain?

Uses the Mayo per-action clips (mayo_action_bundles) as an OUT-OF-DOMAIN TEST SET.
No HB labels. The domain-invariant ground truth is the label-free clinical asymmetry
(Run #14: appearance-driven severity ANTI-correlated with it, rho=-0.50). A model that
GENERALIZES should have its region severity CORRELATE with the clinical asymmetry on
held-out Mayo patients.

We compare the champion (asym features + per-region-decoupled, MARLIN de-emphasized)
against a MARLIN-heavy baseline, scoring each on the Mayo eye/mouth action clips.
"""
from __future__ import annotations
import json, os, glob, sys
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../autoresearch_fp")
import prepare_fp as P
import runner as R

ROOT = Path(__file__).resolve().parent.parent
EYE_ACTS = ("GentleEyeClosure", "TightEyeSqueeze")
MOUTH_ACTS = ("RelaxedSmile", "LipPucker", "LowerTeethShow", "ReanimatedSmile")
DEVICE = "cpu"


def train_model(cfg, seed):
    """Train one model on web data, return the Net (mirrors runner.train_one_seed)."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    train = P.train_records()
    cw = {t: R.class_weights(train, t, P.N_CLASSES[t]) for t in P.TASKS} if cfg["loss"].endswith("_cw") else None
    lw = {t: cfg.get(f"lw_{t}", R.LW[t]) for t in P.TASKS}
    model = R.Net(cfg).to(DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    idx = np.arange(len(train))
    for ep in range(cfg["epochs"]):
        model.train(); rng.shuffle(idx)
        for st in range(0, len(idx), cfg["batch_size"]):
            recs = [train[i] for i in idx[st:st + cfg["batch_size"]]]
            b = P.make_batch(recs, DEVICE); tasks = np.array(b["task"])
            loss = torch.zeros((), device=DEVICE)
            for t in P.TASKS:
                ti = np.where(tasks == t)[0]
                if len(ti) == 0: continue
                jj = torch.tensor(ti, device=DEVICE)
                s = model.severity(b["marlin"].index_select(0, jj), b["mp_seq"].index_select(0, jj),
                                   b["mp_mask"].index_select(0, jj), t)
                lab = b["label"].index_select(0, jj)
                w = cw[t][lab] if cw is not None else None
                loss = loss + lw[t] * R.ordinal_loss(s, model.thr[t](), lab, cfg, w)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0); opt.step()
        sched.step()
    return model.eval()


def load_mayo():
    """Per patient: {eyes:[recs], mouth:[recs]} from mayo_action_bundles."""
    out = {}
    for d in sorted(glob.glob(str(ROOT / "outputs/mayo_action_bundles/*/"))):
        take = os.path.basename(d.rstrip("/"))
        pid = take.split("_", 1)[1]
        rec = {"eyes": [], "mouth": []}
        for npz in glob.glob(os.path.join(d, "*.npz")):
            a = os.path.basename(npz)[:-4]
            task = "eyes" if a in EYE_ACTS else ("mouth" if a in MOUTH_ACTS else None)
            if not task: continue
            z = np.load(npz)
            rec[task].append({"marlin": z["marlin"].astype(np.float32).mean(0),
                              "mp_seq": z["mp_seq"].astype(np.float32),
                              "mp_mask": z["mp_mask"].astype(bool),
                              "label": 0, "task": task})   # dummy label (severity scoring ignores it)
        out[pid] = rec
    return out


def score_mayo(models, mayo):
    """Ensemble severity per patient per region."""
    sev = {}
    for pid, rec in mayo.items():
        row = {}
        for task in ("eyes", "mouth"):
            if not rec[task]: continue
            b = P.make_batch(rec[task], DEVICE)
            ss = []
            for m in models:
                with torch.no_grad():
                    ss.append(m.severity(b["marlin"], b["mp_seq"], b["mp_mask"], task).cpu().numpy())
            row[task] = float(np.mean(ss))            # mean over actions + seeds
        sev[pid] = row
    return sev


def main():
    ef = json.loads((ROOT / "outputs/mayo_eface/eface_scores.json").read_text())
    ef = {k.split("_", 1)[1]: v for k, v in ef.items()}
    mayo = load_mayo()
    pids = [p for p in mayo if p != "MySlate_14" and p in ef]   # drop duplicate + non-take entries

    CHAMP = dict(R.DEFAULT); CHAMP.update(json.loads((ROOT / "autoresearch_fp/best_config.json").read_text()))
    BASE = dict(R.DEFAULT)                                  # v0: raw feat, GRU, full MARLIN (MARLIN-heavy)
    configs = {"champion": CHAMP, "baseline_marlin_heavy": BASE}

    print("Generalization to Mayo (label-free): Spearman(model severity, clinical asymmetry)")
    print("higher/positive = better generalization;  Run #14 old model: eyes -0.01, mouth -0.50\n")
    for name, cfg in configs.items():
        models = [train_model(cfg, s) for s in (0, 1, 2)]
        sev = score_mayo(models, mayo)
        print(f"=== {name} ===")
        for task, asym_key in (("eyes", "eye_asym"), ("mouth", "oral_asym")):
            s = np.array([sev[p].get(task, np.nan) for p in pids])
            a = np.array([ef[p][asym_key] if ef[p].get(asym_key) is not None else np.nan for p in pids])
            m = ~(np.isnan(a) | np.isnan(s))
            rho, pv = spearmanr(s[m], a[m])
            print(f"  {task:5s}: rho={rho:+.2f} p={pv:.2f}  (n={m.sum()})")
        print()


if __name__ == "__main__":
    main()
