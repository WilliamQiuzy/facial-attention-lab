"""Train warm-start v4 on the RunPod A100 — EXPANDED, leak-safe data.

Same model/config as v2-attention (the best by measurable metric), but trained on
the expanded manifest (train_manifest_v4.json): adds a `coarse3` head fed by anisa +
stroke (global 3-level severity) and more `binary` positives from kaggle — all
TRAIN-ONLY and deduped (docs/leakage_policy.md). Eval is UNCHANGED and on the clean
holdouts only (binary=PalsyNet-held, eyes/mouth=FNP-valid/YFP-held subjects), so the
question is: does the extra global-severity data improve the clean held-out metrics
via the shared severity trunk? coarse3 has no clean held-out → train-only (not scored).

Run on pod:  .venv/bin/python scripts/train_v4_pod.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.datasets.patient_multistream import (  # noqa: E402
    ActionBundle, MultiStreamRecord, MultiStreamPatientDataset, collate_multistream,
)
from src.evaluation.hb_metrics import HBMetrics  # noqa: E402
from src.models.facial_palsy_model import FacialPalsyConfig, FacialPalsyModel  # noqa: E402
from src.models.multitask import TaskSpec  # noqa: E402
from src.models.ordinal import cum_probs, predict_grade  # noqa: E402
from src.training.train_multitask import MTTrainConfig, train_multitask  # noqa: E402

OUT = ROOT / "outputs"
MANIFEST = OUT / "train_manifest_v4.json"
MP_FEAT_DIM = 72
ACTION = "clip"


def load_records():
    man = json.loads(MANIFEST.read_text())
    train, val, miss = [], [], 0
    for e in man:
        npz = OUT / e["npz"]
        if not npz.exists():
            miss += 1; continue
        d = np.load(npz)
        b = ActionBundle(marlin=d["marlin"].astype(np.float32),
                         mp_seq=d["mp_seq"].astype(np.float32),
                         mp_mask=d["mp_mask"].astype(bool))
        rec = MultiStreamRecord(patient_id=e["pid"], label=int(e["label"]), task=e["task"], actions=[b])
        (val if e["split"] == "val" else train).append(rec)
    print(f"train={len(train)} val={len(val)} (missing bundles={miss})", flush=True)
    from collections import Counter
    print("train tasks:", dict(Counter(r.task for r in train)), flush=True)
    return train, val


def _source(pid): return pid.split("_", 1)[0]


def evaluate(model, val_recs, device):
    from sklearn.metrics import roc_auc_score
    from torch.utils.data import DataLoader
    ds = MultiStreamPatientDataset(val_recs, actions=[ACTION], mp_feat_dim=MP_FEAT_DIM)
    b = next(iter(DataLoader(ds, batch_size=len(ds), collate_fn=collate_multistream)))
    b = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in b.items()}
    model.eval()
    with torch.no_grad():
        out = model(b["marlin_emb"], b["marlin_mask"], b["mp_seq"], b["mp_mask"], b["action_present"])
    res = {}
    bi = [i for i, r in enumerate(val_recs) if r.task == "binary"]
    if bi:
        idx = torch.tensor(bi, device=device)
        p = cum_probs(out["binary"].index_select(0, idx))[:, 0].cpu().numpy()
        y = np.array([val_recs[i].label for i in bi])
        res["binary"] = {"n": len(bi), "auc": round(float(roc_auc_score(y, p)), 3) if len(set(y)) == 2 else None}
    for task in ("eyes", "mouth"):
        ti = [i for i, r in enumerate(val_recs) if r.task == task]
        if not ti:
            continue
        idx = torch.tensor(ti, device=device)
        pred = predict_grade(out[task].index_select(0, idx)).cpu().numpy()
        true = np.array([val_recs[i].label for i in ti])
        m = HBMetrics.from_predictions(true, pred, n_classes=3)
        entry = {"n": len(ti), "kappa": round(m.quadratic_kappa, 3), "by_source": {}}
        for src in ("fnp", "yfp"):
            si = [j for j, i in enumerate(ti) if _source(val_recs[i].patient_id) == src]
            if si:
                ms = HBMetrics.from_predictions(true[si], pred[si], n_classes=3)
                entry["by_source"][src] = {"n": len(si), "kappa": round(ms.quadratic_kappa, 3)}
        res[task] = entry
    return res


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}", flush=True)
    train_recs, val_recs = load_records()
    train_ds = MultiStreamPatientDataset(train_recs, actions=[ACTION], mp_feat_dim=MP_FEAT_DIM)
    val_ds = MultiStreamPatientDataset(val_recs, actions=[ACTION], mp_feat_dim=MP_FEAT_DIM)

    torch.manual_seed(0)
    cfg = FacialPalsyConfig(
        mp_feat_dim=MP_FEAT_DIM, n_actions=1, temporal_hidden=64, temporal_out=64,
        trunk_hidden=96, dropout=0.1, temporal_pool="attention",
        tasks=[TaskSpec("binary", 2, coupled=True),
               TaskSpec("coarse3", 3, coupled=True),       # <-- NEW: anisa + stroke global severity
               TaskSpec("eyes", 3, coupled=False),
               TaskSpec("mouth", 3, coupled=False)])
    model = FacialPalsyModel(cfg)
    train_multitask(model, train_ds, val_ds, MTTrainConfig(
        epochs=80, batch_size=64, lr=5e-4, weight_decay=3e-2, device=device,
        monitor_task="eyes", monitor_n_classes=3, log_every=20,
        early_stopping_patience=15, early_stopping_warmup=10, seed=0))
    res = evaluate(model, val_recs, device)

    ckpt = OUT / "checkpoints" / "warmstart_v4_expanded.pt"
    torch.save({
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "tasks": [(t.name, t.n_classes, t.coupled) for t in model.multitask.tasks],
        "model_cfg": {"mp_feat_dim": MP_FEAT_DIM, "n_actions": 1, "temporal_hidden": 64,
                      "temporal_out": 64, "trunk_hidden": 96, "dropout": 0.1, "temporal_pool": "attention"},
        "quality": {"mode": "normalize", "work_size": 112},
        "label_scheme": {"binary": ["healthy", "palsy"], "coarse3": ["Normal", "Medium", "Strong"],
                         "eyes": ["Normal", "Slight", "Strong"], "mouth": ["Normal", "Slight", "Strong"]},
        "val_metrics": res, "sources": "PalsyNet+FNP+YFP + anisa/stroke(coarse3)+kaggle(binary), leak-safe",
    }, ckpt)
    (OUT / "run_v4_results.json").write_text(json.dumps(res, indent=2))
    print("\n===== V4 (expanded) vs v2-attention (binary~0.86 / eyes 0.426 / mouth 0.859) =====", flush=True)
    print(f"  v4: binary AUC {res.get('binary',{}).get('auc')}  "
          f"eyes QWK {res.get('eyes',{}).get('kappa')}  mouth QWK {res.get('mouth',{}).get('kappa')}", flush=True)
    print(f"saved {ckpt}", flush=True)


if __name__ == "__main__":
    main()
