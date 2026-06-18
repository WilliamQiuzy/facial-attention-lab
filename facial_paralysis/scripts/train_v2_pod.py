"""Train the unified warm-start model on the RunPod A100 (v2).

Reads the cached public bundles via outputs/train_manifest.json (generated
locally by make_train_manifest.py — no raw data needed on the pod) and trains the
SAME unified model as Run #6 (binary + eyes + mouth), but on GPU and with the
temporal-pooling knob from Run #9. Trains BOTH:
  - pool=mean       -> reproduces v1 on GPU (sanity / baseline)
  - pool=attention  -> the Run #9 improvement (best for the weak eyes head)
and reports per-task, per-source val metrics for each, saving warmstart_v2_<pool>.pt.

Run on pod:  .venv/bin/python scripts/train_v2_pod.py            (both pools)
             .venv/bin/python scripts/train_v2_pod.py attention  (one)
"""
from __future__ import annotations

import json
import sys
from collections import Counter
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
MANIFEST = OUT / "train_manifest.json"
MP_FEAT_DIM = 72
ACTION = "clip"


def load_records():
    man = json.loads(MANIFEST.read_text())
    train, val, missing = [], [], 0
    for e in man:
        npz = OUT / e["npz"]
        if not npz.exists():
            missing += 1; continue
        d = np.load(npz)
        b = ActionBundle(marlin=d["marlin"].astype(np.float32),
                         mp_seq=d["mp_seq"].astype(np.float32),
                         mp_mask=d["mp_mask"].astype(bool))
        rec = MultiStreamRecord(patient_id=e["pid"], label=int(e["label"]),
                                task=e["task"], actions=[b])
        (val if e["split"] == "val" else train).append(rec)
    print(f"loaded train={len(train)} val={len(val)} (missing bundles={missing})", flush=True)
    return train, val


def make_model(pool, seed=0):
    torch.manual_seed(seed)
    return FacialPalsyModel(FacialPalsyConfig(
        mp_feat_dim=MP_FEAT_DIM, n_actions=1, temporal_hidden=64, temporal_out=64,
        trunk_hidden=96, dropout=0.1, temporal_pool=pool,
        tasks=[TaskSpec("binary", 2, coupled=True),
               TaskSpec("eyes", 3, coupled=False),
               TaskSpec("mouth", 3, coupled=False)]))


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
        res["binary"] = {"n": len(bi),
                         "auc": round(float(roc_auc_score(y, p)), 3) if len(set(y)) == 2 else None,
                         "acc": round(float(((p > 0.5).astype(int) == y).mean()), 3)}
    for task in ("eyes", "mouth"):
        ti = [i for i, r in enumerate(val_recs) if r.task == task]
        if not ti:
            continue
        idx = torch.tensor(ti, device=device)
        pred = predict_grade(out[task].index_select(0, idx)).cpu().numpy()
        true = np.array([val_recs[i].label for i in ti])
        m = HBMetrics.from_predictions(true, pred, n_classes=3)
        entry = {"n": len(ti), "kappa": round(m.quadratic_kappa, 3),
                 "mae": round(m.mae_grades, 3), "acc": round(m.accuracy, 3), "by_source": {}}
        for src in ("fnp", "yfp"):
            si = [j for j, i in enumerate(ti) if _source(val_recs[i].patient_id) == src]
            if si:
                ms = HBMetrics.from_predictions(true[si], pred[si], n_classes=3)
                entry["by_source"][src] = {"n": len(si), "kappa": round(ms.quadratic_kappa, 3),
                                           "mae": round(ms.mae_grades, 3)}
        res[task] = entry
    return res


def main():
    pools = sys.argv[1:] or ["mean", "attention"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} pools={pools}", flush=True)
    train_recs, val_recs = load_records()
    train_ds = MultiStreamPatientDataset(train_recs, actions=[ACTION], mp_feat_dim=MP_FEAT_DIM)
    val_ds = MultiStreamPatientDataset(val_recs, actions=[ACTION], mp_feat_dim=MP_FEAT_DIM)

    all_res = {}
    for pool in pools:
        print(f"\n===== training pool={pool} =====", flush=True)
        model = make_model(pool)
        train_multitask(model, train_ds, val_ds, MTTrainConfig(
            epochs=80, batch_size=64, lr=5e-4, weight_decay=3e-2, device=device,
            monitor_task="eyes", monitor_n_classes=3, log_every=20,
            early_stopping_patience=15, early_stopping_warmup=10, seed=0))
        res = evaluate(model, val_recs, device)
        all_res[pool] = res
        print(f"[{pool}] {json.dumps(res)}", flush=True)
        ckpt = OUT / "checkpoints" / f"warmstart_v2_{pool}.pt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "tasks": [(t.name, t.n_classes, t.coupled) for t in model.multitask.tasks],
            "model_cfg": {"mp_feat_dim": MP_FEAT_DIM, "n_actions": 1, "temporal_hidden": 64,
                          "temporal_out": 64, "trunk_hidden": 96, "dropout": 0.1,
                          "temporal_pool": pool},
            "quality": {"mode": "normalize", "work_size": 112},
            "label_scheme": {"binary": ["healthy", "palsy"],
                             "eyes": ["Normal", "Slight", "Strong"],
                             "mouth": ["Normal", "Slight", "Strong"]},
            "val_metrics": res, "sources": "PalsyNet(binary)+FNP+YFP, GPU v2",
        }, ckpt)
        print(f"saved {ckpt}", flush=True)

    (OUT / "run_v2_results.json").write_text(json.dumps(all_res, indent=2))
    print("\n===== V2 SUMMARY (vs v1: eyes 0.38 / mouth 0.82) =====", flush=True)
    for pool in pools:
        r = all_res[pool]
        print(f"  {pool:>9s}: binary AUC {r.get('binary',{}).get('auc')}  "
              f"eyes QWK {r.get('eyes',{}).get('kappa')}  mouth QWK {r.get('mouth',{}).get('kappa')}", flush=True)


if __name__ == "__main__":
    main()
