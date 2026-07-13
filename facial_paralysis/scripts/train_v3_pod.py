"""Train warm-start v3 on the RunPod A100 — REWEIGHT the geometric stream.

Run #14 showed the frozen-MARLIN appearance stream does NOT carry clinical severity
on the Mayo domain (its `s` anti-correlates with true L/R asymmetry); the geometric
stream does. So v3 down-weights MARLIN relative to the geometric/temporal stream:
  - marlin_proj_dim=128  : project MARLIN 768 -> 128 (was 768, dwarfing the 128-d dynamics)
  - stream_layernorm=True: LayerNorm each stream so neither dominates the concat
  - temporal_pool=attention (Run #9 best for the eye head)
Same unified data + manifest as Run #12. Compares to v2-attention (eyes 0.426 / mouth 0.859).

Run on pod:  .venv/bin/python scripts/train_v3_pod.py
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
from scripts._bundle_io import load_action_bundle  # noqa: E402

OUT = ROOT / "outputs"
MANIFEST = OUT / "train_manifest.json"
MP_FEAT_DIM = 72
ACTION = "clip"


def load_records():
    man = json.loads(MANIFEST.read_text())
    train, val = [], []
    for e in man:
        npz = OUT / e["npz"]
        if not npz.exists():
            continue
        b = load_action_bundle(
            npz,
            allow_legacy_schema=True,
            expected_feat_dim=MP_FEAT_DIM,
        )
        rec = MultiStreamRecord(patient_id=e["pid"], label=int(e["label"]),
                                task=e["task"], actions=[b])
        (val if e["split"] == "val" else train).append(rec)
    print(f"train={len(train)} val={len(val)}", flush=True)
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
        entry = {"n": len(ti), "kappa": round(m.quadratic_kappa, 3), "acc": round(m.accuracy, 3), "by_source": {}}
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
        marlin_proj_dim=128, stream_layernorm=True,            # <-- reweight geometric stream
        tasks=[TaskSpec("binary", 2, coupled=True),
               TaskSpec("eyes", 3, coupled=False),
               TaskSpec("mouth", 3, coupled=False)])
    model = FacialPalsyModel(cfg)
    print(f"embed_dim={model.embed_dim} (marlin_proj 128 + dynamics 64); reweighted", flush=True)
    train_multitask(model, train_ds, val_ds, MTTrainConfig(
        epochs=80, batch_size=64, lr=5e-4, weight_decay=3e-2, device=device,
        monitor_task="eyes", monitor_n_classes=3, log_every=20,
        early_stopping_patience=15, early_stopping_warmup=10, seed=0))
    res = evaluate(model, val_recs, device)

    ckpt = OUT / "checkpoints" / "warmstart_v3_reweighted.pt"
    torch.save({
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "tasks": [(t.name, t.n_classes, t.coupled) for t in model.multitask.tasks],
        "model_cfg": {"mp_feat_dim": MP_FEAT_DIM, "n_actions": 1, "temporal_hidden": 64,
                      "temporal_out": 64, "trunk_hidden": 96, "dropout": 0.1,
                      "temporal_pool": "attention", "marlin_proj_dim": 128, "stream_layernorm": True},
        "quality": {"mode": "normalize", "work_size": 112},
        "label_scheme": {"binary": ["healthy", "palsy"], "eyes": ["Normal", "Slight", "Strong"],
                         "mouth": ["Normal", "Slight", "Strong"]},
        "val_metrics": res, "sources": "PalsyNet+FNP+YFP, GPU v3 (reweighted geometric)",
    }, ckpt)
    (OUT / "run_v3_results.json").write_text(json.dumps(res, indent=2))
    print("\n===== V3 SUMMARY (reweighted) vs v2-attention (eyes 0.426 / mouth 0.859) =====", flush=True)
    print(f"  v3: binary AUC {res.get('binary',{}).get('auc')}  "
          f"eyes QWK {res.get('eyes',{}).get('kappa')}  mouth QWK {res.get('mouth',{}).get('kappa')}", flush=True)
    print(f"saved {ckpt}", flush=True)


if __name__ == "__main__":
    main()
