"""autoresearch-style autonomous search for the best facial-palsy model+data config.

Adapts karpathy/autoresearch's loop to OUR task: instead of editing a GPT and
minimizing val_bpb, each experiment picks a CONFIG (model arch + data processing +
optimizer) and we MAXIMIZE the clean-holdout region QWK. One fixed metric,
keep-if-better, logged to outputs/fp_results.tsv.

METRIC = mean(eyes_QWK, mouth_QWK) on the leak-safe holdouts (FNP-valid + YFP-held
subjects). Eval splits are FIXED and clean (docs/leakage_policy.md); only train /
model / data-processing vary. Bundles cached to outputs/fp_cache.pt for fast reload.

Batch mode: pass a JSON file with {"experiments":[{"name":..., ...cfgoverrides}, ...]}.
Runs each sequentially (cache + imports loaded once), prints METRIC, appends results.tsv.

Run on pod:  .venv/bin/python scripts/fp_research.py <experiments.json>
"""
from __future__ import annotations

import json
import sys
import time
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
from src.models.multitask import TaskSpec, multitask_loss  # noqa: E402
from src.models.ordinal import predict_grade  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

OUT = ROOT / "outputs"
MANIFEST = OUT / "train_manifest_v4.json"
CACHE = OUT / "fp_cache.pt"
RESULTS = OUT / "fp_results.tsv"
MP_FEAT_DIM = 72
ACTION = "clip"
N_CLASSES = {"binary": 2, "coarse3": 3, "eyes": 3, "mouth": 3}
COUPLED = {"binary": True, "coarse3": True, "eyes": False, "mouth": False}
LW_DEFAULT = {"binary": 0.5, "coarse3": 0.5, "eyes": 0.3, "mouth": 0.3}
DEFAULT = {
    "temporal_pool": "attention", "marlin_proj_dim": None, "stream_layernorm": False,
    "temporal_hidden": 64, "temporal_out": 64, "trunk_hidden": 96, "dropout": 0.1,
    "action_pool": "mean", "lr": 5e-4, "weight_decay": 3e-2, "batch_size": 256,
    "epochs": 40, "eval_every": 2, "seed": 0, "loss_weights": None,
    "sources": ["palsy", "fnp", "yfp", "anisa", "kaggle"],
    "tasks": ["binary", "coarse3", "eyes", "mouth"],
}


def build_cache():
    man = json.loads(MANIFEST.read_text())
    recs = []
    for e in man:
        p = OUT / e["npz"]
        if not p.exists():
            continue
        d = np.load(p)
        recs.append({"marlin": d["marlin"].astype(np.float32), "mp_seq": d["mp_seq"].astype(np.float32),
                     "mp_mask": d["mp_mask"].astype(bool), "label": int(e["label"]),
                     "task": e["task"], "split": e["split"], "source": e["source"]})
    torch.save(recs, CACHE)
    return recs


def _rec(r, i):
    return MultiStreamRecord(patient_id=f"{r['source']}_{r['task']}_{i}", label=r["label"], task=r["task"],
                             actions=[ActionBundle(marlin=r["marlin"], mp_seq=r["mp_seq"], mp_mask=r["mp_mask"])])


def run_one(cfg, recs, device):
    srcs, tasks = set(cfg["sources"]), set(cfg["tasks"])
    train = [r for r in recs if r["source"] in srcs and r["task"] in tasks and r["split"] != "val"]
    val = [r for r in recs if r["source"] in srcs and r["task"] in tasks and r["split"] == "val"]
    train_ds = MultiStreamPatientDataset([_rec(r, i) for i, r in enumerate(train)], actions=[ACTION], mp_feat_dim=MP_FEAT_DIM)
    val_recs = [_rec(r, i) for i, r in enumerate(val)]
    val_ds = MultiStreamPatientDataset(val_recs, actions=[ACTION], mp_feat_dim=MP_FEAT_DIM)

    torch.manual_seed(cfg["seed"])
    ts = [t for t in ("binary", "coarse3", "eyes", "mouth") if t in tasks]
    lw = cfg["loss_weights"] or LW_DEFAULT
    model = FacialPalsyModel(FacialPalsyConfig(
        mp_feat_dim=MP_FEAT_DIM, n_actions=1, temporal_hidden=cfg["temporal_hidden"],
        temporal_out=cfg["temporal_out"], trunk_hidden=cfg["trunk_hidden"], dropout=cfg["dropout"],
        action_pool=cfg["action_pool"], temporal_pool=cfg["temporal_pool"],
        marlin_proj_dim=cfg["marlin_proj_dim"], stream_layernorm=cfg["stream_layernorm"],
        tasks=[TaskSpec(t, N_CLASSES[t], coupled=COUPLED[t], loss_weight=lw.get(t, 0.3)) for t in ts])).to(device)
    tl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate_multistream)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    vbatch = next(iter(DataLoader(val_ds, batch_size=len(val_ds), collate_fn=collate_multistream)))
    vbatch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in vbatch.items()}
    v_idx = {t: [i for i, r in enumerate(val_recs) if r.task == t] for t in ("eyes", "mouth")}

    def evalq():
        model.eval()
        with torch.no_grad():
            out = model(vbatch["marlin_emb"], vbatch["marlin_mask"], vbatch["mp_seq"], vbatch["mp_mask"], vbatch["action_present"])
        q = {}
        for t in ("eyes", "mouth"):
            if not v_idx[t]:
                continue
            idx = torch.tensor(v_idx[t], device=device)
            pred = predict_grade(out[t].index_select(0, idx)).cpu().numpy()
            true = np.array([val_recs[i].label for i in v_idx[t]])
            q[t] = HBMetrics.from_predictions(true, pred, n_classes=3).quadratic_kappa
        return q

    best, bestq = -1.0, {}
    t0 = time.time()
    ev = cfg.get("eval_every", 2)
    for ep in range(cfg["epochs"]):
        model.train()
        for batch in tl:
            b = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
            opt.zero_grad()
            out = model(b["marlin_emb"], b["marlin_mask"], b["mp_seq"], b["mp_mask"], b["action_present"])
            loss, _ = multitask_loss(out, b["label"], b["task_ids"], model.multitask.tasks)
            loss.backward(); torch.nn.utils.clip_grad_norm_(params, 5.0); opt.step()
        sched.step()
        if (ep + 1) % ev == 0 or ep == cfg["epochs"] - 1:
            q = evalq()
            m = float(np.mean([q.get("eyes", 0.0), q.get("mouth", 0.0)]))
            if m > best:
                best, bestq = m, q
    return round(best, 4), bestq, len(train), len(val), time.time() - t0


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}; loading cache...", flush=True)
    recs = torch.load(CACHE, weights_only=False) if CACHE.exists() else build_cache()
    print(f"cache: {len(recs)} records", flush=True)
    spec = json.loads(Path(sys.argv[1]).read_text())
    exps = spec["experiments"] if "experiments" in spec else [spec]
    if not RESULTS.exists():
        RESULTS.write_text("name\tmetric\teyes\tmouth\tn_train\tsecs\tconfig\n")
    for e in exps:
        name = e.pop("name", "exp")
        cfg = dict(DEFAULT); cfg.update(e)
        try:
            metric, q, ntr, nval, secs = run_one(cfg, recs, device)
            line = (f"{name}\t{metric}\t{q.get('eyes', float('nan')):.3f}\t{q.get('mouth', float('nan')):.3f}"
                    f"\t{ntr}\t{secs:.0f}\t{json.dumps(e)}")
        except Exception as ex:
            line = f"{name}\t0.0\tNaN\tNaN\t0\t0\tCRASH: {type(ex).__name__}: {str(ex)[:120]}"
        print("RESULT\t" + line, flush=True)
        with RESULTS.open("a") as f:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
