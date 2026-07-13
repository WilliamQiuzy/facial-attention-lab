"""Training run #6: the UNIFIED public-data warm-start model.

Stops doing one-off datasets and trains ONE FacialPalsyModel jointly on every
usable source via the heterogeneous-label multi-task design (docs/model_design.md
§4). This is the deployable v1 checkpoint for the "train on public data now →
test on iPhone → fine-tune on Mayo/MEEI later" strategy.

Sources (all already cached as bundles by Runs #1/#3/#5 — no re-encoding):
  PalsyNet  -> `binary` head (2-cls, COUPLED to global severity s); its own
               in-domain healthy negatives => honest.
  FNP + YFP -> shared `eyes` / `mouth` region heads (3-cls ordinal, UNCOUPLED,
               own regional severity). FNP is natively 4-level; we map it to the
               common 3-level {Normal, Slight, Strong} so both sources train the
               SAME heads (more data per head):
                   FNP 0->0(normal), {1 Weak,2 Mid}->1(slight), 3 Severe->2(strong)
                   YFP 0->0,           1 Slight     ->1,         2 Strong  ->2

Holdout (honest, per source): PalsyNet stratified subject split; FNP uses its own
valid split; YFP holds out a subset of subjects. We report per-task AND per-source
val metrics (so source label-prior shortcuts are visible), then SAVE the model.

Run:
  KMP_DUPLICATE_LIB_OK=TRUE \
  /opt/homebrew/Caskroom/miniforge/base/envs/dev/bin/python scripts/run6_unified.py
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
sys.path.insert(0, str(ROOT / "scripts"))

import facial_paralysis.scripts.run3_fnp_region as r3   # noqa: E402  (parse_split_labels, EYE_SEV, MOUTH_SEV, CACHE, SPLITS)
import facial_paralysis.scripts.run5_yfp_region as r5   # noqa: E402  (build_index, CACHE)

ACTION = "clip"
MP_FEAT_DIM = 72
CKPT = ROOT / "outputs" / "checkpoints" / "warmstart_v1.pt"
PALSY_CACHE = ROOT / "outputs" / "palsynet_bundles"

# FNP 4-level -> common 3-level
FNP_TO_3 = {0: 0, 1: 1, 2: 1, 3: 2}


def _bundle(npz_path: Path):
    from facial_paralysis.scripts._bundle_io import load_action_bundle
    return load_action_bundle(
        npz_path,
        allow_legacy_schema=True,
        expected_feat_dim=MP_FEAT_DIM,
    )


# ----------------------------------------------------------------------
# Loaders (read existing caches only)
# ----------------------------------------------------------------------
def load_palsynet():
    """-> list[(record, split)] with stratified subject holdout (20% val)."""
    from sklearn.model_selection import train_test_split
    from facial_paralysis.src.datasets.patient_multistream import MultiStreamRecord
    import csv
    labels = {}
    with (PALSY_CACHE / "labels.csv").open() as f:
        for row in csv.DictReader(f):
            labels[row["patient_id"].strip()] = int(row["label"])
    subs = sorted(labels)
    y = [labels[s] for s in subs]
    tr, va = train_test_split(subs, test_size=0.2, random_state=0, stratify=y)
    vaset = set(va)
    out = []
    for s in subs:
        p = PALSY_CACHE / s / f"{ACTION}.npz"
        if not p.exists():
            continue
        rec = MultiStreamRecord(patient_id=f"palsy_{s}", label=labels[s],
                                task="binary", actions=[_bundle(p)])
        out.append((rec, "val" if s in vaset else "train"))
    return out


def load_fnp():
    """FNP eyes/mouth -> 3-level; train+test split => train, valid => val."""
    from facial_paralysis.src.datasets.patient_multistream import MultiStreamRecord
    out = []
    for sp in r3.SPLITS:
        labels = r3.parse_split_labels(sp)
        split = "val" if sp == "valid" else "train"
        for fn, lab in labels.items():
            stem = Path(fn).stem[:60]
            p = r3.CACHE / f"{sp}/{stem}" / f"{ACTION}.npz"
            if not p.exists():
                continue
            b = _bundle(p)
            for task in ("eyes", "mouth"):
                if lab[task] is not None:
                    out.append((MultiStreamRecord(
                        patient_id=f"fnp_{sp}_{stem}_{task}",
                        label=FNP_TO_3[int(lab[task])], task=task, actions=[b]), split))
    return out


def load_yfp(val_frac=0.25):
    """YFP eyes/mouth (already 3-level); hold out a subset of subjects for val."""
    from facial_paralysis.src.datasets.patient_multistream import MultiStreamRecord
    index = r5.build_index()
    subs = sorted({it["subject"] for it in index}, key=lambda x: int(x))
    rng = np.random.default_rng(0)
    n_val = max(1, int(round(len(subs) * val_frac)))
    val_subs = set(rng.choice(subs, size=n_val, replace=False).tolist())
    out = []
    for it in index:
        p = r5.CACHE / it["subject"] / it["frame"] / f"{ACTION}.npz"
        if not p.exists():
            continue
        b = _bundle(p)
        split = "val" if it["subject"] in val_subs else "train"
        for task in ("eyes", "mouth"):
            if it[task] is not None:
                out.append((MultiStreamRecord(
                    patient_id=f"yfp_{it['subject']}_{it['frame']}_{task}",
                    label=int(it[task]), task=task, actions=[b]), split))
    return out, sorted(val_subs, key=lambda x: int(x))


# ----------------------------------------------------------------------
def make_model(seed=0):
    from facial_paralysis.src.models.facial_palsy_model import FacialPalsyModel, FacialPalsyConfig
    from facial_paralysis.src.models.multitask import TaskSpec
    torch.manual_seed(seed)
    return FacialPalsyModel(FacialPalsyConfig(
        mp_feat_dim=MP_FEAT_DIM, n_actions=1, temporal_hidden=64, temporal_out=64,
        trunk_hidden=96, dropout=0.1,
        tasks=[TaskSpec("binary", 2, coupled=True),
               TaskSpec("eyes", 3, coupled=False),
               TaskSpec("mouth", 3, coupled=False)]))


def _source(pid: str) -> str:
    return pid.split("_", 1)[0]


def evaluate(model, val_recs):
    from facial_paralysis.src.datasets.patient_multistream import MultiStreamPatientDataset, collate_multistream
    from facial_paralysis.src.evaluation.hb_metrics import HBMetrics
    from facial_paralysis.src.models.ordinal import predict_grade, cum_probs
    from sklearn.metrics import roc_auc_score
    from torch.utils.data import DataLoader

    ds = MultiStreamPatientDataset(val_recs, actions=[ACTION], mp_feat_dim=MP_FEAT_DIM)
    b = next(iter(DataLoader(ds, batch_size=len(ds), collate_fn=collate_multistream)))
    model.eval()
    with torch.no_grad():
        out = model(b["marlin_emb"], b["marlin_mask"], b["mp_seq"], b["mp_mask"], b["action_present"])
    res = {}
    # binary (AUC)
    bi = [i for i, r in enumerate(val_recs) if r.task == "binary"]
    if bi:
        idx = torch.tensor(bi)
        p = cum_probs(out["binary"].index_select(0, idx))[:, 0].cpu().numpy()
        y = np.array([val_recs[i].label for i in bi])
        res["binary"] = {"n": len(bi), "auc": round(float(roc_auc_score(y, p)), 3) if len(set(y)) == 2 else None,
                         "acc": round(float(((p > 0.5).astype(int) == y).mean()), 3)}
    # eyes / mouth (kappa) overall + per source
    for task in ("eyes", "mouth"):
        ti = [i for i, r in enumerate(val_recs) if r.task == task]
        if not ti:
            continue
        idx = torch.tensor(ti)
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
    from facial_paralysis.src.datasets.patient_multistream import MultiStreamPatientDataset
    from facial_paralysis.src.training.train_multitask import MTTrainConfig, train_multitask

    print("Loading cached bundles from all sources...")
    pals = load_palsynet()
    fnp = load_fnp()
    yfp, yfp_val_subs = load_yfp()
    print(f"  PalsyNet: {len(pals)} recs   FNP: {len(fnp)} recs   YFP: {len(yfp)} recs "
          f"(YFP val subjects {yfp_val_subs})")

    allrec = pals + fnp + yfp
    train_recs = [r for r, sp in allrec if sp == "train"]
    val_recs = [r for r, sp in allrec if sp == "val"]
    print(f"  TRAIN {len(train_recs)} recs   VAL {len(val_recs)} recs")
    for name, recs in [("train", train_recs), ("val", val_recs)]:
        c = Counter((r.task, r.label) for r in recs)
        print(f"  [{name}] " + "  ".join(
            f"{t}:{sorted({k[1]: v for k, v in c.items() if k[0]==t}.items())}"
            for t in ("binary", "eyes", "mouth")))

    train_ds = MultiStreamPatientDataset(train_recs, actions=[ACTION], mp_feat_dim=MP_FEAT_DIM)
    val_ds = MultiStreamPatientDataset(val_recs, actions=[ACTION], mp_feat_dim=MP_FEAT_DIM)

    print("\nTraining unified warm-start (binary + eyes + mouth)...")
    model = make_model()
    hist = train_multitask(model, train_ds, val_ds, MTTrainConfig(
        epochs=80, batch_size=64, lr=5e-4, weight_decay=3e-2, device="cpu",
        monitor_task="eyes", monitor_n_classes=3, log_every=10,
        early_stopping_patience=15, early_stopping_warmup=10, seed=0))

    res = evaluate(model, val_recs)
    print("\n================= RUN #6 SUMMARY (unified warm-start) =================")
    print(json.dumps(res, indent=2))
    print("======================================================================")

    # Save deployable checkpoint
    CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "tasks": [(t.name, t.n_classes, t.coupled) for t in model.multitask.tasks],
        "model_cfg": {"mp_feat_dim": MP_FEAT_DIM, "n_actions": 1, "temporal_hidden": 64,
                      "temporal_out": 64, "trunk_hidden": 96, "dropout": 0.1},
        "quality": {"mode": "normalize", "work_size": 112},
        "label_scheme": {"binary": ["healthy", "palsy"],
                         "eyes": ["Normal", "Slight", "Strong"],
                         "mouth": ["Normal", "Slight", "Strong"]},
        "val_metrics": res,
        "sources": "PalsyNet(binary)+FNP+YFP(eyes/mouth 3-level)",
    }, CKPT)
    print(f"\nsaved checkpoint -> {CKPT}")
    (ROOT / "outputs" / "run6_results.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
