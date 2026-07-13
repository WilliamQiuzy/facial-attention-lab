"""Run #9: temporal-pooling ablation on YFP clips — does the dynamics signal
survive a better pool? (disambiguates Run #7's negative result)

Run #7 fed YFP as real 16-frame motion clips and found motion did NOT help
(eyes QWK 0.26 vs single-frame 0.47). Its own hypothesis: the GRU's masked-MEAN
over the ~3 s clip *dilutes* a transient event (incomplete eye closure during a
blink) — i.e. the pooling, not the motion, was the problem. This run tests that
directly: SAME cached clips, SAME subject-level splits, SAME model size as
Run #7, varying ONLY the GRU temporal pool ∈ {mean, max, attention}.

  mean       — Run #7's setting (baseline to reproduce ≈0.26)
  max/peak   — keep the most-asymmetric frame (should rescue the blink signal)
  attention  — learn which frames matter

Pure replay from cache: reads outputs/yfp_clip_bundles/<subj>/<anchor>/clip.npz
(MARLIN + 16-frame MediaPipe trajectory, already extracted) and labels from the
YFP Pascal-VOC XML. No MARLIN / MediaPipe / raw frames needed.

Run:
  KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/run9_temporal_pool.py
"""
from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
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
from src.models.ordinal import predict_grade  # noqa: E402
from src.training.train_multitask import MTTrainConfig, train_multitask  # noqa: E402
from scripts._bundle_io import load_action_bundle  # noqa: E402

CACHE = ROOT / "outputs" / "yfp_clip_bundles"
XML_DIR = ROOT / "data" / "external" / "YFP" / "Image_large_XML"
ACTION = "clip"
MP_FEAT_DIM = 72
N_SPLITS = 4
SEED = 0
POOLS = ["mean", "max", "attention"]
TASKS = ("eyes", "mouth")

EYE_SEV = {"Normal_Eyes": 0, "SlightPalsy_Eyes": 1, "StrongPalsy_Eyes": 2}
MOUTH_SEV = {"Normal_Mouth": 0, "SlightPalsy_Mouth": 1, "StrongPalsy_Mouth": 2}


def parse_xml(p: Path) -> dict:
    rec = {"eyes": None, "mouth": None}
    try:
        root = ET.parse(p).getroot()
    except Exception:
        return rec
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        if name in EYE_SEV:
            rec["eyes"] = max(rec["eyes"] or 0, EYE_SEV[name])
        elif name in MOUTH_SEV:
            rec["mouth"] = max(rec["mouth"] or 0, MOUTH_SEV[name])
    return rec


def build_records_from_cache():
    """Enumerate cached clip bundles, attach XML labels, build per-task records.
    Returns (records, groups) where groups[i] is the subject id for CV grouping."""
    recs, groups = [], []
    clips = sorted(CACHE.rglob(f"{ACTION}.npz"))
    n_clips = n_missing_xml = 0
    for npz in clips:
        anchor = npz.parent.name
        subject = npz.parent.parent.name
        xml = XML_DIR / subject / f"{anchor}.xml"
        if not xml.exists():
            n_missing_xml += 1
            continue
        lab = parse_xml(xml)
        if lab["eyes"] is None and lab["mouth"] is None:
            continue
        bundle = load_action_bundle(
            npz,
            allow_legacy_schema=True,
            expected_feat_dim=MP_FEAT_DIM,
        )
        n_clips += 1
        base = f"yfp_{subject}_{anchor}"
        for task in TASKS:
            if lab[task] is not None:
                recs.append(MultiStreamRecord(f"{base}_{task}", int(lab[task]), task, [bundle]))
                groups.append(subject)
    print(f"  {n_clips} clips used ({n_missing_xml} missing XML)")
    return recs, np.array(groups)


def make_model(pool: str, seed: int = 0) -> FacialPalsyModel:
    torch.manual_seed(seed)
    return FacialPalsyModel(FacialPalsyConfig(
        mp_feat_dim=MP_FEAT_DIM, n_actions=1,
        temporal_hidden=64, temporal_out=64, trunk_hidden=96, dropout=0.1,
        temporal_pool=pool,                       # <-- the only knob varied
        tasks=[TaskSpec("eyes", 3, coupled=False), TaskSpec("mouth", 3, coupled=False)],
    ))


def run_cv(pool: str, recs, groups) -> dict:
    from sklearn.model_selection import GroupKFold
    from torch.utils.data import Subset, DataLoader

    full = MultiStreamPatientDataset(recs, actions=[ACTION], mp_feat_dim=MP_FEAT_DIM)
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = {t: {} for t in TASKS}
    for fold, (tr, te) in enumerate(gkf.split(np.zeros(len(recs)), groups=groups), 1):
        model = make_model(pool)
        train_multitask(
            model, Subset(full, tr.tolist()), Subset(full, te.tolist()),
            MTTrainConfig(epochs=60, batch_size=32, lr=5e-4, weight_decay=3e-2,
                          device="cpu", monitor_task="mouth", monitor_n_classes=3,
                          log_every=999, early_stopping_patience=12,
                          early_stopping_warmup=8, seed=0),
        )
        model.eval()
        loader = DataLoader(Subset(full, te.tolist()), batch_size=len(te),
                            collate_fn=collate_multistream)
        b = next(iter(loader))
        with torch.no_grad():
            out = model(b["marlin_emb"], b["marlin_mask"], b["mp_seq"],
                        b["mp_mask"], b["action_present"])
            preds = {t: predict_grade(out[t]).cpu().numpy() for t in TASKS}
        for li, ri in enumerate(te):
            t = recs[ri].task
            oof[t][ri] = int(preds[t][li])
        print(f"  [{pool}] fold {fold}: held-out {sorted(set(groups[te]), key=int)}")

    out = {}
    for t in TASKS:
        idxs = [i for i in range(len(recs)) if recs[i].task == t and i in oof[t]]
        true = np.array([recs[i].label for i in idxs])
        pred = np.array([oof[t][i] for i in idxs])
        m = HBMetrics.from_predictions(true, pred, n_classes=3)
        maj = Counter(true).most_common(1)[0][1] / len(true)
        out[t] = {"n": len(idxs), "kappa": round(m.quadratic_kappa, 3),
                  "mae": round(m.mae_grades, 3), "acc": round(m.accuracy, 3),
                  "majority_acc": round(maj, 3)}
    return out


def main():
    print("Run #9 — temporal-pooling ablation on cached YFP clips")
    recs, groups = build_records_from_cache()
    for t in TASKS:
        c = Counter(r.label for r in recs if r.task == t)
        print(f"  {t}: {sum(c.values())} recs, dist {dict(sorted(c.items()))}")
    print(f"  subjects: {sorted(set(groups), key=int)}")

    results = {"dataset": "YFP 16-frame clips, subject-level GroupKFold, cache replay",
               "vs_run7_mean": {"eyes": 0.26, "mouth": 0.61},
               "n_splits": N_SPLITS, "pools": {}}
    for pool in POOLS:
        print(f"\n--- pool = {pool} ---")
        results["pools"][pool] = run_cv(pool, recs, groups)

    print("\n================= RUN #9 SUMMARY (temporal pooling) =================")
    hdr = f"{'pool':>10s} | {'eyes QWK':>9s} {'eyes acc':>9s} | {'mouth QWK':>10s} {'mouth acc':>10s}"
    print(hdr); print("-" * len(hdr))
    for pool in POOLS:
        r = results["pools"][pool]
        print(f"{pool:>10s} | {r['eyes']['kappa']:>9.3f} {r['eyes']['acc']:>9.3f} | "
              f"{r['mouth']['kappa']:>10.3f} {r['mouth']['acc']:>10.3f}")
    print(f"{'run7 mean':>10s} | {0.26:>9.3f} {'—':>9s} | {0.61:>10.3f} {'—':>10s}")
    print("====================================================================")

    out = CACHE / "pooling_ablation.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
