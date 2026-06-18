"""Run #8: targeted fixes for the FAILING eyes head (acc 0.51 < majority 0.71).

Three research-backed levers, none requiring Mayo data:
  (1a) EAR + region geometry  — explicit eye-closure (eye aspect ratio) and L/R
       region asymmetries added to the MediaPipe stream. Whole-face MARLIN pooling
       loses local eye-closure; EAR measures it directly (PMC7204376, Heinrich 2026).
  (1b) STREAM BALANCING — project MARLIN 768→128 + LayerNorm each stream so the big
       appearance vector stops drowning the small geometric signal in the concat.
  (2)  ASYMMETRY-SAFE FLIP AUGMENTATION — add the horizontally-flipped face as an
       extra TRAIN sample (region-severity label is side-agnostic, so it's valid);
       MediaPipe re-detects the mirror so L/R features swap correctly. ~2x data.

Same data as Run #5 (YFP single-frame, subject-level CV) so eyes/mouth QWK is a
direct A/B vs Run #5 (eyes 0.47 / acc 0.51).

Run:
  KMP_DUPLICATE_LIB_OK=TRUE \
  /opt/homebrew/Caskroom/miniforge/base/envs/dev/bin/python scripts/run8_yfp_targeted.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run5_yfp_region as r5   # noqa: E402  (build_index)

CACHE = ROOT / "outputs" / "yfp_geo_bundles"
ACTION = "clip"
MP_FEAT_DIM = 77               # 52 blendshapes + 20 asym pairs + 5 geometry
N_SPLITS = 4
SEED = 0


def encode(index, enc, mp_ext, normalizer):
    """Encode orig + horizontally-flipped bundle for each indexed frame."""
    ok, n_bad = [], 0
    for it in index:
        for tag, flip in (("", False), ("_flip", True)):
            sid = f"{it['subject']}/{it['frame']}{tag}"
            out = CACHE / sid / f"{ACTION}.npz"
            if not out.exists():
                img = cv2.imread(it["img_path"])
                if img is None:
                    n_bad += 1; break
                if flip:
                    img = cv2.flip(img, 1)
                marlin = enc.encode_clip_bgr([img], normalizer=normalizer)
                seq, mask = mp_ext.extract_sequence([img])
                if marlin is None or not mask.any():
                    n_bad += 1; continue
                out.parent.mkdir(parents=True, exist_ok=True)
                np.savez(out, marlin=marlin[None, :], mp_seq=seq, mp_mask=mask)
        it["ok"] = (CACHE / f"{it['subject']}/{it['frame']}" / f"{ACTION}.npz").exists()
    print(f"  encoded orig+flip ({n_bad} frame-encodes dropped)")
    return [it for it in index if it.get("ok")]


def make_records(entries):
    from src.datasets.patient_multistream import ActionBundle, MultiStreamRecord
    recs, groups, isflip = [], [], []
    for it in entries:
        for tag, flip in (("", 0), ("_flip", 1)):
            p = CACHE / f"{it['subject']}/{it['frame']}{tag}" / f"{ACTION}.npz"
            if not p.exists():
                continue
            d = np.load(p)
            b = ActionBundle(marlin=d["marlin"].astype(np.float32),
                             mp_seq=d["mp_seq"].astype(np.float32),
                             mp_mask=d["mp_mask"].astype(bool))
            for task in ("eyes", "mouth"):
                if it[task] is not None:
                    recs.append(MultiStreamRecord(
                        f"yfp_{it['subject']}_{it['frame']}{tag}_{task}",
                        int(it[task]), task, [b]))
                    groups.append(it["subject"]); isflip.append(flip)
    return recs, np.array(groups), np.array(isflip)


def make_model(seed=0):
    from src.models.facial_palsy_model import FacialPalsyModel, FacialPalsyConfig
    from src.models.multitask import TaskSpec
    torch.manual_seed(seed)
    return FacialPalsyModel(FacialPalsyConfig(
        mp_feat_dim=MP_FEAT_DIM, n_actions=1, temporal_hidden=96, temporal_out=96,
        trunk_hidden=96, dropout=0.1, marlin_proj_dim=128, stream_layernorm=True,
        tasks=[TaskSpec("eyes", 3, coupled=False), TaskSpec("mouth", 3, coupled=False)]))


def main():
    from torch.utils.data import Subset, DataLoader
    from src.preprocessing.action_bundle import MediaPipeFeatureExtractor
    from src.models.backbones.marlin_video import MarlinVideoEncoder
    from src.preprocessing.image_quality import QualityConfig, QualityNormalizer
    from src.training.train_multitask import MTTrainConfig, train_multitask
    from src.datasets.patient_multistream import MultiStreamPatientDataset, collate_multistream
    from src.evaluation.hb_metrics import HBMetrics
    from src.models.ordinal import predict_grade

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("Indexing YFP (same anchors as Run #5)...")
    index = r5.build_index()
    print(f"  {len(index)} frames over {len(set(it['subject'] for it in index))} subjects")
    print(f"Encoding orig+flip with EAR/geometry features on {device}...")
    enc = MarlinVideoEncoder.from_default_weights().to(device).eval()
    mp_ext = MediaPipeFeatureExtractor(with_geometry=True)
    normalizer = QualityNormalizer(QualityConfig(mode="normalize", work_size=112))
    entries = encode(index, enc, mp_ext, normalizer)

    recs, groups, isflip = make_records(entries)
    for task in ("eyes", "mouth"):
        c = Counter(recs[i].label for i in range(len(recs)) if recs[i].task == task and isflip[i] == 0)
        print(f"  {task}: {sum(c.values())} orig recs, dist {dict(sorted(c.items()))}")

    full = MultiStreamPatientDataset(recs, actions=[ACTION], mp_feat_dim=MP_FEAT_DIM)
    subs = sorted(set(groups.tolist()), key=lambda x: int(x))
    rng = np.random.default_rng(SEED); rng.shuffle(subs)
    folds = [subs[i::N_SPLITS] for i in range(N_SPLITS)]
    oof = {t: {} for t in ("eyes", "mouth")}
    for fold in range(N_SPLITS):
        val_subs = set(folds[fold])
        # train = ALL recs (orig+flip) of train subjects; val = ORIG recs of val subjects
        tr = [i for i in range(len(recs)) if groups[i] not in val_subs]
        te = [i for i in range(len(recs)) if groups[i] in val_subs and isflip[i] == 0]
        model = make_model()
        train_multitask(model, Subset(full, tr), Subset(full, te),
                        MTTrainConfig(epochs=60, batch_size=32, lr=5e-4, weight_decay=3e-2,
                                      device="cpu", monitor_task="eyes", monitor_n_classes=3,
                                      log_every=999, early_stopping_patience=12,
                                      early_stopping_warmup=8, seed=0))
        model.eval()
        b = next(iter(DataLoader(Subset(full, te), batch_size=len(te), collate_fn=collate_multistream)))
        with torch.no_grad():
            out = model(b["marlin_emb"], b["marlin_mask"], b["mp_seq"], b["mp_mask"], b["action_present"])
            preds = {t: predict_grade(out[t]).cpu().numpy() for t in ("eyes", "mouth")}
        for li, ri in enumerate(te):
            oof[recs[ri].task][ri] = int(preds[recs[ri].task][li])
        print(f"  fold {fold+1}: held-out {sorted(val_subs, key=int)}")

    res = {"dataset": "YFP single-frame + EAR/geometry + stream-balance + flip-aug",
           "vs_run5": {"eyes": {"kappa": 0.47, "acc": 0.51}}}
    for t in ("eyes", "mouth"):
        idxs = [i for i in range(len(recs)) if recs[i].task == t and i in oof[t]]
        true = np.array([recs[i].label for i in idxs]); pred = np.array([oof[t][i] for i in idxs])
        m = HBMetrics.from_predictions(true, pred, n_classes=3)
        maj = Counter(true.tolist()).most_common(1)[0][1] / len(true)
        res[t] = {"n": len(idxs), "kappa": round(m.quadratic_kappa, 3),
                  "acc": round(m.accuracy, 3), "majority_baseline_acc": round(maj, 3),
                  "mae": round(m.mae_grades, 3)}
    print("\n================= RUN #8 SUMMARY (targeted eyes fixes) =================")
    print(json.dumps({k: v for k, v in res.items() if k != "dataset"}, indent=2))
    print("=======================================================================")
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "results.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
