"""Training run #3: FNP region-ORDINAL task — the first real test of the ordinal
cut-point machinery (every prior run was binary).

FNP Detection (Roboflow) is the only collected dataset with GRADED labels: per
image, eye and mouth regions are annotated as normal / paralyzed{Weak,Mid,Severe},
i.e. a 4-level ordinal severity per region. We map:

    eyes : normalEye=0, paralyzedEyeWeak=1, paralyzedEyeMid=2, paralyzedEyeSevere=3
    mouth: normalMouth=0, paralyzedMouthWeak=1, paralyzedMouthMid=2, paralyzedMouthSevere=3

(if a region has several boxes, take the MAX severity). These attach as two
UNCOUPLED region tasks (own severity projection from the shared trunk `h`, per
docs/model_design.md §4.1) with ordinal cut-point heads. Images go through the
model as the degenerate clip case (MARLIN sees the image tiled to 16 frames;
MediaPipe sequence has length 1). The blurry public crops get the quality
normalizer we built (mode="normalize") — this is exactly the data it targets.

We use FNP's own train/valid/test split (no CV). Report quadratic-weighted kappa /
MAE / accuracy per region on valid + test.

Caveat: web-scraped images; subject overlap across splits is unknown (possible
optimism). Reported as a method check, not a clinical number.

Run:
  KMP_DUPLICATE_LIB_OK=TRUE \
  /opt/homebrew/Caskroom/miniforge/base/envs/dev/bin/python scripts/run3_fnp_region.py
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

FNP = ROOT / "data" / "external" / "roboflow_fnp"
CACHE = ROOT / "outputs" / "fnp_bundles"
ACTION = "clip"
MP_FEAT_DIM = 72
SPLITS = ("train", "valid", "test")

EYE_SEV = {"normalEye": 0, "paralyzedEyeWeak": 1, "paralyzedEyeMid": 2, "paralyzedEyeSevere": 3}
MOUTH_SEV = {"normalMouth": 0, "paralyzedMouthWeak": 1, "paralyzedMouthMid": 2, "paralyzedMouthSevere": 3}


# ----------------------------------------------------------------------
# 1. COCO -> per-image {eyes: sev|None, mouth: sev|None}
# ----------------------------------------------------------------------
def parse_split_labels(split: str) -> dict[str, dict]:
    d = json.loads((FNP / split / "_annotations.coco.json").read_text())
    id2name = {c["id"]: c["name"] for c in d["categories"]}
    id2file = {im["id"]: im["file_name"] for im in d["images"]}
    per_img: dict[str, dict] = {}
    for a in d["annotations"]:
        name = id2name[a["category_id"]]
        fn = id2file[a["image_id"]]
        rec = per_img.setdefault(fn, {"eyes": None, "mouth": None})
        if name in EYE_SEV:
            rec["eyes"] = max(rec["eyes"] or 0, EYE_SEV[name])
        elif name in MOUTH_SEV:
            rec["mouth"] = max(rec["mouth"] or 0, MOUTH_SEV[name])
    return per_img


# ----------------------------------------------------------------------
# 2. Encode each image through MARLIN (+ quality normalizer) and MediaPipe
# ----------------------------------------------------------------------
def encode_split(split: str, enc, mp_ext, normalizer, reextract: bool = False) -> dict[str, str]:
    """Returns {file_name: cache_subdir} for images with a detected face."""
    out_ids: dict[str, str] = {}
    n_noface = 0
    for vp in sorted((FNP / split).glob("*.jpg")):
        sid = f"{split}/{vp.stem[:60]}"
        out = CACHE / sid / f"{ACTION}.npz"
        if out.exists() and not reextract:
            out_ids[vp.name] = sid
            continue
        img = cv2.imread(str(vp))
        if img is None:
            continue
        marlin = enc.encode_clip_bgr([img], normalizer=normalizer)
        seq, mask = mp_ext.extract_sequence([img])
        if marlin is None or not mask.any():
            n_noface += 1
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, marlin=marlin[None, :], mp_seq=seq, mp_mask=mask,
                 mp_feat_dim=mp_ext.feat_dim)
        out_ids[vp.name] = sid
    print(f"  [{split}] {len(out_ids)} images with face, {n_noface} dropped (no face)")
    return out_ids


# ----------------------------------------------------------------------
# 3. Build records (one per (image, region task))
# ----------------------------------------------------------------------
def build_records(split: str, labels: dict, ok_ids: dict[str, str]):
    from src.datasets.patient_multistream import ActionBundle, MultiStreamRecord
    recs = []
    for fn, sid in ok_ids.items():
        lab = labels.get(fn)
        if lab is None:
            continue
        d = np.load(CACHE / sid / f"{ACTION}.npz")
        bundle = ActionBundle(marlin=d["marlin"].astype(np.float32),
                              mp_seq=d["mp_seq"].astype(np.float32),
                              mp_mask=d["mp_mask"].astype(bool))
        for task in ("eyes", "mouth"):
            if lab[task] is not None:
                recs.append(MultiStreamRecord(
                    patient_id=f"fnp_{sid.replace('/', '_')}_{task}",
                    label=int(lab[task]), task=task, actions=[bundle]))
    return recs


# ----------------------------------------------------------------------
# 4. Model / train / eval
# ----------------------------------------------------------------------
def make_model(seed=0):
    from src.models.facial_palsy_model import FacialPalsyModel, FacialPalsyConfig
    from src.models.multitask import TaskSpec
    torch.manual_seed(seed)
    return FacialPalsyModel(FacialPalsyConfig(
        mp_feat_dim=MP_FEAT_DIM, n_actions=1, temporal_hidden=64, temporal_out=64,
        trunk_hidden=64, dropout=0.1,
        tasks=[TaskSpec("eyes", 4, coupled=False), TaskSpec("mouth", 4, coupled=False)]))


def eval_task(model, records, task: str) -> dict:
    """Ordinal metrics for one region task on a record list."""
    from src.datasets.patient_multistream import MultiStreamPatientDataset, collate_multistream
    from src.evaluation.hb_metrics import HBMetrics
    from src.models.ordinal import predict_grade
    from torch.utils.data import DataLoader

    rows = [r for r in records if r.task == task]
    if not rows:
        return {}
    ds = MultiStreamPatientDataset(rows, actions=[ACTION], mp_feat_dim=MP_FEAT_DIM)
    b = next(iter(DataLoader(ds, batch_size=len(ds), collate_fn=collate_multistream)))
    model.eval()
    with torch.no_grad():
        out = model(b["marlin_emb"], b["marlin_mask"], b["mp_seq"], b["mp_mask"], b["action_present"])
        pred = predict_grade(out[task]).cpu().numpy()
    true = b["label"].cpu().numpy()
    m = HBMetrics.from_predictions(true, pred, n_classes=4)
    return {"n": int(len(rows)), "kappa": round(m.quadratic_kappa, 3),
            "mae": round(m.mae_grades, 3), "acc": round(m.accuracy, 3),
            "confusion": m.confusion.tolist()}


def main():
    from src.preprocessing.action_bundle import MediaPipeFeatureExtractor
    from src.models.backbones.marlin_video import MarlinVideoEncoder
    from src.preprocessing.image_quality import QualityConfig, QualityNormalizer
    from src.training.train_multitask import MTTrainConfig, train_multitask
    from src.datasets.patient_multistream import MultiStreamPatientDataset

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Encoding FNP through frozen MARLIN (+ quality normalizer) on {device}...")
    enc = MarlinVideoEncoder.from_default_weights().to(device).eval()
    mp_ext = MediaPipeFeatureExtractor()
    normalizer = QualityNormalizer(QualityConfig(mode="normalize", work_size=112))

    labels = {sp: parse_split_labels(sp) for sp in SPLITS}
    records = {}
    for sp in SPLITS:
        ok = encode_split(sp, enc, mp_ext, normalizer)
        records[sp] = build_records(sp, labels[sp], ok)
        cnt = Counter((r.task, r.label) for r in records[sp])
        print(f"  [{sp}] records: {len(records[sp])}  "
              f"eyes={sorted({k[1]:v for k,v in cnt.items() if k[0]=='eyes'}.items())}  "
              f"mouth={sorted({k[1]:v for k,v in cnt.items() if k[0]=='mouth'}.items())}")

    train_ds = MultiStreamPatientDataset(records["train"], actions=[ACTION], mp_feat_dim=MP_FEAT_DIM)
    val_ds = MultiStreamPatientDataset(records["valid"], actions=[ACTION], mp_feat_dim=MP_FEAT_DIM)

    print("\nTraining region ordinal heads (eyes + mouth, uncoupled)...")
    model = make_model()
    train_multitask(model, train_ds, val_ds, MTTrainConfig(
        epochs=80, batch_size=16, lr=5e-4, weight_decay=3e-2, device="cpu",
        monitor_task="eyes", monitor_n_classes=4, log_every=10,
        early_stopping_patience=15, early_stopping_warmup=10, seed=0))

    res = {"dataset": "roboflow_fnp region-ordinal (4-level eye/mouth severity)",
           "valid": {t: eval_task(model, records["valid"], t) for t in ("eyes", "mouth")},
           "test": {t: eval_task(model, records["test"], t) for t in ("eyes", "mouth")}}
    print("\n================= RUN #3 SUMMARY (region ordinal) =================")
    print(json.dumps({k: v for k, v in res.items() if k != "dataset"}, indent=2))
    print("===================================================================")
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "results.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
