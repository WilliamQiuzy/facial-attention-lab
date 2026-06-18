"""Training run #5: YFP region-ORDINAL severity (eyes + mouth), the second graded
dataset after FNP (Run #3) — but ~24x more labeled data, and with SUBJECT-LEVEL CV.

YFP (YouTube Facial Palsy) labels are Pascal-VOC XML per frame; object names encode
a 3-level ordinal severity per region:
    eyes : Normal_Eyes=0, SlightPalsy_Eyes=1, StrongPalsy_Eyes=2
    mouth: Normal_Mouth=0, SlightPalsy_Mouth=1, StrongPalsy_Mouth=2
(max severity if several boxes for a region). Two UNCOUPLED region tasks with
ordinal cut-point heads (docs/model_design.md §4.1). Images run as the degenerate
clip (MARLIN tiled to 16 frames, length-1 MediaPipe); quality normalizer ON.

YFP has no official split → **subject-level GroupKFold** over the 16 subjects that
have both image + label (no frame from a subject spans train/val). Frames are
heavily redundant (consecutive video frames), so we subsample up to
MAX_PER_SUBJECT per subject (rare Normal/Strong frames kept preferentially).

Run:
  KMP_DUPLICATE_LIB_OK=TRUE \
  /opt/homebrew/Caskroom/miniforge/base/envs/dev/bin/python scripts/run5_yfp_region.py
"""
from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

YFP = ROOT / "data" / "external" / "YFP"
IMG_DIRS = ["Image", "Image2", "Image3", "Image4"]
XML_DIR = YFP / "Image_large_XML"
CACHE = ROOT / "outputs" / "yfp_bundles"
ACTION = "clip"
MP_FEAT_DIM = 72
MAX_PER_SUBJECT = 200
N_SPLITS = 4
SEED = 0

EYE_SEV = {"Normal_Eyes": 0, "SlightPalsy_Eyes": 1, "StrongPalsy_Eyes": 2}
MOUTH_SEV = {"Normal_Mouth": 0, "SlightPalsy_Mouth": 1, "StrongPalsy_Mouth": 2}


def parse_xml(p: Path) -> dict:
    """-> {'eyes': sev|None, 'mouth': sev|None} (max over boxes)."""
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


def build_index() -> list[dict]:
    """Match images to XML labels; subsample per subject. Returns list of
    {subject, frame, img_path, eyes, mouth}."""
    # image (subject, frame) -> path
    img = {}
    for top in IMG_DIRS:
        for p in (YFP / top).rglob("*.bmp"):
            img[(p.parent.name, p.stem)] = p
    # labels
    per_subject = defaultdict(list)
    for xp in XML_DIR.rglob("*.xml"):
        subj = xp.parent.name
        if subj in ("eyes", "mouth"):
            continue
        key = (subj, xp.stem)
        if key not in img:
            continue
        lab = parse_xml(xp)
        if lab["eyes"] is None and lab["mouth"] is None:
            continue
        per_subject[subj].append({"subject": subj, "frame": xp.stem,
                                   "img_path": str(img[key]),
                                   "eyes": lab["eyes"], "mouth": lab["mouth"]})
    # subsample per subject, preferring rare (non-Slight) frames
    rng = np.random.default_rng(SEED)
    index = []
    for subj, items in per_subject.items():
        if len(items) <= MAX_PER_SUBJECT:
            index += items
            continue
        rare = [it for it in items if (it["eyes"] in (0, 2)) or (it["mouth"] in (0, 2))]
        common = [it for it in items if it not in rare]
        keep = list(rare)
        if len(keep) < MAX_PER_SUBJECT and common:
            extra = rng.choice(len(common), size=min(MAX_PER_SUBJECT - len(keep), len(common)),
                               replace=False)
            keep += [common[i] for i in extra]
        if len(keep) > MAX_PER_SUBJECT:
            sel = rng.choice(len(keep), size=MAX_PER_SUBJECT, replace=False)
            keep = [keep[i] for i in sel]
        index += keep
    return index


def encode(index, enc, mp_ext, normalizer, reextract=False) -> list[dict]:
    """Encode each indexed frame; returns index entries that have a usable bundle."""
    ok, n_noface = [], 0
    for it in index:
        sid = f"{it['subject']}/{it['frame']}"
        out = CACHE / sid / f"{ACTION}.npz"
        if not (out.exists() and not reextract):
            img = cv2.imread(it["img_path"])
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
        it["sid"] = sid
        ok.append(it)
    print(f"  encoded {len(ok)} frames ({n_noface} dropped: no face)")
    return ok


def make_records(entries):
    from src.datasets.patient_multistream import ActionBundle, MultiStreamRecord
    recs, groups = [], []
    for it in entries:
        d = np.load(CACHE / it["sid"] / f"{ACTION}.npz")
        bundle = ActionBundle(marlin=d["marlin"].astype(np.float32),
                              mp_seq=d["mp_seq"].astype(np.float32),
                              mp_mask=d["mp_mask"].astype(bool))
        for task in ("eyes", "mouth"):
            if it[task] is not None:
                recs.append(MultiStreamRecord(
                    patient_id=f"yfp_{it['subject']}_{it['frame']}_{task}",
                    label=int(it[task]), task=task, actions=[bundle]))
                groups.append(it["subject"])
    return recs, np.array(groups)


def make_model(seed=0):
    from src.models.facial_palsy_model import FacialPalsyModel, FacialPalsyConfig
    from src.models.multitask import TaskSpec
    torch.manual_seed(seed)
    return FacialPalsyModel(FacialPalsyConfig(
        mp_feat_dim=MP_FEAT_DIM, n_actions=1, temporal_hidden=64, temporal_out=64,
        trunk_hidden=64, dropout=0.1,
        tasks=[TaskSpec("eyes", 3, coupled=False), TaskSpec("mouth", 3, coupled=False)]))


def main():
    from sklearn.model_selection import GroupKFold
    from torch.utils.data import Subset, DataLoader

    from src.preprocessing.action_bundle import MediaPipeFeatureExtractor
    from src.models.backbones.marlin_video import MarlinVideoEncoder
    from src.preprocessing.image_quality import QualityConfig, QualityNormalizer
    from src.training.train_multitask import MTTrainConfig, train_multitask
    from src.datasets.patient_multistream import MultiStreamPatientDataset, collate_multistream
    from src.evaluation.hb_metrics import HBMetrics
    from src.models.ordinal import predict_grade

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("Indexing YFP (match images to XML labels, subsample)...")
    index = build_index()
    subj_counts = Counter(it["subject"] for it in index)
    print(f"  {len(index)} frames over {len(subj_counts)} subjects "
          f"(<= {MAX_PER_SUBJECT}/subj): {dict(sorted(subj_counts.items(), key=lambda x:int(x[0])))}")

    print(f"Encoding through frozen MARLIN (+ quality normalizer) on {device}...")
    enc = MarlinVideoEncoder.from_default_weights().to(device).eval()
    mp_ext = MediaPipeFeatureExtractor()
    normalizer = QualityNormalizer(QualityConfig(mode="normalize", work_size=112))
    entries = encode(index, enc, mp_ext, normalizer)

    recs, groups = make_records(entries)
    for task in ("eyes", "mouth"):
        c = Counter(r.label for r in recs if r.task == task)
        print(f"  {task}: {sum(c.values())} records, level dist {dict(sorted(c.items()))}")

    # subject-level GroupKFold; pool out-of-fold predictions per region
    gkf = GroupKFold(n_splits=N_SPLITS)
    full_ds = MultiStreamPatientDataset(recs, actions=[ACTION], mp_feat_dim=MP_FEAT_DIM)
    oof_pred = {t: {} for t in ("eyes", "mouth")}  # idx -> pred
    for fold, (tr, te) in enumerate(gkf.split(np.zeros(len(recs)), groups=groups), 1):
        model = make_model()
        train_multitask(model, Subset(full_ds, tr.tolist()), Subset(full_ds, te.tolist()),
                        MTTrainConfig(epochs=60, batch_size=32, lr=5e-4, weight_decay=3e-2,
                                      device="cpu", monitor_task="mouth", monitor_n_classes=3,
                                      log_every=999, early_stopping_patience=12,
                                      early_stopping_warmup=8, seed=0))
        model.eval()
        b = next(iter(DataLoader(Subset(full_ds, te.tolist()), batch_size=len(te),
                                 collate_fn=collate_multistream)))
        with torch.no_grad():
            out = model(b["marlin_emb"], b["marlin_mask"], b["mp_seq"], b["mp_mask"], b["action_present"])
            preds = {t: predict_grade(out[t]).cpu().numpy() for t in ("eyes", "mouth")}
        for local_i, rec_i in enumerate(te):
            t = recs[rec_i].task
            oof_pred[t][rec_i] = int(preds[t][local_i])
        sg = sorted(set(groups[te].tolist()), key=lambda x: int(x))
        print(f"  fold {fold}: {len(te)} recs, held-out subjects {sg}")

    res = {"dataset": "YFP region-ordinal (3-level eye/mouth severity), subject-level CV"}
    for t in ("eyes", "mouth"):
        idxs = [i for i in range(len(recs)) if recs[i].task == t and i in oof_pred[t]]
        true = np.array([recs[i].label for i in idxs])
        pred = np.array([oof_pred[t][i] for i in idxs])
        m = HBMetrics.from_predictions(true, pred, n_classes=3)
        res[t] = {"n": len(idxs), "kappa": round(m.quadratic_kappa, 3),
                  "mae": round(m.mae_grades, 3), "acc": round(m.accuracy, 3),
                  "confusion": m.confusion.tolist()}
    print("\n================= RUN #5 SUMMARY (YFP region ordinal, subject-CV) =================")
    print(json.dumps({k: v for k, v in res.items() if k != "dataset"}, indent=2))
    print("==================================================================================")
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "results.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
