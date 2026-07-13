"""Training run #7: YFP with REAL TEMPORAL CLIPS — finally training the dynamics
stream the architecture was built for.

Runs #3/#5 fed each frame as a degenerate 1-frame clip (MARLIN tiled, MediaPipe
length-1) → the temporal GRU and MARLIN's motion modeling were starved. YFP frames
are 6fps video samples (verified: median frame-number gap 5, i.e. 30fps/5), so a
window of 16 consecutive frames = ~2.7 s of smooth motion. We build clips centered
on each labeled anchor frame:
    MARLIN sees 16 REAL frames (intra-clip motion, not a tiled still)
    MediaPipe seq is (16, F) — a real left-right asymmetry TRAJECTORY for the GRU

Tasks: eyes / mouth (3-level ordinal, uncoupled) AS BEFORE, plus a new `side` head
(affected side L/R from the XML <pose>, 2-class, supervised only on palsy clips).
Subject-level GroupKFold (same as Run #5) so we can directly compare clips vs the
single-frame Run #5 (eyes 0.47 / mouth strong).

Run:
  KMP_DUPLICATE_LIB_OK=TRUE \
  /opt/homebrew/Caskroom/miniforge/base/envs/dev/bin/python scripts/run7_yfp_clips.py
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
CACHE = ROOT / "outputs" / "yfp_clip_bundles"
ACTION = "clip"
MP_FEAT_DIM = 72
CLIP_LEN = 16
MAX_PER_SUBJECT = 90
N_SPLITS = 4
SEED = 0

EYE_SEV = {"Normal_Eyes": 0, "SlightPalsy_Eyes": 1, "StrongPalsy_Eyes": 2}
MOUTH_SEV = {"Normal_Mouth": 0, "SlightPalsy_Mouth": 1, "StrongPalsy_Mouth": 2}
SIDE = {"Left": 0, "Right": 1}


def parse_xml(p: Path) -> dict:
    rec = {"eyes": None, "mouth": None, "side": None}
    try:
        root = ET.parse(p).getroot()
    except Exception:
        return rec
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        pose = (obj.findtext("pose") or "").strip()
        if name in EYE_SEV:
            rec["eyes"] = max(rec["eyes"] or 0, EYE_SEV[name])
        elif name in MOUTH_SEV:
            rec["mouth"] = max(rec["mouth"] or 0, MOUTH_SEV[name])
        if pose in SIDE and rec["side"] is None:
            rec["side"] = SIDE[pose]
    return rec


def build_clip_index() -> list[dict]:
    """Per subject: sorted full frame list; clips of CLIP_LEN frames centered on
    each labeled anchor. Returns entries {subject, anchor, frames:[paths], eyes,
    mouth, side}."""
    rng = np.random.default_rng(SEED)
    # full frame list per subject (all bmp, sorted by numeric stem)
    frames_by_subj: dict[str, list[Path]] = defaultdict(list)
    for top in IMG_DIRS:
        for p in (YFP / top).rglob("*.bmp"):
            if p.stem.isdigit():
                frames_by_subj[p.parent.name].append(p)
    for s in frames_by_subj:
        frames_by_subj[s].sort(key=lambda p: int(p.stem))

    index = []
    for subj, frames in frames_by_subj.items():
        pos = {p.stem: i for i, p in enumerate(frames)}
        anchors = []
        for xp in (XML_DIR / subj).glob("*.xml"):
            if xp.stem not in pos:
                continue
            lab = parse_xml(xp)
            if lab["eyes"] is None and lab["mouth"] is None:
                continue
            anchors.append((xp.stem, lab))
        if not anchors:
            continue
        # subsample anchors, prefer rare (non-Slight) severity
        if len(anchors) > MAX_PER_SUBJECT:
            rare = [a for a in anchors if (a[1]["eyes"] in (0, 2)) or (a[1]["mouth"] in (0, 2))]
            common = [a for a in anchors if a not in rare]
            keep = list(rare)
            if len(keep) < MAX_PER_SUBJECT and common:
                idx = rng.choice(len(common), size=min(MAX_PER_SUBJECT - len(keep), len(common)), replace=False)
                keep += [common[i] for i in idx]
            if len(keep) > MAX_PER_SUBJECT:
                idx = rng.choice(len(keep), size=MAX_PER_SUBJECT, replace=False)
                keep = [keep[i] for i in idx]
            anchors = keep
        for stem, lab in anchors:
            i = pos[stem]
            lo = max(0, i - CLIP_LEN // 2)
            window = frames[lo:lo + CLIP_LEN]
            if len(window) < CLIP_LEN:                       # near the end: back-fill
                window = frames[max(0, len(frames) - CLIP_LEN):]
            index.append({"subject": subj, "anchor": stem,
                          "frames": [str(p) for p in window], **lab})
    return index


def encode(index, enc, mp_ext, normalizer) -> list[dict]:
    from src.preprocessing.action_bundle import (
        _assert_existing_cache_schema,
        _bundle_npz_payload,
    )

    ok, n_bad = [], 0
    for it in index:
        sid = f"{it['subject']}/{it['anchor']}"
        out = CACHE / sid / f"{ACTION}.npz"
        if out.exists():
            _assert_existing_cache_schema(
                out,
                mp_ext.feature_schema,
                expected_side_convention=mp_ext.side_convention,
                expected_capture_mirrored="unknown",
            )
        else:
            frames = [cv2.imread(p) for p in it["frames"]]
            frames = [f for f in frames if f is not None]
            if len(frames) < 4:
                n_bad += 1; continue
            marlin = enc.encode_clip_bgr(frames, normalizer=normalizer)   # 16 REAL frames -> motion
            seq, mask = mp_ext.extract_sequence(frames)                   # (T,F) trajectory
            if marlin is None or not mask.any():
                n_bad += 1; continue
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez(out, **_bundle_npz_payload({
                "marlin": marlin[None, :], "mp_seq": seq, "mp_mask": mask,
            }, mp_ext))
        it["sid"] = sid
        ok.append(it)
    print(f"  encoded {len(ok)} clips ({n_bad} dropped)")
    return ok


def make_records(entries):
    from scripts._bundle_io import load_action_bundle
    from src.datasets.patient_multistream import MultiStreamRecord
    recs, groups = [], []
    for it in entries:
        b = load_action_bundle(
            CACHE / it["sid"] / f"{ACTION}.npz",
            allow_legacy_schema=True,
            expected_feat_dim=MP_FEAT_DIM,
        )
        base = f"yfp_{it['subject']}_{it['anchor']}"
        for task in ("eyes", "mouth"):
            if it[task] is not None:
                recs.append(MultiStreamRecord(f"{base}_{task}", int(it[task]), task, [b]))
                groups.append(it["subject"])
        # side head: only where there is actual palsy and a known side
        if it["side"] is not None and ((it["eyes"] or 0) > 0 or (it["mouth"] or 0) > 0):
            recs.append(MultiStreamRecord(f"{base}_side", int(it["side"]), "side", [b]))
            groups.append(it["subject"])
    return recs, np.array(groups)


def make_model(seed=0):
    from src.models.facial_palsy_model import FacialPalsyModel, FacialPalsyConfig
    from src.models.multitask import TaskSpec
    torch.manual_seed(seed)
    return FacialPalsyModel(FacialPalsyConfig(
        mp_feat_dim=MP_FEAT_DIM, n_actions=1, temporal_hidden=64, temporal_out=64,
        trunk_hidden=96, dropout=0.1,
        tasks=[TaskSpec("eyes", 3, coupled=False), TaskSpec("mouth", 3, coupled=False),
               TaskSpec("side", 2, coupled=False)]))


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
    print("Building clip index (16-frame motion windows around labeled anchors)...")
    index = build_clip_index()
    print(f"  {len(index)} clips over {len(set(it['subject'] for it in index))} subjects")
    print(f"Encoding clips through MARLIN (+ quality norm) on {device} — REAL 16-frame motion...")
    enc = MarlinVideoEncoder.from_default_weights().to(device).eval()
    # YFP frame capture orientation is not documented.
    mp_ext = MediaPipeFeatureExtractor(capture_mirrored=None)
    normalizer = QualityNormalizer(QualityConfig(mode="normalize", work_size=112))
    entries = encode(index, enc, mp_ext, normalizer)

    recs, groups = make_records(entries)
    for task in ("eyes", "mouth", "side"):
        c = Counter(r.label for r in recs if r.task == task)
        print(f"  {task}: {sum(c.values())} recs, dist {dict(sorted(c.items()))}")

    gkf = GroupKFold(n_splits=N_SPLITS)
    full = MultiStreamPatientDataset(recs, actions=[ACTION], mp_feat_dim=MP_FEAT_DIM)
    oof = {t: {} for t in ("eyes", "mouth", "side")}
    for fold, (tr, te) in enumerate(gkf.split(np.zeros(len(recs)), groups=groups), 1):
        model = make_model()
        train_multitask(model, Subset(full, tr.tolist()), Subset(full, te.tolist()),
                        MTTrainConfig(epochs=60, batch_size=32, lr=5e-4, weight_decay=3e-2,
                                      device="cpu", monitor_task="mouth", monitor_n_classes=3,
                                      log_every=999, early_stopping_patience=12,
                                      early_stopping_warmup=8, seed=0))
        model.eval()
        b = next(iter(DataLoader(Subset(full, te.tolist()), batch_size=len(te),
                                 collate_fn=collate_multistream)))
        with torch.no_grad():
            out = model(b["marlin_emb"], b["marlin_mask"], b["mp_seq"], b["mp_mask"], b["action_present"])
            preds = {t: predict_grade(out[t]).cpu().numpy() for t in ("eyes", "mouth", "side")}
        for li, ri in enumerate(te):
            t = recs[ri].task
            oof[t][ri] = int(preds[t][li])
        print(f"  fold {fold}: held-out {sorted(set(groups[te]), key=int)}")

    res = {"dataset": "YFP TEMPORAL CLIPS (16-frame motion), subject-CV",
           "vs": "Run#5 single-frame: eyes kappa 0.47"}
    for t, k in (("eyes", 3), ("mouth", 3), ("side", 2)):
        idxs = [i for i in range(len(recs)) if recs[i].task == t and i in oof[t]]
        if not idxs:
            continue
        true = np.array([recs[i].label for i in idxs]); pred = np.array([oof[t][i] for i in idxs])
        m = HBMetrics.from_predictions(true, pred, n_classes=k)
        res[t] = {"n": len(idxs), "kappa": round(m.quadratic_kappa, 3),
                  "mae": round(m.mae_grades, 3), "acc": round(m.accuracy, 3)}
    print("\n================= RUN #7 SUMMARY (YFP temporal clips) =================")
    print(json.dumps({k: v for k, v in res.items() if k != "dataset"}, indent=2))
    print("======================================================================")
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "results.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
