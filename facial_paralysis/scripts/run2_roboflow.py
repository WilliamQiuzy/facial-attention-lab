"""Training run #2: bring in the Roboflow facial-paralysis COCO set.

IMPORTANT — what this dataset actually is (verified): 118 still IMAGES
(82 train / 24 valid / 12 test), every one labeled "facial-paralysis". It is a
single-class palsy *detection* set: NO healthy/negative class, and images not
video. So it cannot, on its own, raise a palsy-vs-healthy AUC the way a balanced
set would. We therefore run the two things positives-only image data CAN support:

  (B) EXTERNAL GENERALIZATION  [the meaningful one]: train the binary model on all
      of PalsyNet, then measure sensitivity (recall) on the 118 independent
      Roboflow palsy images — does a PalsyNet-trained detector recognize palsy
      from a totally different image source?

  (A) AUGMENTED CV  [the literal request]: pool the Roboflow palsy images in as
      extra positives and re-run subject-level stratified CV; report AUC vs
      Run #1 (pooled 0.860). Caveat: only adds positives (negatives still only
      PalsyNet's 22 healthy), so a clean improvement is not expected.

Images go through the model as the degenerate clip case: MARLIN sees the image
tiled to 16 frames; the MediaPipe sequence has length 1 (no dynamics).

Run:
  KMP_DUPLICATE_LIB_OK=TRUE \
  /opt/homebrew/Caskroom/miniforge/base/envs/dev/bin/python scripts/run2_roboflow.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ROBO = ROOT / "data" / "external" / "roboflow_facial_paralysis"
PALSY_CACHE = ROOT / "outputs" / "palsynet_bundles"
ROBO_CACHE = ROOT / "outputs" / "roboflow_bundles"
ACTION = "clip"
MP_FEAT_DIM = 72


def extract_roboflow_bundles(reextract: bool = False) -> list[str]:
    """Extract one bundle per Roboflow image (image = clip tiled to 16 frames for
    MARLIN, length-1 MediaPipe sequence). Returns list of subject ids with a
    detected face."""
    from src.models.backbones.marlin_video import MarlinVideoEncoder
    from src.preprocessing.action_bundle import MediaPipeFeatureExtractor

    ROBO_CACHE.mkdir(parents=True, exist_ok=True)
    imgs = []
    for split in ("train", "valid", "test"):
        imgs += sorted((ROBO / split).glob("*.jpg"))
    enc = mp_ext = None
    ok_ids, n_noface = [], 0
    for vp in imgs:
        sid = vp.stem[:60]
        out = ROBO_CACHE / sid / f"{ACTION}.npz"
        if out.exists() and not reextract:
            ok_ids.append(sid); continue
        if enc is None:
            enc = MarlinVideoEncoder.from_default_weights().eval()
            mp_ext = MediaPipeFeatureExtractor()
        img = cv2.imread(str(vp))
        if img is None:
            continue
        marlin = enc.encode_clip_bgr([img])              # crops face, tiles to 16
        seq, mask = mp_ext.extract_sequence([img])       # (1, F), (1,)
        if marlin is None or not mask.any():
            n_noface += 1; continue
        d = ROBO_CACHE / sid; d.mkdir(parents=True, exist_ok=True)
        np.savez(d / f"{ACTION}.npz", marlin=marlin[None, :], mp_seq=seq, mp_mask=mask,
                 mp_feat_dim=mp_ext.feat_dim)
        ok_ids.append(sid)
    print(f"  roboflow: {len(ok_ids)} images with face, {n_noface} dropped (no face)")
    return ok_ids


def _load_roboflow_records(ok_ids: list[str]):
    from src.datasets.patient_multistream import ActionBundle, MultiStreamRecord
    recs = []
    for sid in ok_ids:
        d = np.load(ROBO_CACHE / sid / f"{ACTION}.npz")
        recs.append(MultiStreamRecord(
            patient_id=f"robo_{sid}", label=1, task="binary",
            actions=[ActionBundle(marlin=d["marlin"].astype(np.float32),
                                  mp_seq=d["mp_seq"].astype(np.float32),
                                  mp_mask=d["mp_mask"].astype(bool))]))
    return recs


def _make_model(seed=0):
    from src.models.facial_palsy_model import FacialPalsyModel, FacialPalsyConfig
    from src.models.multitask import TaskSpec
    torch.manual_seed(seed)
    return FacialPalsyModel(FacialPalsyConfig(
        mp_feat_dim=MP_FEAT_DIM, n_actions=1, temporal_hidden=64, temporal_out=64,
        trunk_hidden=64, dropout=0.1, tasks=[TaskSpec("binary", 2, coupled=True)]))


def _cfg(epochs=60, **kw):
    from src.training.train_multitask import MTTrainConfig
    return MTTrainConfig(epochs=epochs, batch_size=8, lr=5e-4, weight_decay=3e-2,
                         device="cpu", monitor_task="binary", monitor_n_classes=2,
                         log_every=999, seed=0, **kw)


def _palsynet_dataset():
    from src.datasets.patient_multistream import MultiStreamPatientDataset
    return MultiStreamPatientDataset.from_disk(
        PALSY_CACHE, PALSY_CACHE / "labels.csv", actions=[ACTION], mp_feat_dim=MP_FEAT_DIM)


def _palsy_prob(model, records):
    from src.datasets.patient_multistream import MultiStreamPatientDataset, collate_multistream
    from src.models.ordinal import cum_probs
    from torch.utils.data import DataLoader
    ds = MultiStreamPatientDataset(records, actions=[ACTION], mp_feat_dim=MP_FEAT_DIM)
    b = next(iter(DataLoader(ds, batch_size=len(ds), collate_fn=collate_multistream)))
    model.eval()
    with torch.no_grad():
        out = model(b["marlin_emb"], b["marlin_mask"], b["mp_seq"], b["mp_mask"], b["action_present"])
        return cum_probs(out["binary"])[:, 0].cpu().numpy()


def experiment_B_external(robo_recs) -> dict:
    """Train on ALL PalsyNet, test sensitivity on Roboflow palsy images."""
    from src.training.train_multitask import train_multitask
    pals = _palsynet_dataset()
    model = _make_model()
    train_multitask(model, pals, None, _cfg(epochs=50))
    prob = _palsy_prob(model, robo_recs)
    sens = float((prob > 0.5).mean())
    print(f"\n[Exp B] external generalization on {len(robo_recs)} Roboflow palsy imgs:")
    print(f"   sensitivity (flagged palsy) = {sens:.3f}   mean P(palsy) = {prob.mean():.3f}")
    return {"n": len(robo_recs), "sensitivity": round(sens, 3),
            "mean_p_palsy": round(float(prob.mean()), 3)}


def experiment_A_augmented_cv(robo_recs, n_splits=5, seed=0) -> dict:
    """Pool Roboflow positives into PalsyNet; subject-level stratified CV AUC."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, accuracy_score
    from torch.utils.data import Subset, DataLoader
    from src.datasets.patient_multistream import MultiStreamPatientDataset, collate_multistream
    from src.training.train_multitask import train_multitask
    from src.models.ordinal import cum_probs

    pals = _palsynet_dataset()
    all_recs = list(pals.records) + list(robo_recs)
    ds = MultiStreamPatientDataset(all_recs, actions=[ACTION], mp_feat_dim=MP_FEAT_DIM)
    y = np.array([r.label for r in ds.records])
    print(f"\n[Exp A] augmented pool: {len(ds)} subjects, {int((y==1).sum())} palsy / {int((y==0).sum())} healthy")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    prob = np.zeros(len(ds)); fold_auc = []
    for fold, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y), 1):
        model = _make_model()
        train_multitask(model, Subset(ds, tr.tolist()), Subset(ds, te.tolist()),
                        _cfg(epochs=60, early_stopping_patience=12))
        loader = DataLoader(Subset(ds, te.tolist()), batch_size=len(te), collate_fn=collate_multistream)
        b = next(iter(loader)); model.eval()
        with torch.no_grad():
            out = model(b["marlin_emb"], b["marlin_mask"], b["mp_seq"], b["mp_mask"], b["action_present"])
            prob[te] = cum_probs(out["binary"])[:, 0].cpu().numpy()
        # AUC only meaningful where both classes present in the fold
        if len(set(y[te].tolist())) == 2:
            fold_auc.append(roc_auc_score(y[te], prob[te]))
    pooled_auc = roc_auc_score(y, prob)
    pooled_acc = accuracy_score(y, (prob > 0.5).astype(int))
    print(f"   pooled AUC = {pooled_auc:.3f}  (Run#1 PalsyNet-only pooled was 0.860)")
    print(f"   pooled acc = {pooled_acc:.3f}")
    return {"n_subjects": int(len(ds)), "n_palsy": int((y == 1).sum()),
            "n_healthy": int((y == 0).sum()),
            "fold_auc": [round(a, 3) for a in fold_auc],
            "pooled_auc": round(float(pooled_auc), 3),
            "pooled_acc": round(float(pooled_acc), 3)}


def main():
    print("Extracting Roboflow image bundles (MARLIN tiled + MediaPipe single-frame)...")
    ok_ids = extract_roboflow_bundles()
    robo_recs = _load_roboflow_records(ok_ids)
    res = {"dataset": "roboflow_facial_paralysis (single-class palsy, images)",
           "n_images_with_face": len(robo_recs),
           "experiment_B_external": experiment_B_external(robo_recs),
           "experiment_A_augmented_cv": experiment_A_augmented_cv(robo_recs)}
    print("\n================= RUN #2 SUMMARY =================")
    print(json.dumps(res, indent=2))
    print("=================================================")
    ROBO_CACHE.mkdir(parents=True, exist_ok=True)
    (ROBO_CACHE / "results.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
