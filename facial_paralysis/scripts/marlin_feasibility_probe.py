"""Feasibility gate: do FROZEN MARLIN features separate facial palsy?

This decides whether the MARLIN-based design (docs/model_design.md v2) is worth
implementing in full, BEFORE we build the whole pipeline. We already saw the Oo
MLP-Mixer and Shagdar GNN frozen features collapse on our data; this checks
whether MARLIN avoids that fate — first in its OWN domain (PalsyNet = YouTube
talking-head clips, binary palsy/healthy, 49 subjects, one video each).

Logic:
  1. Encode each PalsyNet video into N frozen-MARLIN clip embeddings (768-d).
  2. Linear probe (logistic regression) with SUBJECT-LEVEL CV (GroupKFold by
     video) so clips from the same person never span train/test — no leakage.
  3. Report clip-level and subject-level ROC AUC + accuracy.

Interpretation:
  - subject AUC >~0.80 : frozen MARLIN carries palsy signal -> GREEN, build it.
  - subject AUC ~0.5-0.65 : collapses like Oo -> RED, rethink (unfreeze / V-JEPA).

Run:
  KMP_DUPLICATE_LIB_OK=TRUE \
  /opt/homebrew/Caskroom/miniforge/base/envs/dev/bin/python \
  scripts/marlin_feasibility_probe.py --n-clips 4 [--limit N]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PALSYNET = ROOT / "data" / "external" / "palsynet" / "data"
CACHE = ROOT / "outputs" / "marlin_probe"


def build_embeddings(n_clips: int, limit: int | None) -> dict:
    import torch
    from src.models.backbones.marlin_video import MarlinVideoEncoder
    from utils import MediaPipeFaceLandmarker  # type: ignore

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    enc = MarlinVideoEncoder.from_default_weights().to(device).eval()
    landmarker = MediaPipeFaceLandmarker()

    rows_X, rows_y, rows_subj = [], [], []
    for label, sub in [(1, "affected"), (0, "unaffected")]:
        vids = sorted((PALSYNET / sub).glob("*.mp4"))
        if limit:
            vids = vids[:limit]
        for vp in vids:
            t0 = time.time()
            embs = enc.encode_video_path(vp, n_clips=n_clips, landmarker=landmarker)
            sid = f"{sub}/{vp.stem}"
            if embs is None:
                print(f"  [skip] {sid}: no usable face")
                continue
            for e in embs:
                rows_X.append(e); rows_y.append(label); rows_subj.append(sid)
            print(f"  {sid}: {len(embs)} clips  ({time.time()-t0:.1f}s)")
    X = np.stack(rows_X); y = np.array(rows_y); subj = np.array(rows_subj)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE / "palsynet_marlin.npz", X=X, y=y, subj=subj)
    return {"X": X, "y": y, "subj": subj}


def load_cached() -> dict | None:
    f = CACHE / "palsynet_marlin.npz"
    if not f.exists():
        return None
    d = np.load(f, allow_pickle=True)
    return {"X": d["X"], "y": d["y"], "subj": d["subj"]}


def probe(X: np.ndarray, y: np.ndarray, subj: np.ndarray) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, accuracy_score

    uniq_subj = np.unique(subj)
    n_pos_sub = len(np.unique(subj[y == 1])); n_neg_sub = len(np.unique(subj[y == 0]))
    print(f"\n{len(X)} clips from {len(uniq_subj)} subjects "
          f"({n_pos_sub} palsy / {n_neg_sub} healthy), dim={X.shape[1]}")

    n_splits = min(5, n_pos_sub, n_neg_sub)
    gkf = GroupKFold(n_splits=n_splits)
    clip_prob = np.zeros(len(X)); clip_true = y.copy()
    for tr, te in gkf.split(X, y, groups=subj):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
        clf.fit(sc.transform(X[tr]), y[tr])
        clip_prob[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]

    clip_auc = roc_auc_score(clip_true, clip_prob)
    clip_acc = accuracy_score(clip_true, (clip_prob > 0.5).astype(int))

    # subject-level: mean clip prob per subject
    s_prob, s_true = [], []
    for s in uniq_subj:
        m = subj == s
        s_prob.append(clip_prob[m].mean()); s_true.append(int(y[m][0]))
    s_prob = np.array(s_prob); s_true = np.array(s_true)
    subj_auc = roc_auc_score(s_true, s_prob)
    subj_acc = accuracy_score(s_true, (s_prob > 0.5).astype(int))

    print("\n================= FEASIBILITY RESULT =================")
    print(f"  clip-level   : AUC {clip_auc:.3f}   acc {clip_acc:.3f}  (n={len(X)})")
    print(f"  subject-level: AUC {subj_auc:.3f}   acc {subj_acc:.3f}  (n={len(uniq_subj)})")
    print(f"  CV: {n_splits}-fold GroupKFold by subject (no clip leakage)")
    verdict = ("GREEN — frozen MARLIN carries palsy signal; build the pipeline"
               if subj_auc >= 0.80 else
               "AMBER — weak but present; consider light unfreeze / more clips"
               if subj_auc >= 0.65 else
               "RED — collapses like Oo; rethink encoder/strategy")
    print(f"  VERDICT: {verdict}")
    print("======================================================")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-clips", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="cap videos per class (smoke test)")
    ap.add_argument("--use-cache", action="store_true")
    args = ap.parse_args()

    data = load_cached() if args.use_cache else None
    if data is None:
        print(f"Encoding PalsyNet with frozen MARLIN (n_clips={args.n_clips})...")
        data = build_embeddings(args.n_clips, args.limit)
    probe(data["X"], data["y"], data["subj"])


if __name__ == "__main__":
    main()
