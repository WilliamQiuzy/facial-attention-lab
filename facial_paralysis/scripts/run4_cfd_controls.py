"""Training run #4: add CFD (Chicago Face Database) neutral faces as HEALTHY
controls — the negatives Run #2 said we lacked — with a confound-aware design.

Why confound-aware: CFD is studio photos; our palsy data is YouTube/iPhone. Pool
them naively and the model can separate on "studio vs web" instead of palsy (the
Run #2 trap). Two defenses:
  1. Re-extract BOTH PalsyNet and CFD through the SAME QualityNormalizer
     (work_size=112), so resolution/sharpness is equalized.
  2. Two honesty checks that decide whether any separation is palsy-driven:
       (A) asymmetry-only model — logistic regression on MediaPipe L/R asymmetry
           features (domain-invariant; healthy faces have low asymmetry). If this
           separates palsy from healthy, the signal is real facial asymmetry, not
           appearance/domain.
       (B) domain-shortcut probe — classify CFD-healthy vs PalsyNet-healthy (BOTH
           healthy, different domain) from MARLIN features. If AUC ≈ 1, the encoder
           can trivially tell the domains apart, so the headline palsy-vs-CFD AUC
           is confounded and must be discounted.

Controls: CFD neutral images only (~597, one resting face per identity), the
cleanest analogue to a 'rest' action. Source is referenced in place (not copied);
bundles cached under outputs/.

Run:
  KMP_DUPLICATE_LIB_OK=TRUE \
  /opt/homebrew/Caskroom/miniforge/base/envs/dev/bin/python scripts/run4_cfd_controls.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CFD_DIR = Path("/Users/qiu.ziyue/Library/CloudStorage/OneDrive-MayoClinic/ziyue/"
               "facial_defect/data/external_datasets/CFD/cfd/CFD Version 3.0/Images")
PALSY_VIDEOS = ROOT / "data" / "external" / "palsynet" / "data"
CFD_CACHE = ROOT / "outputs" / "cfd_bundles"
PALSY_NORM_CACHE = ROOT / "outputs" / "palsynet_bundles_norm"
ACTION = "clip"
N_MARLIN_WINDOWS = 4
WORK_SIZE = 112


def _normalizer():
    from src.preprocessing.image_quality import QualityConfig, QualityNormalizer
    return QualityNormalizer(QualityConfig(mode="normalize", work_size=WORK_SIZE))


def extract_all(reextract: bool = False):
    """Extract PalsyNet (videos) + CFD-neutral (images) bundles with the SAME
    quality normalizer. Returns (palsy_records, cfd_records, feat_dim)."""
    from src.models.backbones.marlin_video import MarlinVideoEncoder
    from src.preprocessing.action_bundle import MediaPipeFeatureExtractor, extract_action_bundle

    norm = _normalizer()
    enc = mp_ext = None

    def _ensure():
        nonlocal enc, mp_ext
        if enc is None:
            enc = MarlinVideoEncoder.from_default_weights().eval()
            mp_ext = MediaPipeFeatureExtractor()

    # --- PalsyNet videos (re-extract with normalizer) ---
    palsy_rows = []
    for label, sub in [(1, "affected"), (0, "unaffected")]:
        for vp in sorted((PALSY_VIDEOS / sub).glob("*.mp4")):
            sid = f"{sub}_{vp.stem}"
            out = PALSY_NORM_CACHE / sid / f"{ACTION}.npz"
            palsy_rows.append((sid, label))
            if out.exists() and not reextract:
                continue
            _ensure()
            b = extract_action_bundle(vp, enc, mp_ext, n_marlin_windows=N_MARLIN_WINDOWS,
                                      normalizer=norm)
            if b is None:
                print(f"  [skip] palsy {sid}"); continue
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez(out, marlin=b["marlin"], mp_seq=b["mp_seq"], mp_mask=b["mp_mask"],
                     mp_feat_dim=mp_ext.feat_dim)

    # --- CFD neutral images ---
    cfd_imgs = sorted(CFD_DIR.rglob("*-N.jpg"))
    print(f"  CFD neutral images found: {len(cfd_imgs)}")
    cfd_ids = []
    for vp in cfd_imgs:
        sid = vp.stem                                   # e.g. CFD-MF-330-001-N
        out = CFD_CACHE / sid / f"{ACTION}.npz"
        cfd_ids.append(sid)
        if out.exists() and not reextract:
            continue
        _ensure()
        img = cv2.imread(str(vp))
        if img is None:
            continue
        marlin = enc.encode_clip_bgr([img], normalizer=norm)
        seq, mask = mp_ext.extract_sequence([img])      # NB: MediaPipe runs on raw frame
        if marlin is None or not mask.any():
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, marlin=marlin[None, :], mp_seq=seq, mp_mask=mask, mp_feat_dim=mp_ext.feat_dim)

    feat_dim = None
    # read one bundle to learn feat_dim
    for sid, _ in palsy_rows:
        p = PALSY_NORM_CACHE / sid / f"{ACTION}.npz"
        if p.exists():
            feat_dim = int(np.load(p)["mp_feat_dim"]); break
    return palsy_rows, cfd_ids, feat_dim


def _load_vecs(cache, ids_labels):
    """Return per-subject mean-MARLIN (768), mean-asym features, label, domain."""
    marlin, asym, labels, domains, kept = [], [], [], [], []
    for sid, label, domain in ids_labels:
        p = cache / sid / f"{ACTION}.npz"
        if not p.exists():
            continue
        d = np.load(p)
        m = d["marlin"]
        if m.size == 0:
            continue
        mp = d["mp_seq"]; mask = d["mp_mask"]
        # asymmetry features = last (F-52) dims; mean over valid frames
        valid = mp[mask.astype(bool)] if mask.any() else mp
        asym_dims = valid[:, 52:] if valid.shape[1] > 52 else valid
        marlin.append(m.mean(0)); asym.append(asym_dims.mean(0))
        labels.append(label); domains.append(domain); kept.append(sid)
    return (np.array(marlin), [a for a in asym], np.array(labels),
            np.array(domains), kept)


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    print("Extracting PalsyNet + CFD bundles with quality normalizer...")
    palsy_rows, cfd_ids, feat_dim = extract_all()

    # load per-source (PalsyNet and CFD live in different caches)
    pm, pa, pl, pd, _ = _load_vecs(PALSY_NORM_CACHE, [(s, l, "palsy_web") for s, l in palsy_rows])
    cm, ca, cl, cd, _ = _load_vecs(CFD_CACHE, [(s, 0, "cfd_studio") for s in cfd_ids])
    X_marlin = np.vstack([pm, cm])
    asym = np.vstack([np.array(pa), np.array(ca)])
    y = np.concatenate([pl, cl]); domain = np.concatenate([pd, cd])
    n_pos = int((y == 1).sum()); n_neg = int((y == 0).sum())
    print(f"\npool: {len(y)} subjects, {n_pos} palsy / {n_neg} healthy "
          f"(healthy = {int((domain=='palsy_web').sum() - n_pos)} web + {int((domain=='cfd_studio').sum())} CFD)")

    def cv_auc(X, label, n_splits=5, seed=0):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        prob = np.zeros(len(label))
        for tr, te in skf.split(X, label):
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=3000, class_weight="balanced")
            clf.fit(sc.transform(X[tr]), label[tr])
            prob[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
        return roc_auc_score(label, prob), prob

    # headline: palsy vs all-healthy, MARLIN features
    auc_marlin, _ = cv_auc(X_marlin, y)
    # honesty A: asymmetry-only (domain-invariant)
    auc_asym, _ = cv_auc(asym, y)
    # honesty B: domain shortcut — CFD-healthy vs PalsyNet-healthy (both healthy)
    healthy = y == 0
    dom_label = (domain[healthy] == "cfd_studio").astype(int)
    auc_domain, _ = cv_auc(X_marlin[healthy], dom_label)

    res = {
        "n_subjects": int(len(y)), "n_palsy": n_pos, "n_healthy": n_neg,
        "work_size": WORK_SIZE,
        "headline_palsy_vs_healthy_AUC_marlin": round(float(auc_marlin), 3),
        "honestyA_asymmetry_only_AUC": round(float(auc_asym), 3),
        "honestyB_domain_shortcut_AUC_healthy_cfd_vs_web": round(float(auc_domain), 3),
    }
    print("\n================= RUN #4 (CFD controls) =================")
    print(f"  headline palsy-vs-healthy AUC (MARLIN)      : {res['headline_palsy_vs_healthy_AUC_marlin']:.3f}")
    print(f"  honesty A: asymmetry-only AUC (domain-inv)  : {res['honestyA_asymmetry_only_AUC']:.3f}")
    print(f"  honesty B: domain-shortcut AUC (cfd vs web) : {res['honestyB_domain_shortcut_AUC_healthy_cfd_vs_web']:.3f}")
    print("  read: if B≈1.0 the headline is domain-confounded; trust A as the")
    print("        domain-invariant evidence of real palsy signal.")
    print("========================================================")
    CFD_CACHE.mkdir(parents=True, exist_ok=True)
    (CFD_CACHE / "results.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
