"""Run #10 — corrected Mayo severity / non-collapse analysis with the v1 model.

Supersedes the ad-hoc check in validate_mayo_domain.py, which had two problems:
  (1) it keyed its non-collapse VERDICT on the spread of P(palsy) — which is
      SATURATED (~1.0 for every take) because the PalsyNet-trained threshold isn't
      calibrated to the Mayo domain — and so wrongly reported "COLLAPSED-LIKE".
      The right signal is the spread of the latent severity `s`.
  (2) it only scored binary P(palsy); the deployed v1 model also has eyes/mouth
      region-severity heads, which give a richer per-take readout.

This loads the actual v1 warm-start checkpoint (binary + eyes + mouth heads) and
scores the cached two-stream Mayo bundles. NO labels exist, so this is a
non-collapse + face-validity ranking, NOT a metric run.

CAVEATS (honest):
  - v1 trained on quality-NORMALIZED public data (work_size=112); the cached Mayo
    bundles are UN-normalized (made by validate_mayo to match un-norm PalsyNet).
    So absolute `s` is not calibrated to Mayo — only the RANKING is informative.
    A normalized re-encode (cv2 crop, mediapipe-free) is the follow-up.
  - Take folder names (FACES* vs MySlate*) are NOT palsy/healthy labels.
  - 20260305_FACES018 and 20260305_MySlate_14 are the SAME recording (duplicate).

Run:  KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/run10_mayo_severity.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.datasets.patient_multistream import (  # noqa: E402
    ActionBundle, MultiStreamRecord, MultiStreamPatientDataset, collate_multistream,
)
from src.models.facial_palsy_model import FacialPalsyConfig, FacialPalsyModel  # noqa: E402
from src.models.multitask import TaskSpec  # noqa: E402
from src.models.ordinal import cum_probs, expected_grade, predict_grade  # noqa: E402

CKPT = ROOT / "outputs" / "checkpoints" / "warmstart_v1.pt"
# Bundle dir is overridable: `python run10_mayo_severity.py mayo_bundles_norm`
# lets us re-score the GPU-re-extracted, quality-normalized bundles (closes the
# v1-normalized vs Mayo-unnormalized caveat).
MAYO_CACHE = ROOT / "outputs" / (sys.argv[1] if len(sys.argv) > 1 else "mayo_bundles")
ACTION = "clip"
DUP = {"20260305_FACES018", "20260305_MySlate_14"}  # same recording


def load_v1():
    sd = torch.load(CKPT, map_location="cpu")
    cfg = FacialPalsyConfig(tasks=[TaskSpec(*t) for t in sd["tasks"]], **sd["model_cfg"])
    model = FacialPalsyModel(cfg)
    model.load_state_dict(sd["state_dict"])
    model.eval()
    return model, sd


def load_mayo_records():
    recs = []
    for npz in sorted(MAYO_CACHE.glob("2026*/clip.npz")):
        d = np.load(npz)
        recs.append(MultiStreamRecord(
            patient_id=npz.parent.name, label=0, task="binary",
            actions=[ActionBundle(marlin=d["marlin"].astype(np.float32),
                                  mp_seq=d["mp_seq"].astype(np.float32),
                                  mp_mask=d["mp_mask"].astype(bool))]))
    return recs


def spread(x):
    x = np.asarray(x, dtype=float)
    return {"mean": round(float(x.mean()), 3), "std": round(float(x.std()), 3),
            "min": round(float(x.min()), 3), "max": round(float(x.max()), 3),
            "range": round(float(x.max() - x.min()), 3)}


def main():
    model, sd = load_v1()
    mp_feat_dim = sd["model_cfg"]["mp_feat_dim"]
    recs = load_mayo_records()
    ds = MultiStreamPatientDataset(recs, actions=[ACTION], mp_feat_dim=mp_feat_dim)
    b = next(iter(torch.utils.data.DataLoader(ds, batch_size=len(ds), collate_fn=collate_multistream)))

    with torch.no_grad():
        emb = model.build_action_embeddings(b["marlin_emb"], b["marlin_mask"], b["mp_seq"], b["mp_mask"])
        h, s = model.multitask.trunk.represent(emb, b["action_present"])
        out = model.multitask(emb, b["action_present"])
        p_palsy = cum_probs(out["binary"])[:, 0].numpy()
        eyes_g = predict_grade(out["eyes"]).numpy(); eyes_e = expected_grade(out["eyes"]).numpy()
        mouth_g = predict_grade(out["mouth"]).numpy(); mouth_e = expected_grade(out["mouth"]).numpy()
    s = s.numpy()

    rows = []
    for i, r in enumerate(recs):
        rows.append({"take": r.patient_id, "s": round(float(s[i]), 3),
                     "p_palsy": round(float(p_palsy[i]), 3),
                     "eyes_grade": int(eyes_g[i]), "eyes_sev": round(float(eyes_e[i]), 2),
                     "mouth_grade": int(mouth_g[i]), "mouth_sev": round(float(mouth_e[i]), 2),
                     "dup": r.patient_id in DUP})
    rows.sort(key=lambda r: r["s"], reverse=True)

    # non-collapse measured on s and region severities (NOT saturated P(palsy))
    uniq = [r for r in rows if not (r["dup"] and r["take"] == "20260305_MySlate_14")]
    summary = {
        "n_takes": len(rows), "n_unique": len(uniq),
        "severity_s": spread([r["s"] for r in uniq]),
        "p_palsy": spread([r["p_palsy"] for r in uniq]),
        "eyes_sev": spread([r["eyes_sev"] for r in uniq]),
        "mouth_sev": spread([r["mouth_sev"] for r in uniq]),
        # CORRECT non-collapse test: spread of the latent severity, not P(palsy)
        "non_collapsed_by_s": bool(np.std([r["s"] for r in uniq]) > 0.3),
        "p_palsy_saturated": bool(np.std([r["p_palsy"] for r in uniq]) < 0.05),
    }
    eyes_grades = np.bincount([r["eyes_grade"] for r in uniq], minlength=3)
    mouth_grades = np.bincount([r["mouth_grade"] for r in uniq], minlength=3)
    summary["eyes_grade_hist"] = eyes_grades.tolist()
    summary["mouth_grade_hist"] = mouth_grades.tolist()

    print("================= RUN #10 — Mayo severity (v1) =================")
    print(json.dumps(summary, indent=2))
    print(f"\n{'take':<26s} {'s':>7s} {'P(palsy)':>9s} {'eyes':>5s} {'mouth':>6s}")
    for r in rows:
        tag = "  (dup)" if r["dup"] else ""
        print(f"{r['take']:<26s} {r['s']:7.2f} {r['p_palsy']:9.3f} "
              f"{r['eyes_sev']:5.2f} {r['mouth_sev']:6.2f}{tag}")
    verdict = ("NON-COLLAPSED by s (real spread) — P(palsy) saturated is a "
               "calibration issue, not collapse" if summary["non_collapsed_by_s"]
               else "COLLAPSED — s has no spread; investigate")
    print(f"\nVERDICT: {verdict}")
    print("================================================================")

    MAYO_CACHE.mkdir(parents=True, exist_ok=True)
    (MAYO_CACHE / "run10_severity.json").write_text(
        json.dumps({"summary": summary, "ranked": rows,
                    "preprocessing_caveat": "v1=normalized, mayo bundles=un-normalized"}, indent=2))


if __name__ == "__main__":
    main()
