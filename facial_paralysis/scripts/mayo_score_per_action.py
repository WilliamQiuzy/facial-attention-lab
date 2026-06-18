"""Score Mayo in the model's intended PER-ACTION mode (n_actions=7).

Builds a 7-slot per-action input per take from outputs/mayo_action_bundles/ (one
slot per canonical action; absent actions are masked) and runs the warm-start model
(default v2-attention). This is the first time the model runs on real Mayo data with
the per-action structure it was designed for (docs/model_design.md §5) — vs the
whole-take single-clip scoring of Run #10/#11.

Run:  python3 scripts/mayo_score_per_action.py [checkpoint.pt]
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
from src.models.ordinal import cum_probs, expected_grade  # noqa: E402

BUNDLES = ROOT / "outputs" / "mayo_action_bundles"
ACTION_ORDER = ["EyebrowRise", "GentleEyeClosure", "TightEyeSqueeze",
                "RelaxedSmile", "LipPucker", "LowerTeethShow", "ReanimatedSmile"]
DUP = {"20260305_FACES018", "20260305_MySlate_14"}


def load_model(ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu")
    cfg = FacialPalsyConfig(tasks=[TaskSpec(*t) for t in sd["tasks"]], **sd["model_cfg"])
    model = FacialPalsyModel(cfg)
    model.load_state_dict(sd["state_dict"])
    model.eval()
    return model, sd["model_cfg"]["mp_feat_dim"]


def main():
    ckpt = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "outputs" / "checkpoints" / "warmstart_v2_attention.pt"
    model, mp_feat_dim = load_model(ckpt)
    print(f"checkpoint: {ckpt.name}", flush=True)

    recs, n_actions_present = [], {}
    for take_dir in sorted(BUNDLES.glob("2026*")):
        take = take_dir.name
        bundles, present = [], []
        for action in ACTION_ORDER:
            p = take_dir / f"{action}.npz"
            if p.exists():
                d = np.load(p)
                bundles.append(ActionBundle(marlin=d["marlin"].astype(np.float32),
                                            mp_seq=d["mp_seq"].astype(np.float32),
                                            mp_mask=d["mp_mask"].astype(bool)))
                present.append(action)
            else:
                bundles.append(ActionBundle())
        if not present:
            continue
        recs.append(MultiStreamRecord(patient_id=take, label=0, task="binary", actions=bundles))
        n_actions_present[take] = present

    ds = MultiStreamPatientDataset(recs, actions=ACTION_ORDER, mp_feat_dim=mp_feat_dim)
    b = next(iter(torch.utils.data.DataLoader(ds, batch_size=len(ds), collate_fn=collate_multistream)))
    with torch.no_grad():
        emb = model.build_action_embeddings(b["marlin_emb"], b["marlin_mask"], b["mp_seq"], b["mp_mask"])
        h, s = model.multitask.trunk.represent(emb, b["action_present"])
        out = model.multitask(emb, b["action_present"])
        p_palsy = cum_probs(out["binary"])[:, 0].numpy()
        eyes = expected_grade(out["eyes"]).numpy()
        mouth = expected_grade(out["mouth"]).numpy()
    s = s.numpy()

    rows = []
    for i, r in enumerate(recs):
        rows.append({"take": r.patient_id, "s": round(float(s[i]), 3),
                     "p_palsy": round(float(p_palsy[i]), 3),
                     "eyes_sev": round(float(eyes[i]), 2), "mouth_sev": round(float(mouth[i]), 2),
                     "n_actions": len(n_actions_present[r.patient_id]),
                     "actions": n_actions_present[r.patient_id], "dup": r.patient_id in DUP})
    rows.sort(key=lambda r: r["s"], reverse=True)
    uniq = [r for r in rows if not (r["dup"] and r["take"] == "20260305_MySlate_14")]
    sv = np.array([r["s"] for r in uniq])
    summary = {"n_takes": len(rows), "n_unique": len(uniq),
               "mode": "per-action (n_actions=7)", "checkpoint": ckpt.name,
               "severity_s": {"mean": round(float(sv.mean()), 3), "std": round(float(sv.std()), 3),
                              "min": round(float(sv.min()), 3), "max": round(float(sv.max()), 3)},
               "non_collapsed_by_s": bool(sv.std() > 0.3),
               "mean_actions_per_take": round(float(np.mean([r["n_actions"] for r in uniq])), 1)}
    print("================ PER-ACTION Mayo scoring ================", flush=True)
    print(json.dumps(summary, indent=1), flush=True)
    print(f"\n{'take':<26s} {'s':>6s} {'P(p)':>6s} {'eyes':>5s} {'mouth':>5s} {'#act':>4s}", flush=True)
    for r in rows:
        print(f"{r['take']:<26s} {r['s']:6.2f} {r['p_palsy']:6.3f} {r['eyes_sev']:5.2f} "
              f"{r['mouth_sev']:5.2f} {r['n_actions']:4d}{'  dup' if r['dup'] else ''}", flush=True)
    (BUNDLES / "per_action_scores.json").write_text(json.dumps({"summary": summary, "ranked": rows}, indent=1))
    print(f"\nwrote {BUNDLES}/per_action_scores.json", flush=True)


if __name__ == "__main__":
    main()
