"""FIXED research harness for the facial-palsy autoresearch loop. DO NOT MODIFY.

Analogous to karpathy/autoresearch's `prepare.py`: this file owns the DATA, the
FIXED leak-safe train/val splits, and the GROUND-TRUTH metric. The agent never
edits this file — it only edits `train_fp.py`. Keeping the metric and splits here,
out of the editable file, is what makes experiments honest and comparable.

Task: predict per-region facial-palsy severity (eyes / mouth, 3 ordinal levels)
from a per-action bundle = frozen-MARLIN appearance vector (768) + a MediaPipe
geometric feature SEQUENCE (T x 72: 52 blendshapes + 20 L/R asymmetry deltas).

METRIC (higher = better): mean(eyes_QWK, mouth_QWK) — quadratic-weighted Cohen
kappa of predicted vs. true region grades on the leak-safe validation split
(FNP-valid + held-out YFP subjects), AVERAGED over `SEEDS`. Multi-seed averaging
is deliberate: Run #17 was fooled by single-seed noise (+/-0.015); a real win must
clear that band across seeds.

Reference baseline to beat: v2-attention / Run #17 plateau ~= 0.635
(eyes ~0.42, mouth ~0.85). The whole point of this loop is to find out whether an
open-ended REDESIGN (not just a config tweak) can push past it on the data we have.

Data note (why the task is hard): every FNP/YFP region record has mp_seq length 1
(still image -> zero motion). Only the 49 PalsyNet videos carry real dynamics. So
within current data the temporal stream is degenerate for the metric; the live
levers are FEATURE ENGINEERING on the 72-d geometric vector, FUSION, and LOSS.
"""
from __future__ import annotations

import functools
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score

import os
ROOT = Path(__file__).resolve().parent          # autoresearch_fp/
FP = ROOT.parent                                 # facial_paralysis/
OUT = FP / "outputs"
MANIFEST = OUT / "train_manifest_v4.json"
# FP_CLINICAL=1 (default) uses the cache with raw-landmark clinical features appended
# (72 blendshapes + 23 clinical = 95); FP_CLINICAL=0 = original 72-d blendshapes only.
_CLINICAL = os.environ.get("FP_CLINICAL", "1") == "1"
CACHE = ROOT / ("fp_ar_cache_clinical.pt" if _CLINICAL else "fp_ar_cache.pt")

# ---------------------------------------------------------------------------
# FIXED CONSTANTS — the rules of the game. The agent must not change these.
# ---------------------------------------------------------------------------
SOURCES = ("palsy", "fnp", "yfp")   # metric sources + a global-severity anchor.
                                    # v4 pod-only sources (anisa/kaggle/stroke) are
                                    # excluded: not present locally and net-neutral
                                    # (Run #16). Everything here is present locally.
TASKS = ("binary", "eyes", "mouth")
MP_FEAT_DIM = 95 if _CLINICAL else 72          # 72 blendshapes (+23 raw-landmark clinical)
N_CLASSES = {"binary": 2, "eyes": 3, "mouth": 3}
SEEDS = (0, 1, 2)                   # metric = mean over these seeds.
MAX_EPOCHS = 80                     # fixed training budget (comparable across runs).

# metric-relevant region tasks (what the score is computed on)
REGION_TASKS = ("eyes", "mouth")


# ---------------------------------------------------------------------------
# Data loading (fixed)
# ---------------------------------------------------------------------------
def _load_records() -> list[dict]:
    man = json.loads(MANIFEST.read_text())
    recs: list[dict] = []
    for e in man:
        if e["source"] not in SOURCES or e["task"] not in TASKS:
            continue
        p = OUT / e["npz"]
        if not p.exists():
            continue
        d = np.load(p)
        marlin = d["marlin"].astype(np.float32)             # (W, 768)
        recs.append(
            {
                # frozen MARLIN already models intra-window time; window-pool by mean
                # (fixed preprocessing choice, per docs/model_design.md §5.0).
                "marlin": marlin.mean(0),                    # (768,)
                "mp_seq": d["mp_seq"].astype(np.float32),    # (T, 72)
                "mp_mask": d["mp_mask"].astype(bool),        # (T,)
                "label": int(e["label"]),
                "task": e["task"],
                "source": e["source"],
                "split": e["split"],
            }
        )
    return recs


@functools.lru_cache(maxsize=1)
def load_data() -> list[dict]:
    """Return all records (cached to disk after first load)."""
    if CACHE.exists():
        return torch.load(CACHE, weights_only=False)
    recs = _load_records()
    torch.save(recs, CACHE)
    return recs


def train_records(tasks: tuple[str, ...] = TASKS) -> list[dict]:
    return [r for r in load_data() if r["split"] != "val" and r["task"] in tasks]


def val_records(task: str) -> list[dict]:
    """FIXED, ordered validation records for one task. Predictions must align
    to THIS order. The agent reads inputs from here and returns integer grades."""
    return [r for r in load_data() if r["split"] == "val" and r["task"] == task]


# ---------------------------------------------------------------------------
# Batching utility (offered for convenience; the agent may batch differently)
# ---------------------------------------------------------------------------
def make_batch(recs: list[dict], device: str = "cpu") -> dict:
    """Pad mp_seq to max T, stack marlin, return tensors. Pure utility — using it
    is optional and does not affect the metric."""
    B = len(recs)
    Tmax = max(r["mp_seq"].shape[0] for r in recs)
    F = recs[0]["mp_seq"].shape[1]
    mp = np.zeros((B, Tmax, F), np.float32)
    mask = np.zeros((B, Tmax), bool)
    marlin = np.zeros((B, recs[0]["marlin"].shape[0]), np.float32)
    for i, r in enumerate(recs):
        t = r["mp_seq"].shape[0]
        mp[i, :t] = r["mp_seq"]
        mask[i, :t] = r["mp_mask"]
        marlin[i] = r["marlin"]
    return {
        "marlin": torch.from_numpy(marlin).to(device),          # (B,768)
        "mp_seq": torch.from_numpy(mp).to(device),              # (B,Tmax,F)
        "mp_mask": torch.from_numpy(mask).to(device),           # (B,Tmax)
        "label": torch.tensor([r["label"] for r in recs], device=device),
        "task": [r["task"] for r in recs],
    }


# ---------------------------------------------------------------------------
# THE GROUND-TRUTH METRIC (fixed) — un-gameable, computed here from true labels.
# ---------------------------------------------------------------------------
def quadratic_kappa(y_true, y_pred, k: int) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    if len(np.unique(y_true)) < 2:      # kappa undefined on a constant truth vector
        return 0.0
    return float(cohen_kappa_score(y_true, y_pred, labels=list(range(k)), weights="quadratic"))


def report_metric(preds_per_seed: list[dict], extra: dict | None = None) -> dict:
    """Score a set of per-seed predictions and print the summary block.

    preds_per_seed: list (one per seed) of {"eyes": <grades>, "mouth": <grades>},
    each an int array aligned to val_records(task) order.

    Returns the metric dict and prints a karpathy-style summary that the loop
    greps for `^metric:`.
    """
    truth = {t: np.array([r["label"] for r in val_records(t)]) for t in REGION_TASKS}
    per_seed = {t: [] for t in REGION_TASKS}
    means = []
    for p in preds_per_seed:
        for t in REGION_TASKS:
            per_seed[t].append(quadratic_kappa(truth[t], np.asarray(p[t]), N_CLASSES[t]))
        s_eyes = per_seed["eyes"][-1]
        s_mouth = per_seed["mouth"][-1]
        means.append(0.5 * (s_eyes + s_mouth))
    metric_mean = float(np.mean(means))
    metric_sd = float(np.std(means))
    out = {
        "metric": metric_mean,
        "metric_sd": metric_sd,
        "eyes": float(np.mean(per_seed["eyes"])),
        "mouth": float(np.mean(per_seed["mouth"])),
        "n_seeds": len(preds_per_seed),
    }
    if extra:
        out.update(extra)
    print("---")
    print(f"metric:      {out['metric']:.6f}")
    print(f"metric_sd:   {out['metric_sd']:.6f}")
    print(f"eyes_qwk:    {out['eyes']:.6f}")
    print(f"mouth_qwk:   {out['mouth']:.6f}")
    print(f"n_seeds:     {out['n_seeds']}")
    for k, v in (extra or {}).items():
        print(f"{k}: {v}")
    return out


if __name__ == "__main__":
    recs = load_data()
    import collections

    by = collections.Counter((r["split"], r["task"], r["source"]) for r in recs)
    print(f"loaded {len(recs)} records -> cache {CACHE}")
    for k in sorted(by):
        print(f"  {k}: {by[k]}")
    for t in REGION_TASKS:
        v = val_records(t)
        print(f"val[{t}] = {len(v)}  label dist { dict(collections.Counter(r['label'] for r in v)) }")
