"""Validation tests for the full MARLIN + MediaPipe pipeline model layers:
TemporalLandmarkEncoder, FacialPalsyModel, MultiStreamPatientDataset.

Run:
    KMP_DUPLICATE_LIB_OK=TRUE \
    /opt/homebrew/Caskroom/miniforge/base/envs/dev/bin/python tests/test_pipeline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.temporal import TemporalLandmarkEncoder  # noqa: E402
from src.models.facial_palsy_model import FacialPalsyModel, FacialPalsyConfig  # noqa: E402
from src.models.multitask import DEFAULT_TASKS, multitask_loss  # noqa: E402
from src.datasets.patient_multistream import (  # noqa: E402
    ActionBundle, MultiStreamRecord, MultiStreamPatientDataset,
    collate_multistream, STANDARD_ACTIONS,
)
from torch.utils.data import DataLoader  # noqa: E402

from _testlib import Check  # noqa: E402

F = 73
A = len(STANDARD_ACTIONS)


# ---------------- TemporalLandmarkEncoder ----------------
def test_temporal_shapes(c: Check):
    enc = TemporalLandmarkEncoder(F, hidden_dim=64, out_dim=48)
    out = enc(torch.randn(4, 10, F))
    c.eq(out.shape, (4, 48), "temporal output shape")


def test_temporal_mask_zeros_for_empty(c: Check):
    enc = TemporalLandmarkEncoder(F, out_dim=32).eval()
    x = torch.randn(3, 7, F)
    mask = torch.ones(3, 7, dtype=torch.bool)
    mask[1] = False                                   # row 1 fully padded
    out = enc(x, mask)
    c.true(torch.allclose(out[1], torch.zeros(32), atol=1e-6), "all-padding row -> zeros")
    c.true(not torch.allclose(out[0], torch.zeros(32)), "real row nonzero")


def test_temporal_padding_invariance(c: Check):
    """Appending padding frames must not change the output (mask respected)."""
    enc = TemporalLandmarkEncoder(F, out_dim=32).eval()
    x = torch.randn(2, 5, F)
    base = enc(x, torch.ones(2, 5, dtype=torch.bool))
    x_pad = torch.cat([x, torch.randn(2, 4, F)], dim=1)        # 4 garbage frames
    mask = torch.cat([torch.ones(2, 5), torch.zeros(2, 4)], dim=1).bool()
    padded = enc(x_pad, mask)
    c.true(torch.allclose(base, padded, atol=1e-5), "padding does not change output")


def test_temporal_single_frame(c: Check):
    """Image case: T=1 is valid."""
    enc = TemporalLandmarkEncoder(F, out_dim=16).eval()
    out = enc(torch.randn(2, 1, F))
    c.eq(out.shape, (2, 16), "T=1 works")


def test_temporal_validation(c: Check):
    enc = TemporalLandmarkEncoder(F)
    c.raises(lambda: enc(torch.randn(4, F)), ValueError, "rejects 2-D input")
    c.raises(lambda: enc(torch.randn(4, 5, F + 1)), ValueError, "rejects wrong feat_dim")
    c.raises(lambda: enc(torch.randn(4, 5, F), torch.ones(4, 3, dtype=torch.bool)),
             ValueError, "rejects mismatched mask")


# ---------------- FacialPalsyModel ----------------
def _model(W=2, **kw):
    torch.manual_seed(0)
    return FacialPalsyModel(FacialPalsyConfig(mp_feat_dim=F, n_actions=A, **kw))


def _batch(B=3, W=2, T=8, missing=False):
    marlin = torch.randn(B, A, W, 768)
    marlin_mask = torch.ones(B, A, W, dtype=torch.bool)
    mp = torch.randn(B, A, T, F)
    mp_mask = torch.ones(B, A, T, dtype=torch.bool)
    if missing:
        marlin_mask[0, 2] = False
        mp_mask[0, 2] = False
    return marlin, marlin_mask, mp, mp_mask


def test_model_forward_shapes(c: Check):
    m = _model().eval()
    out = m(*_batch())
    for t in DEFAULT_TASKS:
        c.eq(out[t.name].shape, (3, t.n_classes - 1), f"{t.name} output shape")
    c.eq(m.embed_dim, 768 + 128, "embed_dim = marlin + temporal_out")


def test_model_embed_dim_matches_trunk(c: Check):
    m = _model(temporal_out=64)
    c.eq(m.multitask.trunk.cfg.embed_dim, 768 + 64, "trunk embed_dim wired to concat width")


def test_model_missing_action(c: Check):
    m = _model().eval()
    full = m(*_batch(missing=False))["hb"]
    miss = m(*_batch(missing=True))["hb"]
    c.true(torch.isfinite(miss).all(), "missing-action output finite")
    c.true(not torch.allclose(full[0], miss[0]), "masking action 2 changes patient 0")


def test_model_derives_action_mask(c: Check):
    """If a patient has an action with NO windows and NO frames, it's auto-excluded
    and must not crash (as long as >=1 action remains)."""
    m = _model().eval()
    marlin, marlin_mask, mp, mp_mask = _batch(B=2)
    marlin_mask[1, 0] = False; mp_mask[1, 0] = False     # patient1 action0 absent
    out = m(marlin, marlin_mask, mp, mp_mask)             # action_mask=None -> derived
    c.true(torch.isfinite(out["hb"]).all(), "derived action mask, finite output")


def test_model_gradients(c: Check):
    """Loss reaches the temporal GRU, the trunk, and a head's thresholds."""
    m = _model()
    out = m(*_batch())
    y = torch.randint(0, 6, (3,))
    loss, _ = multitask_loss(out, y, ["hb"] * 3, DEFAULT_TASKS)
    loss.backward()
    g_temporal = next(p.grad for p in m.temporal.gru.parameters() if p.grad is not None)
    c.true(g_temporal.abs().sum() > 0, "gradient reaches temporal GRU")
    c.true(m.multitask.trunk.severity_proj.weight.grad.abs().sum() > 0, "reaches severity proj")
    c.true(m.multitask.thresholds["hb"].first.grad is not None, "reaches HB thresholds")


def test_predict_hb(c: Check):
    m = _model().eval()
    pred = m.predict_hb(*_batch(B=5))
    c.eq(pred.shape, (5,), "predict_hb shape")
    c.true(bool(((pred >= 0) & (pred <= 5)).all()), "HB preds in [0,5]")


def test_model_validation(c: Check):
    m = _model()
    c.raises(lambda: m.build_action_embeddings(torch.randn(2, A, 2, 999),
             torch.ones(2, A, 2, dtype=torch.bool), torch.randn(2, A, 8, F),
             torch.ones(2, A, 8, dtype=torch.bool)), ValueError, "wrong marlin dim rejected")


# ---------------- MultiStreamPatientDataset ----------------
def _rand_record(pid, label, task="hb", W=3, T=9, drop_action=None):
    bundles = []
    for a in range(A):
        if drop_action is not None and a == drop_action:
            bundles.append(ActionBundle()); continue
        bundles.append(ActionBundle(
            marlin=np.random.randn(W, 768).astype(np.float32),
            mp_seq=np.random.randn(T, F).astype(np.float32),
            mp_mask=np.ones(T, dtype=bool),
        ))
    return MultiStreamRecord(patient_id=pid, label=label, task=task, actions=bundles)


def test_dataset_item_shapes(c: Check):
    ds = MultiStreamPatientDataset([_rand_record("p0", 3, W=3, T=9)], mp_feat_dim=F)
    it = ds[0]
    c.eq(it["marlin_emb"].shape, (A, 3, 768), "item marlin shape")
    c.eq(it["mp_seq"].shape, (A, 9, F), "item mp_seq shape")
    c.eq(it["action_present"].shape, (A,), "action_present shape")
    c.true(bool(it["action_present"].all()), "all actions present")


def test_dataset_missing_action(c: Check):
    ds = MultiStreamPatientDataset([_rand_record("p0", 2, drop_action=1)], mp_feat_dim=F)
    it = ds[0]
    c.true(not bool(it["action_present"][1]), "dropped action marked absent")
    c.true(bool(it["action_present"][0]), "other action present")


def test_collate_ragged(c: Check):
    """Batch with different W and T per patient pads to the batch max."""
    recs = [_rand_record("p0", 1, W=2, T=6), _rand_record("p1", 4, W=5, T=11)]
    ds = MultiStreamPatientDataset(recs, mp_feat_dim=F)
    loader = DataLoader(ds, batch_size=2, collate_fn=collate_multistream)
    b = next(iter(loader))
    c.eq(b["marlin_emb"].shape, (2, A, 5, 768), "collate pads W to batch max")
    c.eq(b["mp_seq"].shape, (2, A, 11, F), "collate pads T to batch max")
    c.eq(b["label"].tolist(), [1, 4], "labels preserved")
    c.eq(b["task_ids"], ["hb", "hb"], "task ids preserved")


def test_dataset_into_model(c: Check):
    """Dataset batch flows through the model unchanged."""
    recs = [_rand_record(f"p{i}", i % 6, W=2, T=7) for i in range(4)]
    ds = MultiStreamPatientDataset(recs, mp_feat_dim=F)
    loader = DataLoader(ds, batch_size=4, collate_fn=collate_multistream)
    b = next(iter(loader))
    m = _model().eval()
    out = m(b["marlin_emb"], b["marlin_mask"], b["mp_seq"], b["mp_mask"], b["action_present"])
    c.eq(out["hb"].shape, (4, 5), "end-to-end dataset->model shape")
    c.true(torch.isfinite(out["hb"]).all(), "finite")


def test_mixed_task_records(c: Check):
    recs = [_rand_record("p0", 5, task="hb"), _rand_record("p1", 1, task="binary")]
    ds = MultiStreamPatientDataset(recs, mp_feat_dim=F)
    loader = DataLoader(ds, batch_size=2, collate_fn=collate_multistream)
    b = next(iter(loader))
    c.eq(b["task_ids"], ["hb", "binary"], "mixed task ids carried through")


if __name__ == "__main__":
    from _testlib import run_all
    run_all(__name__, globals())
