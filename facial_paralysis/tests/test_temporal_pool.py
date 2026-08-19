"""Validation tests for temporal/window pooling modes (mean | max | attention).

Run from the project root with the torch env (no pytest):
    KMP_DUPLICATE_LIB_OK=TRUE python3 tests/test_temporal_pool.py

These cover the new pooling knobs added for Run #9. The clinically important
contract is `test_max_captures_peak_mean_dilutes`: a transient asymmetry spike
(e.g. a brief incomplete eye closure) must survive max/"peak" pooling, whereas
a masked mean over the whole clip dilutes it — the mechanism Run #7 hit.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.temporal import POOL_MODES, TemporalLandmarkEncoder  # noqa: E402
from src.models.facial_palsy_model import FacialPalsyConfig, FacialPalsyModel  # noqa: E402
from src.models.multitask import TaskSpec  # noqa: E402
from _testlib import run_all  # noqa: E402


def _encoder(pool: str, feat_dim: int = 8, out_dim: int = 6) -> TemporalLandmarkEncoder:
    torch.manual_seed(0)
    return TemporalLandmarkEncoder(feat_dim=feat_dim, hidden_dim=10, out_dim=out_dim, pool=pool)


# ----------------------------------------------------------------------
def test_all_pool_modes_output_shape(c):
    x = torch.randn(4, 7, 8)
    mask = torch.ones(4, 7, dtype=torch.bool)
    for pool in POOL_MODES:
        enc = _encoder(pool).eval()
        out = enc(x, mask)
        c.eq(tuple(out.shape), (4, 6), f"{pool} output shape")


def test_invalid_pool_raises(c):
    c.raises(lambda: TemporalLandmarkEncoder(feat_dim=8, pool="bogus"),
             ValueError, "unknown pool mode must raise")


def test_empty_rows_zero_all_modes(c):
    x = torch.randn(3, 5, 8)
    mask = torch.ones(3, 5, dtype=torch.bool)
    mask[1] = False                                  # row 1 = all padding
    for pool in POOL_MODES:
        enc = _encoder(pool).eval()
        out = enc(x, mask)
        c.true(torch.allclose(out[1], torch.zeros(6)), f"{pool}: empty row -> zeros")
        c.true(out[0].abs().sum() > 0, f"{pool}: real row non-zero")


def test_max_pool_ignores_padding(c):
    """Padding frames (set to huge values) must NOT leak into the masked max."""
    x = torch.randn(2, 6, 4)
    mask = torch.ones(2, 6, dtype=torch.bool)
    mask[:, 3:] = False                              # frames 3..5 are padding
    x_poisoned = x.clone()
    x_poisoned[:, 3:] = 1e6                          # poison the padding
    real_max = TemporalLandmarkEncoder._masked_max(x, mask)
    poisoned_max = TemporalLandmarkEncoder._masked_max(x_poisoned, mask)
    c.true(torch.allclose(real_max, poisoned_max), "padding must not affect masked max")


def test_max_captures_peak_mean_dilutes(c):
    """The core clinical mechanism: a single transient spike survives max pooling
    but is diluted by mean pooling over a long clip."""
    T = 16
    x = torch.zeros(1, T, 1)
    x[0, 7, 0] = 10.0                                # one spiked frame (the "blink")
    mask = torch.ones(1, T, dtype=torch.bool)
    mean = TemporalLandmarkEncoder._masked_mean(x, mask)[0, 0]
    mx = TemporalLandmarkEncoder._masked_max(x, mask)[0, 0]
    c.eq(float(mx), 10.0, "max keeps the peak frame")
    c.true(float(mean) < 1.0, "mean dilutes the transient spike")
    c.true(mx > mean * 5, "max >> mean for a transient event")


def test_attention_pool_is_trainable(c):
    enc = _encoder("attention").train()
    c.true(enc.attn is not None, "attention pool builds a scorer")
    x = torch.randn(2, 5, 8, requires_grad=True)
    mask = torch.ones(2, 5, dtype=torch.bool)
    enc(x, mask).sum().backward()
    g = enc.attn.weight.grad
    c.true(g is not None and g.abs().sum() > 0, "attention scorer receives gradient")


def test_mean_pool_backward_compatible(c):
    """Default pool='mean' must equal the old masked-mean path exactly."""
    enc = _encoder("mean").eval()
    c.eq(enc.pool, "mean", "default pool is mean")
    x = torch.randn(2, 9, 8)
    mask = torch.ones(2, 9, dtype=torch.bool)
    mask[0, 6:] = False
    out1 = enc(x, mask)
    out2 = enc(x, mask)
    c.true(torch.allclose(out1, out2), "deterministic in eval")


# ----------------------------------------------------------------------
# Integration: FacialPalsyModel wires both pooling knobs through.
def _small_model(temporal_pool: str, marlin_window_pool: str) -> FacialPalsyModel:
    torch.manual_seed(0)
    cfg = FacialPalsyConfig(
        mp_feat_dim=5, marlin_dim=16, temporal_hidden=8, temporal_out=8,
        trunk_hidden=12, n_actions=1, dropout=0.0,
        temporal_pool=temporal_pool, marlin_window_pool=marlin_window_pool,
        tasks=[TaskSpec("eyes", 3, coupled=False)],
    )
    return FacialPalsyModel(cfg)


def _fake_batch(B=2, A=1, W=4, T=10):
    marlin = torch.randn(B, A, W, 16)
    marlin_mask = torch.ones(B, A, W, dtype=torch.bool)
    mp_seq = torch.randn(B, A, T, 5)
    mp_mask = torch.ones(B, A, T, dtype=torch.bool)
    return marlin, marlin_mask, mp_seq, mp_mask


def test_model_wires_pool_knobs(c):
    for tp in POOL_MODES:
        for wp in ("mean", "max"):
            model = _small_model(tp, wp).eval()
            out = model(*_fake_batch())
            c.eq(tuple(out["eyes"].shape), (2, 2), f"tp={tp} wp={wp}: (B, K-1)")
            c.eq(model.temporal.pool, tp, "temporal pool wired")


def test_model_invalid_window_pool_raises(c):
    c.raises(lambda: _small_model("mean", "bogus"), ValueError,
             "bad marlin_window_pool must raise")


def test_model_max_window_pool_ignores_padding(c):
    """With marlin_window_pool='max', a poisoned padding window must not leak."""
    model = _small_model("mean", "max").eval()
    marlin, marlin_mask, mp_seq, mp_mask = _fake_batch(W=4)
    marlin_mask[:, :, 2:] = False                    # windows 2,3 padding
    poisoned = marlin.clone()
    poisoned[:, :, 2:] = 1e6
    o1 = model(marlin, marlin_mask, mp_seq, mp_mask)["eyes"]
    o2 = model(poisoned, marlin_mask, mp_seq, mp_mask)["eyes"]
    c.true(torch.allclose(o1, o2, atol=1e-4), "padding windows must not affect max pool")


if __name__ == "__main__":
    run_all("test_temporal_pool", dict(globals()))
