"""Fixed-width feature-block ablations for the 95-d autoresearch cache."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "autoresearch_fp"))
os.environ["FP_CLINICAL"] = "1"

from runner import apply_landmark_ablation  # noqa: E402
sys.path.insert(0, str(ROOT / "scripts"))
from run_landmark_ablation import _validate_cache_sha256  # noqa: E402
from _testlib import Check, run_all  # noqa: E402


def _x():
    return torch.arange(95, dtype=torch.float32).reshape(1, 1, 95)


def test_fusion_is_identity(c: Check):
    x = _x()
    got = apply_landmark_ablation(x, "fusion")
    c.true(got is x, "fusion path does not copy or alter the tensor")


def test_blendshape_only_zeros_clinical_block(c: Check):
    got = apply_landmark_ablation(_x(), "blendshape_only")
    c.true(bool(torch.equal(got[..., :72], _x()[..., :72])), "base72 retained")
    c.true(bool((got[..., 72:] == 0).all()), "clinical23 zeroed")


def test_landmark_only_zeros_base_block(c: Check):
    got = apply_landmark_ablation(_x(), "landmark_only")
    c.true(bool((got[..., :72] == 0).all()), "base72 zeroed")
    c.true(bool(torch.equal(got[..., 72:], _x()[..., 72:])), "clinical23 retained")


def test_invalid_layout_fails_closed(c: Check):
    c.raises(lambda: apply_landmark_ablation(torch.zeros(1, 1, 72), "fusion"),
             ValueError, "all arms require the same 95-d cache")
    c.raises(lambda: apply_landmark_ablation(_x(), "unknown"),
             ValueError, "unknown arm")


def test_ablation_cache_is_cryptographically_pinned(c: Check):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cache.pt"
        path.write_bytes(b"audited cache")
        observed = _validate_cache_sha256(
            path, "35a60edd7636cba78f0109792b41cbabd8166fbd6d60761c5001ae811fff76fe")
        c.eq(observed,
             "35a60edd7636cba78f0109792b41cbabd8166fbd6d60761c5001ae811fff76fe",
             "matching cache hash returned")
        c.raises(lambda: _validate_cache_sha256(path, "0" * 64), ValueError,
                 "semantic cache changes fail closed")


if __name__ == "__main__":
    run_all("test_landmark_ablation", dict(globals()))
