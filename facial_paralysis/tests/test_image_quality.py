"""Tests for src/preprocessing/image_quality.py (face-crop quality normalization).

Run:
  KMP_DUPLICATE_LIB_OK=TRUE \
  /opt/homebrew/Caskroom/miniforge/base/envs/dev/bin/python tests/test_image_quality.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from _testlib import run_all  # noqa: E402
from src.preprocessing.image_quality import (  # noqa: E402
    QualityConfig,
    QualityNormalizer,
    canonical_normalize,
    quality_augment,
    super_resolve,
    variance_of_laplacian,
)

OUT = 224


def _sharp_face(seed: int = 0) -> np.ndarray:
    """A high-frequency synthetic crop (lots of edges → high Laplacian variance)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(OUT, OUT, 3), dtype=np.uint8)


def _blur(img: np.ndarray, sigma: float = 4.0) -> np.ndarray:
    import cv2
    return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)


# ----------------------------------------------------------------------
def test_config_validates(c):
    c.raises(lambda: QualityConfig(mode="bogus"), ValueError, "bad mode")
    c.raises(lambda: QualityConfig(work_size=999), ValueError, "work>out")
    c.raises(lambda: QualityConfig(aug_min_work=200, aug_max_work=100), ValueError, "min>max")


def test_canonical_shape_dtype(c):
    out = canonical_normalize(_sharp_face(), work_size=112, out_size=OUT)
    c.eq(out.shape, (OUT, OUT, 3), "shape")
    c.eq(out.dtype, np.uint8, "dtype")


def test_canonical_deterministic(c):
    img = _sharp_face(1)
    a = canonical_normalize(img, work_size=96)
    b = canonical_normalize(img, work_size=96)
    c.true(np.array_equal(a, b), "canonical_normalize must be deterministic")


def test_normalize_shrinks_sharpness_gap(c):
    """The whole point: a sharp and a blurry crop should be CLOSER in sharpness
    after canonical_normalize than before."""
    sharp = _sharp_face(2)
    blurry = _blur(sharp, sigma=4.0)
    gap_before = abs(variance_of_laplacian(sharp) - variance_of_laplacian(blurry))

    sN = canonical_normalize(sharp, work_size=112)
    bN = canonical_normalize(blurry, work_size=112)
    gap_after = abs(variance_of_laplacian(sN) - variance_of_laplacian(bN))

    c.true(gap_after < gap_before,
           f"gap should shrink: before={gap_before:.1f} after={gap_after:.1f}")


def test_smaller_worksize_lowers_sharpness(c):
    """A tighter cap removes more high-frequency detail (monotone sanity)."""
    img = _sharp_face(3)
    s_loose = variance_of_laplacian(canonical_normalize(img, work_size=160))
    s_tight = variance_of_laplacian(canonical_normalize(img, work_size=64))
    c.true(s_tight < s_loose, f"tight({s_tight:.1f}) should be < loose({s_loose:.1f})")


def test_augment_seeded_deterministic(c):
    cfg = QualityConfig(augment=True)
    img = _sharp_face(4)
    a = quality_augment(img, np.random.default_rng(7), cfg)
    b = quality_augment(img, np.random.default_rng(7), cfg)
    c.true(np.array_equal(a, b), "same seed → same augmentation")
    c.eq(a.shape, (OUT, OUT, 3), "augment shape")
    c.eq(a.dtype, np.uint8, "augment dtype")


def test_augment_varies_across_seeds(c):
    cfg = QualityConfig(augment=True)
    img = _sharp_face(5)
    a = quality_augment(img, np.random.default_rng(1), cfg)
    b = quality_augment(img, np.random.default_rng(2), cfg)
    c.true(not np.array_equal(a, b), "different seeds should differ")


def test_normalizer_modes(c):
    img = _sharp_face(6)

    off = QualityNormalizer(QualityConfig(mode="off"))(img)
    c.eq(off.shape, (OUT, OUT, 3), "off shape")
    c.true(np.array_equal(off, img), "off must passthrough identical pixels")

    norm = QualityNormalizer(QualityConfig(mode="normalize", work_size=112))(img)
    c.eq(norm.shape, (OUT, OUT, 3), "normalize shape")

    sr = QualityNormalizer(QualityConfig(mode="sr"))(img)
    c.eq(sr.shape, (OUT, OUT, 3), "sr shape (bicubic fallback)")


def test_normalizer_inference_is_deterministic_even_with_augment(c):
    """training=False must NOT augment, even if cfg.augment=True."""
    cfg = QualityConfig(mode="normalize", augment=True, work_size=112)
    n = QualityNormalizer(cfg)
    img = _sharp_face(8)
    a = n(img, training=False)
    b = n(img, training=False)
    c.true(np.array_equal(a, b), "inference path must be deterministic")
    # and must equal the plain canonical normalize
    c.true(np.array_equal(a, canonical_normalize(img, work_size=112)),
           "inference path == canonical_normalize")


def test_super_resolve_shape(c):
    small = _sharp_face(9)[:64, :64]  # a small crop
    out = super_resolve(small, out_size=OUT)
    c.eq(out.shape, (OUT, OUT, 3), "sr output size")
    c.eq(out.dtype, np.uint8, "sr dtype")


def test_rejects_bad_input(c):
    c.raises(lambda: canonical_normalize(np.zeros((10, 10), np.uint8), 8),
             ValueError, "needs 3 channels")
    c.raises(lambda: canonical_normalize(np.zeros((10, 10, 3), np.float32), 8),
             TypeError, "needs uint8")


if __name__ == "__main__":
    run_all("test_image_quality", dict(globals()))
