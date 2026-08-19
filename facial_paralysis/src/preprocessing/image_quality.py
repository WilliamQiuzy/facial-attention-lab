"""Face-crop quality normalization (closes the train/test resolution gap).

The public palsy image sets (Roboflow FNP, sumin) are old, blurry, low-res; our
iPhone/LiveLinkFace captures are much sharper. Feeding both to the frozen MARLIN
encoder unchanged means the appearance embedding partly encodes *capture quality*
instead of *facial function* — a domain gap that hurts transfer. See
docs/model_design.md §2 and data/public_datasets.md "Preprocessing requirement".

The fix runs on the 224×224 aligned face crop, **identically at train and
inference** (that is the whole point). Three modes:

  "normalize" (default, strategy (b)) — cap every crop's *effective resolution* at
      a canonical `work_size` by downscale→upscale. A blurry crop is already below
      the cap and barely changes; a sharp iPhone crop is brought down to the same
      ceiling. Both then share one resolution distribution. Optionally, at train
      time only, `quality_augment` jitters the cap / blur / JPEG so the head is
      robust to a *range* of qualities rather than one fixed level.
  "sr" (strategy (a), ablation) — super-resolve crops upward instead. Risk:
      hallucinated detail that real iPhone texture may not match. Pluggable model;
      defaults to a bicubic baseline so the path is wireable without a heavy dep.
  "off" — passthrough (legacy behavior).

We picked (b) as default because the goal is *distribution match*, not maximal
sharpness, and a frozen encoder cannot adapt to invented SR detail. (a) stays as
an ablation knob.

All functions operate on uint8 HxWx3 RGB and return uint8 HxWx3 RGB.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class QualityConfig:
    """Knobs for the quality normalizer.

    mode:        "normalize" (b) | "sr" (a, ablation) | "off".
    work_size:   canonical effective-resolution cap in px (< out_size). The single
                 most important knob; tune it. 112 = half of 224.
    out_size:    final crop size handed to MARLIN (224).
    augment:     train-only random quality jitter (ignored when training=False).
    aug_*:       augmentation ranges.
    sr_model_path: optional path to a real SR model for mode="sr" (else bicubic).
    """

    mode: str = "normalize"
    work_size: int = 112
    out_size: int = 224
    augment: bool = False
    aug_min_work: int = 64
    aug_max_work: int = 224
    aug_blur_sigma_max: float = 1.2
    aug_jpeg_quality_min: int = 40

    sr_model_path: str | None = None

    def __post_init__(self):
        if self.mode not in ("normalize", "sr", "off"):
            raise ValueError(f"mode must be normalize|sr|off, got {self.mode!r}")
        if not 1 <= self.work_size <= self.out_size:
            raise ValueError(f"work_size must be in [1, out_size], got {self.work_size}")
        if self.aug_min_work > self.aug_max_work:
            raise ValueError("aug_min_work must be <= aug_max_work")


# ----------------------------------------------------------------------
# Core transforms
# ----------------------------------------------------------------------
def _check_rgb(img: np.ndarray) -> None:
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB, got {img.shape}")
    if img.dtype != np.uint8:
        raise TypeError(f"expected uint8, got {img.dtype}")


def canonical_normalize(img: np.ndarray, work_size: int, out_size: int = 224) -> np.ndarray:
    """Cap effective resolution at `work_size`, then resize to `out_size`.

    Downscale with INTER_AREA (proper low-pass) then upscale with INTER_CUBIC.
    Deterministic. This is strategy (b): it equalizes high-frequency content
    across inputs of differing sharpness without inventing detail.
    """
    _check_rgb(img)
    if work_size >= out_size:
        # no cap below out_size: just ensure final size
        if img.shape[:2] != (out_size, out_size):
            return cv2.resize(img, (out_size, out_size), interpolation=cv2.INTER_CUBIC)
        return img.copy()
    small = cv2.resize(img, (work_size, work_size), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (out_size, out_size), interpolation=cv2.INTER_CUBIC)


def _jpeg_recompress(img: np.ndarray, quality: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return img
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)  # decodes as 3-channel


def quality_augment(img: np.ndarray, rng: np.random.Generator, cfg: QualityConfig) -> np.ndarray:
    """Train-only random degradation spanning a range of capture qualities.

    Randomizes the effective-resolution cap, optional Gaussian blur, and optional
    JPEG recompression. Seeded via `rng` for determinism. Output is `out_size`.
    """
    _check_rgb(img)
    work = int(rng.integers(cfg.aug_min_work, cfg.aug_max_work + 1))
    out = canonical_normalize(img, work_size=min(work, cfg.out_size), out_size=cfg.out_size)

    if cfg.aug_blur_sigma_max > 0 and rng.random() < 0.5:
        sigma = float(rng.uniform(0.1, cfg.aug_blur_sigma_max))
        out = cv2.GaussianBlur(out, (0, 0), sigmaX=sigma)

    if cfg.aug_jpeg_quality_min < 100 and rng.random() < 0.5:
        q = int(rng.integers(cfg.aug_jpeg_quality_min, 101))
        out = _jpeg_recompress(out, q)

    return np.ascontiguousarray(out, dtype=np.uint8)


def super_resolve(img: np.ndarray, out_size: int = 224, model=None) -> np.ndarray:
    """Strategy (a) ablation: upscale toward higher apparent resolution.

    If `model` is provided it must expose `.upsample(bgr) -> bgr` (e.g. an OpenCV
    `cv2.dnn_superres.DnnSuperResImpl`). Otherwise falls back to a bicubic 2× then
    resize — a baseline that makes the path runnable without committing a heavy SR
    dependency. Real SR is a future ablation, not the default pipeline.
    """
    _check_rgb(img)
    if model is not None:
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        up = model.upsample(bgr)
        up = cv2.resize(up, (out_size, out_size), interpolation=cv2.INTER_CUBIC)
        return cv2.cvtColor(up, cv2.COLOR_BGR2RGB)
    big = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2), interpolation=cv2.INTER_CUBIC)
    return cv2.resize(big, (out_size, out_size), interpolation=cv2.INTER_CUBIC)


# ----------------------------------------------------------------------
# Normalizer (single entry point used by the encoder / preprocessing)
# ----------------------------------------------------------------------
class QualityNormalizer:
    """Applies the configured quality policy to a face crop.

    Use ONE instance for both train and inference so the deterministic part is
    identical. Pass `training=True` to enable augmentation (mode="normalize" only).
    """

    def __init__(self, cfg: QualityConfig | None = None):
        self.cfg = cfg or QualityConfig()
        self._sr_model = None
        if self.cfg.mode == "sr" and self.cfg.sr_model_path:
            self._sr_model = self._load_sr(self.cfg.sr_model_path)

    @staticmethod
    def _load_sr(path: str):
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel(path)
        # model name/scale are encoded in typical EDSR/ESPCN filenames; caller's
        # responsibility to provide a compatible model. We infer scale=2 default.
        sr.setModel("edsr", 2)
        return sr

    def __call__(self, img_rgb: np.ndarray, *, training: bool = False,
                 rng: np.random.Generator | None = None) -> np.ndarray:
        cfg = self.cfg
        if cfg.mode == "off":
            _check_rgb(img_rgb)
            if img_rgb.shape[:2] != (cfg.out_size, cfg.out_size):
                return cv2.resize(img_rgb, (cfg.out_size, cfg.out_size),
                                  interpolation=cv2.INTER_CUBIC)
            return img_rgb
        if cfg.mode == "sr":
            return super_resolve(img_rgb, out_size=cfg.out_size, model=self._sr_model)
        # mode == "normalize"
        if training and cfg.augment:
            if rng is None:
                rng = np.random.default_rng()
            return quality_augment(img_rgb, rng, cfg)
        return canonical_normalize(img_rgb, work_size=cfg.work_size, out_size=cfg.out_size)


def variance_of_laplacian(img_rgb: np.ndarray) -> float:
    """Sharpness proxy: variance of the Laplacian on the grayscale image.

    Higher = sharper. Used to verify that normalization shrinks the sharpness gap
    between a sharp and a blurry crop (see tests)."""
    _check_rgb(img_rgb)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())
