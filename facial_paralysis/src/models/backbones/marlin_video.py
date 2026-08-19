"""Frozen MARLIN video encoder (appearance + intra-clip motion).

MARLIN (Cai et al., CVPR 2023) is a ViT-Base masked autoencoder pretrained
self-supervised on ~700k facial video clips. We use only its encoder as a frozen
feature extractor: a 16-frame aligned-face clip → a 768-d spatiotemporal
embedding. See docs/model_design.md §3.1.

Why we bypass `transformers.AutoModel`: the HuggingFace dynamic-module loader
fails to copy MARLIN's transitive relative imports (modules.py,
positional_embedding.py, ...) on transformers 4.50.x. The model's own classes
live next to the weights, so we import them directly and load the safetensors
ourselves. This is more robust and has no version coupling.

License note: MARLIN weights are CC BY-NC 4.0 (non-commercial). Research use only.

Usage:
    enc = MarlinVideoEncoder.from_default_weights().to(device).eval()
    vec = enc.encode_clip_bgr(list_of_16_bgr_frames)     # (768,) numpy, or None
    # or batched, already-cropped & normalized:
    emb = enc(clip_tensor)                                # (B, 768) torch
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_MARLIN_DIR = _PROJECT_ROOT / "data" / "external" / "marlin_vit_base_ytf"

# Import MARLIN's own modules as a package (its files use relative imports).
if str(_MARLIN_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_MARLIN_DIR.parent))
from marlin_vit_base_ytf.marlin import Marlin  # type: ignore  # noqa: E402

# The MediaPipe face crop (the Oo baseline's, proven to align faces) is imported
# LAZILY so MARLIN can be imported — and its `forward(clip)` used on pre-cropped
# tensors — WITHOUT mediapipe installed. Only encode_*_bgr (raw-frame cropping)
# need it. See src/preprocessing/face_crop_cv2.py for a mediapipe-free alternative.
_OO_BASELINE = _PROJECT_ROOT / "src" / "baselines" / "oo_multimodal"


def _face_crop_tools():
    if str(_OO_BASELINE) not in sys.path:
        sys.path.insert(0, str(_OO_BASELINE))
    from utils import MediaPipeFaceLandmarker, crop_face_to_square  # type: ignore
    return MediaPipeFaceLandmarker, crop_face_to_square


class MarlinVideoEncoder(nn.Module):
    """Frozen MARLIN encoder. Input clips are (B, 3, T=16, 224, 224) in [0, 1]."""

    OUT_DIM = 768
    INPUT_SIZE = 224
    CLIP_FRAMES = 16

    def __init__(self, config: dict):
        super().__init__()
        self.cfg = config
        # as_feature_extractor=True builds the encoder only (no decoder), matching
        # the released feature-extraction checkpoint.
        self.marlin = Marlin(
            img_size=config["img_size"],
            patch_size=config["patch_size"],
            n_frames=config["n_frames"],
            encoder_embed_dim=config["encoder_embed_dim"],
            encoder_depth=config["encoder_depth"],
            encoder_num_heads=config["encoder_num_heads"],
            decoder_embed_dim=config["decoder_embed_dim"],
            decoder_depth=config["decoder_depth"],
            decoder_num_heads=config["decoder_num_heads"],
            mlp_ratio=config["mlp_ratio"],
            qkv_bias=config["qkv_bias"],
            qk_scale=config["qk_scale"],
            drop_rate=config["drop_rate"],
            attn_drop_rate=config["attn_drop_rate"],
            norm_layer=config["norm_layer"],
            init_values=config["init_values"],
            tubelet_size=config["tubelet_size"],
            as_feature_extractor=True,
        )
        for p in self.parameters():
            p.requires_grad_(False)

    @classmethod
    def from_default_weights(cls, marlin_dir: str | Path | None = None) -> "MarlinVideoEncoder":
        d = Path(marlin_dir) if marlin_dir else _MARLIN_DIR
        cfg_path, w_path = d / "config.json", d / "model.safetensors"
        if not w_path.exists():
            raise FileNotFoundError(f"MARLIN weights not found at {w_path}")
        config = json.loads(cfg_path.read_text())
        model = cls(config)

        state = load_file(str(w_path))
        # Released keys are prefixed `marlin.` (from the MarlinModel wrapper) and
        # encoder-only. Strip the wrapper prefix to match our inner `Marlin`.
        stripped = {k[len("marlin."):]: v for k, v in state.items() if k.startswith("marlin.")}
        missing, unexpected = model.marlin.load_state_dict(stripped, strict=False)
        # The feature-extraction checkpoint has no decoder/proj weights; those are
        # the only allowed "missing" keys. Anything else is a real mismatch.
        bad_missing = [k for k in missing if not (k.startswith("decoder") or k.startswith("enc_dec_proj"))]
        if bad_missing:
            raise RuntimeError(f"unexpected MISSING MARLIN keys: {bad_missing[:8]}")
        if unexpected:
            raise RuntimeError(f"unexpected EXTRA MARLIN keys: {unexpected[:8]}")
        model.eval()
        return model

    def forward(self, clip: torch.Tensor, keep_seq: bool = False) -> torch.Tensor:
        """clip: (B, 3, 16, 224, 224) in [0,1]. Returns (B, 768) pooled, or
        (B, 1568, 768) if keep_seq=True."""
        if clip.dim() != 5 or clip.shape[1] != 3 or clip.shape[2] != self.CLIP_FRAMES:
            raise ValueError(
                f"expected (B,3,{self.CLIP_FRAMES},224,224), got {tuple(clip.shape)}"
            )
        return self.marlin.extract_features(clip, keep_seq=keep_seq)

    # ------------------------------------------------------------------
    # Frame sampling + face crop helpers
    # ------------------------------------------------------------------
    @staticmethod
    def sample_indices(n_available: int, n_want: int = CLIP_FRAMES) -> list[int]:
        """Evenly sample n_want frame indices from n_available (with repetition if
        too few — e.g. a still image tiles to 16)."""
        if n_available <= 0:
            raise ValueError("no frames available")
        if n_available >= n_want:
            return list(np.linspace(0, n_available - 1, n_want).round().astype(int))
        # too few: evenly stretch (repeats frames)
        return list(np.linspace(0, n_available - 1, n_want).round().astype(int))

    @torch.no_grad()
    def encode_clip_bgr(
        self,
        frames_bgr: list[np.ndarray],
        landmarker: MediaPipeFaceLandmarker | None = None,
        normalizer=None,
        training: bool = False,
        rng=None,
    ) -> np.ndarray | None:
        """Crop+align each of 16 BGR frames, stack, forward. Returns (768,) or
        None if no face is detected in a frame (that frame is dropped; if fewer
        than 1 remain we return None).

        `normalizer` (optional `QualityNormalizer`) closes the train/test
        resolution gap on each crop, applied identically at train and inference;
        `training=True` enables its augmentation. None = legacy passthrough."""
        _Landmarker, crop_face_to_square = _face_crop_tools()
        if landmarker is None:
            landmarker = _Landmarker()
        crops = []
        for img in frames_bgr:
            lms = landmarker(img)
            if lms is None:
                continue
            crop_bgr = crop_face_to_square(img, lms, output_size=self.INPUT_SIZE)
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            if normalizer is not None:
                crop_rgb = normalizer(crop_rgb, training=training, rng=rng)
            crops.append(crop_rgb)
        if not crops:
            return None
        # pad (repeat last) up to 16 if some frames lost their face
        while len(crops) < self.CLIP_FRAMES:
            crops.append(crops[-1])
        crops = crops[: self.CLIP_FRAMES]
        arr = np.stack(crops).astype(np.float32) / 255.0          # (16, 224, 224, 3) in [0,1]
        clip = torch.from_numpy(arr).permute(3, 0, 1, 2).unsqueeze(0)  # (1,3,16,224,224)
        device = next(self.parameters()).device
        return self(clip.to(device)).squeeze(0).cpu().numpy()

    @torch.no_grad()
    def encode_video_path(
        self,
        video_path: str | Path,
        n_clips: int = 1,
        landmarker: MediaPipeFaceLandmarker | None = None,
        normalizer=None,
        training: bool = False,
        rng=None,
    ) -> np.ndarray | None:
        """Read a video, split into n_clips evenly-spaced 16-frame windows, encode
        each. Returns (n_encoded, 768) or None if the video/faces are unusable.

        `normalizer`/`training`/`rng` are forwarded to `encode_clip_bgr` (quality
        normalization; None = legacy passthrough)."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"cannot open video: {video_path}")
        frames = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(fr)
        cap.release()
        if len(frames) < 1:
            return None
        if landmarker is None:
            _Landmarker, _ = _face_crop_tools()
            landmarker = _Landmarker()

        # Partition all frames into n_clips contiguous windows; from each window,
        # evenly sample 16 frames.
        vecs = []
        bounds = np.linspace(0, len(frames), n_clips + 1).round().astype(int)
        for i in range(n_clips):
            lo, hi = bounds[i], bounds[i + 1]
            if hi - lo < 1:
                continue
            idx = [lo + j for j in self.sample_indices(hi - lo)]
            window = [frames[j] for j in idx]
            v = self.encode_clip_bgr(window, landmarker=landmarker,
                                     normalizer=normalizer, training=training, rng=rng)
            if v is not None:
                vecs.append(v)
        if not vecs:
            return None
        return np.stack(vecs)
