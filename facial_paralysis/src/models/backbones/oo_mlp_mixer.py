"""Oo et al. 2025 MLP-Mixer B/16 as a frozen 768-d face encoder.

Why this exists separately from `src/baselines/oo_multimodal/`:
  The baselines folder is a faithful reconstruction of Oo's released inference
  repo (architecture + weights + test.py). Validation on our cohort showed the
  published classifier collapses to "always Palsy" on Mayo iPhone data — likely
  because Oo's preprocessing for the 29 handcrafted features was never
  released and our reconstruction differs from theirs.

  However the MLP-Mixer backbone itself was fine-tuned on YFP (palsy YouTube
  videos) + CK+ (healthy faces) and exposes a reasonable 768-d face embedding
  before the (over-fit) classification head. This module strips the head and
  surfaces only the encoder, for use as a frozen feature extractor by our
  downstream House-Brackmann classifier.

Usage:
    enc = OoMLPMixerEncoder.from_default_weights().to(device).eval()
    emb = enc.encode_image_bgr(image_bgr)          # (768,) numpy
    # or batched:
    emb_b = enc(images_tensor)                     # (B, 768) torch
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
from torchvision import transforms

# Reuse the MediaPipe wrapper + crop from the baseline reconstruction. They
# implement face detection and a 224×224 face crop in a way already proven to
# load images correctly. If we change cropping in the future, only this import
# changes.
import sys
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_OO_BASELINE = _PROJECT_ROOT / "src" / "baselines" / "oo_multimodal"
if str(_OO_BASELINE) not in sys.path:
    sys.path.insert(0, str(_OO_BASELINE))
from utils import MediaPipeFaceLandmarker, crop_face_to_square  # type: ignore  # noqa: E402


_DEFAULT_WEIGHTS = (
    _OO_BASELINE
    / "weights"
    / "0_mlp_mixer_rgb_yfp_ck_urp_50_last_training_weights.pth"
)


_MIXER_PREPROC = transforms.Compose(
    [
        transforms.ToTensor(),  # uint8 HxWxC RGB → float32 CxHxW in [0,1]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
)


class OoMLPMixerEncoder(nn.Module):
    """Frozen 768-d face encoder using Oo et al. 2025's MLP-Mixer B/16 weights.

    The forward pass returns the pre-head 768-d token-pooled embedding
    (i.e., what would feed into the original `head` Linear(768→2)). The head
    itself is dropped because validation showed it collapses on our cohort.
    """

    OUT_DIM = 768
    INPUT_SIZE = 224

    def __init__(self):
        super().__init__()
        # Build the same MLP-Mixer architecture the released state_dict expects.
        # We construct it under the `mlp_mixer` submodule prefix so the released
        # keys (mlp_mixer.stem.proj.weight, ...) load with strict=True.
        self.mlp_mixer = timm.create_model(
            "mixer_b16_224", pretrained=False, num_classes=2
        )
        # Freeze all parameters — this is a feature extractor.
        for p in self.parameters():
            p.requires_grad_(False)

    @classmethod
    def from_default_weights(cls, weights_path: str | Path | None = None) -> "OoMLPMixerEncoder":
        path = Path(weights_path) if weights_path else _DEFAULT_WEIGHTS
        if not path.exists():
            raise FileNotFoundError(
                f"MLP-Mixer weights not found at {path}. Download from the Drive "
                "folder linked in src/baselines/oo_multimodal/README.md."
            )
        model = cls()
        state = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(state, strict=True)
        model.eval()
        return model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, 224, 224), values normalized as for timm mixer_b16_224."""
        feats = self.mlp_mixer.forward_features(x)  # (B, 196, 768)
        return self.mlp_mixer.forward_head(feats, pre_logits=True)  # (B, 768)

    # ------------------------------------------------------------------
    # Convenience: end-to-end image → embedding
    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode_image_bgr(
        self,
        image_bgr: np.ndarray,
        landmarker: MediaPipeFaceLandmarker | None = None,
    ) -> np.ndarray | None:
        """Take a BGR uint8 image, run MediaPipe → face crop → preprocess →
        forward. Returns (768,) numpy or None if no face is detected.

        Pass a reused `landmarker` instance when calling in a loop to amortize
        MediaPipe init cost.
        """
        if landmarker is None:
            landmarker = MediaPipeFaceLandmarker()
        lms = landmarker(image_bgr)
        if lms is None:
            return None
        crop_bgr = crop_face_to_square(image_bgr, lms, output_size=self.INPUT_SIZE)
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        device = next(self.parameters()).device
        x = _MIXER_PREPROC(crop_rgb).unsqueeze(0).to(device)
        return self(x).squeeze(0).cpu().numpy()

    @torch.no_grad()
    def encode_image_path(
        self,
        image_path: str | Path,
        landmarker: MediaPipeFaceLandmarker | None = None,
    ) -> np.ndarray | None:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise IOError(f"could not read image: {image_path}")
        return self.encode_image_bgr(image_bgr, landmarker=landmarker)
