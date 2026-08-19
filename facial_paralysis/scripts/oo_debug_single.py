"""Diagnostic: dump every intermediate signal on one image so we can find why
the early-fusion model collapses to always-Palsy.

We compare:
  - FCN-manual standalone logits (single-modality classifier)
  - MLP-Mixer standalone logits (single-modality classifier)
  - EarlyFusion with concat order [mixer, manual]   ← our current guess
  - EarlyFusion with concat order [manual, mixer]   ← swapped
  - CPU vs MPS
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
OO_DIR = ROOT / "src" / "baselines" / "oo_multimodal"
sys.path.insert(0, str(OO_DIR))
from models import EarlyFusion, FCNManual, MLPMixerRGB
from utils import MediaPipeFaceLandmarker, compute_29_features, crop_face_to_square


_MIXER_PREPROC = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])


def _load_models(device, w):
    mixer = MLPMixerRGB().to(device).eval()
    fcn = FCNManual().to(device).eval()
    fusion = EarlyFusion().to(device).eval()
    mixer.load_state_dict(torch.load(w / "0_mlp_mixer_rgb_yfp_ck_urp_50_last_training_weights.pth",
                                     map_location=device, weights_only=False), strict=True)
    fcn.load_state_dict(torch.load(w / "fcn_manual_val_0_last_training_weights.pth",
                                   map_location=device, weights_only=False), strict=True)
    fusion.load_state_dict(torch.load(w / "early_fusion_fcn_rgb_mixer_val_0_training_weights_epoch_9.pth",
                                      map_location=device, weights_only=False), strict=True)
    return mixer, fcn, fusion


def _features_for(image_path, landmarker):
    image_bgr = cv2.imread(str(image_path))
    lms = landmarker(image_bgr)
    crop = crop_face_to_square(image_bgr, lms, output_size=224)
    crop_lms = landmarker(crop)
    if crop_lms is None:
        crop_lms = lms
    feats29 = compute_29_features(crop_lms)
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return rgb, feats29


def _diagnose(image_path, device, mixer, fcn, fusion):
    landmarker = MediaPipeFaceLandmarker()
    rgb, feats29 = _features_for(image_path, landmarker)

    print(f"\n=== {image_path.name} on {device} ===")
    print(f"29 handcrafted features (min/median/max/mean): "
          f"{feats29.min():.3f} / {np.median(feats29):.3f} / {feats29.max():.3f} / {feats29.mean():.3f}")
    print(f"   per-feature: {np.round(feats29, 2).tolist()}")

    mixer_in = _MIXER_PREPROC(rgb).unsqueeze(0).to(device)
    manual_in = torch.from_numpy(feats29).unsqueeze(0).to(device)

    with torch.no_grad():
        # --- MLP-Mixer standalone classifier (uses .head built into timm) ---
        mixer_logits = mixer(mixer_in).cpu().numpy().squeeze()
        mixer_probs = F.softmax(torch.from_numpy(mixer_logits), dim=0).numpy()
        # --- FCN-manual standalone classifier ---
        fcn_logits = fcn(manual_in).cpu().numpy().squeeze()
        fcn_probs = F.softmax(torch.from_numpy(fcn_logits), dim=0).numpy()
        # --- Embeddings for fusion ---
        mfeat = mixer.extract_features(mixer_in)  # (1, 768)
        hfeat = fcn.extract_features(manual_in)    # (1, 59)

    print(f"MLP-Mixer alone logits:  {mixer_logits}  probs: {mixer_probs}")
    print(f"FCN-manual alone logits: {fcn_logits}   probs: {fcn_probs}")
    print(f"mixer embed stats:  min={mfeat.min().item():.2f}  max={mfeat.max().item():.2f}  "
          f"mean={mfeat.mean().item():.2f}  std={mfeat.std().item():.2f}")
    print(f"manual embed stats: min={hfeat.min().item():.2f}  max={hfeat.max().item():.2f}  "
          f"mean={hfeat.mean().item():.2f}  std={hfeat.std().item():.2f}")

    # Try both concatenation orders
    for order_name, concat in [
        ("[mixer, manual]", torch.cat([mfeat, hfeat], dim=1)),
        ("[manual, mixer]", torch.cat([hfeat, mfeat], dim=1)),
    ]:
        with torch.no_grad():
            logits = fusion.layers(concat).cpu().numpy().squeeze()
            probs = F.softmax(torch.from_numpy(logits), dim=0).numpy()
        print(f"EarlyFusion {order_name:20s} logits: {logits}  probs: {probs}")


def main():
    w = OO_DIR / "weights"
    # Try a FACES (patient) and a MySlate (probably-healthy) and a public healthy face if we add one
    samples = [
        ROOT / "data" / "livelinkface_data" / "20260219_FACES017" / "thumbnail.jpg",  # patient
        ROOT / "data" / "livelinkface_data" / "20260109_MySlate_2"  / "thumbnail.jpg",  # likely healthy
    ]

    for device_name in ("cpu", "mps"):
        device = torch.device(device_name)
        try:
            mixer, fcn, fusion = _load_models(device, w)
        except Exception as e:
            print(f"\n=== {device_name} unavailable: {e} ===")
            continue
        for path in samples:
            _diagnose(path, device, mixer, fcn, fusion)


if __name__ == "__main__":
    main()
