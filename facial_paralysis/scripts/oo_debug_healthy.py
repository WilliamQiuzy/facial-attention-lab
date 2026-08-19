"""Sanity test: a verifiably-healthy public-domain face (Obama official WH
portrait, 2012, public domain) vs. one Mayo FACES sample. Compare:

  - MLP-Mixer alone   (no 29-feature dependency)
  - FCN-manual alone with RAW Parra-Dominguez formulas
  - FCN-manual alone with a [0,1] normalization variant:
        angles → /180  (range [0,1])
        max-ratios → 1/max = min(a/b, b/a)  (range (0, 1])
        slopes left alone
        C/A, X/A left alone
  - EarlyFusion under both concat orders, both feature variants
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


def normalize_to_unit(feats29: np.ndarray) -> np.ndarray:
    """Apply a [0,1] normalization variant: divide angles by 180, invert max-ratios."""
    f = feats29.copy()
    # Angle features (per Parra-Dominguez Table 1)
    angle_idx = [0, 1, 2, 7, 14, 22, 23]
    for i in angle_idx:
        f[i] = f[i] / 180.0
    # Max-ratio features (always >= 1) → invert to min(a/b, b/a) ∈ (0, 1]
    max_ratio_idx = [3, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26]
    for i in max_ratio_idx:
        if abs(f[i]) > 1e-8:
            f[i] = 1.0 / f[i]
    # Slopes (4, 5, 6) and C/A (27), X/A (28) untouched
    return f


def _features_for(path, landmarker):
    img = cv2.imread(str(path))
    lms = landmarker(img)
    if lms is None:
        return None, None
    crop = crop_face_to_square(img, lms, output_size=224)
    crop_lms = landmarker(crop)
    if crop_lms is None:
        crop_lms = lms
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), compute_29_features(crop_lms)


def main():
    device = torch.device("cpu")  # deterministic
    w = OO_DIR / "weights"
    mixer = MLPMixerRGB().to(device).eval()
    fcn = FCNManual().to(device).eval()
    fusion = EarlyFusion().to(device).eval()
    mixer.load_state_dict(torch.load(w / "0_mlp_mixer_rgb_yfp_ck_urp_50_last_training_weights.pth",
                                     map_location=device, weights_only=False), strict=True)
    fcn.load_state_dict(torch.load(w / "fcn_manual_val_0_last_training_weights.pth",
                                   map_location=device, weights_only=False), strict=True)
    fusion.load_state_dict(torch.load(w / "early_fusion_fcn_rgb_mixer_val_0_training_weights_epoch_9.pth",
                                      map_location=device, weights_only=False), strict=True)

    landmarker = MediaPipeFaceLandmarker()
    samples = [
        ("HEALTHY (Obama)", ROOT / "src/baselines/oo_multimodal/input_images/healthy_portrait.jpg"),
        ("FACES_017",       ROOT / "data/livelinkface_data/20260219_FACES017/thumbnail.jpg"),
        ("MySlate_2",       ROOT / "data/livelinkface_data/20260109_MySlate_2/thumbnail.jpg"),
    ]

    for label, path in samples:
        rgb, feats_raw = _features_for(path, landmarker)
        if rgb is None:
            print(f"\n=== {label}: NO_FACE ===")
            continue
        feats_norm = normalize_to_unit(feats_raw)
        print(f"\n=== {label} ===")
        print(f"raw feature range:  min={feats_raw.min():.2f} max={feats_raw.max():.2f} mean={feats_raw.mean():.2f}")
        print(f"norm feature range: min={feats_norm.min():.2f} max={feats_norm.max():.2f} mean={feats_norm.mean():.2f}")

        mixer_in = _MIXER_PREPROC(rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            ml = mixer(mixer_in).cpu().numpy().squeeze()
            mp = F.softmax(torch.from_numpy(ml), dim=0).numpy()
            mfeat = mixer.extract_features(mixer_in)

        print(f"MLP-Mixer alone:                logits={ml}  -> p(palsy)={mp[1]:.3f}")

        for tag, feats in [("RAW", feats_raw), ("NORM[0,1]", feats_norm)]:
            manual_in = torch.from_numpy(feats).unsqueeze(0).to(device)
            with torch.no_grad():
                fl = fcn(manual_in).cpu().numpy().squeeze()
                fp = F.softmax(torch.from_numpy(fl), dim=0).numpy()
                hfeat = fcn.extract_features(manual_in)
            for order_name, concat in [
                ("[mixer,manual]", torch.cat([mfeat, hfeat], dim=1)),
                ("[manual,mixer]", torch.cat([hfeat, mfeat], dim=1)),
            ]:
                with torch.no_grad():
                    el = fusion.layers(concat).cpu().numpy().squeeze()
                ep = F.softmax(torch.from_numpy(el), dim=0).numpy()
                print(f"  feat={tag:<10s}  FCN p(palsy)={fp[1]:.3f}  Fusion {order_name}: "
                      f"logits=({el[0]:>8.1f},{el[1]:>8.1f})  p(palsy)={ep[1]:.3f}")


if __name__ == "__main__":
    main()
