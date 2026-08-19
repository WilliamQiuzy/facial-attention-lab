"""Batch-run the reconstructed Oo 2025 early-fusion baseline over every take in
`data/livelinkface_data/`. For each take we use that take's `thumbnail.jpg`
(LiveLinkFace generates one per take). Reports prediction, Palsy probability
(softmax), and the per-step timing so we can characterize throughput.

Run from project root:
    KMP_DUPLICATE_LIB_OK=TRUE conda run -n dev python scripts/oo_infer_takes.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
OO_DIR = ROOT / "src" / "baselines" / "oo_multimodal"
DATA_DIR = ROOT / "data" / "livelinkface_data"

sys.path.insert(0, str(OO_DIR))
from models import EarlyFusion, FCNManual, MLPMixerRGB
from utils import MediaPipeFaceLandmarker, compute_29_features, crop_face_to_square


_MIXER_PREPROC = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])


def _load_models(device):
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
    return mixer, fcn, fusion


def _predict_with_timing(image_bgr, landmarker, mixer, fcn, fusion, device):
    """Return (verdict, palsy_prob, step_times_ms)."""
    t = {}

    t0 = time.perf_counter()
    landmarks = landmarker(image_bgr)
    t["mediapipe_full"] = (time.perf_counter() - t0) * 1000
    if landmarks is None:
        return "NO_FACE", None, t

    t0 = time.perf_counter()
    crop_bgr = crop_face_to_square(image_bgr, landmarks, output_size=224)
    t["crop"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    crop_lms = landmarker(crop_bgr)
    if crop_lms is None:
        crop_lms = landmarks
    t["mediapipe_crop"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    feats29 = compute_29_features(crop_lms)
    t["handcrafted_29"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    mixer_in = _MIXER_PREPROC(rgb).unsqueeze(0).to(device)
    manual_in = torch.from_numpy(feats29).unsqueeze(0).to(device)
    with torch.no_grad():
        mfeat = mixer.extract_features(mixer_in)        # (1, 768)
        hfeat = fcn.extract_features(manual_in)          # (1, 59)
        logits = fusion(mfeat, hfeat)                    # (1, 2)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    t["forward"] = (time.perf_counter() - t0) * 1000

    palsy_p = float(probs[1])
    verdict = "Palsy" if palsy_p >= 0.5 else "Healthy"
    return verdict, palsy_p, t


def main():
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    print(f"device: {device}")

    landmarker = MediaPipeFaceLandmarker()
    mixer, fcn, fusion = _load_models(device)

    takes = sorted(p.name for p in DATA_DIR.iterdir() if p.is_dir())
    print(f"# takes found: {len(takes)}\n")

    rows = []
    print(f"{'#':>2}  {'take':<24s}  {'group':<7s}  {'verdict':<8s}  {'p(palsy)':>9s}  "
          f"{'face':>5s}  {'crop':>5s}  {'feat':>5s}  {'fwd':>6s}  {'total':>6s}")
    print("-" * 92)
    for i, take in enumerate(takes, 1):
        thumb = DATA_DIR / take / "thumbnail.jpg"
        group = "FACES" if "FACES" in take else "MySlate"
        if not thumb.exists():
            print(f"{i:>2}  {take:<24s}  {group:<7s}  NO_THUMB")
            continue
        image_bgr = cv2.imread(str(thumb))
        verdict, p_palsy, t = _predict_with_timing(image_bgr, landmarker, mixer, fcn, fusion, device)
        if verdict == "NO_FACE":
            print(f"{i:>2}  {take:<24s}  {group:<7s}  NO_FACE")
            continue
        total = sum(t.values())
        print(f"{i:>2}  {take:<24s}  {group:<7s}  {verdict:<8s}  {p_palsy:>9.3f}  "
              f"{t['mediapipe_full']:>5.0f}  {t['crop']:>5.1f}  {t['handcrafted_29']:>5.2f}  "
              f"{t['forward']:>6.0f}  {total:>6.0f}")
        rows.append((take, group, verdict, p_palsy, total))

    print()
    print("============ SUMMARY ============")
    n = len(rows)
    if n:
        n_palsy = sum(1 for r in rows if r[2] == "Palsy")
        n_healthy = n - n_palsy
        print(f"successful predictions: {n} / {len(takes)}")
        print(f"  Palsy:   {n_palsy}")
        print(f"  Healthy: {n_healthy}")
        for grp in ("FACES", "MySlate"):
            sub = [r for r in rows if r[1] == grp]
            if sub:
                gp = sum(1 for r in sub if r[2] == "Palsy")
                print(f"  {grp:<8s}: {gp}/{len(sub)} predicted Palsy")
        times = [r[4] for r in rows]
        print(f"\nlatency per image (full pipeline, ms): "
              f"mean={np.mean(times):.0f}  median={np.median(times):.0f}  "
              f"min={np.min(times):.0f}  max={np.max(times):.0f}")
        per_s = 1000.0 / np.mean(times)
        print(f"throughput: ~{per_s:.2f} images/sec")
        print(f"\nProjected costs:")
        print(f"  1 image:                {np.mean(times):>7.0f} ms")
        print(f"  60 frames (1s @ 60fps): {np.mean(times)*60/1000:>7.1f} s")
        print(f"  360 frames (1min @ 6fps, Oo's training rate): "
              f"{np.mean(times)*360/1000:>7.1f} s")
        print(f"  3600 frames (1min @ 60fps): "
              f"{np.mean(times)*3600/1000:>7.1f} s = {np.mean(times)*3600/60000:.1f} min")


if __name__ == "__main__":
    main()
