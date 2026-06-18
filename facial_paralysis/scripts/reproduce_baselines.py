"""Reproduce both Oo 2025 and Shagdar 2024 baselines on the same input set,
output a JSON summary. Designed to run inside a Vertex AI Custom Job container
where the only setup is `pip install -r requirements_reproduce.txt`.

Inputs (all from <bundle_root>/input_images/):
  - OBAMA_healthy:  public-domain studio portrait (expected: healthy/non-stroke)
  - BELLS_palsy:    Wikimedia Bell's palsy photo (expected: palsy/stroke)
  - sample_take:    one Mayo iPhone thumbnail (no ground truth in this script)

Outputs (printed line-by-line + written to result.json):
  - For each input: device used, Oo verdict + p(palsy), Shagdar mean p(stroke)
  - Architecture sanity: all state_dicts load strict=True
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
print(f"[setup] bundle root: {ROOT}", flush=True)

# ---------------------------------------------------------------------- Oo
from src.baselines.oo_multimodal.models import EarlyFusion, FCNManual, MLPMixerRGB  # noqa: E402
from src.baselines.oo_multimodal.utils import (  # noqa: E402
    MediaPipeFaceLandmarker, compute_29_features, crop_face_to_square,
)
from torchvision import transforms  # noqa: E402

# ---------------------------------------------------------------------- Shagdar
SHAGDAR_DIR = ROOT / "src" / "baselines" / "shagdar_gnn"
sys.path.insert(0, str(SHAGDAR_DIR))
from graph_model import GCNMid  # noqa: E402

from torch_geometric.data import Data, Batch  # noqa: E402
import scipy.spatial  # noqa: E402
import mediapipe as mp  # noqa: E402
from mediapipe.tasks import python as mp_python  # noqa: E402
from mediapipe.tasks.python import vision as mp_vision  # noqa: E402


_MIXER_PREPROC = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])


def _load_oo(device):
    w = ROOT / "src" / "baselines" / "oo_multimodal" / "weights"
    mixer = MLPMixerRGB().to(device).eval()
    fcn = FCNManual().to(device).eval()
    fusion = EarlyFusion().to(device).eval()
    sd1 = torch.load(w / "0_mlp_mixer_rgb_yfp_ck_urp_50_last_training_weights.pth",
                     map_location=device, weights_only=False)
    sd2 = torch.load(w / "fcn_manual_val_0_last_training_weights.pth",
                     map_location=device, weights_only=False)
    sd3 = torch.load(w / "early_fusion_fcn_rgb_mixer_val_0_training_weights_epoch_9.pth",
                     map_location=device, weights_only=False)
    # Verify strict load
    m1, u1 = mixer.load_state_dict(sd1, strict=False)
    m2, u2 = fcn.load_state_dict(sd2, strict=False)
    m3, u3 = fusion.load_state_dict(sd3, strict=False)
    strict_ok = all(len(m) == 0 and len(u) == 0 for m, u in [(m1,u1),(m2,u2),(m3,u3)])
    return mixer, fcn, fusion, strict_ok


@torch.no_grad()
def _oo_predict(image_bgr, landmarker, mixer, fcn, fusion, device):
    lms = landmarker(image_bgr)
    if lms is None:
        return None
    crop_bgr = crop_face_to_square(image_bgr, lms, output_size=224)
    crop_lms = landmarker(crop_bgr)
    if crop_lms is None:
        crop_lms = lms
    feats29 = compute_29_features(crop_lms)
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    x_img = _MIXER_PREPROC(rgb).unsqueeze(0).to(device)
    x_manual = torch.from_numpy(feats29).unsqueeze(0).to(device)
    mfeat = mixer.extract_features(x_img)
    hfeat = fcn.extract_features(x_manual)
    logits = fusion(mfeat, hfeat)
    probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    return {"verdict": "Palsy" if int(np.argmax(probs)) == 1 else "Healthy",
            "p_palsy": float(probs[1])}


# ---------------------------------------------------------------------- Shagdar
def _shagdar_make_detector():
    return mp_vision.FaceLandmarker.create_from_options(
        mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(SHAGDAR_DIR / "face_landmarker_v2_with_blendshapes.task")),
            num_faces=1,
        )
    )


def _shagdar_image_to_graph(image_bgr, detector):
    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    detection = detector.detect(mp_image)
    if not detection.face_landmarks:
        return None
    landmarks = np.array(
        [(int(lm.x * w), int(lm.y * h)) for lm in detection.face_landmarks[0]],
        dtype=np.int64,
    )
    tri = scipy.spatial.Delaunay(landmarks).simplices
    edges = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]], axis=0)
    edges = np.sort(edges, axis=1)
    edge_index = np.unique(edges, axis=0)
    x = torch.tensor(landmarks, dtype=torch.float32)
    ei = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=ei)


def _load_shagdar(device):
    models = []
    strict_ok = True
    for i in range(4):
        path = SHAGDAR_DIR / "model_weights" / f"clean_split_{i}.pt"
        m = GCNMid().to(device).eval()
        sd = torch.load(path, map_location=device, weights_only=False)
        missing, unexpected = m.load_state_dict(sd, strict=False)
        if missing or unexpected:
            strict_ok = False
        models.append(m)
    return models, strict_ok


@torch.no_grad()
def _shagdar_predict(graph, models, device):
    batch = Batch.from_data_list([graph]).to(device)
    per_split = []
    for m in models:
        logits = m(batch.x, batch.edge_index, batch.batch)
        per_split.append(F.softmax(logits, dim=1).squeeze(0).cpu().numpy())
    mean_p = np.mean(per_split, axis=0)
    return {"vote": "stroke" if mean_p[1] >= 0.5 else "nonstroke",
            "mean_p_stroke": float(mean_p[1]),
            "per_split_p_stroke": [float(p[1]) for p in per_split]}


# ---------------------------------------------------------------------- driver
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] torch: {torch.__version__}, device: {device}, "
          f"cuda: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"[setup] gpu: {torch.cuda.get_device_name(0)}", flush=True)

    t0 = time.perf_counter()
    print("[setup] loading Oo baseline ...", flush=True)
    oo_mixer, oo_fcn, oo_fusion, oo_strict = _load_oo(device)
    print(f"  Oo all strict_load=True? {oo_strict}", flush=True)
    print(f"  Oo load time: {time.perf_counter()-t0:.1f}s", flush=True)

    t0 = time.perf_counter()
    print("[setup] loading Shagdar 4-split ensemble ...", flush=True)
    shagdar_models, shagdar_strict = _load_shagdar(device)
    print(f"  Shagdar all strict_load=True? {shagdar_strict}", flush=True)
    print(f"  Shagdar load time: {time.perf_counter()-t0:.1f}s", flush=True)

    print("[setup] preparing MediaPipe detectors ...", flush=True)
    oo_lm = MediaPipeFaceLandmarker()
    sh_lm = _shagdar_make_detector()

    samples_dir = ROOT / "input_images"
    samples = sorted(samples_dir.glob("*.jpg")) + sorted(samples_dir.glob("*.png"))
    print(f"[setup] {len(samples)} test images:", flush=True)
    for p in samples:
        print(f"    {p.name}", flush=True)

    results: list[dict] = []
    for path in samples:
        bgr = cv2.imread(str(path))
        if bgr is None:
            print(f"  SKIP {path.name}: cv2 failed to read", flush=True)
            continue
        h, w = bgr.shape[:2]
        print(f"\n--- {path.name} ({w}x{h}) ---", flush=True)

        # Oo
        t0 = time.perf_counter()
        oo_out = _oo_predict(bgr, oo_lm, oo_mixer, oo_fcn, oo_fusion, device)
        t_oo = (time.perf_counter() - t0) * 1000
        if oo_out is None:
            print(f"  Oo:      NO_FACE", flush=True)
            oo_str = "NO_FACE"
        else:
            oo_str = f"{oo_out['verdict']} (p_palsy={oo_out['p_palsy']:.3f})"
            print(f"  Oo:      {oo_str}   [{t_oo:.0f} ms]", flush=True)

        # Shagdar
        t0 = time.perf_counter()
        graph = _shagdar_image_to_graph(bgr, sh_lm)
        sh_out = _shagdar_predict(graph, shagdar_models, device) if graph else None
        t_sh = (time.perf_counter() - t0) * 1000
        if sh_out is None:
            print(f"  Shagdar: NO_FACE", flush=True)
            sh_str = "NO_FACE"
        else:
            sh_str = (f"{sh_out['vote']} (mean_p_stroke={sh_out['mean_p_stroke']:.3f}, "
                      f"per_split={[round(p,3) for p in sh_out['per_split_p_stroke']]})")
            print(f"  Shagdar: {sh_str}   [{t_sh:.0f} ms]", flush=True)

        results.append({
            "image": path.name,
            "resolution": f"{w}x{h}",
            "oo": oo_out,
            "oo_ms": round(t_oo, 1),
            "shagdar": sh_out,
            "shagdar_ms": round(t_sh, 1),
        })

    out_path = ROOT / "result.json"
    payload = {
        "device": str(device),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "oo_strict_load": oo_strict,
        "shagdar_strict_load": shagdar_strict,
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\n[done] wrote {out_path}", flush=True)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
