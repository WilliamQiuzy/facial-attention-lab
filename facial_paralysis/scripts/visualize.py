"""Explainability / "feel the model" tool: score a video or image and render a
FAITHFUL visual report of WHY the model concludes facial paralysis.

It shows what the model actually keys on (not a decorative heatmap):
  1. Verdict on the face — P(palsy) + eyes/mouth severity from the model's own
     per-region ordinal heads, with the affected side marked.
  2. Region attribution by OCCLUSION — black out eyes / mouth / left / right of the
     aligned face, re-encode through frozen MARLIN, and measure how much P(palsy)
     drops. A big drop ⇒ the model relied on that region. This is model-faithful.
  3. WHEN — a per-frame left-vs-right asymmetry timeline (a real model input from
     MediaPipe), with the peak (most asymmetric) frame marked.
  4. WHICH muscles — the top left−right blendshape asymmetries, named.

Output: a single PNG panel (and optionally an annotated MP4 with --video-out).
This is the same tool intended for clinician feedback in the production pipeline.

Caveat: scores come from the public-data warm-start v1 (web-still trained). Use it
to sanity-check and explain behavior, not as a clinical instrument yet.

Usage:
  KMP_DUPLICATE_LIB_OK=TRUE \
  /opt/homebrew/Caskroom/miniforge/base/envs/dev/bin/python scripts/visualize.py \
      --input path/to/face.{mov,mp4,jpg,png,bmp} --out outputs/viz/report.png [--video-out out.mp4]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXT = {".mov", ".mp4", ".avi", ".m4v"}
SEV_COLOR = {0: "#2ca02c", 1: "#ff7f0e", 2: "#d62728"}   # Normal / Slight / Strong
SEV_NAME = ["Normal", "Slight", "Strong"]
CROP = 224
# region bands in aligned-224 crop space (face is centered+aligned by the crop)
BANDS = {
    "eyes":  (slice(56, 120), slice(0, 224)),
    "mouth": (slice(150, 208), slice(0, 224)),
    "left half (img)":  (slice(0, 224), slice(0, 112)),
    "right half (img)": (slice(0, 224), slice(112, 224)),
}

# MediaPipe FaceMesh landmark indices per facial region (for real region polygons)
REGION_LMK = {
    "right_eye": [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
    "left_eye":  [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466],
    "right_brow": [70, 63, 105, 66, 107, 46, 53, 52, 65, 55],
    "left_brow":  [300, 293, 334, 296, 336, 276, 283, 282, 295, 285],
    "nose": [1, 2, 98, 327, 168, 6, 197, 195, 5, 4, 45, 275, 220, 440],
    "mouth": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0,
              37, 39, 40, 185, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308],
}
# which clinical regions to score for asymmetry, and the polygons that draw them
ASYM_REGIONS = {  # display region -> (blendshape substring test, landmark groups)
    "brow":  (lambda n: "brow" in n,  ["right_brow", "left_brow"]),
    "eyes":  (lambda n: "eye" in n and "brow" not in n, ["right_eye", "left_eye"]),
    "nose":  (lambda n: "nose" in n,  ["nose"]),
    "mouth": (lambda n: "mouth" in n or "jaw" in n, ["mouth"]),
}


# ---------------------------------------------------------------- model / encoders
def load_everything(ckpt_path: Path, device: str):
    from src.models.facial_palsy_model import FacialPalsyModel, FacialPalsyConfig
    from src.models.multitask import TaskSpec
    from src.models.backbones.marlin_video import MarlinVideoEncoder
    from src.preprocessing.action_bundle import MediaPipeFeatureExtractor
    from src.preprocessing.image_quality import QualityConfig, QualityNormalizer

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    tasks = [TaskSpec(n, k, coupled=c) for (n, k, c) in ck["tasks"]]
    model = FacialPalsyModel(FacialPalsyConfig(tasks=tasks, **ck["model_cfg"])).to(device).eval()
    model.load_state_dict(ck["state_dict"])
    enc = MarlinVideoEncoder.from_default_weights().to(device).eval()
    mp_ext = MediaPipeFeatureExtractor()
    q = ck.get("quality", {"mode": "normalize", "work_size": 112})
    norm = QualityNormalizer(QualityConfig(mode=q["mode"], work_size=q["work_size"]))
    return model, enc, mp_ext, norm, ck


def read_frames(path: Path, max_frames: int = 64):
    ext = path.suffix.lower()
    if ext in IMAGE_EXT:
        img = cv2.imread(str(path))
        if img is None:
            raise IOError(f"cannot read {path}")
        return [img]
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    if len(frames) > max_frames:
        idx = np.linspace(0, len(frames) - 1, max_frames).round().astype(int)
        frames = [frames[i] for i in idx]
    return frames


# ---------------------------------------------------------------- MARLIN on crops
def aligned_crop(frame_bgr, landmarker):
    from utils import crop_face_to_square  # type: ignore
    lms = landmarker(frame_bgr)
    if lms is None:
        return None
    crop_bgr = crop_face_to_square(frame_bgr, lms, output_size=CROP)
    return cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)


def marlin_vec_from_crops(enc, crops_rgb, normalizer, device):
    """crops_rgb: list of (224,224,3) uint8 RGB -> (768,) MARLIN vec."""
    cs = [normalizer(c, training=False) if normalizer else c for c in crops_rgb]
    while len(cs) < enc.CLIP_FRAMES:
        cs.append(cs[-1])
    cs = cs[: enc.CLIP_FRAMES]
    arr = np.stack(cs).astype(np.float32) / 255.0
    clip = torch.from_numpy(arr).permute(3, 0, 1, 2).unsqueeze(0).to(device)
    with torch.no_grad():
        return enc(clip).squeeze(0).cpu().numpy()


def score_from_streams(model, marlin_vec, mp_seq, mp_mask, device):
    from src.models.ordinal import predict_grade, cum_probs
    me = torch.from_numpy(marlin_vec[None, None, None]).to(device)      # (1,1,1,768)
    mm = torch.ones(1, 1, 1, dtype=torch.bool, device=device)
    ms = torch.from_numpy(mp_seq[None, None]).to(device)               # (1,1,T,F)
    mk = torch.from_numpy(mp_mask[None, None]).to(device)
    ap = torch.ones(1, 1, dtype=torch.bool, device=device)
    with torch.no_grad():
        out = model(me, mm, ms, mk, ap)
    # raw cumulative logit (s − θ): unbounded, so occlusion attribution stays
    # sensitive even when the probability saturates near 0 or 1.
    res = {"palsy": float(cum_probs(out["binary"])[0, 0]),
           "palsy_logit": float(out["binary"][0, 0])}
    for t in ("eyes", "mouth"):
        if t in out:
            res[t] = {"level": int(predict_grade(out[t])[0]),
                      "expected": float(cum_probs(out[t]).sum(1)[0])}
    return res


# ---------------------------------------------------------------- main analysis
def analyze(input_path, model, enc, mp_ext, norm, device):
    from utils import MediaPipeFaceLandmarker  # type: ignore
    landmarker = MediaPipeFaceLandmarker()
    frames = read_frames(Path(input_path))

    # per-frame asymmetry (cheap, MediaPipe) + collect aligned crops
    seq, mask = mp_ext.extract_sequence(frames)            # (T,F),(T,)
    n_bs = len(mp_ext._bs_names)
    asym = seq[:, n_bs:]                                   # (T, n_pairs) L-R deltas
    asym_mag = np.abs(asym).sum(axis=1) * mask             # per-frame asymmetry magnitude
    peak = int(np.argmax(asym_mag)) if mask.any() else 0

    # crops around the peak for MARLIN (use up to 16 frames centered on peak)
    lo = max(0, peak - 8); hi = min(len(frames), lo + 16)
    crops = [c for c in (aligned_crop(frames[i], landmarker) for i in range(lo, hi)) if c is not None]
    if not crops:
        raise RuntimeError("no face detected")

    base_vec = marlin_vec_from_crops(enc, crops, norm, device)
    base = score_from_streams(model, base_vec, seq, mask, device)

    # occlusion attribution on the peak-window crops
    occl = {}
    for name, (rs, cs) in BANDS.items():
        occ_crops = []
        for c in crops:
            cc = c.copy(); cc[rs, cs] = 128
            occ_crops.append(cc)
        v = marlin_vec_from_crops(enc, occ_crops, norm, device)
        p = score_from_streams(model, v, seq, mask, device)["palsy_logit"]
        occl[name] = base["palsy_logit"] - p              # drop in logit = reliance

    # top asymmetric blendshapes at peak
    pair_names = [mp_ext._bs_names[l][:-4] for l, _ in mp_ext._pairs]
    peak_asym = asym[peak]
    order = np.argsort(-np.abs(peak_asym))[:8]
    top_asym = [(pair_names[i], float(peak_asym[i])) for i in order]

    # per-region asymmetry (sum |L-R| of the blendshapes governing that region),
    # averaged over a small window around the peak for stability
    w0, w1 = max(0, peak - 3), min(len(asym), peak + 4)
    win_asym = np.abs(asym[w0:w1]).mean(axis=0)
    region_asym = {}
    for rname, (test, _) in ASYM_REGIONS.items():
        region_asym[rname] = float(sum(win_asym[i] for i, n in enumerate(pair_names) if test(n)))

    # landmarks on the full peak frame (for real region polygons)
    peak_lmk = landmarker(frames[peak])

    return {"frames": frames, "peak": peak, "peak_crop": crops[min(len(crops) - 1, peak - lo)],
            "peak_frame": cv2.cvtColor(frames[peak], cv2.COLOR_BGR2RGB),
            "peak_lmk": peak_lmk, "region_asym": region_asym,
            "asym_mag": asym_mag, "mask": mask, "base": base, "occl": occl,
            "top_asym": top_asym, "n_frames": len(frames)}


def render(a, input_path, out_path, scheme):
    base, occl = a["base"], a["occl"]
    fig = plt.figure(figsize=(15, 9))
    fig.suptitle(f"Facial-paralysis model report — {Path(input_path).name}", fontsize=15, weight="bold")

    # ---- verdict banner
    palsy = base["palsy"]
    verdict = "PALSY" if palsy > 0.5 else "no palsy"
    vcol = "#d62728" if palsy > 0.5 else "#2ca02c"
    ax0 = fig.add_axes([0.05, 0.90, 0.9, 0.06]); ax0.axis("off")
    txt = f"P(palsy) = {palsy:.2f}  →  {verdict}"
    for t in ("eyes", "mouth"):
        if t in base:
            lv = base[t]["level"]
            txt += f"     |  {t}: {SEV_NAME[lv]} ({base[t]['expected']:.2f})"
    ax0.text(0.0, 0.5, txt, fontsize=14, weight="bold", color=vcol, va="center")

    # ---- A: peak FULL frame with REAL landmark-defined regions, filled by
    #         per-region asymmetry (red = asymmetric); eyes/mouth outlined in the
    #         model's own severity color. Covers more regions than the 2 heads.
    from matplotlib.patches import Polygon as MplPoly
    axA = fig.add_axes([0.03, 0.36, 0.36, 0.50]); axA.imshow(a["peak_frame"]); axA.axis("off")
    axA.set_title(f"Most-asymmetric frame (#{a['peak']}/{a['n_frames']})\n"
                  "fill = local L-R asymmetry · outline = model severity", fontsize=10)
    lmk = a["peak_lmk"]; ra = a["region_asym"]
    amax = max(ra.values()) if ra else 1.0
    cmap = plt.cm.RdYlGn_r
    if lmk is not None:
        for rname, (_, groups) in ASYM_REGIONS.items():
            heat = ra.get(rname, 0.0) / (amax + 1e-6)
            for g in groups:
                pts = lmk[REGION_LMK[g]].astype(np.int32)
                hull = cv2.convexHull(pts).reshape(-1, 2)
                axA.add_patch(MplPoly(hull, closed=True, facecolor=cmap(heat),
                                      alpha=0.45, edgecolor="none", zorder=2))
        # eyes / mouth severity outline (model heads)
        sev_groups = {"eyes": ["right_eye", "left_eye"], "mouth": ["mouth"]}
        for t, groups in sev_groups.items():
            if t not in base:
                continue
            col = SEV_COLOR[base[t]["level"]]
            allp = np.concatenate([lmk[REGION_LMK[g]] for g in groups]).astype(np.int32)
            hull = cv2.convexHull(allp).reshape(-1, 2)
            axA.add_patch(MplPoly(hull, closed=True, fill=False, edgecolor=col, lw=2.5, zorder=3))
            yk = hull[:, 1].min()
            axA.text(hull[:, 0].mean(), yk - 6, f"{t}: {SEV_NAME[base[t]['level']]}",
                     color=col, fontsize=9, weight="bold", ha="center", zorder=4)

    # ---- B: occlusion attribution (which region the palsy score relies on)
    axB = fig.add_axes([0.44, 0.55, 0.52, 0.31])
    names = list(occl.keys()); vals = [occl[n] for n in names]
    cols = ["#d62728" if v > 0 else "#1f77b4" for v in vals]
    axB.barh(names, vals, color=cols); axB.axvline(0, color="k", lw=0.8)
    axB.set_title("Region attribution — drop in palsy score (logit) when region is hidden\n(bigger = model relies on it more)", fontsize=10)
    axB.invert_yaxis()

    # ---- C: asymmetry timeline (WHEN)
    axC = fig.add_axes([0.44, 0.08, 0.52, 0.34])
    axC.plot(a["asym_mag"], color="#9467bd"); axC.axvline(a["peak"], color="#d62728", ls="--", label="peak")
    axC.set_title("Left–right asymmetry over time (MediaPipe input)", fontsize=10)
    axC.set_xlabel("frame"); axC.set_ylabel("|L−R| blendshape sum"); axC.legend(fontsize=8)

    # ---- D: per-region asymmetry (WHICH region) — matches the face overlay
    axD = fig.add_axes([0.05, 0.07, 0.32, 0.22])
    ra = a["region_asym"]
    rnames = list(ra.keys()); rvals = [ra[n] for n in rnames]
    amax = max(rvals) if rvals else 1.0
    axD.barh(rnames, rvals, color=[plt.cm.RdYlGn_r(v / (amax + 1e-6)) for v in rvals])
    axD.set_title("Left−right asymmetry by region (avg near peak)", fontsize=10)
    axD.tick_params(labelsize=9); axD.invert_yaxis()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110); plt.close(fig)
    return out_path


def write_annotated_video(a, mp_ext, input_path, video_out, model, enc, norm, device):
    """Annotated MP4: per-frame asymmetry bar + verdict banner (overall scores)."""
    from utils import MediaPipeFaceLandmarker  # type: ignore
    frames = a["frames"]; mag = a["asym_mag"]; mmax = max(mag.max(), 1e-6)
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(video_out), cv2.VideoWriter_fourcc(*"mp4v"), 12, (w, h))
    palsy = a["base"]["palsy"]
    banner = f"P(palsy)={palsy:.2f} {'PALSY' if palsy>0.5 else 'ok'}"
    for i, fr in enumerate(frames):
        img = fr.copy()
        col = (0, 0, 255) if palsy > 0.5 else (0, 160, 0)
        cv2.rectangle(img, (0, 0), (w, 40), (0, 0, 0), -1)
        cv2.putText(img, banner, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
        # asymmetry bar (right side), red when high
        bh = int((h - 60) * (mag[i] / mmax))
        cv2.rectangle(img, (w - 30, h - 10 - bh), (w - 10, h - 10),
                      (0, 0, 255) if mag[i] > 0.5 * mmax else (0, 200, 200), -1)
        if i == a["peak"]:
            cv2.putText(img, "PEAK asymmetry", (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        vw.write(img)
    vw.release()
    return video_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--video-out", default=None)
    ap.add_argument("--checkpoint", default=str(ROOT / "outputs" / "checkpoints" / "warmstart_v1.pt"))
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    model, enc, mp_ext, norm, ck = load_everything(Path(args.checkpoint), args.device)
    scheme = ck.get("label_scheme", {})
    a = analyze(args.input, model, enc, mp_ext, norm, args.device)

    out = args.out or str(ROOT / "outputs" / "viz" / (Path(args.input).stem + "_report.png"))
    render(a, args.input, out, scheme)
    print(f"report -> {out}")
    print(f"  P(palsy)={a['base']['palsy']:.3f}  "
          + "  ".join(f"{t}={SEV_NAME[a['base'][t]['level']]}" for t in ('eyes', 'mouth') if t in a['base']))
    print("  region reliance (Δpalsy when hidden): "
          + ", ".join(f"{k}={v:+.3f}" for k, v in a["occl"].items()))
    if args.video_out:
        write_annotated_video(a, mp_ext, args.input, args.video_out, model, enc, norm, args.device)
        print(f"annotated video -> {args.video_out}")


if __name__ == "__main__":
    main()
