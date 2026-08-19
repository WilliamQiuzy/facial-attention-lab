"""POD script (RunPod A100): 60fps MediaPipe EAR re-extraction of Mayo eye/mouth actions.

The cached blendshapes are 6fps — they resolve the 3-s hold but coarsely sample the
fast closure TRANSITION. This re-extracts the eye-closure and smile action windows at
native 60fps with the MediaPipe FaceLandmarker and computes geometric landmark
measures that blendshapes don't give directly:

  EAR (eye aspect ratio) per eye per frame  → closure depth + closure VELOCITY + time-to-close
  mouth-corner vertical droop L/R           → smile excursion asymmetry

giving publication-grade closure-dynamics (velocity/transient L-R asymmetry) and a
per-action high-fps feature set for a future supervised model.

Run on the pod:  .venv/bin/python scripts/mayo_ear_pod.py \
    --videos <dir_with_mov> --segments outputs/mayo_blendshapes/segments.json \
    --model data/mediapipe_out/_models/face_landmarker.task --out outputs/mayo_ear/ear.json
Copies each mov to /dev/shm and cv2-SEEKs only the action-window frames (network FS is slow).
"""
from __future__ import annotations
import argparse, json, os, shutil
from pathlib import Path
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# MediaPipe FaceMesh 6-point EAR landmark indices (per eye)
EYE = {
    "right": [33, 160, 158, 133, 153, 144],    # subject's right eye
    "left":  [362, 385, 387, 263, 373, 380],
}
MOUTH_CORNER = {"left": 61, "right": 291}
IOD = (33, 263)                                  # outer eye corners → inter-ocular distance (scale norm)
ACTIONS = ("GentleEyeClosure", "TightEyeSqueeze", "RelaxedSmile", "ReanimatedSmile")


def ear(pts, idx):
    p = [pts[i] for i in idx]
    v = np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])
    h = 2.0 * np.linalg.norm(p[0] - p[3]) + 1e-6
    return float(v / h)


def make_landmarker(model):
    opts = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model),
        running_mode=vision.RunningMode.VIDEO, num_faces=1)
    return vision.FaceLandmarker.create_from_options(opts)


def extract_take(mov, segs, lm):
    tmp = f"/dev/shm/{Path(mov).name}"
    shutil.copy(mov, tmp)
    cap = cv2.VideoCapture(tmp)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    out = {}
    for s in segs:
        if s["action"] not in ACTIONS:
            continue
        f0, f1 = int(s["t_start"] * fps), int(s["t_end"] * fps)
        traj = {"t": [], "ear_left": [], "ear_right": [], "corner_left": [], "corner_right": []}
        for fi in range(f0, f1 + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = lm.detect_for_video(img, int(fi / fps * 1000))
            if not res.face_landmarks:
                continue
            h, w = frame.shape[:2]
            pts = np.array([[p.x * w, p.y * h] for p in res.face_landmarks[0]])
            iod = np.linalg.norm(pts[IOD[0]] - pts[IOD[1]]) + 1e-6
            traj["t"].append(round(fi / fps, 4))
            traj["ear_left"].append(round(ear(pts, EYE["left"]), 4))
            traj["ear_right"].append(round(ear(pts, EYE["right"]), 4))
            traj["corner_left"].append(round(float(pts[MOUTH_CORNER["left"], 1]) / iod, 4))
            traj["corner_right"].append(round(float(pts[MOUTH_CORNER["right"], 1]) / iod, 4))
        out[s["action"]] = traj
    cap.release(); os.remove(tmp)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--segments", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    segments = json.loads(Path(a.segments).read_text())
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    movs = {p.stem: p for p in Path(a.videos).rglob("*.mov")}
    lm = make_landmarker(a.model)
    result = {}
    for take, segs in segments.items():
        # match take id to a video file (segments keys look like 20260313_FACES020)
        cand = [v for k, v in movs.items() if take in k or take.split("_", 1)[-1] in k]
        if not segs or not cand:
            print(f"skip {take} (segs={len(segs)} vid={len(cand)})", flush=True)
            continue
        result[take] = extract_take(str(cand[0]), segs, lm)
        print(f"done {take}: {list(result[take])}", flush=True)
        Path(a.out).write_text(json.dumps(result, indent=1))   # checkpoint each take
    print(f"wrote {a.out} ({len(result)} takes)", flush=True)


if __name__ == "__main__":
    main()
