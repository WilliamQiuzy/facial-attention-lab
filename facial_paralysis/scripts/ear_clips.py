"""POD: run MediaPipe FaceLandmarker on pre-trimmed per-action clips at native fps,
compute per-frame EAR (eye aspect ratio) L/R + mouth-corner droop L/R.

Input: a dir of clips named <take>__<action>.mov + clip_manifest.json.
Output: ear.json = {clip: {t, ear_left, ear_right, corner_left, corner_right, fps}}.
"""
import argparse, json, os
from pathlib import Path
import numpy as np, cv2
import mediapipe as mp
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision

EYE = {"right": [33, 160, 158, 133, 153, 144], "left": [362, 385, 387, 263, 373, 380]}
CORNER = {"left": 61, "right": 291}
IOD = (33, 263)


def ear(pts, idx):
    p = [pts[i] for i in idx]
    v = np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])
    return float(v / (2.0 * np.linalg.norm(p[0] - p[3]) + 1e-6))


def run(clip, lm):
    cap = cv2.VideoCapture(clip)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    tr = {"fps": round(fps, 2), "t": [], "ear_left": [], "ear_right": [], "corner_left": [], "corner_right": []}
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        fi += 1
        if not res.face_landmarks:
            continue
        h, w = frame.shape[:2]
        pts = np.array([[p.x * w, p.y * h] for p in res.face_landmarks[0]])
        iod = np.linalg.norm(pts[IOD[0]] - pts[IOD[1]]) + 1e-6
        tr["t"].append(round((fi - 1) / fps, 4))
        tr["ear_left"].append(round(ear(pts, EYE["left"]), 4))
        tr["ear_right"].append(round(ear(pts, EYE["right"]), 4))
        tr["corner_left"].append(round(float(pts[CORNER["left"], 1]) / iod, 4))
        tr["corner_right"].append(round(float(pts[CORNER["right"], 1]) / iod, 4))
    cap.release()
    return tr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=a.model),
        running_mode=vision.RunningMode.IMAGE, num_faces=1))
    clips = sorted(Path(a.clips).glob("*.mov"))
    out = {}
    for c in clips:
        out[c.name] = run(str(c), lm)
        n = len(out[c.name]["t"])
        print(f"done {c.name}: {n} frames @ {out[c.name]['fps']}fps", flush=True)
        Path(a.out).write_text(json.dumps(out))
    print(f"wrote {a.out} ({len(out)} clips)", flush=True)


if __name__ == "__main__":
    main()
