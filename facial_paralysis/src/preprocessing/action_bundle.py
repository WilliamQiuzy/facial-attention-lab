"""Preprocessing: per-action clip → cached bundle (.npz) for the MARLIN pipeline.

For each segmented ~3 s action clip this produces the two streams of
docs/model_design.md §3, written to `<cache>/<patient_id>/<action>.npz`:
    marlin   (W, 768)  — frozen MARLIN clip embedding(s) over W windows
    mp_seq   (T, F)    — MediaPipe per-frame features: 52 blendshapes + L/R
                         asymmetry deltas (F computed dynamically; see below)
    mp_mask  (T,) bool — True where a face was detected

`F` depends on how many Left/Right blendshape pairs MediaPipe exposes (computed
at runtime, typically ~70). The value is stored as `mp_feat_dim` in the npz and
printed; set `FacialPalsyConfig.mp_feat_dim` to match when training.

Segmentation (long video → per-action clips) happens UPSTREAM of this module —
see docs/model_design.md §2. This extractor only ever sees short per-action clips.

CLI:
  KMP_DUPLICATE_LIB_OK=TRUE python -m src.preprocessing.action_bundle \
      --data-root <root with <patient>/<action>.mov> --cache-root <out>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_DEFAULT_MP_MODEL = _PROJECT_ROOT / "data" / "mediapipe_out" / "_models" / "face_landmarker.task"


def _read_frames(path: str | Path, max_frames: int | None = None) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open video: {path}")
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    if max_frames and len(frames) > max_frames:
        idx = np.linspace(0, len(frames) - 1, max_frames).round().astype(int)
        frames = [frames[i] for i in idx]
    return frames


class MediaPipeFeatureExtractor:
    """Per-frame MediaPipe features: 52 blendshapes + left/right asymmetry deltas.

    The asymmetry features are signed (left - right) differences for every
    blendshape that has a mirrored partner (e.g. mouthSmileLeft - mouthSmileRight),
    which is exactly the clinically diagnostic asymmetry signal.
    """

    # geometry feature names (appended when with_geometry=True). EAR = eye aspect
    # ratio (closure); the rest are scale-normalized L/R region asymmetries.
    GEOM_NAMES = ["ear_right", "ear_left", "ear_asym", "brow_asym", "mouthcorner_asym"]

    def __init__(self, model_path: str | Path | None = None, with_geometry: bool = False):
        import mediapipe as mp
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision as mpv

        model_path = Path(model_path) if model_path else _DEFAULT_MP_MODEL
        if not model_path.exists():
            raise FileNotFoundError(f"MediaPipe model not found at {model_path}")
        self._mp = mp
        self.with_geometry = with_geometry
        self._landmarker = mpv.FaceLandmarker.create_from_options(
            mpv.FaceLandmarkerOptions(
                base_options=mpp.BaseOptions(model_asset_path=str(model_path)),
                running_mode=mpv.RunningMode.IMAGE,
                output_face_blendshapes=True,
                num_faces=1,
            )
        )
        self._bs_names: list[str] | None = None
        self._pairs: list[tuple[int, int]] | None = None  # (left_idx, right_idx)

    def _init_layout(self, names: list[str]) -> None:
        self._bs_names = names
        idx = {n: i for i, n in enumerate(names)}
        pairs = []
        for n, i in idx.items():
            if n.endswith("Left"):
                r = n[:-4] + "Right"
                if r in idx:
                    pairs.append((i, idx[r]))
        self._pairs = pairs

    @property
    def feat_dim(self) -> int:
        if self._bs_names is None:
            raise RuntimeError("feat_dim unknown until the first frame is processed")
        return len(self._bs_names) + len(self._pairs) + (len(self.GEOM_NAMES) if self.with_geometry else 0)

    @property
    def feature_names(self) -> list[str]:
        base = list(self._bs_names)
        asym = [f"asym_{self._bs_names[l][:-4]}" for l, _ in self._pairs]
        return base + asym + (list(self.GEOM_NAMES) if self.with_geometry else [])

    @staticmethod
    def _geometry(lms) -> np.ndarray:
        """Eye Aspect Ratio (closure) per eye + scale-normalized L/R asymmetries,
        from MediaPipe normalized landmarks. EAR directly captures eye-closure —
        the local, transient signal that whole-face pooling loses (the eyes head's
        failure mode). All distances normalized by inter-ocular distance."""
        P = np.array([[p.x, p.y] for p in lms], dtype=np.float32)
        d = lambda a, b: float(np.linalg.norm(P[a] - P[b]))
        iod = d(33, 263) + 1e-6                              # outer eye corners
        ear_r = d(159, 145) / (d(33, 133) + 1e-6)            # right eye (image)
        ear_l = d(386, 374) / (d(362, 263) + 1e-6)           # left eye
        ear_asym = ear_r - ear_l
        brow_r = d(105, 159) / iod                           # brow→eye gap, right
        brow_l = d(334, 386) / iod                           # left
        brow_asym = brow_r - brow_l
        mouthcorner_asym = float(P[61, 1] - P[291, 1]) / iod  # corner vertical droop L/R
        return np.array([ear_r, ear_l, ear_asym, brow_asym, mouthcorner_asym], dtype=np.float32)

    def _frame_features(self, bgr: np.ndarray) -> np.ndarray | None:
        img = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
        )
        res = self._landmarker.detect(img)
        if not res.face_blendshapes:
            return None
        cats = res.face_blendshapes[0]
        if self._bs_names is None:
            self._init_layout([c.category_name for c in cats])
        scores = np.array([c.score for c in cats], dtype=np.float32)
        asym = np.array([scores[l] - scores[r] for l, r in self._pairs], dtype=np.float32)
        parts = [scores, asym]
        if self.with_geometry:
            if not res.face_landmarks:
                return None
            parts.append(self._geometry(res.face_landmarks[0]))
        return np.concatenate(parts)

    def extract_sequence(self, frames_bgr: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Returns (seq (T, F) float32, mask (T,) bool). Frames with no detected
        face become zero rows with mask=False."""
        feats: list[np.ndarray] = []
        present: list[bool] = []
        for fr in frames_bgr:
            v = self._frame_features(fr)
            feats.append(v)
            present.append(v is not None)
        if not any(present):
            return np.zeros((len(frames_bgr), 1), dtype=np.float32), np.zeros(len(frames_bgr), bool)
        F = self.feat_dim
        seq = np.zeros((len(feats), F), dtype=np.float32)
        for i, v in enumerate(feats):
            if v is not None:
                seq[i] = v
        return seq, np.array(present, dtype=bool)


def extract_action_bundle(
    clip_path: str | Path,
    marlin_encoder,
    mp_extractor: MediaPipeFeatureExtractor,
    n_marlin_windows: int = 1,
    max_mp_frames: int = 60,
    landmarker=None,
    normalizer=None,
) -> dict | None:
    """One clip → {marlin (W,768), mp_seq (T,F), mp_mask (T,)} or None if unusable.

    `normalizer` (optional `QualityNormalizer`) is applied to every MARLIN face
    crop to close the train/test resolution gap (deterministic here — caching is
    inference-time, so no augmentation). None = legacy passthrough."""
    marlin = marlin_encoder.encode_video_path(
        clip_path, n_clips=n_marlin_windows, landmarker=landmarker, normalizer=normalizer)
    frames = _read_frames(clip_path, max_frames=max_mp_frames)
    if not frames:
        return None
    mp_seq, mp_mask = mp_extractor.extract_sequence(frames)
    if marlin is None and not mp_mask.any():
        return None
    return {
        "marlin": (np.zeros((0, marlin_encoder.OUT_DIM), np.float32) if marlin is None else marlin),
        "mp_seq": mp_seq,
        "mp_mask": mp_mask,
    }


def process_dataset(
    data_root: str | Path,
    cache_root: str | Path,
    actions: list[str] | None = None,
    n_marlin_windows: int = 1,
    overwrite: bool = False,
    quality_mode: str = "normalize",
    quality_work_size: int = 112,
) -> None:
    from facial_paralysis.src.models.backbones.marlin_video import MarlinVideoEncoder
    from facial_paralysis.src.datasets.patient_multistream import STANDARD_ACTIONS
    from facial_paralysis.src.preprocessing.image_quality import QualityConfig, QualityNormalizer

    data_root, cache_root = Path(data_root), Path(cache_root)
    actions = actions or list(STANDARD_ACTIONS)
    enc = MarlinVideoEncoder.from_default_weights().eval()
    mp_ext = MediaPipeFeatureExtractor()
    # Quality normalization closes the public(blurry)/iPhone(sharp) resolution gap;
    # deterministic at cache time (no augment). See src/preprocessing/image_quality.py.
    normalizer = QualityNormalizer(
        QualityConfig(mode=quality_mode, work_size=quality_work_size))

    for patient_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        for action in actions:
            clip = patient_dir / f"{action}.mov"
            if not clip.exists():
                clip = patient_dir / f"{action}.mp4"
            if not clip.exists():
                continue
            out = cache_root / patient_dir.name / f"{action}.npz"
            if out.exists() and not overwrite:
                continue
            bundle = extract_action_bundle(clip, enc, mp_ext, n_marlin_windows=n_marlin_windows,
                                           normalizer=normalizer)
            if bundle is None:
                print(f"[skip] {patient_dir.name}/{action}: unusable")
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez(out, marlin=bundle["marlin"], mp_seq=bundle["mp_seq"],
                     mp_mask=bundle["mp_mask"], mp_feat_dim=mp_ext.feat_dim)
            print(f"  {patient_dir.name}/{action}: marlin{bundle['marlin'].shape} "
                  f"mp_seq{bundle['mp_seq'].shape}")
    print(f"\nMediaPipe feat_dim = {mp_ext.feat_dim} "
          f"(set FacialPalsyConfig.mp_feat_dim to this)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--n-marlin-windows", type=int, default=1)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--quality-mode", default="normalize",
                    choices=["normalize", "sr", "off"],
                    help="crop quality normalization (b=normalize default, a=sr ablation)")
    ap.add_argument("--quality-work-size", type=int, default=112,
                    help="canonical effective-resolution cap in px for --quality-mode normalize")
    args = ap.parse_args()
    process_dataset(args.data_root, args.cache_root,
                    n_marlin_windows=args.n_marlin_windows, overwrite=args.overwrite,
                    quality_mode=args.quality_mode, quality_work_size=args.quality_work_size)


if __name__ == "__main__":
    main()
