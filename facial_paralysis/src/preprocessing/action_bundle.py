"""Preprocessing: per-action clip → cached bundle (.npz) for the MARLIN pipeline.

For each segmented ~3 s action clip this produces the two streams of
docs/model_design.md §3, written to `<cache>/<patient_id>/<action>.npz`:
    marlin   (W, 768)  — frozen MARLIN clip embedding(s) over W windows
    mp_seq   (T, F)    — MediaPipe per-frame features: 52 blendshapes + L/R
                         asymmetry deltas, optionally plus clinical landmarks
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

from .clinical_landmarks import (
    CLINICAL_LANDMARK_NAMES,
    CLINICAL_SIDE_CONVENTION,
    clinical_landmark_features,
)

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

    # Legacy geometry names (appended when with_geometry=True). EAR = eye aspect
    # ratio (closure); the rest are scale-normalized L/R region asymmetries.
    GEOM_NAMES = ["ear_right", "ear_left", "ear_asym", "brow_asym", "mouthcorner_asym"]

    LANDMARK_FEATURE_MODES = ("none", "legacy5", "clinical23")

    def __init__(
        self,
        model_path: str | Path | None = None,
        with_geometry: bool | None = None,
        landmark_features: str = "none",
        capture_mirrored: bool | None = None,
    ):
        import mediapipe as mp
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision as mpv

        model_path = Path(model_path) if model_path else _DEFAULT_MP_MODEL
        if not model_path.exists():
            raise FileNotFoundError(f"MediaPipe model not found at {model_path}")
        if with_geometry:
            if landmark_features != "none":
                raise ValueError("with_geometry=True conflicts with landmark_features")
            landmark_features = "legacy5"
        self._mp = mp
        self.landmark_features = self._validate_landmark_features(landmark_features)
        self.capture_mirrored = capture_mirrored
        # Backward-compatible public attribute; new code should inspect the mode.
        self.with_geometry = self.landmark_features != "none"
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

    @classmethod
    def _validate_landmark_features(cls, mode: str) -> str:
        if mode not in cls.LANDMARK_FEATURE_MODES:
            raise ValueError(
                f"landmark_features must be one of {cls.LANDMARK_FEATURE_MODES}, got {mode!r}"
            )
        return mode

    @property
    def _landmark_names(self) -> list[str]:
        mode = self._validate_landmark_features(self.landmark_features)
        if mode == "none":
            return []
        if mode == "legacy5":
            return list(self.GEOM_NAMES)
        return list(CLINICAL_LANDMARK_NAMES)

    def _init_layout(self, names: list[str]) -> None:
        # A schema id is an exact ordered-column contract.  MediaPipe normally
        # emits this fixed 52-category layout, but accepting an arbitrary first
        # frame here would let the producer mint a self-contradictory cache.
        # Import lazily because the dataset registry imports this package's
        # clinical-landmark definitions during module initialization.
        from ..datasets.patient_multistream import MP_FEATURE_NAMES_BY_SCHEMA

        expected = list(MP_FEATURE_NAMES_BY_SCHEMA["mediapipe_bs_lr_v1"][:52])
        if names != expected:
            raise RuntimeError(
                "MediaPipe blendshape category layout does not match the "
                "registered mediapipe_bs_lr_v1 schema"
            )
        self._bs_names = names
        idx = {n: i for i, n in enumerate(names)}
        pairs = []
        for n, i in idx.items():
            if n.endswith("Left"):
                r = n[:-4] + "Right"
                if r in idx:
                    pairs.append((i, idx[r]))
        self._pairs = pairs

    def _ensure_layout(self, names: list[str]) -> None:
        if self._bs_names is None:
            self._init_layout(names)
            return
        if names != self._bs_names:
            raise RuntimeError(
                "MediaPipe blendshape category order changed within one extractor"
            )

    @property
    def feat_dim(self) -> int:
        if self._bs_names is None:
            raise RuntimeError("feat_dim unknown until the first frame is processed")
        return len(self._bs_names) + len(self._pairs) + len(self._landmark_names)

    @property
    def feature_names(self) -> list[str]:
        base = list(self._bs_names)
        asym = [f"delta_left_minus_right_{self._bs_names[l][:-4]}"
                for l, _ in self._pairs]
        return base + asym + self._landmark_names

    @property
    def feature_schema(self) -> str:
        mode = self._validate_landmark_features(self.landmark_features)
        suffix = {"none": "", "legacy5": "+legacy_geometry5_v1",
                  "clinical23": "+clinical23_v2"}[mode]
        return "mediapipe_bs_lr_v1" + suffix

    @property
    def side_convention(self) -> str:
        if self.landmark_features == "clinical23":
            return CLINICAL_SIDE_CONVENTION
        if self.landmark_features == "legacy5":
            return "mediapipe_labels_plus_legacy_mesh_topology_capture_mirror_required"
        return "mediapipe_left_right_labels_capture_mirror_required"

    @staticmethod
    def _legacy_geometry(lms) -> np.ndarray:
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

    # Historical private name retained for callers that used it directly.
    _geometry = _legacy_geometry

    def _assemble_features(
        self,
        scores: np.ndarray,
        landmarks,
        image_width: int,
        image_height: int,
    ) -> np.ndarray:
        """Combine blendshapes, mirrored deltas, and the selected landmark block."""
        if self._bs_names is None or self._pairs is None:
            raise RuntimeError("feature layout has not been initialized")
        scores = np.asarray(scores, dtype=np.float32)
        if scores.shape != (len(self._bs_names),):
            raise ValueError(
                f"blendshape scores must have shape ({len(self._bs_names)},), got {scores.shape}"
            )
        asym = np.asarray([scores[left] - scores[right] for left, right in self._pairs],
                          dtype=np.float32)
        parts = [scores, asym]
        mode = self._validate_landmark_features(self.landmark_features)
        if mode != "none":
            if landmarks is None:
                raise ValueError(f"landmarks are required for landmark_features={mode!r}")
            if mode == "legacy5":
                parts.append(self._legacy_geometry(landmarks))
            else:
                parts.append(clinical_landmark_features(
                    landmarks, image_width=image_width, image_height=image_height))
        return np.concatenate(parts).astype(np.float32, copy=False)

    def _frame_features(self, bgr: np.ndarray) -> np.ndarray | None:
        img = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
        )
        res = self._landmarker.detect(img)
        if not res.face_blendshapes:
            return None
        cats = res.face_blendshapes[0]
        self._ensure_layout([c.category_name for c in cats])
        scores = np.array([c.score for c in cats], dtype=np.float32)
        landmarks = res.face_landmarks[0] if res.face_landmarks else None
        if self.landmark_features != "none" and landmarks is None:
            return None
        try:
            return self._assemble_features(
                scores, landmarks, image_width=bgr.shape[1], image_height=bgr.shape[0])
        except ValueError:
            # A malformed landmark frame is a detector miss, not a neutral face.
            return None

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
            # If a blendshape frame initialized the layout before a malformed
            # landmark caused the miss, retain the declared width.  If no face
            # ever initialized the layout, width zero explicitly means the MP
            # stream is absent; a fake one-column stream is not a valid schema.
            width = self.feat_dim if self._bs_names is not None else 0
            return (np.zeros((len(frames_bgr), width), dtype=np.float32),
                    np.zeros(len(frames_bgr), bool))
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


def _assert_existing_cache_schema(
    path: str | Path,
    expected_schema: str,
    expected_side_convention: str | None = None,
    expected_capture_mirrored: str | None = None,
) -> None:
    """Fail closed instead of silently mixing an old cache with a new layout."""
    path = Path(path)
    try:
        with np.load(path, allow_pickle=False) as cached:
            fields = set(cached.files)
            marlin_key = "marlin" if "marlin" in fields else (
                "embeddings" if "embeddings" in fields else None
            )
            if marlin_key is not None:
                marlin = np.asarray(cached[marlin_key])
                if not np.issubdtype(marlin.dtype, np.number) or not np.isfinite(marlin).all():
                    raise RuntimeError(
                        f"existing cache {path} has non-finite {marlin_key} values"
                    )
            if "mp_seq" not in fields:
                if not ({"marlin", "embeddings"} & fields):
                    raise RuntimeError(f"existing cache {path} has no usable stream")
                if fields & {
                    "mp_feature_schema", "mp_feature_names", "mp_feat_dim",
                    "mp_side_convention", "mp_capture_mirrored", "mp_mask",
                }:
                    raise RuntimeError(
                        f"existing MARLIN-only cache {path} has partial MediaPipe metadata"
                    )
                return
            required = {
                "mp_mask", "mp_feat_dim", "mp_feature_schema", "mp_feature_names",
                "mp_side_convention", "mp_capture_mirrored",
            }
            missing = required - fields
            if missing:
                raise RuntimeError(
                    f"existing cache {path} is missing {sorted(missing)}; "
                    "rerun with --overwrite"
                )
            observed = str(np.asarray(cached["mp_feature_schema"]).item())
            seq = np.asarray(cached["mp_seq"])
            mask = np.asarray(cached["mp_mask"])
            stored_dim = int(np.asarray(cached["mp_feat_dim"]).item())
            stored_names = tuple(str(x) for x in np.asarray(
                cached["mp_feature_names"]).tolist())
            stored_side = str(np.asarray(cached["mp_side_convention"]).item())
            stored_mirror = str(np.asarray(cached["mp_capture_mirrored"]).item())
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"cannot validate existing cache schema at {path}") from exc
    if observed != expected_schema:
        raise RuntimeError(
            f"existing cache {path} uses {observed!r}, requested {expected_schema!r}; "
            "rerun with --overwrite"
        )
    # Lazy import avoids a module-initialization cycle: the dataset registry
    # itself imports this extractor's clinical landmark contract.
    from ..datasets.patient_multistream import (
        MP_FEATURE_NAMES_BY_SCHEMA,
        MP_SIDE_CONVENTION_BY_SCHEMA,
    )
    expected_names = MP_FEATURE_NAMES_BY_SCHEMA.get(expected_schema)
    if expected_names is None:
        raise RuntimeError(f"unknown requested MediaPipe schema {expected_schema!r}")
    expected_side = (expected_side_convention
                     or MP_SIDE_CONVENTION_BY_SCHEMA[expected_schema])
    if seq.ndim != 2 or mask.shape != (seq.shape[0],):
        raise RuntimeError(
            f"existing cache {path} has invalid seq/mask shapes {seq.shape}/{mask.shape}"
        )
    if mask.dtype != np.bool_:
        raise RuntimeError(f"existing cache {path} has a non-boolean MediaPipe mask")
    if not np.issubdtype(seq.dtype, np.number) or not np.isfinite(seq).all():
        raise RuntimeError(f"existing cache {path} has non-finite MediaPipe values")
    if stored_dim != seq.shape[1] or stored_dim != len(expected_names):
        raise RuntimeError(
            f"existing cache {path} has inconsistent MediaPipe dimension {stored_dim}"
        )
    if stored_names != expected_names:
        raise RuntimeError(f"existing cache {path} has wrong ordered feature names")
    if stored_side != expected_side:
        raise RuntimeError(f"existing cache {path} has wrong side convention")
    if stored_mirror not in {"true", "false", "unknown"}:
        raise RuntimeError(f"existing cache {path} has invalid capture-mirror provenance")
    if (expected_capture_mirrored is not None
            and stored_mirror != expected_capture_mirrored):
        raise RuntimeError(
            f"existing cache {path} capture mirror {stored_mirror!r} != requested "
            f"{expected_capture_mirrored!r}; rerun with --overwrite"
        )


def _bundle_npz_payload(bundle: dict, mp_extractor: MediaPipeFeatureExtractor) -> dict:
    """Validate the stream contract and construct an object-free NPZ payload."""
    marlin = np.asarray(bundle["marlin"], dtype=np.float32)
    if not np.isfinite(marlin).all():
        raise ValueError("MARLIN embeddings must be finite")
    payload = {"marlin": marlin}
    seq = np.asarray(bundle["mp_seq"], dtype=np.float32).copy()
    raw_mask = np.asarray(bundle["mp_mask"])
    if raw_mask.dtype != np.bool_:
        raise ValueError(f"MediaPipe mask must be boolean, got {raw_mask.dtype}")
    mask = raw_mask.astype(bool, copy=False)
    if seq.ndim != 2 or mask.shape != (seq.shape[0],):
        raise ValueError(
            f"MediaPipe sequence/mask mismatch: seq={seq.shape}, mask={mask.shape}"
        )
    if seq.shape[1] == 0:
        if mask.any():
            raise ValueError("zero-width MediaPipe stream cannot contain valid frames")
        return payload
    expected_dim = mp_extractor.feat_dim
    if seq.shape[1] != expected_dim:
        raise ValueError(
            f"MediaPipe sequence has {seq.shape[1]} columns but schema declares {expected_dim}"
        )
    from ..datasets.patient_multistream import (
        MP_FEATURE_NAMES_BY_SCHEMA,
        MP_SIDE_CONVENTION_BY_SCHEMA,
    )
    schema = mp_extractor.feature_schema
    expected_names = MP_FEATURE_NAMES_BY_SCHEMA.get(schema)
    if expected_names is None:
        raise ValueError(f"unregistered MediaPipe feature schema {schema!r}")
    if tuple(mp_extractor.feature_names) != expected_names:
        raise ValueError(f"feature names do not match registered schema {schema!r}")
    if mp_extractor.side_convention != MP_SIDE_CONVENTION_BY_SCHEMA[schema]:
        raise ValueError(f"side convention does not match registered schema {schema!r}")
    if not np.isfinite(seq[mask]).all():
        raise ValueError("valid MediaPipe frames must contain only finite values")
    # Action caches use finite zero padding.  Audit trajectories may retain NaN
    # for human inspection, but those rows must be canonicalized at this model
    # input boundary so NaN * 0 can never poison a recurrent encoder.
    seq[~mask] = 0.0
    capture_mirrored = getattr(mp_extractor, "capture_mirrored", None)
    payload.update({
        "mp_seq": seq,
        "mp_mask": mask,
        "mp_feat_dim": np.asarray(expected_dim, np.int32),
        "mp_feature_schema": np.asarray(schema),
        "mp_feature_names": np.asarray(mp_extractor.feature_names),
        "mp_side_convention": np.asarray(mp_extractor.side_convention),
        "mp_capture_mirrored": np.asarray(
            "unknown" if capture_mirrored is None else str(bool(capture_mirrored)).lower()
        ),
    })
    return payload


def process_dataset(
    data_root: str | Path,
    cache_root: str | Path,
    actions: list[str] | None = None,
    n_marlin_windows: int = 1,
    overwrite: bool = False,
    quality_mode: str = "normalize",
    quality_work_size: int = 112,
    landmark_features: str = "none",
    capture_mirrored: bool | None = None,
) -> None:
    from facial_paralysis.src.models.backbones.marlin_video import MarlinVideoEncoder
    from facial_paralysis.src.datasets.patient_multistream import STANDARD_ACTIONS
    from facial_paralysis.src.preprocessing.image_quality import QualityConfig, QualityNormalizer

    data_root, cache_root = Path(data_root), Path(cache_root)
    actions = actions or list(STANDARD_ACTIONS)
    enc = MarlinVideoEncoder.from_default_weights().eval()
    mp_ext = MediaPipeFeatureExtractor(
        landmark_features=landmark_features, capture_mirrored=capture_mirrored)
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
                expected_mirror = (
                    "unknown" if capture_mirrored is None
                    else str(bool(capture_mirrored)).lower()
                )
                _assert_existing_cache_schema(
                    out, mp_ext.feature_schema,
                    expected_side_convention=mp_ext.side_convention,
                    expected_capture_mirrored=expected_mirror,
                )
                continue
            bundle = extract_action_bundle(clip, enc, mp_ext, n_marlin_windows=n_marlin_windows,
                                           normalizer=normalizer)
            if bundle is None:
                print(f"[skip] {patient_dir.name}/{action}: unusable")
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            payload = _bundle_npz_payload(bundle, mp_ext)
            np.savez(out, **payload)
            print(f"  {patient_dir.name}/{action}: marlin{bundle['marlin'].shape} "
                  f"mp_seq{bundle['mp_seq'].shape}")
    if mp_ext._bs_names is None:
        print("\nMediaPipe stream absent: no clip initialized a feature layout")
    else:
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
    ap.add_argument(
        "--landmark-features",
        default="none",
        choices=MediaPipeFeatureExtractor.LANDMARK_FEATURE_MODES,
        help="append no landmarks, the historical 5-d block, or clinical23",
    )
    ap.add_argument(
        "--capture-mirrored",
        choices=("unknown", "true", "false"),
        default="unknown",
        help="whether input pixels are mirrored; required before patient-side interpretation",
    )
    args = ap.parse_args()
    process_dataset(args.data_root, args.cache_root,
                    n_marlin_windows=args.n_marlin_windows, overwrite=args.overwrite,
                    quality_mode=args.quality_mode, quality_work_size=args.quality_work_size,
                    landmark_features=args.landmark_features,
                    capture_mirrored=(None if args.capture_mirrored == "unknown"
                                      else args.capture_mirrored == "true"))


if __name__ == "__main__":
    main()
