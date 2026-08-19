"""Video → MediaPipe outputs → per-take embeddings preprocessing pipeline.

Two stages:
  Stage 1 (mediapipe_extract): for each .mov in the input tree, run MediaPipe
    FaceLandmarker in VIDEO mode and write landmarks.csv / blendshapes.csv /
    blendshapes_wide.csv / transform_matrices.npy / <stem>_landmarks.mp4 +
    a done.json resume marker.
  Stage 2 (peak_embeddings): pick representative frames per take using
    blendshape activity peaks, run the Oo MLP-Mixer encoder on each, and
    save (n_frames, 768) embeddings + metadata per take.
"""
import importlib

# Public name -> submodule that defines it. Imported lazily (PEP 562) so that
# importing a mediapipe-free sibling (e.g. `image_quality`) does NOT pull in the
# heavy `mediapipe` dependency that `mediapipe_extract` imports at module load.
_LAZY = {
    "discover_videos": ".mediapipe_extract",
    "process_video": ".mediapipe_extract",
    "run_batch": ".mediapipe_extract",
    "Stage1Config": ".mediapipe_extract",
    "extract_keyframe_embeddings": ".peak_embeddings",
    "select_keyframes": ".peak_embeddings",
    "Stage2Config": ".peak_embeddings",
}


def __getattr__(name: str):
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(mod, __name__), name)


def __dir__():
    return sorted([*globals(), *_LAZY])


__all__ = [
    "discover_videos",
    "process_video",
    "run_batch",
    "Stage1Config",
    "extract_keyframe_embeddings",
    "select_keyframes",
    "Stage2Config",
]
