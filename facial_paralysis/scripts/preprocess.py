"""Unified preprocessing CLI: video → MediaPipe outputs → 768-d embeddings.

Stages:
  1. mediapipe_extract: walks <input_root>/<take_or_patient>/<*.mov>, runs
     MediaPipe FaceLandmarker in VIDEO mode, writes landmarks.csv /
     blendshapes.csv / blendshapes_wide.csv / transform_matrices.npy /
     annotated mp4 / done.json per slot.
  2. peak_embeddings: walks the slots from stage 1, picks neutral + peak
     frames per take by blendshape activity, encodes each with the Oo
     MLP-Mixer backbone, and writes <embedding_root>/<slot_id>.npz.

Both stages are idempotent (resume by file presence).

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE conda run -n dev python scripts/preprocess.py \\
        --input-root  data/livelinkface_data \\
        --output-root data/mediapipe_out \\
        --embedding-root outputs/embeddings

Defaults match the current project layout, so a bare `python scripts/preprocess.py`
reproduces what the notebook + scripts/extract_video_embeddings.py do today.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing import (  # noqa: E402
    Stage1Config, Stage2Config, run_batch, extract_keyframe_embeddings,
)
from src.preprocessing.peak_embeddings import run_stage2_batch  # noqa: E402
from src.models.backbones import OoMLPMixerEncoder  # noqa: E402
from src.baselines.oo_multimodal.utils import MediaPipeFaceLandmarker  # type: ignore  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-root", type=Path, default=ROOT / "data" / "livelinkface_data",
                   help="root containing <take>/*.mov OR <patient_id>/<action>.mov")
    p.add_argument("--output-root", type=Path, default=ROOT / "data" / "mediapipe_out",
                   help="where to write per-slot MediaPipe outputs")
    p.add_argument("--embedding-root", type=Path, default=ROOT / "outputs" / "embeddings",
                   help="where to write per-slot 768-d embedding .npz files")
    p.add_argument("--skip-stage1", action="store_true",
                   help="skip MediaPipe extraction; only run embedding extraction on existing outputs")
    p.add_argument("--skip-stage2", action="store_true",
                   help="skip embedding extraction; only run MediaPipe stage 1")
    p.add_argument("--force-reprocess", action="store_true",
                   help="re-run stage 1 even if done.json exists (does not affect stage 2 cache)")
    p.add_argument("--no-annotated", action="store_true",
                   help="don't write the annotated mp4 in stage 1 (~1.4x faster)")
    p.add_argument("--num-workers", type=int, default=None,
                   help="stage 1 parallelism (default: min(4, cpu_count-1))")
    p.add_argument("--k-peaks", type=int, default=8,
                   help="stage 2 K when strategy uses peaks")
    p.add_argument("--strategy", default="auto",
                   choices=["auto", "all", "uniform_fps", "peaks", "peaks_plus_uniform"],
                   help="stage 2 frame selection strategy (default 'auto': uniform_fps for "
                        "videos < short_video_threshold_frames, peaks otherwise)")
    p.add_argument("--target-fps", type=float, default=6.0,
                   help="stage 2 target fps for uniform_fps strategy (default matches Oo's 6 fps training)")
    return p.parse_args()


def main():
    args = _parse_args()

    if not args.skip_stage1:
        print("=" * 60)
        print("Stage 1: MediaPipe FaceLandmarker on every input video")
        print("=" * 60)
        cfg1 = Stage1Config(
            input_root=args.input_root,
            output_root=args.output_root,
            write_annotated=not args.no_annotated,
            force_reprocess=args.force_reprocess,
            num_workers=args.num_workers or min(4, max(1, 3)),
        )
        run_batch(cfg1)

    if args.skip_stage2:
        return

    print("\n" + "=" * 60)
    print("Stage 2: peak-frame selection + Oo MLP-Mixer encoding")
    print("=" * 60)

    # Build the (slot_id, mov_path) list from stage 1's outputs (per_video_meta.csv).
    # This avoids re-walking the input tree and inherits stage 1's video discovery.
    import pandas as pd
    meta_csv = args.output_root / "per_video_meta.csv"
    if not meta_csv.exists():
        print(f"no manifest at {meta_csv}; nothing to do for stage 2")
        return
    df = pd.read_csv(meta_csv)
    df = df[df["error"].isna()] if "error" in df.columns else df
    items: list[tuple[str, Path]] = []
    # The notebook stored video_path relative to its cwd (data/). New runs use
    # absolute paths. We probe candidates in order so both forms work.
    search_roots = [Path("/"), args.input_root.parent, ROOT, ROOT / "data"]
    for _, row in df.iterrows():
        slot_id = str(row["slot_id"])
        raw = Path(row["video_path"])
        resolved: Path | None = None
        if raw.is_absolute() and raw.exists():
            resolved = raw
        else:
            for root in search_roots:
                cand = (root / raw) if not raw.is_absolute() else raw
                if cand.exists():
                    resolved = cand
                    break
        if resolved is None:
            print(f"[warn] mov referenced in manifest not found: {raw}")
            continue
        items.append((slot_id, resolved))

    print(f"\n{len(items)} slots have stage 1 outputs and an on-disk .mov\n")

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"device: {device}")

    encoder = OoMLPMixerEncoder.from_default_weights().to(device).eval()
    landmarker = MediaPipeFaceLandmarker()
    cfg2 = Stage2Config(
        mediapipe_root=args.output_root,
        embedding_root=args.embedding_root,
        strategy=args.strategy,
        k_peaks=args.k_peaks,
        target_fps=args.target_fps,
    )
    run_stage2_batch(items, encoder, landmarker, cfg2)


if __name__ == "__main__":
    main()
