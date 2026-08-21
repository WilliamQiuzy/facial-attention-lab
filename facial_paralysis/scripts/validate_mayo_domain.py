"""#2 — End-to-end face-validity / non-collapse check on the Mayo iPhone domain.

The MARLIN feasibility gate showed the frozen ENCODER does not collapse on Mayo
iPhone takes (centered identity margin +0.806). This checks the next link: does
the *full trained model* (encoder -> trunk -> severity head) also stay
non-collapsed and produce a sensible spread on real iPhone data?

We have no Mayo HB labels yet, so this is NOT a metric run. We:
  1. Encode the 15 local Mayo takes end-to-end (MARLIN windows + MediaPipe seq).
  2. Train the binary palsy model on ALL of PalsyNet (released public data).
  3. Score P(palsy) and the latent severity `s` for each take, and check:
     - NON-COLLAPSE: the scores are spread out, not all ~equal (the Oo failure was
       "always palsy, p=0.913" for everyone, incl. a healthy portrait).
     - FACE VALIDITY: ranked list of takes by P(palsy) for eyeballing. NOTE: take
       folder names (*FACES* vs *MySlate*) are NOT reliable palsy/healthy labels
       (see memory project_data_duplicates), so this is a sanity ranking only.

Consistency note: the existing PalsyNet bundles are UN-normalized (predate the
quality normalizer), so we encode Mayo un-normalized too — train and test must
share preprocessing. A normalized re-run is a follow-up.

Run:
  KMP_DUPLICATE_LIB_OK=TRUE \
  /opt/homebrew/Caskroom/miniforge/base/envs/dev/bin/python scripts/validate_mayo_domain.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MAYO = ROOT / "data" / "livelinkface_data"
PALSY_CACHE = ROOT / "outputs" / "palsynet_bundles"
MAYO_CACHE = ROOT / "outputs" / "mayo_bundles"
ACTION = "clip"
MP_FEAT_DIM = 72


def encode_mayo(enc, mp_ext, n_clips=4, max_mp=60, reextract=False) -> list[str]:
    """One bundle per take (.mov). Un-normalized to match PalsyNet bundles."""
    from facial_paralysis.src.preprocessing.action_bundle import (
        _assert_existing_cache_schema,
        _bundle_npz_payload,
        _read_frames,
    )
    MAYO_CACHE.mkdir(parents=True, exist_ok=True)
    ok = []
    for vp in sorted(MAYO.glob("*/*.mov")):
        take = vp.parent.name
        out = MAYO_CACHE / take / f"{ACTION}.npz"
        if out.exists() and not reextract:
            _assert_existing_cache_schema(
                out, mp_ext.feature_schema,
                expected_side_convention=mp_ext.side_convention,
                expected_capture_mirrored="unknown",
            )
            ok.append(take); continue
        marlin = enc.encode_video_path(vp, n_clips=n_clips)
        frames = _read_frames(vp, max_frames=max_mp)
        seq, mask = mp_ext.extract_sequence(frames) if frames else (None, None)
        if marlin is None or seq is None or not mask.any():
            print(f"  [skip] {take}: unusable"); continue
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, **_bundle_npz_payload({
            "marlin": marlin, "mp_seq": seq, "mp_mask": mask,
        }, mp_ext))
        ok.append(take)
        print(f"  {take}: marlin{marlin.shape} mp_seq{seq.shape}")
    return ok


def make_binary_model(seed=0):
    from facial_paralysis.src.models.facial_palsy_model import FacialPalsyModel, FacialPalsyConfig
    from facial_paralysis.src.models.multitask import TaskSpec
    torch.manual_seed(seed)
    return FacialPalsyModel(FacialPalsyConfig(
        mp_feat_dim=MP_FEAT_DIM, n_actions=1, temporal_hidden=64, temporal_out=64,
        trunk_hidden=64, dropout=0.1, tasks=[TaskSpec("binary", 2, coupled=True)]))


def load_mayo_records(takes: list[str]):
    from facial_paralysis.src.datasets.patient_multistream import (
        ActionBundle, MultiStreamRecord,
    )
    from facial_paralysis.src.preprocessing.action_bundle import (
        _assert_existing_cache_schema,
    )
    recs = []
    for take in takes:
        path = MAYO_CACHE / take / f"{ACTION}.npz"
        _assert_existing_cache_schema(
            path, "mediapipe_bs_lr_v1", expected_capture_mirrored="unknown"
        )
        with np.load(path, allow_pickle=False) as d:
            marlin = d["marlin"].astype(np.float32)
            mp_seq = d["mp_seq"].astype(np.float32)
            mp_mask = d["mp_mask"]
            schema = str(d["mp_feature_schema"].item())
            names = tuple(str(x) for x in d["mp_feature_names"])
            side = str(d["mp_side_convention"].item())
            mirror = str(d["mp_capture_mirrored"].item())
        recs.append(MultiStreamRecord(
            patient_id=take, label=0, task="binary",
            actions=[ActionBundle(marlin=marlin,
                                  mp_seq=mp_seq,
                                  mp_mask=mp_mask,
                                  mp_feature_schema=schema,
                                  mp_feature_names=names,
                                  mp_side_convention=side,
                                  mp_capture_mirrored=mirror)]))
    return recs


def score_mayo(model, recs) -> list[dict]:
    from facial_paralysis.src.datasets.patient_multistream import MultiStreamPatientDataset, collate_multistream
    from facial_paralysis.src.models.ordinal import cum_probs
    from torch.utils.data import DataLoader
    ds = MultiStreamPatientDataset(recs, actions=[ACTION], mp_feat_dim=MP_FEAT_DIM)
    b = next(iter(DataLoader(ds, batch_size=len(ds), collate_fn=collate_multistream)))
    model.eval()
    with torch.no_grad():
        action_emb = model.build_action_embeddings(b["marlin_emb"], b["marlin_mask"],
                                                   b["mp_seq"], b["mp_mask"])
        h, s = model.multitask.trunk.represent(action_emb, b["action_present"])
        out = model.multitask(action_emb, b["action_present"])
        p = cum_probs(out["binary"])[:, 0].cpu().numpy()
    s = s.cpu().numpy()
    return [{"take": r.patient_id, "p_palsy": float(p[i]), "severity_s": float(s[i])}
            for i, r in enumerate(recs)]


def main():
    from facial_paralysis.src.preprocessing.action_bundle import MediaPipeFeatureExtractor
    from facial_paralysis.src.models.backbones.marlin_video import MarlinVideoEncoder
    from facial_paralysis.src.datasets.patient_multistream import MultiStreamPatientDataset
    from facial_paralysis.src.training.train_multitask import MTTrainConfig, train_multitask

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Encoding Mayo takes end-to-end on {device} (un-normalized)...")
    enc = MarlinVideoEncoder.from_default_weights().to(device).eval()
    mp_ext = MediaPipeFeatureExtractor(capture_mirrored=None)
    takes = encode_mayo(enc, mp_ext)

    print(f"\nTraining binary palsy model on ALL PalsyNet...")
    pals = MultiStreamPatientDataset.from_disk(
        PALSY_CACHE, PALSY_CACHE / "labels.csv", actions=[ACTION],
        mp_feat_dim=MP_FEAT_DIM, mp_feature_schema="mediapipe_bs_lr_v1")
    model = make_binary_model()
    train_multitask(model, pals, None, MTTrainConfig(
        epochs=50, batch_size=8, lr=5e-4, weight_decay=3e-2, device="cpu",
        monitor_task="binary", monitor_n_classes=2, log_every=999, seed=0))

    scored = score_mayo(model, load_mayo_records(takes))
    scored.sort(key=lambda r: r["p_palsy"], reverse=True)
    p = np.array([r["p_palsy"] for r in scored])
    s = np.array([r["severity_s"] for r in scored])

    collapse = {
        "n_takes": len(scored),
        "p_palsy_mean": round(float(p.mean()), 3), "p_palsy_std": round(float(p.std()), 3),
        "p_palsy_min": round(float(p.min()), 3), "p_palsy_max": round(float(p.max()), 3),
        "p_palsy_range": round(float(p.max() - p.min()), 3),
        "severity_std": round(float(s.std()), 3),
        "severity_range": round(float(s.max() - s.min()), 3),
        # non-collapse heuristic: a meaningful spread of predictions across takes
        "non_collapsed": bool(p.std() > 0.05 and (p.max() - p.min()) > 0.2),
    }
    res = {"check": "Mayo iPhone end-to-end face-validity / non-collapse",
           "collapse_summary": collapse, "ranked_takes": scored}
    print("\n================= MAYO DOMAIN VALIDATION =================")
    print(json.dumps(collapse, indent=2))
    print("\nranked by P(palsy):")
    for r in scored:
        print(f"  {r['take']:<28s} P(palsy)={r['p_palsy']:.3f}  s={r['severity_s']:+.3f}")
    verdict = ("NON-COLLAPSED — full model gives a real spread on iPhone domain"
               if collapse["non_collapsed"] else
               "COLLAPSED-LIKE — predictions clustered; investigate (cf. Oo failure)")
    print(f"\nVERDICT: {verdict}")
    print("=========================================================")
    MAYO_CACHE.mkdir(parents=True, exist_ok=True)
    (MAYO_CACHE / "validation.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
