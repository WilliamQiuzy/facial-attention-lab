"""Characterize every Mayo LiveLinkFace take: what data each contains and which
takes are usable for the video pipeline.

The LiveLinkFace export is RAW capture (RGB .mov + depth .bin + timing log +
metadata) — NO ARKit blendshape CSV and NO HB labels. Not every take has video
or audio. This builds an honest manifest so downstream steps know exactly what
they can use.

Run:  python3 scripts/mayo_manifest.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LLF = ROOT / "data" / "livelinkface_data"
OUT = ROOT / "outputs" / "mayo_manifest.json"


def ffprobe(path: Path) -> dict:
    """Return {duration, fps, n_frames, has_audio, width, height} for a video."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_streams", "-show_format",
             "-of", "json", str(path)],
            capture_output=True, text=True, timeout=60,
        )
        meta = json.loads(r.stdout)
    except Exception as e:
        return {"error": str(e)}
    streams = meta.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    out = {"has_audio": has_audio,
           "duration": round(float(meta.get("format", {}).get("duration", 0)), 1)}
    if v:
        num, den = (v.get("avg_frame_rate", "0/1").split("/") + ["1"])[:2]
        out["fps"] = round(float(num) / float(den), 1) if float(den) else None
        out["n_frames"] = int(v.get("nb_frames", 0) or 0)
        out["width"], out["height"] = v.get("width"), v.get("height")
    return out


def main():
    takes = sorted(p for p in LLF.iterdir() if p.is_dir())
    manifest = []
    for t in takes:
        movs = list(t.glob("*.mov"))
        depth = t / "depth_data.bin"
        rec = {
            "take": t.name,
            "has_video": bool(movs),
            "has_depth": depth.exists(),
            "depth_mb": round(depth.stat().st_size / 1e6, 1) if depth.exists() else 0,
        }
        tj = t / "take.json"
        if tj.exists():
            try:
                j = json.loads(tj.read_text())
                rec["declared_frames"] = j.get("frames")
                rec["date"] = j.get("date")
                rec["device"] = j.get("deviceModel")
            except Exception:
                pass
        if movs:
            rec.update(ffprobe(movs[0]))
        manifest.append(rec)

    # usability verdict
    for r in manifest:
        usable = (r.get("has_video") and r.get("duration", 0) > 20
                  and r.get("n_frames", 0) > 500)
        r["usable_video"] = bool(usable)

    n_usable = sum(r["usable_video"] for r in manifest)
    OUT.write_text(json.dumps({"n_takes": len(manifest), "n_usable_video": n_usable,
                               "takes": manifest}, indent=2))

    print(f"{'take':<26s} {'vid':>3s} {'dur':>5s} {'frames':>6s} {'fps':>4s} "
          f"{'aud':>3s} {'depth':>6s} {'usable':>6s}")
    print("-" * 72)
    for r in manifest:
        print(f"{r['take']:<26s} {'Y' if r['has_video'] else '-':>3s} "
              f"{r.get('duration', 0):5.0f} {r.get('n_frames', 0):6d} "
              f"{r.get('fps', 0) or 0:4.0f} {'Y' if r.get('has_audio') else '-':>3s} "
              f"{r['depth_mb']:5.0f}M {'YES' if r['usable_video'] else 'no':>6s}")
    print(f"\n{len(manifest)} takes, {n_usable} usable for the video pipeline. wrote {OUT}")


if __name__ == "__main__":
    main()
