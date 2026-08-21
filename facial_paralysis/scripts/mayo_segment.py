"""Surface the per-action structure of the audio-prompted Mayo recordings.

The design assumes the capture app guides the patient with audio prompts through
the fixed HB action sequence, so the recording can be segmented into one clip per
action in preprocessing. This script tests how cleanly that segmentation can be
recovered *without the protocol transcript*:

  - Audio prompt onsets: short-time speech-energy bursts (the spoken prompts).
  - Face motion energy: per-frame abs-diff inside the Haar face box (the patient
    performing each action — small vs head motion, so face-localized).

For each usable take it saves a stacked timeline PNG (audio energy + detected
prompts + face motion) and a JSON of CANDIDATE action windows (one per prompt
gap). These are the artifacts a human can validate/label against the real
protocol. We deliberately do NOT claim action identities — see the honest finding
that prompt counts (16-31) exceed the ~5-7 canonical actions.

Run:  python3 scripts/mayo_segment.py            (all usable takes)
      python3 scripts/mayo_segment.py <take>     (one take)
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LLF = ROOT / "data" / "livelinkface_data"
OUTDIR = ROOT / "outputs" / "mayo_segments"
_HAAR = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def audio_envelope(mov: Path, sr: int = 16000, win_s: float = 0.05):
    """Return (times, normalized speech-energy envelope) or (None, None)."""
    wav = tempfile.mktemp(suffix=".wav")
    r = subprocess.run(["ffmpeg", "-y", "-i", str(mov), "-ac", "1", "-ar", str(sr),
                        wav, "-loglevel", "error"], capture_output=True)
    if r.returncode != 0 or not Path(wav).exists():
        return None, None
    w = wave.open(wav, "rb")
    a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    w.close(); Path(wav).unlink(missing_ok=True)
    win = int(win_s * sr)
    env = np.array([np.sqrt((a[i:i + win] ** 2).mean() + 1e-12)
                    for i in range(0, len(a) - win, win)])
    t = np.arange(len(env)) * win_s
    e = (env - env.min()) / (env.max() - env.min() + 1e-9)
    return t, e


def detect_prompts(t, e, thr=0.12, min_dur=0.2, merge_gap=0.6):
    """Contiguous above-threshold speech bursts -> list of (start, end) seconds."""
    active = e > thr
    bursts, i = [], 0
    dt = t[1] - t[0] if len(t) > 1 else 0.05
    while i < len(active):
        if active[i]:
            j = i
            while j < len(active) and active[j]:
                j += 1
            if (j - i) * dt > min_dur:
                bursts.append([t[i], t[min(j, len(t) - 1)]])
            i = j
        else:
            i += 1
    merged = []
    for b in bursts:
        if merged and b[0] - merged[-1][1] < merge_gap:
            merged[-1][1] = b[1]
        else:
            merged.append(b)
    return merged


def face_motion(mov: Path, step: int = 3):
    """Per-(sampled)frame motion energy inside the median Haar face box."""
    cap = cv2.VideoCapture(str(mov))
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    # locate face from a few early frames
    boxes = []
    for _ in range(30):
        ok, fr = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        f = _HAAR.detectMultiScale(g, 1.1, 5, minSize=(120, 120))
        if len(f):
            boxes.append(max(f, key=lambda b: b[2] * b[3]))
    box = np.median(np.array(boxes), axis=0).astype(int) if boxes else None
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    prev, energy, times, i = None, [], [], 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % step == 0:
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            if box is not None:
                x, y, w, h = box
                g = g[max(0, y):y + h, max(0, x):x + w]
            g = cv2.resize(g, (96, 96))
            if prev is not None:
                energy.append(float(np.abs(g.astype(np.int16) - prev).mean()))
                times.append(i / fps)
            prev = g
        i += 1
    cap.release()
    e = np.array(energy)
    if e.size:
        e = (e - e.min()) / (e.max() - e.min() + 1e-9)
    return np.array(times), e


def candidate_windows(prompts, dur, lead=0.5, max_len=4.0):
    """One window per prompt: from prompt end+lead to next prompt (capped)."""
    wins = []
    for k, (s, en) in enumerate(prompts):
        nxt = prompts[k + 1][0] if k + 1 < len(prompts) else dur
        start = en + lead
        end = min(start + max_len, nxt)
        if end - start > 0.5:
            wins.append([round(start, 2), round(end, 2)])
    return wins


def process(take: str) -> dict | None:
    movs = list((LLF / take).glob("*.mov"))
    if not movs:
        return None
    mov = movs[0]
    cap = cv2.VideoCapture(str(mov)); dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 60); cap.release()
    ta, ea = audio_envelope(mov)
    prompts = detect_prompts(ta, ea) if ta is not None else []
    tm, em = face_motion(mov)
    wins = candidate_windows(prompts, dur)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    if ta is not None:
        ax[0].plot(ta, ea, lw=0.6, color="tab:blue"); ax[0].set_ylabel("audio energy")
        for s, e in prompts:
            ax[0].axvspan(s, e, color="red", alpha=0.4)
        ax[0].set_title(f"{take} — {len(prompts)} prompts, dur {dur:.0f}s")
    else:
        ax[0].set_title(f"{take} — NO AUDIO")
    ax[1].plot(tm, em, lw=0.6, color="tab:green"); ax[1].set_ylabel("face motion"); ax[1].set_xlabel("seconds")
    for s, e in wins:
        ax[1].axvspan(s, e, color="orange", alpha=0.25)
    fig.tight_layout(); fig.savefig(OUTDIR / f"{take}.png", dpi=80); plt.close(fig)

    return {"take": take, "duration": round(dur, 1), "has_audio": ta is not None,
            "n_prompts": len(prompts), "n_candidate_windows": len(wins),
            "prompts_s": [[round(s, 1), round(e, 1)] for s, e in prompts],
            "candidate_windows_s": wins}


def main():
    takes = sys.argv[1:] or [p.name for p in sorted(LLF.iterdir())
                             if p.is_dir() and list(p.glob("*.mov"))]
    results = []
    for t in takes:
        r = process(t)
        if r is None:
            print(f"  [skip] {t}: no video"); continue
        results.append(r)
        print(f"  {t:<26s} dur={r['duration']:5.0f}s  prompts={r['n_prompts']:2d}  "
              f"windows={r['n_candidate_windows']:2d}  audio={'Y' if r['has_audio'] else '-'}")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "segments.json").write_text(json.dumps(results, indent=2))
    counts = [r["n_prompts"] for r in results if r["has_audio"]]
    if counts:
        print(f"\nprompt counts (audio takes): min={min(counts)} median={int(np.median(counts))} max={max(counts)}")
    print(f"timelines + segments.json in {OUTDIR}")


if __name__ == "__main__":
    main()
