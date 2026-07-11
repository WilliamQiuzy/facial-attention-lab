"""Auto-curate the collected YouTube palsy videos for #4 (self-supervised pretraining).

Two layers:
  (1) HEURISTIC quality (no key, runs now): per-video face-crop sharpness (Laplacian var),
      face size, and count -> drop videos with too few usable frames (slideshows, tiny/blurry
      faces, non-face content).
  (2) LLM content screen (needs a key): send each video's title to a text LLM ->
      {is_facial_palsy, subject_type (patient/education/other), weak_severity, keep, reason}.
      Works with DeepSeek (DEEPSEEK_API_KEY, base https://api.deepseek.com, model deepseek-chat)
      or any OpenAI-compatible endpoint (OPENAI_API_KEY). Gracefully skips if no key.

Output: youtube/curated.json  (per video: quality metrics + LLM verdict + weak label).
Usage: python youtube_curate.py
"""
from __future__ import annotations
import json, os, glob, statistics
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "youtube"
FACES = OUT / "faces"

# ---- (1) heuristic quality ------------------------------------------------
def crop_quality(path):
    img = cv2.imread(path)
    if img is None:
        return None
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return {"sharp": float(cv2.Laplacian(g, cv2.CV_64F).var()),
            "bright": float(g.mean()), "size": img.shape[0]}


def video_quality(vid, sharp_thr=60.0):
    crops = glob.glob(str(FACES / f"{vid}_*.jpg"))
    qs = [q for q in (crop_quality(c) for c in crops) if q]
    if not qs:
        return {"n_crops": 0, "n_sharp": 0, "usable": False}
    n_sharp = sum(1 for q in qs if q["sharp"] > sharp_thr and 40 < q["bright"] < 230)
    return {"n_crops": len(qs), "n_sharp": n_sharp,
            "med_sharp": round(statistics.median(q["sharp"] for q in qs), 1),
            "usable": n_sharp >= 8}                       # need >=8 sharp frames to be worth it


# ---- (2) LLM content screen (pluggable) -----------------------------------
def llm_client():
    """Return (base_url, key, model) for a text LLM, or None if no key configured."""
    if os.environ.get("DEEPSEEK_API_KEY"):
        return "https://api.deepseek.com/chat/completions", os.environ["DEEPSEEK_API_KEY"], "deepseek-chat"
    if os.environ.get("OPENAI_API_KEY"):
        return "https://api.openai.com/v1/chat/completions", os.environ["OPENAI_API_KEY"], "gpt-4o-mini"
    return None


PROMPT = (
    "You are screening YouTube videos for a facial-palsy VIDEO dataset (for computer-vision "
    "research on facial movement). For each numbered title, decide if the video likely shows a "
    "REAL human face with facial palsy / facial nerve weakness doing facial movements (patient "
    "footage or clinical/therapy demo), NOT lectures/slides/animation/unrelated. Return a JSON "
    "array; one object per title with fields: idx (int), is_facial_palsy (bool), "
    "subject (\"patient\"|\"education\"|\"other\"), weak_severity (\"mild\"|\"moderate\"|"
    "\"severe\"|\"unknown\"), keep (bool), reason (short). Titles:\n"
)


def llm_screen(titles):
    import urllib.request
    cli = llm_client()
    if cli is None:
        return None
    url, key, model = cli
    body = json.dumps({
        "model": model, "temperature": 0,
        "messages": [{"role": "user",
                      "content": PROMPT + "\n".join(f"{i}. {t}" for i, t in enumerate(titles))}],
    }).encode()
    req = urllib.request.Request(url, body, {"Authorization": f"Bearer {key}",
                                             "Content-Type": "application/json"})
    txt = json.loads(urllib.request.urlopen(req, timeout=120).read())["choices"][0]["message"]["content"]
    txt = txt.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    arr = json.loads(txt)
    return {o["idx"]: o for o in arr}


def main():
    man = json.loads((OUT / "manifest.json").read_text())
    print(f"curating {len(man)} videos ...")
    for m in man:
        m["quality"] = video_quality(m["id"])
    verdicts = None
    try:
        verdicts = llm_screen([m.get("title", "") for m in man])
        print(f"LLM screen: {len(verdicts)} verdicts")
    except Exception as e:  # noqa: BLE001
        print(f"LLM screen skipped ({type(e).__name__}: {str(e)[:80]}) -- set DEEPSEEK_API_KEY or OPENAI_API_KEY")

    keep = []
    for i, m in enumerate(man):
        v = (verdicts or {}).get(i)
        m["llm"] = v
        # keep = enough sharp frames AND (LLM says keep, or LLM unavailable)
        m["keep"] = bool(m["quality"]["usable"] and (v is None or v.get("keep", True)))
        if m["keep"]:
            keep.append(m)
    (OUT / "curated.json").write_text(json.dumps(man, indent=1))
    tot_sharp = sum(m["quality"]["n_sharp"] for m in keep)
    print(f"\nKEPT {len(keep)}/{len(man)} videos ({tot_sharp} sharp face frames) -> youtube/curated.json")
    if verdicts:
        from collections import Counter
        print("weak severity (kept):", Counter(m["llm"].get("weak_severity") for m in keep if m.get("llm")))


if __name__ == "__main__":
    main()
