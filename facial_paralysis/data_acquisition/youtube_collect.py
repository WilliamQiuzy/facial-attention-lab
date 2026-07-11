"""Collect public YouTube facial-palsy videos -> face crops + manifest.

Free, no application, and (unlike web stills) VIDEO with real movement dynamics — the
same approach that built PalsyNet. Searches with yt-dlp, downloads a short low-res
segment per hit, extracts faces with cv2 Haar (no mediapipe needed locally), writes a
manifest. Modest budget by default; scale up N_PER_QUERY / SECONDS once verified.

ETHICS/CONSENT NOTE: these are public patient/education videos. For any use beyond
method development, review consent/usage terms and IRB scope. Manifest keeps source
URLs for provenance/takedown.

Usage: python youtube_collect.py [n_per_query] [seconds]
"""
from __future__ import annotations
import json, os, subprocess, sys
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "youtube"
CLIPS = OUT / "clips"; FACES = OUT / "faces"
for d in (CLIPS, FACES):
    d.mkdir(parents=True, exist_ok=True)

QUERIES = [
    "facial palsy exercises", "Bell's palsy face movement", "facial paralysis patient smile",
    "facial nerve palsy eye closure", "facial synkinesis", "House Brackmann grading",
    "Bell's palsy recovery week", "facial paralysis rehabilitation", "facial nerve palsy patient",
    "facial reanimation surgery before after", "synkinesis facial exercises", "facial droop patient",
    "facial palsy physical therapy face", "Ramsay Hunt syndrome face", "facial paralysis smile exercise",
    "acoustic neuroma facial weakness", "facial palsy eFACE assessment", "facial nerve recovery exercises",
]
N_PER_QUERY = int(sys.argv[1]) if len(sys.argv) > 1 else 2
SECONDS = int(sys.argv[2]) if len(sys.argv) > 2 else 45
YTDLP = "/Users/williamqiu/opt/anaconda3/bin/yt-dlp"
CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def search_ids(query, n):
    r = subprocess.run([YTDLP, f"ytsearch{n}:{query}", "--flat-playlist",
                        "--print", "%(id)s\t%(title)s\t%(duration)s"],
                       capture_output=True, text=True, timeout=120)
    out = []
    for ln in r.stdout.strip().splitlines():
        p = ln.split("\t")
        if len(p) >= 1 and p[0]:
            out.append((p[0], p[1] if len(p) > 1 else "", p[2] if len(p) > 2 else ""))
    return out


def download(vid, dst):
    # YouTube forces SABR on web client (403); the android client still works, no PO token.
    subprocess.run([YTDLP, f"https://youtube.com/watch?v={vid}",
                    "-f", "18/worst[ext=mp4]/worst",
                    "--extractor-args", "youtube:player_client=android",
                    "-o", str(dst), "--no-playlist", "--quiet", "--no-warnings"],
                   capture_output=True, text=True, timeout=300)
    return dst.exists()


def extract_faces(mov, vid, every=15):
    cap = cv2.VideoCapture(str(mov)); n = 0; fi = 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    max_frames = int(SECONDS * fps)          # cap to first SECONDS of the video
    while True:
        ok, frame = cap.read()
        if not ok or fi > max_frames:
            break
        if fi % every == 0:
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = CASCADE.detectMultiScale(g, 1.1, 5, minSize=(80, 80))
            for (x, y, w, h) in faces[:1]:
                pad = int(0.3 * w)
                crop = frame[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad]
                if crop.size:
                    cv2.imwrite(str(FACES / f"{vid}_{fi}.jpg"), cv2.resize(crop, (224, 224)))
                    n += 1
        fi += 1
    cap.release()
    return n


def main():
    mpath = OUT / "manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else []   # resume: keep prior
    seen = {m["id"] for m in manifest}
    for q in QUERIES:
        try:
            hits = search_ids(q, N_PER_QUERY)
        except Exception as e:  # noqa: BLE001
            print(f"search fail [{q}]: {e}", flush=True); continue
        for vid, title, dur in hits:
            if vid in seen:
                continue
            seen.add(vid)
            mov = CLIPS / f"{vid}.mp4"
            try:
                if not mov.exists() and not download(vid, mov):
                    print(f"  dl fail {vid}", flush=True); continue
                nf = extract_faces(mov, vid)
                manifest.append({"id": vid, "query": q, "title": title, "duration": dur,
                                 "url": f"https://youtube.com/watch?v={vid}", "n_faces": nf})
                print(f"  ok {vid}: {nf} faces  [{title[:50]}]", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"  err {vid}: {type(e).__name__}", flush=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    tot = sum(m["n_faces"] for m in manifest)
    print(f"\nCOLLECTED {len(manifest)} videos, {tot} face crops -> {FACES}")


if __name__ == "__main__":
    main()
