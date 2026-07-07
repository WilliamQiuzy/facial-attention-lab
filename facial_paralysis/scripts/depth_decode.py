"""Decode LiveLinkFace Oodle-Kraken depth_data.bin (direction #4). BLOCKER SOLVED.

Container: record 0x02 = file header (intrinsics JSON), records 0x05 = per-frame depth.
Each 0x05 record: [0x05][15-char ts][16-byte sub-header][Kraken block] -> 640x360 fp16
depth in METERS. Decompressed with the locally-built ooz (liboodle.dylib, kraken_decompress).

RUN WITH arm64 python: /usr/bin/python3 (the dylib is arm64; anaconda python is x86_64).

Usage: /usr/bin/python3 depth_decode.py [take_dir] [stride]
"""
from __future__ import annotations
import ctypes, re, sys, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LIB = ctypes.CDLL(str(ROOT / "ooz" / "liboodle.dylib"))
LIB.kraken_decompress.restype = ctypes.c_int
LIB.kraken_decompress.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t]
W, H = 640, 360
OUT_LEN = W * H * 2          # fp16
KRAKEN_OFF = 16             # Kraken block starts 16 bytes into the 0x05 payload


def decode_block(src: bytes):
    """Kraken-decompress -> 16-bit horizontal Sub de-filter -> little-endian fp16 (meters).
    The app applies a per-row 16-bit Sub predictor before Kraken (verified: this decode
    gives frame-to-frame temporal corr 0.93 and a smooth face surface; raw fp16 gives noise).
    """
    dst = ctypes.create_string_buffer(OUT_LEN)
    n = LIB.kraken_decompress(src, len(src), dst, OUT_LEN)
    if n != OUT_LEN:
        return None
    u = np.frombuffer(dst.raw, '<u2').reshape(H, W)                       # filtered uint16
    rec = (np.cumsum(u.astype(np.uint32), axis=1) % 65536).astype(np.uint16)  # inverse Sub (per row)
    d = np.frombuffer(rec.tobytes(), '<f2').astype(np.float32).reshape(H, W)
    d[(d < 0.05) | (d > 3.0)] = 0.0                                       # invalid -> 0
    return d


def depth_records(path: Path):
    data = path.read_bytes()
    marks = [(m.start() - 1) for m in re.finditer(rb'(0000000\d{8})', data) if m.start() >= 1]
    marks.append(len(data))
    for i in range(len(marks) - 1):
        p = marks[i]
        if data[p] == 0x05:
            yield data[p + 16 + KRAKEN_OFF: marks[i + 1]]


def decode_take(take_dir: Path, stride: int = 1):
    """Return (frames[T,H,W], intrinsics_dict)."""
    db = take_dir / "depth_data.bin"
    hdr = json.loads(re.search(rb'\{.*?"DepthDimensions".*?\}\}', db.read_bytes()[:4096], re.S).group().decode("utf-8", "replace"))
    frames = []
    for i, blk in enumerate(depth_records(db)):
        if i % stride:
            continue
        d = decode_block(blk)
        if d is not None:
            frames.append(d)
    return np.stack(frames) if frames else None, hdr


def main():
    take = Path(sys.argv[1]) if len(sys.argv) > 1 else next(
        d.parent for d in (ROOT / "data/livelinkface_data").rglob("depth_data.bin"))
    stride = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    frames, hdr = decode_take(take, stride)
    print(f"take {take.name}: decoded {len(frames)} frames (stride {stride}), shape {frames.shape}")
    print(f"  device {hdr.get('DeviceModel')}, depth {hdr['DepthDimensions']}, meters")
    fg = frames[frames > 0]
    print(f"  foreground depth: {fg.min():.3f}–{fg.max():.3f} m, {100*(frames>0).mean():.0f}% nonzero")
    # QC render: middle frame as a depth colormap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = ROOT / "outputs" / "depth"; out.mkdir(parents=True, exist_ok=True)
    fr = frames[len(frames) // 2].copy()
    fr[fr == 0] = np.nan
    plt.figure(figsize=(6, 3.5)); plt.imshow(fr, cmap="turbo"); plt.colorbar(label="depth (m)")
    plt.title(f"{take.name} — decoded iPhone depth (fp16, Oodle-Kraken)"); plt.axis("off")
    plt.tight_layout(); plt.savefig(out / f"{take.name}_depth.png", dpi=120)
    print(f"  QC image -> {out / (take.name + '_depth.png')}")


if __name__ == "__main__":
    main()
