"""Direction #4 payoff: appearance/camera-INVARIANT 3D facial asymmetry from the decoded
iPhone depth. Sidesteps the MARLIN domain confound (domain-AUC 1.0) entirely.

Pipeline per frame: decode depth -> clean -> upright (Orientation 4) -> extract face ->
nose-ridge midsagittal line (palsy-invariant: nose is central) -> mirror L/R -> 3D
asymmetry = median |depth - mirrored_depth| over the face overlap.

RUN WITH arm64 /usr/bin/python3 (the ooz dylib is arm64).
"""
from __future__ import annotations
import ctypes, re, sys, json
from pathlib import Path
import numpy as np
from scipy.ndimage import median_filter, label, binary_closing, binary_fill_holes

ROOT = Path(__file__).resolve().parent.parent
LIB = ctypes.CDLL(str(ROOT / "ooz" / "liboodle.dylib"))
LIB.kraken_decompress.restype = ctypes.c_int
LIB.kraken_decompress.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t]
W, H, OUT = 640, 360, 640 * 360 * 2


def frames_of(take_dir: Path):
    data = (take_dir / "depth_data.bin").read_bytes()
    # record = [control-byte marker][15-digit zero-padded timestamp]; markers are <0x20,
    # in-JSON numbers are preceded by ASCII (>=0x20), so this cleanly finds records.
    marks = [m.start() for m in re.finditer(rb'[\x00-\x1f]\d{15}(?!\d)', data)] + [len(data)]
    for i in range(len(marks) - 1):
        if data[marks[i]] == 0x05:
            yield data[marks[i] + 32: marks[i + 1]]


def decode(blk: bytes):
    dst = ctypes.create_string_buffer(OUT)
    if LIB.kraken_decompress(blk, len(blk), dst, OUT) != OUT:
        return None
    u = np.frombuffer(dst.raw, '<u2').reshape(H, W)
    rec = (np.cumsum(u.astype(np.uint32), axis=1) % 65536).astype(np.uint16)
    d = np.frombuffer(rec.tobytes(), '<f2').astype(np.float32).reshape(H, W)
    d[(d < 0.05) | (d > 3.0)] = 0.0
    return d


def _head_mask(dep):
    fg = dep > 0.05
    if fg.sum() < 500:
        return None
    near = np.percentile(dep[fg], 3)
    band = (dep > near - 0.02) & (dep < near + 0.20)          # ~20cm: whole head
    band = binary_closing(band, structure=np.ones((5, 9)))    # bridge streak gaps
    lab, n = label(band)
    if n == 0:
        return None
    biggest = 1 + int(np.argmax([(lab == i).sum() for i in range(1, n + 1)]))
    mask = binary_fill_holes(lab == biggest)
    return mask if mask.sum() > 800 else None


def face_upright(d):
    """Clean streaks, isolate the head, rotate so it is upright (taller than wide)."""
    d = median_filter(d, size=(5, 7))                         # stronger de-streak
    best = None
    for k in (1, 3):                                          # Orientation 4 -> 90 or 270
        up = np.rot90(d, k=k)
        mask = _head_mask(up)
        if mask is None:
            continue
        ys, xs = np.where(mask)
        aspect = (ys.max() - ys.min() + 1) / (xs.max() - xs.min() + 1)   # upright head: >1
        if best is None or aspect > best[0]:
            best = (aspect, up, mask)
    if best is None:
        return None, None
    return best[1], best[2]


def mirror_residual(dep, mask, c0):
    """Median |depth(c0+k) - depth(c0-k)| over the head, mirroring about column c0."""
    dd = np.where(mask, dep, np.nan)
    W = dd.shape[1]
    half = int(min(c0, W - 1 - c0))
    if half < 20:
        return np.nan, 0
    left = dd[:, c0 - half:c0][:, ::-1]                       # reflect
    right = dd[:, c0 + 1:c0 + 1 + half]
    both = ~np.isnan(left) & ~np.isnan(right)
    if both.sum() < 300:
        return np.nan, 0
    return float(np.median(np.abs(left[both] - right[both]))), int(both.sum())


def asymmetry_3d(up, mask):
    """Optimize the vertical mirror axis; min residual = the irreducible 3D asymmetry.
    Robust to noise (global, no per-row landmark). Also centers the head first."""
    ys, xs = np.where(mask)
    cx = int(xs.mean())
    best = (np.inf, cx, 0)
    for c0 in range(max(cx - 60, 25), min(cx + 60, up.shape[1] - 25), 2):
        res, n = mirror_residual(up, mask, c0)
        if not np.isnan(res) and n > 300 and res < best[0]:
            best = (res, c0, n)
    resid, c0, n = best
    # per-pixel asymmetry map about the best axis
    diff = np.full(up.shape, np.nan, np.float32)
    half = int(min(c0, up.shape[1] - 1 - c0))
    dd = np.where(mask, up, np.nan)
    for k in range(1, half):
        l, r = dd[:, c0 - k], dd[:, c0 + k]
        v = ~np.isnan(l) & ~np.isnan(r)
        diff[v, c0 + k] = np.abs(l - r)[v]; diff[v, c0 - k] = np.abs(l - r)[v]
    return (resid if resid != np.inf else np.nan), c0, diff, n


def process_frame(blk):
    d = decode(blk)
    if d is None:
        return None
    up, mask = face_upright(d)
    if up is None or mask.sum() < 800:
        return None
    asym, c0, diff, n = asymmetry_3d(up, mask)
    if np.isnan(asym):
        return None
    return {"asym": asym, "up": up, "mask": mask, "axis": c0, "diff": diff,
            "n_face": int(mask.sum()), "n_pairs": n}


def main():
    take = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/livelinkface_data/20260313_FACES020")
    blks = list(frames_of(ROOT / take if not take.is_absolute() else take))
    # try a few early (resting) frames, keep the one with the most face pixels
    best = None
    for k in (40, 60, 80, 100, 120):
        if k < len(blks):
            r = process_frame(blks[k])
            if r and (best is None or r["n_face"] > best["n_face"]):
                best, bk = r, k
    if not best:
        print("no usable frame"); return
    print(f"{take.name} frame {bk}: face_px={best['n_face']}, 3D asymmetry(min mirror residual)={best['asym']:.4f} m, "
          f"axis_col={best['axis']}, n_pairs={best['n_pairs']}")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    out = ROOT / "outputs" / "depth"; out.mkdir(parents=True, exist_ok=True)
    up, mask, c0, diff = best["up"], best["mask"], best["axis"], best["diff"]
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    fd = np.where(mask, up, np.nan)
    ax[0].imshow(fd, cmap="turbo_r"); ax[0].set_title("upright face depth"); ax[0].axis("off")
    ax[1].imshow(fd, cmap="gray"); ax[1].axvline(c0, color="r", lw=1.5)
    ax[1].set_title("best symmetry axis"); ax[1].axis("off")
    ax[2].imshow(diff, cmap="hot", vmax=np.nanpercentile(diff, 95) if np.isfinite(np.nanpercentile(diff, 95)) else 0.05)
    ax[2].set_title(f"3D L-R asymmetry ({best['asym']:.3f} m)"); ax[2].axis("off")
    fig.tight_layout(); fig.savefig(out / f"{take.name}_3d.png", dpi=120)
    print(f"  viz -> {out / (take.name + '_3d.png')}")


if __name__ == "__main__":
    main()
