"""Consolidate #4 tracks 1-4 into a final de-identified figure + JSON. arm64 /usr/bin/python3."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import asym3d_analyze as AN, asym3d_landmark as A

ROOT = Path(__file__).resolve().parent.parent
D3 = ROOT / "outputs" / "depth3d"
MAN = AN.MAN
frames = {fid: AN.resid_of(fid) for fid in MAN}
frames = {k: v for k, v in frames.items() if v is not None}
takes = sorted({MAN[f]["take"] for f in frames})


def pooled_overall(fids):
    idx = set(); [idx.update(frames[f]) for f in fids]
    pr = {i: np.median([frames[f][i] for f in fids if i in frames[f]]) for i in idx}
    vs = [AN.region_asym(pr, rg) for rg in A.PAIRS]
    vs = [x for x in vs if not np.isnan(x)]
    return float(np.median(vs)) if vs else np.nan


# split-half reliability + per-frame ICC (reuse analyze for per-frame)
sh_a, sh_b = [], []
for t in takes:
    rf = sorted(f for f in frames if MAN[f]["take"] == t and MAN[f]["kind"] == "rest")
    if len(rf) >= 4:
        a, b = pooled_overall(rf[::2]), pooled_overall(rf[1::2])
        if not (np.isnan(a) or np.isnan(b)):
            sh_a.append(a); sh_b.append(b)
r_sh = float(np.corrcoef(sh_a, sh_b)[0, 1]); rel_full = 2 * r_sh / (1 + r_sh)

# simulation rows
sim = json.load(open(D3 / "sim_measurement_error.json"))
# dynamic
tracks = json.load(open(D3 / "asym3d_tracks.json"))

fig, ax = plt.subplots(1, 4, figsize=(17, 4.2))

# A. reliability: per-frame vs pooled
ax[0].bar(["per-frame\n(ICC)", "pooled\n(Spearman-Brown)"], [tracks["reliability"]["icc"], rel_full],
          color=["#c44", "#4a4"])
ax[0].axhline(0.75, ls="--", c="gray", lw=1); ax[0].set_ylim(0, 1)
ax[0].set_ylabel("reliability"); ax[0].set_title("A. Track 3: pooling frames\nrescues reliability (0.10 -> %.2f)" % rel_full, fontsize=9)

# B. simulation: measured vs true
rows = sim["rows"]
for name in sorted({r[0] for r in rows}):
    xs = [r[1] for r in rows if r[0] == name]; ys = [r[2] for r in rows if r[0] == name]
    ax[1].plot(xs, ys, "o-", ms=4, label=name.split("_", 1)[1])
ax[1].plot([0, 10], [0, 10], "k--", lw=1, label="ideal")
ax[1].axhline(sim["noise_floor_mm"], color="gray", ls=":", label=f"floor {sim['noise_floor_mm']}mm")
ax[1].set_xlabel("injected true delta (mm)"); ax[1].set_ylabel("measured (mm)")
ax[1].set_title("B. Track 4: measurement error\nfloor ~%.1fmm, unbiased where dense" % sim["noise_floor_mm"], fontsize=9)
ax[1].legend(fontsize=6)

# C. dynamic rest vs action
regs = list(tracks["dynamic"]); rest = [tracks["dynamic"][r]["rest_mm"] for r in regs]
act = [tracks["dynamic"][r]["action_mm"] for r in regs]
x = np.arange(len(regs)); w = 0.35
ax[2].bar(x - w/2, rest, w, label="rest", color="#88a")
ax[2].bar(x + w/2, act, w, label="action peak", color="#a55")
ax[2].set_xticks(x); ax[2].set_xticklabels(regs); ax[2].legend(fontsize=7)
ax[2].set_ylabel("3D asymmetry (mm)")
ax[2].set_title("C. Track 1: 'asymmetry on demand'\nno significant rest->action rise (all p>0.29)", fontsize=9)

# D. honest verdict: reliable pooled 3D still orthogonal to 2D
ef = json.load(open(ROOT / "outputs/mayo_eface/eface_scores.json")); ef = {k.split("_", 1)[1]: v for k, v in ef.items()}
def d2(p):
    r = ef.get(p)
    if not r: return None
    vs = [r[k] for k in ("brow_asym", "eye_asym", "oral_asym") if r.get(k) is not None]
    return np.mean(vs) if vs else None
pool = {}
for t in takes:
    rf = [f for f in frames if MAN[f]["take"] == t and MAN[f]["kind"] == "rest"]
    if len(rf) >= 3:
        v = pooled_overall(rf)
        if not np.isnan(v): pool[t.split("_", 1)[1]] = v
P = [p for p in pool if p != "MySlate_14" and d2(p) is not None]
xx = [pool[p] for p in P]; yy = [d2(p) for p in P]
rho, pv = spearmanr(xx, yy)
ax[3].scatter(xx, yy, c="steelblue")
for p, a_, b_ in zip(P, xx, yy):
    ax[3].annotate(p.replace("MySlate_", "M").replace("FACES", "F"), (a_, b_), fontsize=6)
ax[3].set_xlabel("reliable pooled 3D asym (mm)"); ax[3].set_ylabel("2D blendshape asym")
ax[3].set_title("D. Even reliable, 3D is orthogonal\nto 2D: spearman %+.2f (p=%.2f, n=%d)" % (rho, pv, len(P)), fontsize=9)

fig.suptitle("Direction #4 without more patients/labels: the measure can be made RELIABLE (pooling), "
             "error is characterized, but clinical validity stays gated on labels/3D-ground-truth.", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(D3 / "asym3d_tracks_summary.png", dpi=125)
out = dict(tracks); out["reliability"]["pooled_split_half_r"] = round(r_sh, 3)
out["reliability"]["pooled_full_reliability"] = round(rel_full, 3)
out["pooled_vs_2d_spearman"] = round(float(rho), 3); out["pooled_vs_2d_p"] = round(float(pv), 3)
out["pooled_per_patient_mm"] = {k: round(v, 2) for k, v in pool.items()}
(D3 / "asym3d_tracks.json").write_text(json.dumps(out, indent=1))
print(f"pooled reliability {rel_full:.2f}; pooled-vs-2D spearman {rho:+.2f} (n={len(P)})")
print(f"wrote {D3/'asym3d_tracks_summary.png'}")
