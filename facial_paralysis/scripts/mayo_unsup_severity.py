"""Unsupervised, label-free facial-palsy severity model for the Mayo FACES cohort.

No HB labels exist, so we don't predict a grade — we place each patient in an
interpretable, clinically-grounded 2-axis space built ONLY from the label-free,
domain-invariant measures (Run #14 showed asymmetry — not MARLIN appearance — is the
trustworthy Mayo signal):

  severity axis  = how asymmetric / how reduced the voluntary movement is
  synkinesis axis = involuntary cross-region co-movement

This separates the two clinically canonical phenotypes (flaccid vs synkinetic) plus a
mild/near-recovered group — the split clinicians actually care about — and yields a
defensible severity RANKING for triage and for prioritizing HB labeling.

Features come from scripts/mayo_eface.py output (outputs/mayo_eface/eface_scores.json).
Runs locally; no labels, no mediapipe, no GPU.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parent.parent
R = json.loads((ROOT / "outputs" / "mayo_eface" / "eface_scores.json").read_text())
OUT = ROOT / "outputs" / "mayo_eface"

takes = [k for k in R if k != "20260305_MySlate_14"]          # drop duplicate
short = [k.split("_", 1)[1] for k in takes]


def feat(k):
    r = R[k]
    ed, sk, acts = r["eye_dynamics"], r["synkinesis"], r["actions"]
    # movement excursion (low = weak voluntary movement, a severity sign)
    exc = [acts[a]["excursion"] for a in acts]
    mean_exc = float(np.mean(exc)) if exc else 0.0
    return {
        "brow": r["brow_asym"], "eye": r["eye_asym"], "oral": r["oral_asym"],
        "forced_deficit": -min(ed.get("forced_recruitment", 0.0), 0.0),   # >0 when forcing FAILS (flaccid)
        "low_excursion": max(0.0, 0.5 - mean_exc),                        # reduced movement
        "synk": max(sk.get("oral_ocular_synk", 0.0), 0.0) + max(sk.get("ocular_oral_synk", 0.0), 0.0),
    }


F = [feat(k) for k in takes]
# impute missing (some takes miss a region) with cohort median
sev_cols = ["brow", "eye", "oral", "forced_deficit", "low_excursion"]
M = np.array([[ (f[c] if f[c] is not None else np.nan) for c in sev_cols] for f in F], float)
col_med = np.nanmedian(M, axis=0)
M = np.where(np.isnan(M), col_med, M)
synk = np.array([f["synk"] for f in F])

# severity axis = PC1 of the standardized asymmetry/deficit features
Z = StandardScaler().fit_transform(M)
pc1 = PCA(n_components=2).fit_transform(Z)[:, 0]
if np.corrcoef(pc1, M[:, 1])[0, 1] < 0:   # orient so higher = more asymmetric (eye col)
    pc1 = -pc1
severity = (pc1 - pc1.min()) / (pc1.ptp() + 1e-9)                       # 0..1

# phenotype clustering in (severity, synkinesis) space
X2 = StandardScaler().fit_transform(np.c_[severity, synk])
km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(X2)
# name clusters by their centroid: high-synk=synkinetic, high-sev/low-synk=flaccid, low both=mild
cent = km.cluster_centers_
names = {}
for c in range(3):
    s, y = cent[c]
    names[c] = "synkinetic" if y == max(cent[:, 1]) else ("flaccid/severe" if s == max(cent[:, 0]) else "mild")
lab = [names[c] for c in km.labels_]

rank = np.argsort(-severity)
print(f"{'patient':14s} {'severity':>8} {'synk':>6} {'eye':>5} {'oral':>5} {'forcedDef':>9}  phenotype")
print("-" * 70)
for i in rank:
    print(f"{short[i]:14s} {severity[i]:8.2f} {synk[i]:6.2f} {M[i,1]:5.2f} {M[i,2]:5.2f} "
          f"{M[i,3]:9.2f}  {lab[i]}")

out = {short[i]: {"severity": round(float(severity[i]), 3), "synkinesis": round(float(synk[i]), 3),
                  "phenotype": lab[i]} for i in range(len(takes))}
(OUT / "unsup_severity.json").write_text(json.dumps(out, indent=1))

# figure: clinical 2-axis phenotype map
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(9, 7))
colors = {"mild": "#2ca02c", "flaccid/severe": "#d62728", "synkinetic": "#9467bd"}
for i in range(len(takes)):
    ax.scatter(severity[i], synk[i], s=90, color=colors[lab[i]], edgecolor="k", zorder=3)
    ax.annotate(short[i], (severity[i], synk[i]), fontsize=8, xytext=(4, 4), textcoords="offset points")
for ph, col in colors.items():
    ax.scatter([], [], color=col, label=ph)
ax.set_xlabel("label-free SEVERITY axis  (asymmetry + reduced movement + failed recruitment)", fontsize=10)
ax.set_ylabel("SYNKINESIS axis  (involuntary co-movement)", fontsize=10)
ax.set_title("Mayo FACES cohort (n=13): unsupervised phenotype map\n"
             "built only from label-free dynamic measures — no HB labels used", fontsize=11)
ax.legend(title="phenotype (KMeans)", fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "phenotype_map.png", dpi=130)
print(f"\nwrote {OUT/'unsup_severity.json'} and {OUT/'phenotype_map.png'}")
