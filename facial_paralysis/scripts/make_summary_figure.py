"""One figure telling the whole arc: autoresearch ceiling -> no web->Mayo transfer ->
the bootstrap correction -> the power analysis (how much data is needed)."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "summary_figure.png"
fig, axes = plt.subplots(2, 3, figsize=(20, 9))
ax = axes  # 2x3 grid

# Panel 1: autoresearch trajectory (web region-QWK)
labels = ["v0\nbaseline", "asym\n+mlp", "+MARLIN\nproj", "+CORN", "per-region\ntrunks", "decoupled\n(champion)"]
vals = [0.530, 0.580, 0.597, 0.610, 0.625, 0.648]
ax[0, 0].plot(range(len(vals)), vals, "-o", color="#1f77b4", lw=2, ms=8)
ax[0, 0].set_xticks(range(len(vals))); ax[0, 0].set_xticklabels(labels, fontsize=8)
ax[0, 0].set_ylabel("web region QWK"); ax[0, 0].set_title("1. autoresearch found the web ceiling (~100 models)")
ax[0, 0].axhline(0.649, ls=":", color="gray"); ax[0, 0].set_ylim(0.50, 0.68); ax[0, 0].grid(alpha=0.3)
ax[0, 0].annotate("plateau: ceiling is DATA,\nnot architecture", (4, 0.60), fontsize=9, color="#333")

# Panel 2: web works, Mayo doesn't (transfer rho, independent target)
groups = ["champion\neyes", "champion\nmouth", "geom-only\neyes", "geom-only\nmouth"]
rho = [-0.42, 0.03, 0.05, 0.26]
ci = [(-0.85, 0.23), (-0.70, 0.89), (-0.55, 0.63), (-0.56, 0.90)]
x = np.arange(len(groups))
ax[0, 1].bar(x, rho, color=["#d62728", "#d62728", "#2ca02c", "#2ca02c"], alpha=0.7)
for i, (lo, hi) in enumerate(ci):
    ax[0, 1].plot([i, i], [lo, hi], "k-", lw=1.5)
ax[0, 1].axhline(0, color="k", lw=0.8); ax[0, 1].set_xticks(x); ax[0, 1].set_xticklabels(groups, fontsize=8)
ax[0, 1].set_ylabel("Spearman rho (severity vs clinical asym)")
ax[0, 1].set_title("2. web->Mayo transfer: every 95% CI crosses 0 (n=13)")
ax[0, 1].set_ylim(-1, 1); ax[0, 1].grid(alpha=0.3)

# Panel 3: the correction
ax[1, 0].bar([0, 1], [0.48, 0.05], color=["#ff7f0e", "#2ca02c"])
ax[1, 0].plot([1, 1], [-0.55, 0.63], "k-", lw=2)
ax[1, 0].set_xticks([0, 1])
ax[1, 0].set_xticklabels(["vs blendshape target\n(shares model inputs =\npartly circular)",
                          "vs independent EAR\n(landmark) target\n+ 95% CI"], fontsize=8)
ax[1, 0].axhline(0, color="k", lw=0.8); ax[1, 0].set_ylabel("geom-only eyes transfer rho")
ax[1, 0].set_title("3. rigor correction: the +0.48 was fragile"); ax[1, 0].set_ylim(-0.7, 0.8); ax[1, 0].grid(alpha=0.3)

# Panel 4: power curve
true_rho = [0.3, 0.4, 0.5, 0.6, 0.7]; n_need = [94, 52, 34, 22, 16]
ax[1, 1].plot(true_rho, n_need, "-o", color="#9467bd", lw=2, ms=8)
ax[1, 1].axhline(13, ls="--", color="red"); ax[1, 1].text(0.55, 16, "our n=13", color="red", fontsize=9)
ax[1, 1].fill_between([0.3, 0.45], 0, 120, color="green", alpha=0.08)
ax[1, 1].set_xlabel("true transfer strength (rho)"); ax[1, 1].set_ylabel("patients needed (80% power)")
ax[1, 1].set_title("4. to prove transfer: ~35-50 patients (or ~40-60 HB labels)")
ax[1, 1].set_ylim(0, 110); ax[1, 1].grid(alpha=0.3)

# Panel 5: feature-importance mechanism
grp = ["MARLIN\n(appearance)", "blendshapes", "asym\ndeltas"]
eyes_imp = [0.134, 0.156, 0.096]; mouth_imp = [-0.052, 0.217, 0.552]
xx = np.arange(3); w = 0.38
ax[0, 2].bar(xx - w / 2, eyes_imp, w, label="eyes", color="#1f77b4")
ax[0, 2].bar(xx + w / 2, mouth_imp, w, label="mouth", color="#d62728")
ax[0, 2].axhline(0, color="k", lw=0.8); ax[0, 2].set_xticks(xx); ax[0, 2].set_xticklabels(grp, fontsize=8)
ax[0, 2].set_ylabel("QWK drop when scrambled (importance)")
ax[0, 2].set_title("5. mechanism: mouth rides ASYMMETRY,\nMARLIN useless-harmful for it; eyes needs MARLIN")
ax[0, 2].legend(fontsize=9); ax[0, 2].grid(alpha=0.3)

# Panel 6: bottom line text
ax[1, 2].axis("off")
ax[1, 2].text(0.0, 1.0, "Bottom line", fontsize=13, fontweight="bold", va="top")
lines = [
    "• Web model optimized to 0.668 QWK (~100 configs, ablated,",
    "  feature-attributed). Ceiling is DATA, not model.",
    "• Web→Mayo transfer is NOT establishable at n=13 (CIs cross 0).",
    "• MARLIN appearance = domain-confound; geometric L/R asymmetry",
    "  is the domain-invariant signal (drives mouth entirely).",
    "• Deployable now: label-free scorecard (asymmetry / EAR-lagophthalmos",
    "  / synkinesis) for triage of CLEAR cases — not a validated grade.",
    "",
    "Data asks (quantified):",
    "  1. in-domain healthy controls  → trustworthy detector",
    "  2. ~40–60 HB labels            → usable accuracy CI",
    "  3. ~35–50 patients             → power to prove transfer",
    "  4. DISFA/BP4D (AU dynamics)    → temporal-stream pretraining",
]
for i, ln in enumerate(lines):
    ax[1, 2].text(0.0, 0.90 - i * 0.075, ln, fontsize=9, va="top",
                  family="monospace" if ln.startswith("  ") else None)

fig.suptitle("Facial-palsy on Mayo: strong web model, real dynamic measures, but transfer is data-limited (n=13)",
             fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT, dpi=130)
print("wrote", OUT)
