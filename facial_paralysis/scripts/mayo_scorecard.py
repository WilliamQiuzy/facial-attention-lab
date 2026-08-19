"""Deployable per-patient clinical scorecard for the Mayo FACES cohort.

Consolidates everything that is trustworthy on Mayo WITHOUT HB labels into one report:
  - label-free regional asymmetry (brow / eye / oral)            [mayo_eface]
  - 60fps eye-closure dynamics: lagophthalmos + closure asymmetry [mayo_ear]
  - synkinesis, forced recruitment, phenotype, severity rank      [mayo_eface / unsup]
  - geometry-only learned severity (the transferable model)       [deploy_config]

Output: outputs/mayo_scorecard/scorecard.json + a cohort table + one facogram card
per patient. This is the tool clinicians can use now for triage/ranking; when HB labels
or in-domain healthy controls arrive, the supervised heads slot straight in.
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../autoresearch_fp")
import runner as R
from mayo_generalization import train_model, load_mayo, score_mayo

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "mayo_scorecard"; OUT.mkdir(parents=True, exist_ok=True)


def learned_severity():
    cfg = dict(R.DEFAULT); cfg.update(json.loads((ROOT / "autoresearch_fp/deploy_config.json").read_text()))
    models = [train_model(cfg, s) for s in (0, 1, 2)]
    return score_mayo(models, load_mayo())


def flags(rec) -> list[str]:
    f = []
    ed = rec["eface"]["eye_dynamics"]; ear = rec.get("ear", {}); sk = rec["unsup"]["synkinesis"]
    ts = ear.get("TightEyeSqueeze", {})
    wk = ts.get("weaker")
    if wk and ts.get(f"residual_{wk}", 0) > 0.06:
        f.append(f"incomplete eye closure on {wk} at max effort (lagophthalmos / corneal-exposure risk)")
    if ed.get("forced_recruitment", 0) < -0.1:
        f.append("flaccid: forcing does not recruit the weak eye (poorer prognosis)")
    elif ed.get("forced_recruitment", 0) > 0.15:
        f.append("effort recruits residual eye closure (better prognosis)")
    if sk > 0.3:
        f.append(f"synkinesis present (index {sk:.2f})")
    for reg, key in (("brow", "brow_asym"), ("eye", "eye_asym"), ("oral", "oral_asym")):
        v = rec["eface"].get(key)
        if v is not None and v > 0.30:
            f.append(f"marked {reg} asymmetry ({v:.2f})")
    return f or ["no marked deficit on the label-free measures"]


def card(pid, rec, path):
    ef = rec["eface"]
    regions = [("Forehead / Brow", ef.get("brow_asym")), ("Eye", ef.get("eye_asym")), ("Mouth / Oral", ef.get("oral_asym"))]
    fig, ax = plt.subplots(figsize=(8.5, 6)); ax.axis("off")
    ax.text(0.02, 0.96, f"FACES scorecard — {pid}", fontsize=15, fontweight="bold")
    ax.text(0.02, 0.90, f"phenotype: {rec['unsup']['phenotype']}    severity rank: {rec['rank']}/{rec['n']}"
                        f"    label-free severity: {rec['unsup']['severity']:.2f}", fontsize=10)
    cmap = plt.cm.RdYlGn_r
    for i, (name, v) in enumerate(regions):
        y = 0.70 - i * 0.16
        col = cmap(min((v or 0) / 0.5, 1.0))
        ax.add_patch(FancyBboxPatch((0.05, y), 0.42, 0.12, boxstyle="round,pad=0.01",
                                    facecolor=col, edgecolor="k", transform=ax.transAxes))
        txt = f"{name}: {'n/a' if v is None else f'{v:.2f}'}"
        ax.text(0.26, y + 0.06, txt, ha="center", va="center", fontsize=11, transform=ax.transAxes)
    # eye-closure detail (60fps)
    ts = rec.get("ear", {}).get("TightEyeSqueeze", {})
    if ts:
        ax.text(0.52, 0.72, "Eye closure @ max squeeze (60fps EAR):", fontsize=10, fontweight="bold")
        ax.text(0.54, 0.66, f"residual L={ts.get('residual_left')}  R={ts.get('residual_right')}"
                            f"   (higher = less closed)", fontsize=9)
        ax.text(0.54, 0.61, f"closure asymmetry: {ts.get('closure_asym')}   weaker: {ts.get('weaker')}", fontsize=9)
    ax.text(0.02, 0.30, "Clinical flags (label-free):", fontsize=11, fontweight="bold")
    for j, fl in enumerate(rec["flags"]):
        ax.text(0.04, 0.25 - j * 0.045, f"• {fl}", fontsize=9)
    ax.text(0.02, 0.02, "No HB labels used. Measures are objective facial-symmetry/dynamics from the FACES action battery.",
            fontsize=7, style="italic")
    fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)


def main():
    ef = json.loads((ROOT / "outputs/mayo_eface/eface_scores.json").read_text())
    ef = {k.split("_", 1)[1]: v for k, v in ef.items()}
    unsup = json.loads((ROOT / "outputs/mayo_eface/unsup_severity.json").read_text())
    ear = json.loads((ROOT / "outputs/mayo_ear/ear_dynamics.json").read_text())
    learned = learned_severity()

    pids = [p for p in unsup if p in ef]
    ranked = sorted(pids, key=lambda p: -unsup[p]["severity"])
    rank_of = {p: i + 1 for i, p in enumerate(ranked)}

    scorecard = {}
    for p in pids:
        rec = {"eface": ef[p], "unsup": unsup[p], "ear": ear.get(p, {}),
               "learned_severity_geom": {k: round(v, 3) for k, v in learned.get(p, {}).items()},
               "rank": rank_of[p], "n": len(pids)}
        rec["flags"] = flags(rec)
        scorecard[p] = rec
        card(p, rec, OUT / f"card_{p}.png")

    # persist a compact JSON (drop the bulky per-frame 'actions')
    compact = {p: {k: v for k, v in r.items() if k != "eface"} |
                  {"regions": {x: r["eface"].get(f"{x}_asym") for x in ("brow", "eye", "oral")}}
               for p, r in scorecard.items()}
    (OUT / "scorecard.json").write_text(json.dumps(compact, indent=1))

    print(f"{'patient':12s} {'pheno':>11s} {'rank':>4s} {'brow':>5s} {'eye':>5s} {'oral':>5s} "
          f"{'synk':>5s} {'geomSev(e/m)':>13s}  top flag")
    print("-" * 105)
    for p in ranked:
        r = scorecard[p]; ls = r["learned_severity_geom"]
        print(f"{p:12s} {r['unsup']['phenotype']:>11s} {r['rank']:>4d} "
              f"{str(r['eface'].get('brow_asym')):>5s} {str(r['eface'].get('eye_asym')):>5s} "
              f"{str(r['eface'].get('oral_asym')):>5s} {r['unsup']['synkinesis']:>5.2f} "
              f"{ls.get('eyes',float('nan')):>6.2f}/{ls.get('mouth',float('nan')):<6.2f} {r['flags'][0][:42]}")
    print(f"\nwrote {OUT}/scorecard.json + {len(pids)} patient cards (card_*.png)")


if __name__ == "__main__":
    main()
