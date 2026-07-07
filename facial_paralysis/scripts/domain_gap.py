"""Domain-gap diagnostic (no labels, well-powered): how separable are web vs Mayo
features, per feature group? A domain classifier's AUC = how far apart the domains are.
If MARLIN (appearance) is far more separable than the geometric features, that is HARD
evidence (distribution distance, not just correlation) that MARLIN is the domain confound.
Also the baseline against which DANN / augmentation is measured.
"""
from __future__ import annotations
import os, sys, json, glob
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../autoresearch_fp")
import prepare_fp as P
ROOT = Path(__file__).resolve().parent.parent


def web_feats(n=300, seed=0):
    recs = P.load_data(); rng = np.random.default_rng(seed)
    idx = rng.choice(len(recs), min(n, len(recs)), replace=False)
    M, G, A = [], [], []
    for i in idx:
        r = recs[i]
        M.append(r["marlin"]); mp = r["mp_seq"].mean(0)
        G.append(mp); A.append(mp[52:72])
    return np.array(M), np.array(G), np.array(A)


def mayo_feats():
    M, G, A = [], [], []
    for npz in glob.glob(str(ROOT / "outputs/mayo_action_bundles/*/*.npz")):
        z = np.load(npz)
        M.append(z["marlin"].astype(np.float32).mean(0)); mp = z["mp_seq"].astype(np.float32).mean(0)
        G.append(mp); A.append(mp[52:72])
    return np.array(M), np.array(G), np.array(A)


def domain_auc(web, mayo):
    X = np.vstack([web, mayo]); y = np.r_[np.zeros(len(web)), np.ones(len(mayo))]
    X = StandardScaler().fit_transform(X)
    auc = cross_val_score(LogisticRegression(max_iter=2000, C=0.5), X, y, cv=5, scoring="roc_auc")
    return auc.mean(), auc.std()


def main():
    wM, wG, wA = web_feats(); mM, mG, mA = mayo_feats()
    print(f"web n={len(wM)}, mayo n={len(mM)}")
    print("Domain separability (web vs Mayo) — AUC 1.0 = maximally different domains, 0.5 = same:\n")
    for name, w, m in (("MARLIN appearance (768)", wM, mM),
                       ("geometry all (72)", wG, mG),
                       ("asym deltas (20)", wA, mA)):
        a, s = domain_auc(w, m)
        print(f"  {name:26s} domain-AUC = {a:.3f} ± {s:.3f}   {'<-- big gap' if a>0.9 else ('<-- small gap' if a<0.75 else '')}")
    print("\nInterpretation: the higher the AUC, the more that feature group ENCODES the domain")
    print("(and thus fails to transfer). MARLIN >> geometry would confirm the confound directly.")


if __name__ == "__main__":
    main()
