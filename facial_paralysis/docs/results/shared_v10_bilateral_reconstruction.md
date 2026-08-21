# Shared V10 Bilateral Reconstruction

## Decision

**No V10 candidate is promoted. BLV9-009 remains the research baseline, and V8 remains the deployment model.**

The screen completed six frozen candidates, three seeds, six within-source
participant-disjoint folds, and three leave-one-source-out fits: **162 fits**
in 265.4 seconds on one NVIDIA H200. PalsyNet protected reads: **0**. Mayo reads: **0**; Mayo predictions: **0**.

## Why these candidates were tested

The selected V9 auxiliary reconstructs the average of original and mirrored
110D views. V10 tested whether retaining more bilateral information could
improve robustness without assuming that either side was healthy:

- **bilateral decomposition:** reconstruct the symmetric mean and absolute
  difference, representing movement capacity plus asymmetry magnitude;
- **unordered twin:** reconstruct original and mirrored views as an unordered
  pair, preserving unilateral patterns without assigning left/right disease;
- **SAM:** test whether the same representation benefits from a flatter
  training solution on the small cohorts.

All reconstruction decoders are training-only. Source identity remains absent
from the shared encoder, and no laterality or House-Brackmann label is inferred.

## Three-seed results

| Candidate | PalsyNet acc/spec/AUROC | NeuroFace acc/spec/AUROC | MEEI acc/spec/AUROC | Minimum AUROC |
|---|---:|---:|---:|---:|
| BRV10-000 V9 average | **0.921/0.882/0.952** | **0.889/0.818/0.920** | 0.839/0.700/**0.926** | **0.920** |
| BRV10-001 V9 average + SAM | 0.895/0.882/0.952 | 0.833/0.818/0.869 | 0.821/0.800/0.904 | 0.869 |
| BRV10-002 bilateral decomposition | **0.921/0.882/0.952** | 0.861/0.727/0.905 | **0.857**/0.700/**0.926** | 0.905 |
| BRV10-003 bilateral decomposition + SAM | 0.895/0.882/0.952 | 0.861/0.818/0.895 | 0.821/0.800/0.904 | 0.895 |
| BRV10-004 unordered twin | **0.921/0.882/0.952** | 0.861/0.727/0.909 | 0.839/0.700/0.924 | 0.909 |
| BRV10-005 unordered twin + SAM | 0.895/0.882/0.952 | 0.833/0.727/0.865 | 0.821/0.800/0.913 | 0.865 |

SAM consistently raised the worst-source specificity to 0.80 for the averaged
and bilateral-decomposition targets, but it reduced accuracy and AUROC. The
bilateral decomposition improved MEEI accuracy from 0.839 to 0.857 without
changing its AUROC, but reduced NeuroFace performance. The unordered twin also
failed to improve the three-source minimum. Therefore none passes the locked
gate, and the stronger minimum AUROC of BLV9-009 remains the safer research
choice.

## Locked gate and next boundary

Promotion required every source to reach accuracy >=0.90, specificity >=0.80,
sensitivity >=0.85, and AUROC >=0.92, with no material regression from the V9
baseline and a strict worst-source specificity improvement. No candidate met
that joint requirement.

This result suggests that reconstructing more bilateral detail alone cannot
resolve the remaining cross-source trade-off. A subsequent experiment should
not keep recombining objectives on these same 130 exposed participants. The
next defensible improvement needs new participant-disjoint controls/labels or
a frozen evaluation cohort; Mayo model selection remains prohibited until its
labels and split protocol are locked.

## Reproducibility

- Machine report:
  `docs/results/artifacts/shared_v10_bilateral_reconstruction/report.json`
- Report SHA-256:
  `e13cd5a5d72e77c94f3181a90a86d7e84f9fe6238d41cc2c5099589e3159157a`
- Runtime: Python 3.11.13, PyTorch 2.7.1+cu128, NumPy 2.4.6,
  NVIDIA H200, 20 epochs per fit.
