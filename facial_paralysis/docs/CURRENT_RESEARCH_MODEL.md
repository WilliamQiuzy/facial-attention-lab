# Current Research Model

## Identity

- Research version: **V9**
- Candidate: **BLV9-009 — Masked Clinical Reconstruction**
- Exact V10 comparator alias: **BRV10-000**
- Status: participant-disjoint development research; not clinically validated

BLV9-009 is the baseline for all subsequent model research. It keeps the V8
shared 478D+110D facial-motor inference architecture and adds a training-only
masked 110D reconstruction decoder. The decoder encourages the shared trunk to
retain distributed brow, eye, oral, temporal, and bilateral geometry; it is
removed at inference.

## Locked development evidence

| Source | Accuracy | Specificity | Sensitivity | AUROC |
|---|---:|---:|---:|---:|
| PalsyNet development | 0.921 | 0.882 | 0.952 | 0.952 |
| NeuroFace | 0.889 | 0.818 | 0.920 | 0.920 |
| MEEI | 0.839 | 0.700 | 0.870 | 0.926 |

These are three-seed mean-probability, participant-disjoint development
results at the fixed 0.5 threshold. BLV9-009 was selected because it had the
strongest minimum-source AUROC (0.920) in the frozen 20-model V9 screen and
materially improved NeuroFace without reducing PalsyNet performance.

## Deployment boundary

**V8 remains the deployment model.** V9 has not met the locked accuracy and
specificity floors on every development source, has no protected Mayo-label
evaluation, and does not replace `docs/CURRENT_DEPLOYMENT_MODEL.md`.

The complete three-seed V9 research ensemble is now public under
`releases/shared-v9-research-v1/`. Each member contains the exact fitted model
tensors and common 110D scaler and is bound by SHA-256 in `manifest.json`.
Publishing these research weights does not change the clinical or deployment
claim boundary.

Mayo performance is unknown until participant-level labels and a protected,
participant-disjoint evaluation are available. No House-Brackmann or clinical
accuracy claim is authorized by this research selection.

## Evidence

- Selection report: `docs/results/broad_literature_shared_v9.md`
- V9 machine artifact:
  `docs/results/artifacts/broad_literature_shared_v9/report.json`
- Public research weights: `releases/shared-v9-research-v1/`
- V10 follow-up: `docs/results/shared_v10_bilateral_reconstruction.md`
