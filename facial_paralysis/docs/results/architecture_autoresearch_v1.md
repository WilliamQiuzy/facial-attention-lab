# Architecture Autoresearch v1

## Decision

The frozen mirror-invariant Landmark 110D Logistic model remains the development
champion. Nine model families were evaluated on exactly the same 39 PalsyNet
development recordings / 38 reviewed person groups and four fixed group-
disjoint folds. The 10-recording protected outer partition was neither loaded
nor predicted. No neural, tree, hybrid, or adaptive ensemble simultaneously
improved AUROC and balanced accuracy, so the current model was not replaced.

## Architecture screen

| Candidate | AUROC | Balanced accuracy | Sensitivity | Specificity | Brier | Parameters |
|---|---:|---:|---:|---:|---:|---:|
| **Logistic 110D** | **0.980** | **0.952** | 0.905 | **1.000** | 0.117 | 111 |
| Hybrid 110D + TCN | 0.978 | 0.899 | 0.857 | 0.941 | **0.060** | 12,707 |
| MLP 110D | 0.961 | 0.923 | 0.905 | 0.941 | 0.067 | 8,817 |
| Extra Trees 110D | 0.952 | 0.947 | **0.952** | 0.941 | 0.087 | 256 trees |
| HistGradientBoosting 110D | 0.908 | 0.840 | 0.857 | 0.824 | 0.125 | 1,500 nominal |
| BiGRU Clinical23 | 0.899 | 0.800 | **0.952** | 0.647 | 0.137 | 16,753 |
| Transformer Clinical23 | 0.894 | 0.805 | 0.905 | 0.706 | 0.128 | 40,753 |
| Region TCN Clinical23 | 0.838 | 0.623 | **0.952** | 0.294 | 0.214 | 7,211 |
| TCN Clinical23 | 0.748 | 0.529 | **1.000** | 0.059 | 0.212 | 5,035 |

The Logistic winner was confirmed with seeds 0, 1, and 2; all three runs were
identical because the fixed solver is deterministic. Four post-screen adaptive
mean ensembles materially reduced Brier calibration error, but none improved
both discrimination metrics. For example, Logistic + Extra Trees retained
balanced accuracy 0.952 and improved Brier to 0.098, but reduced AUROC to 0.964.

## Split stability and null test

The frozen Logistic model was rerun on 50 new deterministic, stratified,
patient/group-disjoint four-fold partitions. AUROC had median 0.966, 5th–95th
percentile range 0.956–0.981, and full range 0.941–0.986; 49/50 repeats (98%)
were at least 0.95. Balanced accuracy had median 0.917 and range 0.881–0.952.

In a separate 500-repeat group-label permutation test, the null AUROC mean was
0.489 and its 95th percentile was 0.686. The real fixed-fold AUROC was 0.980,
giving an add-one permutation p-value of 0.001996. This rejects the explanation
that the observed development ranking is a chance alignment of person groups
and labels, while remaining a development result rather than outer validation.

## Mayo positive-cohort challenge

The 65-session Mayo folder contains 50 MOV files. Exact SHA-256 deduplication
produced 49 unique video contents; one insufficient-frame file and one video
below the fixed 75% face-coverage gate were excluded, leaving 47 challenge
videos. Container rotation metadata was applied and four time-spread,
face-present 32-frame windows were extracted locally. Raw Mayo videos were not
uploaded to H200.

The frozen PalsyNet-development 110D model called 45/47 videos positive at the
fixed 0.5 threshold: positive-call rate 0.957 with Wilson 95% interval
0.858–0.988. Confidence mean was 0.665, median 0.649, and IQR 0.598–0.716;
MediaPipe coverage median was 1.000 and minimum 0.922. Because this cohort has
no verified negative class, this is not binary accuracy, specificity, AUROC,
or independent clinical validation. Mayo was not used for model selection.

## Provenance and safety boundary

- Branch: `codex/110d-generalization-v1`
- Implementation commit: `036e033`
- H200 immutable release: `architecture-autoresearch-v1-036e033-r4`
- Code archive SHA-256: `df07df417e9981503ed3803d209cfd3724ea75cfcd89bce3b496f34403c3f3e5`
- Architecture report SHA-256: `4f5d77c6c50b2f5d313f0d907416c6ac958837b677bf0b7ecf018fe5a949e8e7`
- Mayo challenge report SHA-256: `bd05143e6221d3bf3c95a6014d2f67cf757848d340c1c52980e30a7b7a593060`
- Mayo cache manifest SHA-256: `873c2439ced56957b1f7c7dade13b5a40f9485117b066513790e5bd9ee2b3c76`
- Protected PalsyNet cache loads and predictions: `0`
- Protected outer evaluation authorized: `false`
- HB grading, clinical validation, and deployment authorized: `false`
