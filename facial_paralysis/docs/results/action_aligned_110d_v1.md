# Action-Aligned Landmark 110D v1

## Decision

Keep the frozen four-time-window Landmark 110D as the current research model. The seven-action-window representation is archived as a completed negative generalization experiment: it improved PalsyNet development ranking slightly but reduced Mayo assumed-positive confidence, so it is not adopted.

## Locked protocol

- Candidates: frozen four-time-window Landmark 110D and seven deterministic action-window Landmark 110D.
- Classifier: the same standardized L2 logistic regression (`C=0.01`) for both candidates, with original-plus-mirror training and mean original/mirror inference.
- Selection data: 39 PalsyNet development recordings from 38 reviewed identity groups, evaluated with four group-disjoint out-of-fold splits.
- Protected data: 10 PalsyNet recordings remained unopened; protected feature reads, fits, and predictions were all zero.
- Mayo use: one post-lock aggregate challenge only. Mayo did not tune the representation, model, threshold, or decision rule.

## PalsyNet development result

| Representation | AUROC | Average precision | Balanced accuracy | Sensitivity | Specificity | Brier score |
|---|---:|---:|---:|---:|---:|---:|
| Four time windows | 0.9804 | 0.9872 | 0.9524 | 0.9048 | 1.0000 | 0.1174 |
| Seven action windows | 0.9888 | 0.9924 | 0.9524 | 0.9048 | 1.0000 | 0.1087 |

The action representation met the predeclared development gate through a +0.0084 AUROC change and a -0.0087 Brier change. However, the paired 95% bootstrap intervals crossed zero for AUROC, balanced accuracy, and Brier, so this is not evidence of a statistically confirmed or clinical improvement.

## Mayo assumed-positive challenge

The content-deduplicated challenge contains 47 videos assumed to be affected and no verified negative controls. Therefore accuracy, AUROC, sensitivity, specificity, and HB-grade performance are undefined.

| Representation | Positive calls at 0.5 | Positive-call rate | Mean confidence | Median confidence |
|---|---:|---:|---:|---:|
| Four time windows | 45/47 | 95.74% | 0.6646 | 0.6494 |
| Seven action windows | 39/47 | 82.98% | 0.6264 | 0.6046 |

The action representation changed one baseline-negative call to positive but changed seven baseline-positive calls to negative. This external confidence regression is the reason it is not adopted. It is a stress-test result, not Mayo clinical accuracy.

## Reproduction and audit

- The final PalsyNet development report has SHA-256 `6f6ee89bb4f1646354492ca4cef49d486db56fdc94bb5d30692fc447e157e3df` and implementation SHA-256 `b8c30a112690c5a7d235dce59c39e2cad4499474188ea7fcbc69f48440817f88`.
- Nebius H200 reproduced the same decisions and metrics within `1e-14` numerical tolerance in 13.14 seconds on an NVIDIA H200 with 143,771 MiB memory.
- The action sampling contract is native-frame-rate invariant: 30 fps and 60 fps videos are mapped to the same 30 Hz action time grid while retaining exact timestamps and detector-validity masks.
- No raw Mayo video, row-level Mayo probability, or protected PalsyNet prediction is included in the repository artifacts.

## Next research direction

The next fast experiment should keep the four-window 110D classifier fixed and test a narrowly scoped scale-robust eye-geometry representation. Selection must remain on identity-disjoint PalsyNet development data; Mayo should remain an aggregate monitoring cohort until verified labels and controls are available.
