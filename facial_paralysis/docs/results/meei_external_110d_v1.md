# Frozen 110D MEEI external validation

The unchanged final PalsyNet `landmark_mi_110d` artifact was evaluated exactly
once on 60 MEEI participants: 50 with facial palsy and 10 normal controls. All
60 participants had one authenticated eligible video and passed the frozen
label-blind extraction gate. No photographs were scored and no MEEI fitting,
calibration, threshold selection, feature selection, or candidate selection
occurred.

| Participant-level metric | Point | Descriptive 95% bootstrap interval |
|---|---:|---:|
| AUROC | 0.776 | 0.622–0.902 |
| Average precision | 0.949 | 0.910–0.980 |
| Balanced accuracy | 0.650 | 0.500–0.810 |
| Sensitivity | 0.900 | 0.820–0.980 |
| Specificity | 0.400 | 0.100–0.700 |
| Brier score | 0.143 | 0.111–0.178 |

At the locked 0.5 threshold, 45/50 affected and 4/10 normal participants were
correct, so ordinary accuracy is 49/60 = 81.7%. The cohort is 83.3% positive;
therefore average precision is prevalence-sensitive and must not be called
accuracy. Balanced accuracy shows the external behavior more honestly.

The result does not meet the 95% target. Its dominant failure is specificity:
the frozen model transferred affected-case sensitivity reasonably but called
6/10 external normal participants positive. This should direct the next study
toward source-robust normal-reference geometry using only training/development
data, followed by a new untouched external cohort. It must not trigger MEEI
threshold tuning or MEEI-specific refitting.

Evidence:

- final artifact SHA-256: `cbc49d0aa54b504915bebd00fdbe005458378e5675b57461ce83d3385f9b60f9`
- result-free authorization SHA-256: `8bf70dd1b04381a35af2de99b530e744425384a060b90a48c56156ec6b3ae6af`
- cache-byte collection SHA-256: `9f9ac765eb0ce0aff317c27099ed9c4e9828f9f4303b51c1a03372774a3c3f6f`
- aggregate report SHA-256: `445fa3c770addbeb70820a3395c13304e602b91445bb218d404cd1dff5f54baf`
- execution audit: 60 hashes, 60 loads, 60 feature extractions, 60 predictions,
  120 mirror transforms, and zero scaler/model/calibration fits
- H200 release: `/home/ssh-ziyue/facial-paralysis-h200/releases/meei-external-v1-81a83df`

The machine-readable report is a closed aggregate and contains no participant
or recording IDs, paths, filenames, labels, or row-level probabilities.
