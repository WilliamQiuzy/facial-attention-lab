# MEEI Facial Palsy Photo and Video Standard Set

Status as of 2026-08-11: acquisition, participant audit, frozen dynamic
extraction, and the one authorized external scoring are complete. MEEI remained
quarantined from PalsyNet candidate selection and no model component was fit on
MEEI.

## Official sources

- [Hadlock Facial Plastic Surgery resources](https://www.hadlockfacialplasticsurgery.com/resources/)
  links the public [Facial Palsy Open Source Dataset](https://drive.google.com/drive/folders/1fSWluSuOJ5_SwiLfOr1yK63NSzy-95uZ).
  No application gate is asserted for this publisher-linked download. The
  public link does not by itself establish a redistribution license; this packet
  tracks no raw data and makes no redistribution or broader-use claim.
- Greene et al., “The spectrum of facial palsy: The MEEI facial palsy photo and
  video standard set,” [PubMed 31021433](https://pubmed.ncbi.nlm.nih.gov/31021433/),
  DOI [`10.1002/lary.27986`](https://doi.org/10.1002/lary.27986).

The paper reports 60 participants: 25 with flaccid palsy, 25 with synkinetic
palsy, and 10 normal participants. It describes a standardized video of facial
movements and clinical grading with eFACE, House-Brackmann, and Sunnybrook. The
current protocol does not reinterpret these materials as frame-level Action or
Phase labels.

## Hash-first local inventory

The ignored acquisition contains 547 files totaling 1,316,600,371 bytes:

| Type | Count | Dynamic endpoint |
| --- | ---: | --- |
| JPG | 480 | Excluded; static photographs are never tiled or converted into trajectories |
| MP4 | 60 | The only eligible media type, subject to the future authenticated cache and label-blind QC |
| TIF | 2 | Excluded cohort-level supporting montage |
| PDF | 1 | Supporting document only |
| XLSX | 1 | Metadata only |
| `.DS_Store` | 3 | Supporting/non-endpoint files only |

- Aggregate member-manifest SHA-256:
  `098ab51327be335ae08ddd16268cdfbd899a1c581cec95b951af3fd9bff93546`
- Metadata XLSX SHA-256:
  `52f60e8fc73d00bdbb0888ee9b2dc592b2172a234de9049480f66f4e28cfbbd6`
- Paper PDF SHA-256:
  `57e483f2c44b74d75f4fa033f1e5721dc804b6f404cb15863ee90b0c1a23d243`

No media, face data, source filenames, local paths, participant keys,
credentials, or access tokens are tracked by this documentation packet.

## Metadata reconciliation

The XLSX contains 51 participant rows, while the media tree contains 60 video
participants. There are no spreadsheet rows without a video; nine video
participants have no spreadsheet metadata. Their unavailable metadata remains
`null` and was not inferred, imputed, copied, or reconstructed from filenames.

## Frozen evaluation and result boundary

The future external binary endpoint is participant-level normal versus facial
palsy. Only the 60 MP4 files may enter dynamic extraction. Static photographs
remain excluded from decoding and scoring; HB, eFACE, and Sunnybrook may be
reported only as secondary descriptive strata where authenticated metadata is
present.

This lane was independent of PalsyNet model selection. The one-shot registry
bound the final artifact, participant manifest, dynamic-cache manifest, all 60
NPZ byte artifacts, implementation, protocol, population counts, and fixed
output. The H200 runner loaded and predicted each participant once and refused
all fitting, calibration, threshold, seed, and output overrides.

The closed aggregate result is AUROC 0.776, balanced accuracy 0.650,
sensitivity 0.900, specificity 0.400, average precision 0.949, and Brier 0.143.
The ordinary fixed-threshold accuracy derived from aggregate class counts is
49/60 = 81.7%. The result is external evidence of a cross-source specificity
gap, not a 95% result, HB accuracy, or clinical validation. MEEI is now an
exposed external test and must not be used to tune a successor model.
