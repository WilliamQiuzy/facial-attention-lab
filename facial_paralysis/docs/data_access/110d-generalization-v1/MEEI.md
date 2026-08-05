# MEEI Facial Palsy Photo and Video Standard Set

Status as of 2026-08-05: the publisher-linked public acquisition is complete in
local ignored storage. It is quarantined from PalsyNet candidate selection and
does not authorize MEEI scoring.

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
| TIF | 2 | Excluded photograph |
| PDF | 1 | Supporting document only |
| XLSX | 1 | Metadata only |
| Extensionless | 3 | Supporting/non-endpoint files only |

- Aggregate member-manifest SHA-256:
  `0583d9cc349029ccca927438ef4bdb2cba2e4cec1a21fd7d55e922597b2d3bf2`
- Metadata XLSX SHA-256:
  `52f60e8fc73d00bdbb0888ee9b2dc592b2172a234de9049480f66f4e28cfbbd6`
- Paper PDF SHA-256:
  `57e483f2c44b74d75f4fa033f1e5721dc804b6f404cb15863ee90b0c1a23d243`

No media, face data, raw source filenames, local paths, private identities,
credentials, or access tokens are tracked by this documentation packet. The
nine normalized, filename-derived public participant keys below are
intentionally recorded only for metadata reconciliation; they are not raw
filenames or private identities.

## Metadata reconciliation

The XLSX contains 51 participant rows, while the media tree contains 60 video
participants. There are no spreadsheet keys without a video. These nine video
keys have no spreadsheet row:

- `mildflaccid4`
- `mildflaccid5`
- `moderateflaccid2`
- `nearnormalflaccid3`
- `severeflaccid4`
- `synkinetic_complete5`
- `synkinetic_mild4`
- `synkinetic_moderate4`
- `synkinetic_severe5`

Their unavailable spreadsheet metadata remains `null`. It must not be inferred,
imputed, copied from another participant, or reconstructed from a filename.

## Frozen evaluation boundary

The future external binary endpoint is participant-level normal versus facial
palsy. Only the 60 MP4 files may enter dynamic extraction. Static photographs
remain excluded from decoding and scoring; HB, eFACE, and Sunnybrook may be
reported only as secondary descriptive strata where authenticated metadata is
present.

This lane is independent of PalsyNet model selection. MEEI cannot influence the
candidate, threshold, scaler, calibration, or feature registry. Scoring remains
unauthorized until the one-shot PalsyNet outer result is sealed, the final
PalsyNet artifact is frozen, the participant and dynamic-cache manifests are
authenticated, and a separate one-shot MEEI authorization is created. No MEEI
outcome has been exposed by this documentation work.
