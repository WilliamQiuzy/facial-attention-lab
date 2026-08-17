# NeuroFace Script-Conditioned Action Capacity v1

## Result

The preregistered script-conditioned feasibility experiment completed on the
NeuroFace cohort. It used 36 participants (11 ALS, 11 healthy controls, and 14
post-stroke), with one KISS, OPEN, and SPREAD recording per participant (108
recordings total). All selection and scoring were participant-disjoint. No
PalsyNet input, cache, prediction, or protected test split was read.

The candidate is three separate action experts. Each expert receives an 18D
capacity vector derived from the frozen landmark pipeline: IQR, range, and
maximum absolute velocity for bilateral mouth corners, mouth width, and mouth
opening. Each expert is a standardized L2 logistic model (`C=0.01`,
`liblinear`), trained in six fixed participant-disjoint folds. Original and
horizontally mirrored observations each receive weight 0.5; their held-out
probabilities are averaged within action, then the three action probabilities
are averaged per participant.

| Participant-level metric | Point estimate | 95% stratified-bootstrap CI |
|---|---:|---:|
| AUROC | 0.753 | 0.578–0.902 |
| Average precision | 0.891 | 0.803–0.964 |
| Balanced accuracy | 0.744 | 0.573–0.895 |
| Sensitivity | 0.760 | 0.560–0.920 |
| Specificity | 0.727 | 0.455–1.000 |
| Brier score | 0.235 | 0.228–0.243 |

The fixed 5,000-draw, cohort-stratified bootstrap had 5,000 valid draws. Its
AUROC lower bound was above 0.50, so this experiment passes the preregistered
*cross-disease orofacial-capacity feasibility* criterion.

## Action diagnostic

The action-specific AUROCs were 0.629 for KISS, 0.822 for OPEN, and 0.465 for
SPREAD. OPEN therefore carries the clearest capacity signal in this cohort;
SPREAD does not generalize reliably by itself. This argues for keeping
action-specific experts and explicit missing-action abstention, rather than
treating every movement as exchangeable or assuming a single whole-video
asymmetry score is sufficient.

The previously released frozen 110D descriptive comparator had AUROC 0.349 on
the same three named NeuroFace task types. It was not rerun or used for model
selection. The two results are descriptive only: the capacity branch was fitted
and evaluated out of fold on NeuroFace, whereas 110D was transferred frozen.
They are therefore not a causal or head-to-head representation comparison and
do not isolate representation from endpoint- or domain-specific fitting. The
new result does **not** authorize replacement or fusion with the
frozen 110D free-recording expert inside Universal Clinical Router v4.

## Reproducibility and audit

- Formal H200 image: `sha256:25940c2e52d566bbe241b78d8cee2ae72349fb92ed461eb05f2cef540d857c27`.
- Implementation commitment: `d75e865d6a701caca753dbb54f1e79c08845abb5c80300e7f4ea14e920e5904f`.
- Public report SHA-256: `be246d848e78598c47d0470ff5c4175dbe1a4b6b012fadedbbf0f7494f34c290`.
- Participant-free OOF commitment published in the report: `245db33fcae711722212a76f037930f7a1d9cad0b9ea7694879612eb73e689da`.
- Full private OOF file SHA-256 (the file remains only on H200): `9bdbd3ce3b0f800b7226a183026298c6fe461d921b5b5f17385bb148ca46b4a4`.
- The container ran as UID/GID 1001:1001 with exactly two host binds: frozen
  NeuroFace input read-only and formal output read-write. A signed 64 MiB
  in-memory `/tmp` was used for SciPy; it was not a host-data mount.
- An independent read-only recomputation reproduced every point metric,
  original/mirror mean, three-task participant mean, per-task AUROC, and all
  5,000 bootstrap intervals exactly (maximum absolute difference 0.0).
- The public report contains no participant identifiers; the private OOF file
  remains owner-only on the H200 release.

## Claim boundary and next decision

This endpoint is ALS/post-stroke versus healthy-control *orofacial capacity*.
It is not Bell's palsy detection, House–Brackmann grading, Mayo accuracy, or a
clinical-use result. The frozen 110D model remains the current Bell's palsy
candidate. The next useful step is to obtain externally anchored action timing
for the scripted Mayo recordings and collect labeled Mayo patients plus healthy
controls; only then should 110D asymmetry and the action-capacity experts be
evaluated participant-disjoint and reported separately before any fusion.
Fusion must then be preregistered and validated on an untouched split or cohort
before a Bell's-palsy transfer claim.

Machine-readable public result:
`outputs/dynamic_landmark/benchmarks/external/neuroface-action-capacity-v1/report.json`.
