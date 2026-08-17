# Universal Clinical Router v5 selective candidate — rejected

## Outcome

We tested four fixed, source-blind confidence/consensus rules on top of the
unchanged UCR4 probabilities. None met the preregistered requirement of at
least 0.95 selective accuracy and 0.95 selective balanced accuracy on all three
evidence profiles at at least 70% coverage. The candidate is therefore rejected
and **UCR4 remains current**.

This is useful negative evidence: PalsyNet and NeuroFace errors are partly
concentrated near the decision boundary, but MEEI errors remain among
high-confidence cases. A confidence layer cannot repair a missing action-
dynamics representation.

## Primary 70% coverage result

The closest fixed rule was absolute probability margin. Coverage is the
fraction for which the model returned a decision; every other case abstained.

| Evidence profile | Retained / total | Coverage | Selective accuracy | Selective balanced accuracy |
|---|---:|---:|---:|---:|
| Free asymmetry — PalsyNet development | 27 / 38 | 0.711 | 0.963 | 0.964 |
| Scripted multimechanism — NeuroFace | 26 / 36 | 0.722 | 0.962 | 0.929 |
| Cue-aligned upper/action — MEEI | 40 / 56 | 0.714 | 0.875 | 0.924 |

The other three rules penalized expert range, required unanimous component
agreement, or normalized margin by component dispersion. None passed the gate;
on MEEI, expert disagreement was not a reliable marker of wrong predictions.

## Protocol and maintenance boundary

- UCR4 probabilities, heads, model JSON, current import and current registry
  were not changed.
- Evaluation remained participant/group-disjoint and used only the three
  already-exposed development cohorts. The sealed PalsyNet outer partition and
  Mayo were not read or scored.
- MEEI development accuracy uses its original nested protocol: each held fold
  is classified with a threshold learned from that fold's training participants
  (range 0.493–0.538). The final UCR4 artifact stores a single aggregated
  threshold, so the nested development result and final single-threshold
  runtime are related but not identical estimands.
- The H200 run reconstructed 38 PalsyNet, 36 NeuroFace and 56 MEEI anonymous
  OOF profiles. A second isolated aggregate run reproduced the exact public
  report SHA-256 `97555485fdfc14253ffc6deb782a0f6ca2cf443a339d79bad588f18876d3c33a`.

The next useful experiment must change the cue-aligned action representation
itself—especially cross-action temporal capacity on MEEI—rather than rescore
the same UCR4 probabilities. Any such work starts from UCR4 as a new candidate
and still requires untouched external confirmation before promotion.
