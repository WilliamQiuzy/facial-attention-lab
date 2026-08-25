# Facial Process Web — Direct PDF and Evidence Report Acceptance v4

**Date:** 2026-08-25

**Route:** Facial Process Web → raw-video gateway → MediaPipe → pinned Shared V9 BLV9-009

**Local acceptance URL:** `http://127.0.0.1:8080`

## Outcome

The report now uses a single aligned three-action toolbar: **Save PDF**,
**Download recorded video**, and **Start a new session**. The prior explanatory
block titled “How the model formed the score” was removed. The three controls
share the same width and height on desktop, use a single concise helper strip,
and become equal-width stacked controls on narrow screens.

**Save PDF now directly downloads**
`faces-research-movement-report.pdf`; it does not open the browser print dialog.
The generated A4 document contains the score, recording coverage,
interpretation limits, every available recorded context image, per-action face
tracking, and the same descriptive geometry shown on screen.

Evidence values now provide three levels of interpretation:

1. a plain-language measurement type such as side-to-side difference or
   change from neutral;
2. the value as a percentage of eye-to-eye width plus the exact normalized
   ratio;
3. a short explanation of what smaller values or neutral-relative changes mean.

Tracking is shown as both a count and percentage out of 32 registered points.
These measurements remain descriptive: no normal range, severity cutpoint,
affected side, or causal attribution is invented.

## False-positive safety boundary

The application no longer labels a thresholded result as “healthy-control
class” or “facial-palsy class.” It reports only **above or below the MEEI
research cutpoint**. The report explicitly states that the model is not
calibrated on FACES recordings and that a high score in a healthy person can
be a false positive.

This wording change is a safety correction, not a hidden model change. The
locked BLV9-009 weights and fixed 0.5 research cutpoint remain unchanged. The
available participant-disjoint development evidence already shows why a
healthy false positive is plausible:

| Development source | Accuracy | Specificity | AUROC |
|---|---:|---:|---:|
| PalsyNet development | 0.921 | 0.882 | 0.952 |
| NeuroFace | 0.889 | 0.818 | 0.920 |
| MEEI | 0.839 | 0.700 | 0.926 |

Specificity is the fraction of healthy controls called negative; therefore the
MEEI result includes a 30% healthy-control false-positive rate at the frozen
cutpoint. The candidate did not pass its preregistered promotion gate, is not
FACES- or Mayo-calibrated, and is not a diagnostic product.

A scientifically valid model correction requires new independent FACES/Mayo
healthy controls and labeled affected participants. Those data must be split
by participant before any recalibration, threshold selection, or representation
update. A single healthy recording must remain a post-lock failure case and
must not be used to tune the model that is then reported on that same person.

## Verification

- Frontend unit/component/strict-contract suite: **113/113 passed**.
- TypeScript check and production build: **passed**.
- Production dependency audit after upgrading jsPDF to 4.2.1: **no known
  vulnerabilities**.
- Docker web build reran the same **113/113** tests and the local three-service
  stack returned healthy readiness.
- Browser full-loop acceptance exercised seven- and eight-step sessions,
  rapid duplicate activation, report navigation, same-origin networking,
  Blob-URL cleanup, direct PDF download, source-video download, and responsive
  action alignment.
- Poppler verified a five-page A4 PDF containing **seven JPEG context images**;
  rendered-page review found no clipped cards or missing evidence pages.
- The desktop and mobile evidence below use only the public synthetic camera
  fixture and contain no patient or user face.

## Public-safe evidence

- [Desktop research report](artifacts/facial_process_web_ui/evidence-report-synthetic-desktop.png)
- [Mobile research report](artifacts/facial_process_web_ui/evidence-report-synthetic-mobile.png)
- [Directly downloaded PDF with context images](artifacts/facial_process_web_ui/facial-process-shared-v9-synthetic-report-with-images.pdf)

## Remaining release boundary

This is a validated research-prototype interaction, not a clinical release.
Institutional authentication/TLS, approved PHI governance, monitoring and high
availability, plus participant-disjoint Mayo/FACES labels and controls remain
required before clinical validation or patient-facing diagnostic use.
