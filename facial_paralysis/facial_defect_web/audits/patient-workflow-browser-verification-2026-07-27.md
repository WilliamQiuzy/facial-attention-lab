# Patient Workflow Browser Verification — 2026-07-27

## Outcome

The session-only clinician prototype completed the full synthetic/test path:

`Patients → New patient → Photo visit → Demo capture → Four quality checks → Simulated analysis → Image-first result → Clinician review → Patient timeline`

The verified local service was:

`http://127.0.0.1:5173/`

This is not evidence that a real patient backend or spatial-attention model is
connected. The patient workflow uses an in-memory media vault and a local,
deterministic fixed-template simulation.

## Automated verification

- Vitest: 51 files passed, 821 tests passed.
- TypeScript: `tsc --noEmit --pretty false` passed.
- Production build: passed.
- Scoped `git diff --check`: passed.
- Vite emitted one non-blocking bundle-size warning for the 538.54 kB
  minified main JavaScript chunk.

## Browser workflow

The Chromium audit created a session-test patient, added the approved
standalone demo image, confirmed all four quality checks, observed queued and
running states, reviewed the one-page visual result, completed the review, and
confirmed that the patient timeline changed to `Complete`.

The result page presented, in order:

1. original photograph;
2. blue simulated overlay;
3. blue density-only field;
4. fixed-template facial-area summary;
5. collapsed technical details;
6. simple clinician review.

The result visibly stated that the demo field is seeded by the capture hash,
does not detect the photographed person's scar, and was not produced by the
checked-in facial-paralysis model.

## Browser checks

| Check | Evidence |
| --- | --- |
| Desktop | 1440 × 1000 full workflow passed |
| Tablet | 768 × 900 result page had `scrollWidth = clientWidth = 768` |
| Mobile | 390 × 844 result page had `scrollWidth = clientWidth = 390` |
| Narrow mobile | 360 × 800 result page had `scrollWidth = clientWidth = 360` |
| Visit-create tablet edge | 600 px page had `scrollWidth = 600`; identity stacked without clipping |
| Interactive targets | Minimum audited visible click target was 48 px; no target below 44 px |
| Keyboard error recovery | Missing review decision focused the `Reviewed` radio |
| Search continuity | Patient search was restored from React memory after opening a patient and using the back link; the query was absent from URL and `history.state.usr` |
| Duplicate IDs | None |
| Console/page/request failures | None |
| External requests | None |
| Browser persistence | localStorage 0, sessionStorage 0, IndexedDB empty, cookies empty |
| Default result identifiers | No run ID or SHA-256 visible |
| Visualization styling | No CSS background gradient on attention points; density follows source aspect ratio |
| Research route isolation | Legacy `/reviews/:id` redirected to `/research/reviews/:id` with query preserved |

## Longitudinal demo safeguard

After the first visit used the standalone catalog demo, the audit added a
second postoperative visit for the same record. The second visit:

- did not render `Use synthetic demo photo`;
- displayed that catalog demos cannot establish longitudinal identity; and
- required a separately taken or uploaded test image.

The same restriction is enforced in the provider, not only in the page.
Re-selecting the same current demo within one visit is also rejected instead
of creating a meaningless duplicate capture version.

## Screenshots

Screenshots were retained outside this public source snapshot as local
verification artifacts.

Key files:

- `patient-audit-01-patients-desktop.png`
- `patient-audit-02-capture-desktop.png`
- `patient-audit-03-quality-desktop.png`
- `patient-audit-04-processing-desktop.png`
- `patient-audit-05-result-1440.png`
- `patient-audit-05-result-768.png`
- `patient-audit-05-result-390.png`
- `patient-audit-05-result-360.png`
- `patient-audit-06-complete-desktop.png`
- `patient-audit-07-second-visit-boundary-600.png`
- `patient-audit-08-reviews-desktop.png`

## Remaining blockers before real patient use

Real patient use remains blocked until the project has:

- authenticated, role-based access and a real patient identity source;
- durable encrypted record and media storage;
- authorization, audit, retention, deletion, and recovery controls;
- a versioned patient-media inference request and response contract;
- an actual validated spatial observer-attention model output;
- validated facial registration if anatomical AOI percentages are retained;
- verified longitudinal identity and acquisition-protocol compatibility;
- prospective clinical, human-factors, privacy, security, quality-system, and
  regulatory review.

The current fixed-template AOI and density field are UI-rehearsal outputs, not
clinical measurements or evidence of surgical outcome.
