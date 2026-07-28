# Browser verification — 2026-07-22

This is a fresh, single-session verification of the simplified clinician workflow at
`http://127.0.0.1:5173/`. It verifies a synthetic research interface, not a clinical
model, patient-data workflow, or public deployment.

## Environment

- Playwright Chromium `145.0.7632.6`, headless
- Desktop viewport: `1440 × 1050`
- Mobile viewport: `390 × 844`
- Tablet navigation check: `768` px wide
- Jobs breakpoint sweep: `721`, `739`, `780`, and `1101` px wide
- Reduced motion enabled where focus/reveal behavior was evaluated
- One browser context retained across ROI, inference, run, batch, comparison, and review lifecycles

## Result

The browser audit passed **63 of 63 checks** with no failed check.

| Clinician workflow | Fresh evidence | Result |
| --- | --- | --- |
| Cases | Ten exact synthetic cases; URL-backed search/category/ROI filters; square contained previews; one next action per card | Pass |
| ROI | Author save/submit and independent reviewer approval; desktop/mobile controls and back target | Pass |
| Analysis | Exact case/run URL; result focus/reveal; settings-change rerun gate; no result carried across cases | Pass |
| Runs | Case-led recent list; exact detail/back binding; one valid lifecycle action; technical data disclosed on demand | Pass |
| Jobs | Ready-case selection, visible exact exclusion, explicit acknowledgement, atomic start, 9 succeeded + 1 blocked | Pass |
| Models | Approved-case selector; two same-case/same-ROI outputs; four metrics; technical data disclosed on demand | Pass |
| Reviews | Queue separation, required structured notes, append-only changes/resubmit/approve/revoke lifecycle | Pass |
| Responsive/accessibility | No horizontal overflow or clipped primary action; visible focus; effective 44 px primary targets | Pass |

## Attention-presentation simplification follow-up

A focused browser pass exercised the revised Analysis and approved patient-explanation
flows after the 63-check baseline:

- `Summary` opens first and expresses the nine image-relative regions with both color
  and the text bands `No displayed signal`, `Lower`, `Moderate`, or `Higher relative signal`.
- `Separate` keeps the source image and the blue simulated signal field in distinct
  panels. `Overlay` remains available as the original optional presentation.
- The non-prediction boundary remains visible in every mode. Overlay-only controls do
  not appear in Summary or Separate.
- Analysis keeps the result and next lifecycle action visible while metrics, hashes,
  configuration, preflight, provenance, and timeline stay in the closed
  `Advanced details` disclosure.
- The patient explanation now reaches the Result section at approximately 659 px on
  the `390 × 844` viewport. Discussion guidance and technical export remain closed
  until requested.
- The `390` px layout has no horizontal overflow and retains 44 px result choices.
  At `768` px, Cases, Reviews, and Help remain directly visible without a menu step.
- No page error, console warning/error, failed request, external request, or browser
  persistence write was observed in the focused pass.

Focused screenshots are stored outside the repository at:

- `/tmp/recheck-analysis-summary-desktop.png`
- `/tmp/recheck-patient-desktop.png`
- `/tmp/recheck-patient-mobile.png`
- `/tmp/recheck-patient-details-mobile.png`

## Runtime evidence

- Page errors: **0**
- Console warnings/errors: **0**
- Failed requests or HTTP errors: **0**
- External HTTP/WebSocket requests: **0**
- Browser persistence writes: **0**
- `localStorage` / `sessionStorage` entries: **0**
- IndexedDB databases and cookies: **0**
- Observed requests: **92**, all from `http://127.0.0.1:5173`

The structured report, audit script, and screenshots were retained as local
verification artifacts outside this public source snapshot.

## Intentional boundary

All images remain exact hash-approved, AI-generated, standalone synthetic identities.
All displayed attention remains deterministic simulation and permanently states that
it is not human gaze. The session resets on refresh; connected outputs, stale results,
malformed state, and revoked reviews cannot unlock patient preview or export. Clinical
use remains blocked.

## Image-first Run follow-up — 2026-07-23

A fresh Chromium session exercised the revised workflow from Cases through an approved
patient explanation:

`Cases → Run → completed Analysis → exact Run Detail → Review → approved patient explanation`

The targeted audit passed **32 of 32 checks**.

| View | Fresh evidence | Result |
| --- | --- | --- |
| Cases | 10 exact synthetic cards; one direct Run action per card; no visible ROI workflow | Pass |
| Analysis | Source image → signal-only field → overlay → regional visual summary in one story; boundary copy and Next step follow the visuals | Pass |
| Run Detail | The same stack renders only for the exact current result selected by the central fail-closed selector | Pass |
| Patient explanation | Compact Summary-first view remains available only after exact mock review approval | Pass |
| Responsive layout | `1440 × 1050`, `768 × 900`, and `390 × 844`; no horizontal overflow | Pass |
| Runtime boundary | No warning/error console message, page exception, failed/error response, external request, cookie, IndexedDB entry, or storage write | Pass |

The completed clinician result contains no radio/tab choice and no visible ROI or
“selected region” wording. The browser verified that source, signal, overlay, and
regional summary precede safety copy and lifecycle actions in DOM order. Analysis was
4,186 px high on desktop, 3,920 px on tablet, and 3,382 px on mobile, confirming that
all visual methods remain in one normal vertically scrollable document rather than a
carousel or separate results route.

Fresh structured evidence and responsive screenshots were retained outside the
repository as local verification artifacts. They included desktop, tablet, and
mobile Analysis; desktop Run Detail; and mobile patient-explanation views.

All screenshots were visually inspected after the automated pass. The centered
760 px clinician story, restrained blue-white surfaces, readable 390 px regional
labels, and closed technical disclosures matched the intended simple clinician
workflow.

This evidence verifies only the synthetic UI and its safety gates. The current
`facial_paralysis` severity model emits a different output schema and is not connected
to this spatial-attention workbench.
