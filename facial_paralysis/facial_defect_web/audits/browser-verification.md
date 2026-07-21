# Browser verification — 2026-07-20

## Environment

- Vite development server on `http://127.0.0.1:5173`
- Desktop viewport: 1440 × 1000
- Mobile emulation: 390 × 844, device scale factor 1
- Mobile geometry check: `innerWidth = clientWidth = scrollWidth = 390` on both analysis and patient routes

## Exercised paths

| Route / flow | Evidence checked | Result |
| --- | --- | --- |
| `/` | Research notice, independent brand, hero hierarchy, two pathways, safeguards | Pass |
| `/cases` | Search, no-result state, clear search, open synthetic case | Pass |
| `/analysis?case=demo-001` | Provenance, unpaired disclosure, two watermarks, layer toggle, ROI toggle, metrics, QC | Pass |
| Analysis → patient handoff | Case query retained and clinician return URL encoded | Pass |
| `/patient-report?...` | Plain-language limits, unpaired synthetic disclosure, print action, return link | Pass |
| Route navigation | Scroll position resets at route changes | Pass |
| Mobile header | Primary navigation collapses to a menu control | Pass |
| Mobile analysis / patient | No horizontal overflow at 390 px | Pass |
| Six-route console pass | Runtime exceptions, console errors/warnings | Pass — no issues captured |
| Production preview | Direct-load analysis route, two images, two simulation watermarks | Pass |
| Print-media patient view | Research-only notice, patient safety card, interpretation limits, print-safe contrast | Pass |

Both complete journeys were exercised: worklist → analysis → patient explanation → clinician return preserved `case=demo-001`, and overview → patient explanation → methods → overview returned to the public landing route. The print handler and range control are covered by DOM interaction tests; the browser bridge could click buttons and links but seals page globals and could not safely intercept the native print dialog or synthesize a native range drag.

## Captures

The local verification workspace retained four screenshots (desktop home,
mobile analysis, mobile patient explanation, and patient print). They are not
part of the repository because they are test artifacts rather than runtime
assets.

The first mobile command-line capture was discarded because Chrome on macOS clamped the requested layout viewport to 500 px and cropped the bitmap to 390 px. The accepted captures use DevTools device emulation and report the actual 390 px layout metrics above.
