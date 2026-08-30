# Facial Process Web: End-to-End Release Matrix

This matrix is the release contract for the five-stage FACES capture journey. A case is not considered covered merely because a component test exists: critical patient paths require a real browser, and recording/report paths require a real `MediaRecorder` run through the deployed web/gateway boundary.

## Coverage levels

- **Unit:** deterministic component, hook, and response-contract behavior in Vitest.
- **Browser-mocked:** real DOM, browser permissions/files/navigation, with only the model boundary mocked.
- **Browser-live:** deployed Docker web/gateway/model path or an explicitly declared real-backend rejection path.
- **Cross-browser:** Chromium, Firefox, and WebKit.

## Required state matrix

| ID | User state or transition | Required assertion | Evidence suite |
|---|---|---|---|
| P01 | First load, Step 8 unanswered | Camera setup cannot be entered; the missing choice is named | `wizard_acceptance.py`, App unit tests |
| P02 | Step 8 not applicable | Seven-step choice persists through preparation/setup navigation | `journey_edge_acceptance.py` |
| P03 | Include Step 8 | Eight-step choice persists through preparation/setup navigation | `journey_edge_acceptance.py` |
| P04 | Back/forward repeatedly | No choice, file, or stage becomes inconsistent | `journey_edge_acceptance.py` |
| P05 | Refresh before recording | Session returns to a clean Prepare state | `journey_edge_acceptance.py` |
| P06 | Voice preview start/stop | Controls recover without an alert or active speech | VoiceGuide unit tests |
| E01 | Readiness delayed | Setup remains blocked and exposes a waiting state | App unit tests |
| E02 | Readiness fails then recovers | Retry performs a new readiness request and unlocks setup | `upload_network_edge_acceptance.py` |
| C01 | Camera permission granted | Preview becomes ready and Continue unlocks | `wizard_acceptance.py` |
| C02 | Camera permission denied | Specific recovery text appears; upload remains available | `journey_edge_acceptance.py` |
| C03 | Camera missing/unreadable | Specific recovery text appears; retry and upload remain available | `journey_edge_acceptance.py` |
| C04 | Camera → Upload | Camera tracks end; no disabled Continue dead end appears | `capture_source_recovery_acceptance.py` |
| C05 | Upload → Camera | Explicit return works; camera can be re-enabled | `capture_source_recovery_acceptance.py` |
| C06 | Repeated source switching | Step 8 persists and no state or object URL leaks | `capture_source_recovery_acceptance.py` |
| C07 | Speech synthesis unavailable | Exact blocker is announced before recording | `journey_edge_acceptance.py`, workspace unit tests |
| C08 | Resize/orientation change | Camera, face guide, cues, and actions do not stretch or overlap | `responsive_capture_acceptance.py` |
| R01 | Rapid Start double click | One recorder and one guided sequence start | `journey_edge_acceptance.py`, workspace unit tests |
| R02 | Active recording | Source tabs/navigation are locked; Stop remains reachable | `wizard_acceptance.py`, `live_full_loop.py` |
| R03 | Stop and discard | No partial video is published; camera/source controls recover | `wizard_acceptance.py`, `journey_edge_acceptance.py` |
| R04 | Voice playback error | Recorder is stopped/discarded and a recoverable error is shown | workspace unit tests |
| R05 | Recorder returns no bytes | No empty video is published; clear recovery is shown | camera/workspace unit tests |
| R06 | Automatic seven-step completion | Timeline contains neutral plus six active movements | `live_full_loop.py --steps 7` |
| R07 | Automatic eight-step completion | Timeline contains neutral plus seven active movements | `live_full_loop.py --steps 8` |
| U01 | Upload chooser cancelled | No error, no state loss, and camera return remains available | `capture_source_recovery_acceptance.py` |
| U02 | Supported video | File is retained only in the browser session | `upload_network_edge_acceptance.py` |
| U03 | Unsupported/MIME mismatch | File is rejected without erasing Step 8 | `upload_network_edge_acceptance.py`, MediaCapture unit tests |
| U04 | Empty or >512 MiB video | File is rejected with bounded-size guidance | `upload_network_edge_acceptance.py`, MediaCapture unit tests |
| U05 | Replacement after valid upload | Old preview/object URL is removed; only replacement remains | `upload_network_edge_acceptance.py`, MediaCapture unit tests |
| T01 | Valid seven-step sidecar | Digest/action order bind; Step 8 remains unavailable | `upload_network_edge_acceptance.py`, inference unit tests |
| T02 | Valid eight-step sidecar | Digest/action order bind; Step 8 is included | `upload_network_edge_acceptance.py`, inference unit tests |
| T03 | Malformed/oversized sidecar | Analysis stays disabled and a specific error appears | `upload_network_edge_acceptance.py`, inference unit tests |
| T04 | Sidecar digest mismatch | Sidecar is rejected against selected video | `upload_network_edge_acceptance.py`, inference unit tests |
| A01 | Consent unchecked | Inference cannot be submitted | `upload_network_edge_acceptance.py`, App unit tests |
| A02 | Rapid Run double click | Exactly one POST and one idempotency key are emitted | `live_full_loop.py` |
| A03 | Network abort/HTTP 500 | Recording is retained and retry remains available | `upload_network_edge_acceptance.py`, App unit tests |
| A04 | HTTP 422/non-retryable rejection | Same invalid video cannot be resubmitted; reset is available | `upload_network_edge_acceptance.py`, `live_full_loop.py` |
| A05 | Malformed HTTP 200 | Response is rejected; no synthetic report appears | `upload_network_edge_acceptance.py`, inference unit tests |
| A06 | Stale response after reset/replacement | Late response cannot overwrite current session | App unit tests |
| O01 | Accepted report | Score, evidence, coverage, and report actions render | `live_full_loop.py --stub-success` |
| O02 | Save PDF | Direct PDF download contains evidence images | `live_full_loop.py --stub-success` |
| O03 | Download recording | Fixed de-identified filename; Blob URL is revoked | `live_full_loop.py --stub-success` |
| O04 | Report back/forward/reload | No inference rerun; refreshed report is not retained | `live_full_loop.py --stub-success` |
| S01 | Refresh/close/reset | Camera tracks, object URLs, and in-memory media are released | `journey_edge_acceptance.py`, `live_full_loop.py` |
| S02 | External requests | Browser contacts only the configured same origin | `live_full_loop.py` |
| S03 | Console/runtime health | No unexpected console error or uncaught page exception | every browser suite |
| X01 | Keyboard-only journey | Visible focus, tabs, radio controls, skip link, and recovery work | `accessibility_runtime_acceptance.py` |
| X02 | Live announcements | Blocking/recovery status is exposed through ARIA live/status semantics | `accessibility_runtime_acceptance.py` |
| X03 | Touch/viewport | Critical controls are at least 48 px and remain in the viewport | `responsive_capture_acceptance.py`, `accessibility_runtime_acceptance.py` |
| X04 | Reduced motion/high contrast | Navigation does not force smooth motion; controls remain visible | `accessibility_runtime_acceptance.py` |

## Required execution matrix

| Layer | Engines/devices | Minimum frequency |
|---|---|---|
| Unit/build | Node 22 + pnpm 11 | Every change |
| Core journey | Chromium desktop/tablet/mobile | Every change |
| Source recovery | Chromium desktop + touch mobile, both Step 8 choices, repeated loops | Every change |
| Responsive layout | Chromium, Firefox, WebKit × six viewports/orientations | Every UI change |
| Accessibility/runtime | Chromium desktop/mobile + WebKit keyboard path | Every release |
| Full recording/report | Chromium real MediaRecorder, seven and eight steps | Every release |
| Real backend | Deployed gateway/model, expected accepted or explicit tracking rejection | Every deployment candidate |

Passing this matrix is engineering acceptance of the web workflow. It is not clinical validation of the model or of Mayo patient performance.
