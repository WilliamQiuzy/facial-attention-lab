# 8081 UX iteration — September 8, 2026

Scope: the five-stage **Facial Process Web** candidate, branch
`codex/facial-process-wizard-ab`. The stable 8080 service, model weights,
prediction thresholds, backend response contract, and clinical action IDs are unchanged.

## Seven changes

1. **One contextual primary action.** The persistent bottom bar offers Enable camera,
   permission guidance, Continue, or Start recording according to the real state.
   Duplicate camera/start controls are removed from this journey. Camera setup
   headings and preview dimensions are more compact.
2. **Synchronized voice and movement.** Both practice and capture use the same
   executor: action cue ending in “Hold now”, a three-second sampled hold, then
   an audible release cue. Eye closure ends with an explicit open-eyes cue.
   Optional sound testing needs neither a camera nor a recording. Speech ownership,
   cancellation, and watchdogs prevent practice from interrupting a live sequence.
3. **Protect the completed recording.** Replacing, switching source, recording again,
   changing the movement plan, or starting a new session asks the user to keep,
   download, or discard the current video. Invalid replacements preserve the usable
   recording. Canceled replacements can select the same file again. PDF and video
   filenames share an anonymous session identifier; download feedback states that
   a download was requested, not that disk persistence was verified.
4. **Predictable navigation.** Visited valid stages are clickable, and browser
   Back/Forward works within the current session. Preparation visits preserve
   uploaded media/timing. Five workflow steps are distinct from seven/eight
   facial movements; the movement preview follows the selected script.
5. **Small-screen readability.** Low-height windows use a compact current-stage
   menu, mobile primary buttons are at least 16px/48px, and preparation tips
   stack vertically. Scroll/focus offsets account for the persistent bars without
   double-counting their height. The face guide keeps its own aspect ratio and
   video uses `object-fit: contain`.
6. **Optional on-device framing hints.** A self-hosted, pinned face-box detector
   offers one debounced prompt about lighting, face count, centering, or distance.
   It does not evaluate movement strength, asymmetry, or disease and never gates
   recording. Unsupported devices receive written framing guidance. Processing
   stops while recording or when the preview is hidden.
7. **Clear waiting and recovery.** Analysis shows elapsed time and preserves a
   retry action for retryable failures without inventing progress percentages.
   Upload requirements (video + matching action timeline, file size limit) are
   shown before file selection; a rejected capture retains its download option.

Report presentation was also refined: a coverage overview and up to three stable
model influences precede expandable measurement/influence/stability layers.
Evidence links keep the report open and can play the registered hold in the
retained video. Collapsing evidence does not remove PDF content. This changes
presentation, not attribution calculations or predictions.

## Retention and verification boundaries

Navigation within the page preserves the session. Browser reload/close warnings
are best-effort: mobile OS termination, browser crashes, and forced shutdowns
cannot be intercepted. There is no new persistent patient-media storage; download
the video/report before leaving when retention is needed.

Camera hints are composition heuristics, not a validated assessment of image
quality across skin tones, devices, or impairment. No score or model-performance
improvement is claimed for this UI iteration.

Run the manifest-driven acceptance gate described in the web README. It includes
unit/contract checks, Chromium/Firefox/WebKit viewport and recovery checks,
seven/eight-movement real browser recording loops with a synthetic camera and
stubbed model success, report/PDF/video exports, and synthetic no-face rejection
through the real backend. Stubbed success tests demonstrate product behavior,
not clinical model accuracy.

## Verified result

- Docker build gate: **209/209 tests**, 22 test files; TypeScript checking and
  production build passed. Test workers are capped at two to avoid resource
  contention; assertions and test timeouts were not relaxed.
- Browser release gate: **12/12 suites**, 67 distinct acceptance IDs, passed on
  the candidate image. Includes Chromium, Firefox, and WebKit layout/recovery
  checks, seven/eight-movement browser recording and report-export loops, and a
  real-backend rejection of a synthetic recording containing no face. Successful
  report payloads are stubbed; this is not a new patient/model accuracy study.
- A newly reproduced quick-Start race was covered by a failing regression test
  before the fix. The unchanged immediate-double-click browser case then passed
  five consecutive times. Additional fixes preserve media on preparation visits
  and canceled/repeated file selections, and keep evidence links inside the report.
- Tested and deployed local image:
  `facial-process-wizard-ab:ux-20260908`,
  `sha256:dd458c4fe3fb537bab22bd0d0732eee5c15ae9e058063f1639b9ff0cdea612fe`.
  8081 health and proxied model readiness passed after deployment. The stable
  8080 HTML hash was unchanged before and after deployment.
- Post-deployment browser smoke passed at 1440×1000 and 390×844: preparation,
  standard movement-plan selection, setup, and the contextual Enable camera
  control. Export inspection found six/seven embedded action images and all
  18/21 evidence-layer headings in the seven/eight-movement PDFs respectively.
- Local rollback: the previous 8081 container is retained, stopped, as
  `facial-process-wizard-ab-before-ux-20260908`. The new container has automatic
  restart and bounded logs (two 10 MB files). No model/backend service was restarted.

Physical-phone camera/audio behavior and clinician/older-adult usability still
need hands-on acceptance; browser-engine tests are not a substitute for that.
