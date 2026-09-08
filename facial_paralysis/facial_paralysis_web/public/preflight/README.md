# Optional, local camera preflight assets

These files support **non-blocking framing hints**, not diagnosis, face identity,
movement scoring, or recording eligibility. Only boxes and transient brightness
are used. No image/frame is uploaded, logged, encoded, or persisted by preflight.
Camera/video submission in the main application is a separate user action.

## Provenance and exact versions

- Runtime: official Google `@mediapipe/tasks-vision@0.10.32` npm package.
  [Exact tarball](https://registry.npmjs.org/@mediapipe/tasks-vision/-/tasks-vision-0.10.32.tgz).
  The four `mediapipe-0.10.32/` files are unmodified copies from that package's
  `wasm/` directory. `vision_bundle.cjs.js` is the unmodified `vision_bundle.cjs`
  from that package, renamed only to serve a JavaScript MIME type. The dependency
  stays pinned in package.json and the lockfile for reproducible vendoring.
  `preflight.worker.js` is application-owned source, covered by worker tests.
- Model: Google BlazeFace short-range float16 **version 1**.
  [Exact model download](https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite).
  [Official model guide](https://developers.google.com/edge/mediapipe/solutions/vision/face_detector)
  and [model card](https://storage.googleapis.com/mediapipe-assets/MediaPipe%20BlazeFace%20Model%20Card%20%28Short%20Range%29.pdf).
- Both the npm package and model are Apache-2.0; the model card explicitly
  identifies that license. `LICENSE-Apache-2.0.txt` is copied from the
  [MediaPipe v0.10.32 license](https://github.com/google-ai-edge/mediapipe/blob/v0.10.32/LICENSE),
  including its upstream notices. Runtime source copyright headers are retained.
- Acquired and checksummed 2026-09-08. `SHA256SUMS` records every runtime/model
  file. From this directory: `shasum -a 256 -c SHA256SUMS`.

Both SIMD and non-SIMD runtime variants are included for browser compatibility;
the resolver downloads only the supported variant. Total vendored footprint is
about 22 MiB (the model itself is about 225 KiB). No CDN is used at runtime.

## Operational boundaries

- The detector is loaded only while camera preview hints are active. The CPU
  delegate and WASM initialization run inside a same-origin classic Worker,
  never on the UI thread. It processes at most two downsized frames per second
  (longest side 320 px), with at most one transferred ImageBitmap in flight.
  No checks run during recording. A hidden document or unchanged frame is skipped.
- Camera boxes produce only face-visible, single-face, center, and distance hints.
  Luma is measured inside the detected face box, or the central half of the
  actual video when no single face is available. A bright background therefore
  cannot substitute for a dark detected face. At most one transient pixel buffer
  is retained while awaiting a box result, then zeroed; the canvas is cleared.
  The worker explicitly closes each processed/rejected ImageBitmap. Termination
  frees any transferred bitmap and graph memory during cancelled WASM work.
- Heuristic thresholds are composition aids, not clinical/photometric validation:
  dark-enter luma 42/255, dark-exit 60/255; small-face-enter minimum side ratio
  0.24, exit 0.30; center-enter offset 0.17 of the frame, exit 0.13; 2% edge margin.
  Three consecutive samples are required to change the visible hint.
- Detection can miss faces, especially at extreme angles, far away, in poor light,
  or outside its training distribution. The model is not clinically validated
  for facial-paralysis patients. Hints are optional and cannot disable recording.
- Model-load timeout is 12 seconds; in-flight detection timeout is 8 seconds.
  Unsupported Worker/OffscreenCanvas/ImageBitmap/WASM, load failure, and detection
  failure show unavailable. There is no UI-blocking fallback. Current browsers
  with worker WebGL support can run the detector; older Safari may be unavailable.
  Deactivation/unmount terminates the worker immediately, including in-progress
  WASM initialization/inference. Late bitmap creation is closed without transfer;
  pending results, timers, and event listeners are cancelled.
- The Nginx CSP adds only `'wasm-unsafe-eval'` to `script-src 'self'` because
  WebAssembly compilation needs it. It does **not** add JavaScript `'unsafe-eval'`,
  external origins, blob scripts, or outbound frame traffic. `worker-src 'self'`
  explicitly limits worker creation to same-origin files. Missing files below
  `/preflight/` return 404 instead of SPA HTML.

Runtime integration follows the [official Web guide](https://developers.google.com/edge/mediapipe/solutions/vision/face_detector/web_js).
