/* Application-owned classic worker. See README.md for pinned upstream assets/licenses. */
let task = null
let initializing = false
let failed = false

function fail() {
  failed = true
  try { task?.close() } catch { /* A lost context must not prevent cleanup. */ }
  task = null
  self.postMessage({ type: 'error' })
}

self.onmessage = async ({ data }) => {
  if (data?.type === 'init') {
    if (initializing || task || failed) return
    initializing = true
    try {
      const base = new URL('.', self.location.href).href
      // The exact npm CommonJS bytes are self-hosted with a .js MIME-safe name.
      // A classic worker supports the upstream WASM loader's importScripts.
      self.exports = {}
      importScripts(`${base}mediapipe-0.10.32/vision_bundle.cjs.js`)
      const { FaceDetector, FilesetResolver } = self.exports
      const response = await fetch(`${base}blaze_face_short_range_v1.tflite`, { credentials: 'omit' })
      if (!response.ok) throw new Error('Local preflight model is unavailable')
      const model = new Uint8Array(await response.arrayBuffer())
      const fileset = await FilesetResolver.forVisionTasks(`${base}mediapipe-0.10.32`)
      task = await FaceDetector.createFromOptions(fileset, {
        baseOptions: { modelAssetBuffer: model, delegate: 'CPU' },
        runningMode: 'VIDEO',
        minDetectionConfidence: 0.6,
      })
      self.postMessage({ type: 'ready' })
    } catch { fail() }
    return
  }
  if (data?.type !== 'detect') {
    data?.bitmap?.close()
    return
  }
  try {
    if (!task || failed) throw new Error('Camera hints are not ready')
    const faces = task.detectForVideo(data.bitmap, data.timestampMs).detections.flatMap(({ boundingBox }) => boundingBox
      ? [{ x: boundingBox.originX, y: boundingBox.originY, width: boundingBox.width, height: boundingBox.height }]
      : [])
    self.postMessage({ type: 'result', id: data.id, faces })
  } catch { fail() }
  finally { data.bitmap?.close() }
}
