import { useEffect, useState, type RefObject } from 'react'

import { createLocalFaceDetector, type LocalFaceDetector } from '../camera/localFaceDetector'
import { assessFraming, measureLuminance, stabilizeHint, type PreflightHint, type StableHint } from '../camera/preflight'

const SAMPLE_INTERVAL_MS = 500
const LOAD_TIMEOUT_MS = 12_000
const MAX_FRAME_SIDE = 320

// Read afresh across async detection; visibility can change while awaiting a worker.
const documentHidden = () => document.visibilityState === 'hidden'

/** Advisory only; never controls recording or evaluates facial movement. */
export function useCameraPreflight(videoRef: RefObject<HTMLVideoElement | null>, active: boolean): PreflightHint {
  const [hint, setHint] = useState<PreflightHint>('checking')

  useEffect(() => {
    if (!active) return
    setHint('checking')
    const controller = new AbortController()
    let cancelled = false
    let detector: LocalFaceDetector | undefined
    let timer: ReturnType<typeof setTimeout> | undefined
    let loadTimer: ReturnType<typeof setTimeout> | undefined
    let stable: StableHint | undefined
    let lastVideoTime = -1
    let lastFrameAt = -1
    const canvas = document.createElement('canvas')
    let context: CanvasRenderingContext2D | null = null

    const release = () => {
      controller.abort()
      clearTimeout(timer)
      clearTimeout(loadTimer)
      try { detector?.close() } catch { /* Keep camera controls usable on SDK failure. */ }
      detector = undefined
      canvas.width = 0
      canvas.height = 0
    }
    const unavailable = () => {
      if (cancelled) return
      cancelled = true
      release()
      setHint('unavailable')
    }
    const sample = async () => {
      if (cancelled) return
      const video = videoRef.current
      const now = performance.now()
      if (!video || !context || !detector) { unavailable(); return }
      if (documentHidden() || video.readyState < 2 || !video.videoWidth || !video.videoHeight
        || video.currentTime === lastVideoTime) {
        // Do not leave a stale positive result on a stopped/hidden preview.
        if (lastFrameAt < 0 || now - lastFrameAt > 2000) { stable = undefined; setHint('checking') }
        timer = setTimeout(sample, SAMPLE_INTERVAL_MS)
        return
      }
      let pixels: Uint8ClampedArray | undefined
      try {
        lastVideoTime = video.currentTime
        lastFrameAt = now
        const scale = Math.min(1, MAX_FRAME_SIDE / Math.max(video.videoWidth, video.videoHeight))
        const width = Math.max(1, Math.round(video.videoWidth * scale))
        const height = Math.max(1, Math.round(video.videoHeight * scale))
        if (canvas.width !== width) canvas.width = width
        if (canvas.height !== height) canvas.height = height
        context.drawImage(video, 0, 0, width, height)
        pixels = context.getImageData(0, 0, width, height).data
        const faces = await detector.detect(canvas, now)
        if (cancelled) return
        if (documentHidden()) {
          stable = undefined
          setHint('checking')
          timer = setTimeout(sample, SAMPLE_INTERVAL_MS)
          return
        }
        const luminance = measureLuminance(pixels, width, height, faces.length === 1 ? faces[0] : undefined)
        const next = assessFraming({ width, height, faces, luminance }, stable?.hint)
        if (next === 'unavailable') { unavailable(); return }
        stable = stabilizeHint(stable, next)
        setHint(stable.hint)
      } catch {
        unavailable()
      } finally {
        // No frame is encoded, persisted, logged, or uploaded. Erase transient samples.
        pixels?.fill(0)
        try { context.clearRect(0, 0, canvas.width, canvas.height) } catch { /* Context may be lost. */ }
      }
      if (!cancelled) timer = setTimeout(sample, SAMPLE_INTERVAL_MS)
    }

    try { context = canvas.getContext('2d', { willReadFrequently: true }) } catch { /* Unsupported browser. */ }
    if (!context) unavailable()
    else {
      loadTimer = setTimeout(unavailable, LOAD_TIMEOUT_MS)
      void createLocalFaceDetector(controller.signal).then((loaded) => {
        if (cancelled) {
          try { loaded.close() } catch { /* Late SDK cleanup. */ }
          return
        }
        detector = loaded
        clearTimeout(loadTimer)
        timer = setTimeout(sample, SAMPLE_INTERVAL_MS)
      }).catch(unavailable)
    }

    return () => { cancelled = true; release() }
  }, [active, videoRef])

  return active ? hint : 'checking'
}
