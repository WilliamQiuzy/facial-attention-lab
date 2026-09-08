import type { FaceBox } from './preflight'

export type LocalFaceDetector = {
  detect: (frame: HTMLCanvasElement, timestampMs: number) => Promise<FaceBox[]>
  close: () => void
}

type PendingFrame = {
  id: number
  resolve: (faces: FaceBox[]) => void
  reject: (reason: Error) => void
  timer: ReturnType<typeof setTimeout>
}

/** Inference and WASM initialization run in a same-origin, independently terminable worker. */
export async function createLocalFaceDetector(signal: AbortSignal): Promise<LocalFaceDetector> {
  signal.throwIfAborted()
  if (typeof Worker === 'undefined' || typeof OffscreenCanvas === 'undefined'
    || typeof createImageBitmap === 'undefined' || typeof WebAssembly === 'undefined') {
    throw new Error('Local camera hints are not supported')
  }
  const worker = new Worker(`${import.meta.env.BASE_URL}preflight/preflight.worker.js`)
  return new Promise<LocalFaceDetector>((resolve, reject) => {
    let closed = false
    let ready = false
    let sequence = 0
    let pending: PendingFrame | undefined
    let loadTimer: ReturnType<typeof setTimeout>

    const shutdown = (reason = new Error('Camera hints stopped')) => {
      if (closed) return
      closed = true
      clearTimeout(loadTimer)
      if (pending) { clearTimeout(pending.timer); pending.reject(reason); pending = undefined }
      signal.removeEventListener('abort', abort)
      worker.onmessage = null
      worker.onerror = null
      worker.onmessageerror = null
      // Termination stops even synchronous WASM work and frees transferred bitmap/graph memory.
      worker.terminate()
      reject(reason)
    }
    const abort = () => shutdown(new Error('Camera hints cancelled'))
    const detector: LocalFaceDetector = {
      close: () => shutdown(),
      detect: (frame, timestampMs) => {
        if (closed || !ready || pending) return Promise.reject(new Error('Camera hints are not ready for a frame'))
        const id = ++sequence
        return new Promise<FaceBox[]>((resolveFrame, rejectFrame) => {
          pending = {
            id, resolve: resolveFrame, reject: rejectFrame,
            timer: setTimeout(() => shutdown(new Error('Camera hint detection timed out')), 8000),
          }
          let bitmapPromise: Promise<ImageBitmap>
          try { bitmapPromise = createImageBitmap(frame) }
          catch { shutdown(new Error('Camera frame sampling failed')); return }
          void bitmapPromise.then((bitmap) => {
            if (closed || pending?.id !== id) { bitmap.close(); return }
            try {
              worker.postMessage({ type: 'detect', id, bitmap, timestampMs }, [bitmap])
            } catch {
              bitmap.close()
              shutdown(new Error('Camera frame transfer failed'))
            }
          }).catch(() => shutdown(new Error('Camera frame sampling failed')))
        })
      },
    }
    worker.onmessage = ({ data }) => {
      if (closed) return
      if (data?.type === 'ready' && !ready) {
        ready = true
        clearTimeout(loadTimer)
        resolve(detector)
      } else if (data?.type === 'result' && pending && data.id === pending.id) {
        const current = pending
        pending = undefined
        clearTimeout(current.timer)
        current.resolve(data.faces as FaceBox[])
      } else if (data?.type === 'error') shutdown(new Error('Local camera hints are unavailable'))
    }
    worker.onerror = (event) => { event.preventDefault?.(); shutdown(new Error('Camera hint worker failed')) }
    worker.onmessageerror = () => shutdown(new Error('Camera hint message failed'))
    signal.addEventListener('abort', abort, { once: true })
    loadTimer = setTimeout(() => shutdown(new Error('Camera hint loading timed out')), 12_000)
    try { worker.postMessage({ type: 'init' }) } catch { shutdown(new Error('Camera hint worker failed')) }
  })
}
