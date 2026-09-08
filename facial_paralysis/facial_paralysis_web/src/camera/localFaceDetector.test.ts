import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { createLocalFaceDetector } from './localFaceDetector'

class FakeWorker {
  static instances: FakeWorker[] = []
  onmessage: ((event: MessageEvent) => void) | null = null
  onerror: ((event: ErrorEvent) => void) | null = null
  onmessageerror: (() => void) | null = null
  terminate = vi.fn()
  postMessage = vi.fn()
  constructor(public url: string) { FakeWorker.instances.push(this) }
  emit(data: unknown) { this.onmessage?.({ data } as MessageEvent) }
}

describe('self-hosted worker detector', () => {
  const bitmap = { close: vi.fn() }
  beforeEach(() => {
    vi.useFakeTimers()
    vi.clearAllMocks()
    FakeWorker.instances = []
    vi.stubGlobal('Worker', FakeWorker)
    vi.stubGlobal('OffscreenCanvas', class {})
    vi.stubGlobal('createImageBitmap', vi.fn().mockResolvedValue(bitmap))
  })
  afterEach(() => { vi.useRealTimers(); vi.unstubAllGlobals() })

  async function ready(signal = new AbortController().signal) {
    const pending = createLocalFaceDetector(signal)
    const worker = FakeWorker.instances.at(-1)!
    worker.emit({ type: 'ready' })
    return { detector: await pending, worker }
  }

  it('loads a same-origin classic worker and sends only one transferable frame at a time', async () => {
    const { detector, worker } = await ready()
    expect(worker.url).toBe('/preflight/preflight.worker.js')
    expect(worker.postMessage).toHaveBeenCalledWith({ type: 'init' })
    const pending = detector.detect(document.createElement('canvas'), 10)
    await expect(detector.detect(document.createElement('canvas'), 11)).rejects.toThrow()
    await vi.advanceTimersByTimeAsync(0)
    expect(worker.postMessage).toHaveBeenLastCalledWith({ type: 'detect', id: 1, bitmap, timestampMs: 10 }, [bitmap])
    worker.emit({ type: 'result', id: 1, faces: [{ x: 10, y: 20, width: 30, height: 40 }] })
    await expect(pending).resolves.toEqual([{ x: 10, y: 20, width: 30, height: 40 }])
    detector.close()
    detector.close()
    expect(worker.terminate).toHaveBeenCalledTimes(1)
  })

  it('terminates immediately when initialization is cancelled', async () => {
    const controller = new AbortController()
    const pending = createLocalFaceDetector(controller.signal)
    const rejection = expect(pending).rejects.toThrow()
    controller.abort()
    await rejection
    expect(FakeWorker.instances[0].terminate).toHaveBeenCalledTimes(1)
  })

  it('rejects worker load failure and releases worker listeners', async () => {
    const pending = createLocalFaceDetector(new AbortController().signal)
    const rejection = expect(pending).rejects.toThrow()
    const worker = FakeWorker.instances[0]
    worker.emit({ type: 'error' })
    await rejection
    expect(worker.terminate).toHaveBeenCalledTimes(1)
    expect(worker.onmessage).toBeNull()
    expect(worker.onerror).toBeNull()
  })

  it('closes a bitmap created after cancellation without transferring it', async () => {
    const controller = new AbortController()
    const { detector, worker } = await ready(controller.signal)
    let resolve: (value: ImageBitmap) => void = () => {}
    vi.mocked(createImageBitmap).mockImplementation(() => new Promise((done) => { resolve = done }))
    const pending = detector.detect(document.createElement('canvas'), 1)
    const rejection = expect(pending).rejects.toThrow()
    controller.abort()
    resolve(bitmap as unknown as ImageBitmap)
    await rejection
    expect(bitmap.close).toHaveBeenCalledTimes(1)
    expect(worker.postMessage).toHaveBeenCalledTimes(1)
    expect(worker.terminate).toHaveBeenCalledTimes(1)
  })

  it('closes an untransferred bitmap if postMessage fails', async () => {
    const { detector, worker } = await ready()
    worker.postMessage.mockImplementation(() => { throw new Error('transfer failed') })
    await expect(detector.detect(document.createElement('canvas'), 1)).rejects.toThrow()
    expect(bitmap.close).toHaveBeenCalledTimes(1)
    expect(worker.terminate).toHaveBeenCalledTimes(1)
  })

  it('terminates immediately if the browser throws synchronously while sampling a bitmap', async () => {
    const { detector, worker } = await ready()
    vi.mocked(createImageBitmap).mockImplementation(() => { throw new Error('invalid canvas') })
    await expect(detector.detect(document.createElement('canvas'), 1)).rejects.toThrow()
    expect(worker.terminate).toHaveBeenCalledTimes(1)
    expect(vi.getTimerCount()).toBe(0)
  })

  it('bounds an in-flight inference and terminates a stalled worker', async () => {
    const { detector, worker } = await ready()
    const pending = detector.detect(document.createElement('canvas'), 1)
    const rejection = expect(pending).rejects.toThrow()
    await vi.advanceTimersByTimeAsync(8000)
    await rejection
    expect(worker.terminate).toHaveBeenCalledTimes(1)
    expect(vi.getTimerCount()).toBe(0)
  })

  it('cancels pending inference and ignores stale results after close', async () => {
    const { detector, worker } = await ready()
    const pending = detector.detect(document.createElement('canvas'), 1)
    const rejection = expect(pending).rejects.toThrow()
    await vi.advanceTimersByTimeAsync(0)
    const staleMessage = worker.onmessage
    detector.close()
    staleMessage?.({ data: { type: 'result', id: 1, faces: [] } } as MessageEvent)
    await rejection
    expect(worker.terminate).toHaveBeenCalledTimes(1)
  })

  it('fails truthfully without a supported worker bitmap pipeline', async () => {
    vi.stubGlobal('OffscreenCanvas', undefined)
    await expect(createLocalFaceDetector(new AbortController().signal)).rejects.toThrow()
    expect(FakeWorker.instances).toHaveLength(0)
  })
})
