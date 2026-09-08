import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { runInNewContext } from 'node:vm'
import { describe, expect, it, vi } from 'vitest'

function workerHarness() {
  const task = { detectForVideo: vi.fn(() => ({ detections: [{ boundingBox: { originX: 1, originY: 2, width: 3, height: 4 }, keypoints: [{ x: 0.5, y: 0.5 }] }] })), close: vi.fn() }
  const createFromOptions = vi.fn(async () => task)
  const forVisionTasks = vi.fn(async () => ({}))
  const scope = {
    location: { href: 'https://capture.example/preflight/preflight.worker.js' },
    exports: {} as Record<string, unknown>,
    onmessage: null as null | ((event: { data: unknown }) => Promise<void>),
    postMessage: vi.fn(), close: vi.fn(),
  }
  const importScripts = vi.fn(() => {
    scope.exports = { FaceDetector: { createFromOptions }, FilesetResolver: { forVisionTasks } }
  })
  const fetch = vi.fn(async () => ({ ok: true, arrayBuffer: async () => new ArrayBuffer(8) }))
  runInNewContext(readFileSync(resolve(process.cwd(), 'public/preflight/preflight.worker.js'), 'utf8'), {
    self: scope, importScripts, fetch, URL, Uint8Array,
  })
  return { scope, task, importScripts, fetch, createFromOptions, forVisionTasks }
}

describe('classic preflight worker (real worker source)', () => {
  it('initializes pinned same-origin assets with CPU and returns boxes only', async () => {
    const { scope, importScripts, fetch, createFromOptions, forVisionTasks } = workerHarness()
    await scope.onmessage!({ data: { type: 'init' } })
    expect(importScripts).toHaveBeenCalledWith('https://capture.example/preflight/mediapipe-0.10.32/vision_bundle.cjs.js')
    expect(fetch).toHaveBeenCalledWith('https://capture.example/preflight/blaze_face_short_range_v1.tflite', { credentials: 'omit' })
    expect(forVisionTasks).toHaveBeenCalledWith('https://capture.example/preflight/mediapipe-0.10.32')
    expect(createFromOptions).toHaveBeenCalledWith(expect.anything(), expect.objectContaining({ baseOptions: { modelAssetBuffer: expect.any(Uint8Array), delegate: 'CPU' }, runningMode: 'VIDEO' }))
    const bitmap = { close: vi.fn() }
    await scope.onmessage!({ data: { type: 'detect', id: 7, timestampMs: 100, bitmap } })
    expect(scope.postMessage).toHaveBeenLastCalledWith({ type: 'result', id: 7, faces: [{ x: 1, y: 2, width: 3, height: 4 }] })
    expect(bitmap.close).toHaveBeenCalledTimes(1)
  })

  it('closes frames received before readiness', async () => {
    const { scope } = workerHarness()
    const bitmap = { close: vi.fn() }
    await scope.onmessage!({ data: { type: 'detect', id: 1, timestampMs: 1, bitmap } })
    expect(bitmap.close).toHaveBeenCalledTimes(1)
    expect(scope.postMessage).toHaveBeenLastCalledWith({ type: 'error' })
  })

  it('closes bitmap and task when inference throws', async () => {
    const { scope, task } = workerHarness()
    await scope.onmessage!({ data: { type: 'init' } })
    task.detectForVideo.mockImplementation(() => { throw new Error('lost context') })
    const bitmap = { close: vi.fn() }
    await scope.onmessage!({ data: { type: 'detect', id: 1, timestampMs: 1, bitmap } })
    expect(bitmap.close).toHaveBeenCalledTimes(1)
    expect(task.close).toHaveBeenCalledTimes(1)
    expect(scope.postMessage).toHaveBeenLastCalledWith({ type: 'error' })
  })

  it('reports unavailable without creating a task after model-fetch failure', async () => {
    const { scope, fetch, createFromOptions } = workerHarness()
    fetch.mockResolvedValue({ ok: false, arrayBuffer: async () => new ArrayBuffer(0) })
    await scope.onmessage!({ data: { type: 'init' } })
    expect(createFromOptions).not.toHaveBeenCalled()
    expect(scope.postMessage).toHaveBeenLastCalledWith({ type: 'error' })
  })
})
