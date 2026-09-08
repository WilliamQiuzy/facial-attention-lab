import { act, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { createLocalFaceDetector, type LocalFaceDetector } from '../camera/localFaceDetector'
import { useCameraPreflight } from './useCameraPreflight'

vi.mock('../camera/localFaceDetector', () => ({ createLocalFaceDetector: vi.fn() }))

const createDetector = vi.mocked(createLocalFaceDetector)

function videoRef() {
  const video = document.createElement('video')
  Object.defineProperties(video, {
    videoWidth: { value: 640 }, videoHeight: { value: 480 }, readyState: { value: 2 },
    currentTime: { configurable: true, get: () => Date.now() / 1000 },
  })
  return { current: video }
}

describe('useCameraPreflight lifecycle', () => {
  let detector: LocalFaceDetector
  let context: { drawImage: ReturnType<typeof vi.fn>; getImageData: ReturnType<typeof vi.fn>; clearRect: ReturnType<typeof vi.fn> }

  beforeEach(() => {
    vi.useFakeTimers()
    createDetector.mockReset()
    detector = { detect: vi.fn(async () => [{ x: 96, y: 48, width: 128, height: 144 }]), close: vi.fn() }
    createDetector.mockResolvedValue(detector)
    context = {
      drawImage: vi.fn(), clearRect: vi.fn(),
      getImageData: vi.fn(() => ({ data: new Uint8ClampedArray(320 * 240 * 4).fill(120) })),
    }
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(context as unknown as CanvasRenderingContext2D)
  })

  afterEach(() => { vi.useRealTimers() })

  it('does not load the model or draw frames when inactive', async () => {
    renderHook(() => useCameraPreflight(videoRef(), false))
    await act(async () => { await vi.advanceTimersByTimeAsync(2500) })
    expect(createDetector).not.toHaveBeenCalled()
    expect(context.drawImage).not.toHaveBeenCalled()
  })

  it('samples at most twice per second, resizes frames, and never uploads or stores frames', async () => {
    const fetchSpy = vi.spyOn(window, 'fetch')
    const storageSpy = vi.spyOn(Storage.prototype, 'setItem')
    const ref = videoRef()
    const { result, unmount } = renderHook(() => useCameraPreflight(ref, true))
    await act(async () => { await vi.advanceTimersByTimeAsync(2000) })
    expect(detector.detect).toHaveBeenCalledTimes(4)
    expect(context.drawImage).toHaveBeenCalledWith(ref.current, 0, 0, 320, 240)
    expect(context.clearRect).toHaveBeenCalled()
    expect(result.current).toBe('ready')
    expect(fetchSpy).not.toHaveBeenCalled()
    expect(storageSpy).not.toHaveBeenCalled()
    unmount()
    expect(detector.close).toHaveBeenCalledTimes(1)
    await act(async () => { await vi.advanceTimersByTimeAsync(2000) })
    expect(detector.detect).toHaveBeenCalledTimes(4)
  })

  it('closes immediately and cancels sampling when capture becomes inactive', async () => {
    const ref = videoRef()
    const { rerender, result } = renderHook(({ active }) => useCameraPreflight(ref, active), { initialProps: { active: true } })
    await act(async () => { await vi.advanceTimersByTimeAsync(1500) })
    rerender({ active: false })
    expect(result.current).toBe('checking')
    expect(detector.close).toHaveBeenCalledTimes(1)
    await act(async () => { await vi.advanceTimersByTimeAsync(5000) })
    expect(detector.detect).toHaveBeenCalledTimes(3)
  })

  it('aborts loading and closes a late detector without processing a stale frame', async () => {
    let resolve: (value: LocalFaceDetector) => void = () => {}
    createDetector.mockImplementation(() => new Promise((done) => { resolve = done }))
    const { unmount } = renderHook(() => useCameraPreflight(videoRef(), true))
    const signal = createDetector.mock.calls[0][0]
    unmount()
    expect(signal.aborted).toBe(true)
    await act(async () => { resolve(detector) })
    expect(detector.close).toHaveBeenCalledTimes(1)
    expect(detector.detect).not.toHaveBeenCalled()
    expect(vi.getTimerCount()).toBe(0)
  })

  it('reports model load failure as unavailable, not face valid', async () => {
    createDetector.mockRejectedValue(new Error('model unavailable'))
    const ref = videoRef()
    const { result } = renderHook(() => useCameraPreflight(ref, true))
    await act(async () => {})
    expect(result.current).toBe('unavailable')
    expect(vi.getTimerCount()).toBe(0)
  })

  it('bounds model loading and frees a detector that finishes after the timeout', async () => {
    let resolve: (value: LocalFaceDetector) => void = () => {}
    createDetector.mockImplementation(() => new Promise((done) => { resolve = done }))
    const ref = videoRef()
    const { result } = renderHook(() => useCameraPreflight(ref, true))
    await act(async () => { await vi.advanceTimersByTimeAsync(15000) })
    expect(result.current).toBe('unavailable')
    expect(createDetector.mock.calls[0][0].aborted).toBe(true)
    await act(async () => { resolve(detector) })
    expect(detector.close).toHaveBeenCalledTimes(1)
    expect(detector.detect).not.toHaveBeenCalled()
  })

  it('closes the detector and discards pixel data on detection failure', async () => {
    vi.mocked(detector.detect).mockImplementation(() => { throw new Error('WebGL lost') })
    const ref = videoRef()
    const { result } = renderHook(() => useCameraPreflight(ref, true))
    await act(async () => { await vi.advanceTimersByTimeAsync(1000) })
    expect(result.current).toBe('unavailable')
    expect(detector.close).toHaveBeenCalledTimes(1)
    expect(context.clearRect).toHaveBeenCalled()
    expect(vi.getTimerCount()).toBe(0)
  })

  it('never initializes a model when canvas sampling is unsupported', async () => {
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null)
    const ref = videoRef()
    const { result } = renderHook(() => useCameraPreflight(ref, true))
    await act(async () => {})
    expect(result.current).toBe('unavailable')
    expect(createDetector).not.toHaveBeenCalled()
  })

  it('does not sample hidden documents or duplicate video frames', async () => {
    const ref = videoRef()
    Object.defineProperty(ref.current, 'currentTime', { configurable: true, value: 1 })
    const hidden = vi.spyOn(document, 'visibilityState', 'get').mockReturnValue('hidden')
    renderHook(() => useCameraPreflight(ref, true))
    await act(async () => { await vi.advanceTimersByTimeAsync(1500) })
    expect(detector.detect).not.toHaveBeenCalled()
    hidden.mockReturnValue('visible')
    await act(async () => { await vi.advanceTimersByTimeAsync(1500) })
    expect(detector.detect).toHaveBeenCalledTimes(1)
  })

  it('resumes after the document hides during an in-flight frame without overlapping work', async () => {
    let resolve: (faces: []) => void = () => {}
    vi.mocked(detector.detect).mockImplementationOnce(() => new Promise((done) => { resolve = done }))
    const visibility = vi.spyOn(document, 'visibilityState', 'get').mockReturnValue('visible')
    const ref = videoRef()
    renderHook(() => useCameraPreflight(ref, true))
    await act(async () => { await vi.advanceTimersByTimeAsync(1500) })
    expect(detector.detect).toHaveBeenCalledTimes(1)
    visibility.mockReturnValue('hidden')
    await act(async () => { resolve([]) })
    visibility.mockReturnValue('visible')
    await act(async () => { await vi.advanceTimersByTimeAsync(1000) })
    expect(detector.detect).toHaveBeenCalledTimes(3)
  })

  it('discards late inference and clears the only pixel buffer after deactivation', async () => {
    let resolve: (faces: []) => void = () => {}
    vi.mocked(detector.detect).mockImplementationOnce(() => new Promise((done) => { resolve = done }))
    const pixels = new Uint8ClampedArray(320 * 240 * 4).fill(120)
    context.getImageData.mockReturnValue({ data: pixels })
    const ref = videoRef()
    const { result, rerender } = renderHook(({ active }) => useCameraPreflight(ref, active), { initialProps: { active: true } })
    await act(async () => { await vi.advanceTimersByTimeAsync(500) })
    rerender({ active: false })
    await act(async () => { resolve([]) })
    expect(result.current).toBe('checking')
    expect(detector.close).toHaveBeenCalledTimes(1)
    expect(pixels.every((value) => value === 0)).toBe(true)
    expect(vi.getTimerCount()).toBe(0)
  })
})
