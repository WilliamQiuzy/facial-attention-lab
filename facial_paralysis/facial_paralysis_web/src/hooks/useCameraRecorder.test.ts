import { act, renderHook, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { selectSupportedVideoMimeType, stopMediaStream, useCameraRecorder } from './useCameraRecorder'

afterEach(() => {
  vi.useRealTimers()
  vi.unstubAllGlobals()
})

describe('camera recorder helpers', () => {
  it('selects the first supported MediaRecorder MIME type', () => {
    const isTypeSupported = vi.fn((type: string) => type === 'video/webm;codecs=vp8')
    expect(selectSupportedVideoMimeType(isTypeSupported)).toBe('video/webm;codecs=vp8')
  })

  it('returns an empty MIME type when the browser reports no supported candidate', () => {
    expect(selectSupportedVideoMimeType(() => false)).toBe('')
  })

  it('stops every active media track during cleanup', () => {
    const tracks = [{ stop: vi.fn() }, { stop: vi.fn() }]
    stopMediaStream({ getTracks: () => tracks } as unknown as MediaStream)
    expect(tracks.every((track) => track.stop.mock.calls.length === 1)).toBe(true)
  })

  it('releases the camera stream after recording stops', async () => {
    const stopTrack = vi.fn()
    const stream = { getTracks: () => [{ stop: stopTrack }] } as unknown as MediaStream

    class FakeMediaRecorder {
      static isTypeSupported = () => true
      state: RecordingState = 'inactive'
      mimeType = 'video/webm'
      ondataavailable: ((event: BlobEvent) => void) | null = null
      onstop: (() => void) | null = null
      onerror: (() => void) | null = null

      start() {
        this.state = 'recording'
      }

      stop() {
        this.state = 'inactive'
        this.ondataavailable?.({ data: new Blob(['video'], { type: this.mimeType }) } as BlobEvent)
        this.onstop?.()
      }
    }

    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: { getUserMedia: vi.fn().mockResolvedValue(stream) },
    })
    vi.stubGlobal('MediaRecorder', FakeMediaRecorder)

    const { result } = renderHook(() => useCameraRecorder())
    await act(async () => result.current.enableCamera())
    act(() => result.current.startRecording())
    act(() => result.current.stopRecording())

    await waitFor(() => expect(result.current.status).toBe('recorded'))
    expect(stopTrack).toHaveBeenCalledTimes(1)
    expect(result.current.recordingFile).not.toBeNull()
  })

  it('discards an interrupted recording without publishing a partial file', async () => {
    const stopTrack = vi.fn()
    const stream = { getTracks: () => [{ stop: stopTrack }] } as unknown as MediaStream

    class FakeMediaRecorder {
      static isTypeSupported = () => true
      static latest: FakeMediaRecorder | null = null
      state: RecordingState = 'inactive'
      mimeType = 'video/webm'
      ondataavailable: ((event: BlobEvent) => void) | null = null
      onstop: (() => void) | null = null
      onerror: (() => void) | null = null

      constructor() {
        FakeMediaRecorder.latest = this
      }

      start() {
        this.state = 'recording'
      }

      stop() {
        this.state = 'inactive'
        this.ondataavailable?.({ data: new Blob(['partial'], { type: this.mimeType }) } as BlobEvent)
        this.onstop?.()
      }
    }

    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: { getUserMedia: vi.fn().mockResolvedValue(stream) },
    })
    vi.stubGlobal('MediaRecorder', FakeMediaRecorder)

    const { result } = renderHook(() => useCameraRecorder())
    await act(async () => result.current.enableCamera())
    act(() => result.current.startRecording())
    const staleDataAvailable = FakeMediaRecorder.latest?.ondataavailable
    const staleStop = FakeMediaRecorder.latest?.onstop
    const staleError = FakeMediaRecorder.latest?.onerror
    act(() => result.current.discardRecording())
    act(() => {
      staleDataAvailable?.({ data: new Blob(['late-partial'], { type: 'video/webm' }) } as BlobEvent)
      staleStop?.()
      staleError?.()
    })

    await waitFor(() => expect(result.current.status).toBe('idle'))
    expect(result.current.recordingFile).toBeNull()
    expect(stopTrack).toHaveBeenCalledTimes(1)
  })

  it('releases a late camera stream when the component unmounts during permission request', async () => {
    const stopTrack = vi.fn()
    const stream = { getTracks: () => [{ stop: stopTrack }] } as unknown as MediaStream
    let resolveStream: ((stream: MediaStream) => void) | undefined
    const getUserMedia = vi.fn(
      () =>
        new Promise<MediaStream>((resolve) => {
          resolveStream = resolve
        }),
    )
    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: { getUserMedia },
    })

    const { result, unmount } = renderHook(() => useCameraRecorder())
    let pending: Promise<void> | undefined
    act(() => {
      pending = result.current.enableCamera()
    })
    unmount()
    resolveStream?.(stream)
    await act(async () => pending)

    expect(stopTrack).toHaveBeenCalledTimes(1)
  })

  it('surfaces synchronous MediaRecorder setup failure and releases the camera', async () => {
    const stopTrack = vi.fn()
    const stream = { getTracks: () => [{ stop: stopTrack }] } as unknown as MediaStream

    class ThrowingMediaRecorder {
      static isTypeSupported = () => true

      constructor() {
        throw new DOMException('unsupported encoder', 'NotSupportedError')
      }
    }

    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: { getUserMedia: vi.fn().mockResolvedValue(stream) },
    })
    vi.stubGlobal('MediaRecorder', ThrowingMediaRecorder)

    const { result } = renderHook(() => useCameraRecorder())
    await act(async () => result.current.enableCamera())
    expect(() => act(() => result.current.startRecording())).not.toThrow()

    expect(result.current.status).toBe('error')
    expect(result.current.error).toMatch(/record/i)
    expect(stopTrack).toHaveBeenCalledTimes(1)
  })

  it('fails closed when MediaRecorder never confirms that recording started', async () => {
    vi.useFakeTimers()
    const stopTrack = vi.fn()
    const stream = { getTracks: () => [{ stop: stopTrack }] } as unknown as MediaStream

    class SilentMediaRecorder {
      static isTypeSupported = () => true
      state: RecordingState = 'inactive'
      mimeType = 'video/webm'
      ondataavailable: ((event: BlobEvent) => void) | null = null
      onstart: (() => void) | null = null
      onstop: (() => void) | null = null
      onerror: (() => void) | null = null

      start() {
        this.state = 'recording'
      }

      stop() {
        this.state = 'inactive'
      }
    }

    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: { getUserMedia: vi.fn().mockResolvedValue(stream) },
    })
    vi.stubGlobal('MediaRecorder', SilentMediaRecorder)

    const { result } = renderHook(() => useCameraRecorder())
    await act(async () => result.current.enableCamera())
    act(() => result.current.startRecording())
    expect(result.current.status).toBe('starting')

    act(() => vi.advanceTimersByTime(5_000))

    expect(result.current.status).toBe('error')
    expect(result.current.error).toMatch(/did not start in time/i)
    expect(result.current.recordingFile).toBeNull()
    expect(stopTrack).toHaveBeenCalledTimes(1)
  })

  it('fails closed when MediaRecorder never confirms finalization', async () => {
    vi.useFakeTimers()
    const stopTrack = vi.fn()
    const stream = { getTracks: () => [{ stop: stopTrack }] } as unknown as MediaStream

    class NeverFinalizesMediaRecorder {
      static isTypeSupported = () => true
      state: RecordingState = 'inactive'
      mimeType = 'video/webm'
      ondataavailable: ((event: BlobEvent) => void) | null = null
      onstart: (() => void) | null = null
      onstop: (() => void) | null = null
      onerror: (() => void) | null = null

      start() {
        this.state = 'recording'
        this.onstart?.()
      }

      stop() {
        this.state = 'inactive'
      }
    }

    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: { getUserMedia: vi.fn().mockResolvedValue(stream) },
    })
    vi.stubGlobal('MediaRecorder', NeverFinalizesMediaRecorder)

    const { result } = renderHook(() => useCameraRecorder())
    await act(async () => result.current.enableCamera())
    act(() => result.current.startRecording())
    expect(result.current.status).toBe('recording')
    act(() => result.current.stopRecording())

    act(() => vi.advanceTimersByTime(5_000))

    expect(result.current.status).toBe('error')
    expect(result.current.error).toMatch(/did not finish in time/i)
    expect(result.current.recordingFile).toBeNull()
    expect(stopTrack).toHaveBeenCalledTimes(1)
  })

  it('fails closed when MediaRecorder stops before finalization is requested', async () => {
    const stopTrack = vi.fn()
    const stream = { getTracks: () => [{ stop: stopTrack }] } as unknown as MediaStream

    class UnexpectedStopMediaRecorder {
      static isTypeSupported = () => true
      static latest: UnexpectedStopMediaRecorder | null = null
      state: RecordingState = 'inactive'
      mimeType = 'video/webm'
      ondataavailable: ((event: BlobEvent) => void) | null = null
      onstart: (() => void) | null = null
      onstop: (() => void) | null = null
      onerror: (() => void) | null = null

      constructor() {
        UnexpectedStopMediaRecorder.latest = this
      }

      start() {
        this.state = 'recording'
        this.onstart?.()
      }

      stop() {
        this.state = 'inactive'
        this.onstop?.()
      }
    }

    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: { getUserMedia: vi.fn().mockResolvedValue(stream) },
    })
    vi.stubGlobal('MediaRecorder', UnexpectedStopMediaRecorder)

    const { result } = renderHook(() => useCameraRecorder())
    await act(async () => result.current.enableCamera())
    act(() => result.current.startRecording())
    act(() => UnexpectedStopMediaRecorder.latest?.onstop?.())

    expect(result.current.status).toBe('error')
    expect(result.current.error).toMatch(/stopped unexpectedly/i)
    expect(result.current.recordingFile).toBeNull()
    expect(stopTrack).toHaveBeenCalledTimes(1)
  })
})
