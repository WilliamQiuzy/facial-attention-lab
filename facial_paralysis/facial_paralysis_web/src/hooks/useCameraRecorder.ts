import { useCallback, useEffect, useRef, useState } from 'react'

const MIME_CANDIDATES = [
  'video/webm;codecs=vp9',
  'video/webm;codecs=vp8',
  'video/webm',
  'video/mp4',
] as const

const RECORDER_START_TIMEOUT_MS = 5_000
const RECORDER_FINALIZE_TIMEOUT_MS = 5_000

export function selectSupportedVideoMimeType(
  isTypeSupported: (mimeType: string) => boolean = MediaRecorder.isTypeSupported,
): string {
  return MIME_CANDIDATES.find((mimeType) => isTypeSupported(mimeType)) ?? ''
}

export function stopMediaStream(stream: MediaStream | null): void {
  stream?.getTracks().forEach((track) => track.stop())
}

export type CameraStatus = 'idle' | 'requesting' | 'ready' | 'starting' | 'recording' | 'recorded' | 'error'

export interface CameraRecorderState {
  readonly status: CameraStatus
  readonly error: string | null
  readonly recordingFile: File | null
  readonly recordingStartedAtMs: number | null
  readonly videoRef: React.RefObject<HTMLVideoElement | null>
  readonly enableCamera: () => Promise<void>
  readonly startRecording: () => void
  readonly stopRecording: () => void
  readonly discardRecording: () => void
  readonly resetRecording: () => void
  readonly closeCamera: () => void
}

function cameraErrorMessage(error: unknown): string {
  if (error instanceof DOMException && error.name === 'NotAllowedError') {
    return 'Camera permission was denied. Allow access in browser settings or upload a LifeLink Face video.'
  }
  if (error instanceof DOMException && error.name === 'NotFoundError') {
    return 'No front-facing camera was found. Connect a camera or upload a LifeLink Face video.'
  }
  return error instanceof Error
    ? error.message
    : 'The camera could not be started. Upload a LifeLink Face video instead.'
}

export function useCameraRecorder(): CameraRecorderState {
  const [status, setStatus] = useState<CameraStatus>('idle')
  const [error, setError] = useState<string | null>(null)
  const [recordingFile, setRecordingFile] = useState<File | null>(null)
  const [recordingStartedAtMs, setRecordingStartedAtMs] = useState<number | null>(null)
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const recorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const requestGenerationRef = useRef(0)
  const recordingGenerationRef = useRef(0)
  const commitGenerationRef = useRef<number | null>(null)
  const startWatchdogRef = useRef<number | null>(null)
  const finalizationWatchdogRef = useRef<number | null>(null)
  const mountedRef = useRef(true)

  const clearStartWatchdog = useCallback(() => {
    if (startWatchdogRef.current !== null) {
      window.clearTimeout(startWatchdogRef.current)
      startWatchdogRef.current = null
    }
  }, [])

  const clearFinalizationWatchdog = useCallback(() => {
    if (finalizationWatchdogRef.current !== null) {
      window.clearTimeout(finalizationWatchdogRef.current)
      finalizationWatchdogRef.current = null
    }
  }, [])

  const releaseActiveCamera = useCallback(() => {
    stopMediaStream(streamRef.current)
    streamRef.current = null
    if (videoRef.current) videoRef.current.srcObject = null
  }, [])

  const discardActiveRecorder = useCallback(() => {
    clearStartWatchdog()
    clearFinalizationWatchdog()
    recordingGenerationRef.current += 1
    commitGenerationRef.current = null
    chunksRef.current = []
    const recorder = recorderRef.current
    recorderRef.current = null
    if (recorder) {
      recorder.ondataavailable = null
      recorder.onstop = null
      recorder.onerror = null
      recorder.onstart = null
      if (recorder.state !== 'inactive') {
        try {
          recorder.stop()
        } catch {
          // The recording is already invalidated; resource cleanup continues below.
        }
      }
    }
    releaseActiveCamera()
  }, [clearFinalizationWatchdog, clearStartWatchdog, releaseActiveCamera])

  const closeCamera = useCallback(() => {
    requestGenerationRef.current += 1
    discardActiveRecorder()
    setRecordingStartedAtMs(null)
    setStatus((current) => (current === 'recorded' ? current : 'idle'))
  }, [discardActiveRecorder])

  const enableCamera = useCallback(async () => {
    const requestGeneration = requestGenerationRef.current + 1
    requestGenerationRef.current = requestGeneration
    setError(null)
    if (!navigator.mediaDevices?.getUserMedia) {
      setError('Camera capture is unavailable in this browser. Upload a LifeLink Face video instead.')
      setStatus('error')
      return
    }
    setStatus('requesting')
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'user',
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      })
      if (!mountedRef.current || requestGenerationRef.current !== requestGeneration) {
        stopMediaStream(stream)
        return
      }
      stopMediaStream(streamRef.current)
      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play().catch(() => undefined)
      }
      setStatus('ready')
    } catch (caught) {
      if (!mountedRef.current || requestGenerationRef.current !== requestGeneration) return
      setError(cameraErrorMessage(caught))
      setStatus('error')
    }
  }, [])

  const startRecording = useCallback(() => {
    const stream = streamRef.current
    if (!stream || typeof MediaRecorder === 'undefined') {
      setError('Video recording is unavailable in this browser.')
      setStatus('error')
      return
    }
    const generation = recordingGenerationRef.current + 1
    recordingGenerationRef.current = generation
    commitGenerationRef.current = null
    chunksRef.current = []
    setRecordingFile(null)
    setRecordingStartedAtMs(null)
    setError(null)
    const mimeType = selectSupportedVideoMimeType()
    let recorder: MediaRecorder | null = null
    try {
      recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined)
      recorderRef.current = recorder
      recorder.ondataavailable = (event) => {
        if (recordingGenerationRef.current !== generation) return
        if (event.data.size > 0) chunksRef.current.push(event.data)
      }
      recorder.onstart = () => {
        if (!mountedRef.current || recordingGenerationRef.current !== generation) return
        clearStartWatchdog()
        setRecordingStartedAtMs(performance.now())
        setStatus('recording')
      }
      recorder.onstop = () => {
        if (!mountedRef.current || recordingGenerationRef.current !== generation) return
        clearStartWatchdog()
        clearFinalizationWatchdog()
        if (commitGenerationRef.current !== generation) {
          recordingGenerationRef.current += 1
          commitGenerationRef.current = null
          chunksRef.current = []
          setRecordingStartedAtMs(null)
          releaseActiveCamera()
          recorderRef.current = null
          setError('The video recorder stopped unexpectedly. The incomplete recording was discarded.')
          setStatus('error')
          return
        }
        const type = recorder?.mimeType || mimeType || 'video/webm'
        const extension = type.includes('mp4') ? 'mp4' : 'webm'
        const blob = new Blob(chunksRef.current, { type })
        chunksRef.current = []
        setRecordingStartedAtMs(null)
        commitGenerationRef.current = null
        if (blob.size === 0) {
          releaseActiveCamera()
          recorderRef.current = null
          setError('The browser ended the recording without producing video data. Please record again.')
          setStatus('error')
          return
        }
        const file = new File([blob], `faces-capture-${Date.now()}.${extension}`, {
          type,
          lastModified: Date.now(),
        })
        releaseActiveCamera()
        recorderRef.current = null
        setRecordingFile(file)
        setStatus('recorded')
      }
      recorder.onerror = () => {
        if (recordingGenerationRef.current !== generation) return
        clearStartWatchdog()
        clearFinalizationWatchdog()
        recordingGenerationRef.current += 1
        commitGenerationRef.current = null
        chunksRef.current = []
        setRecordingStartedAtMs(null)
        releaseActiveCamera()
        recorderRef.current = null
        setError('The browser could not record this camera stream.')
        setStatus('error')
      }
      setStatus('starting')
      startWatchdogRef.current = window.setTimeout(() => {
        if (
          !mountedRef.current ||
          recordingGenerationRef.current !== generation ||
          recorderRef.current !== recorder
        ) return
        discardActiveRecorder()
        setRecordingStartedAtMs(null)
        setError('The video recorder did not start in time. The incomplete recording was discarded.')
        setStatus('error')
      }, RECORDER_START_TIMEOUT_MS)
      recorder.start(500)
    } catch {
      if (recorder) {
        recorder.ondataavailable = null
        recorder.onstop = null
        recorder.onerror = null
        recorder.onstart = null
      }
      recordingGenerationRef.current += 1
      clearStartWatchdog()
      clearFinalizationWatchdog()
      commitGenerationRef.current = null
      chunksRef.current = []
      setRecordingStartedAtMs(null)
      releaseActiveCamera()
      recorderRef.current = null
      setError('Video recording could not start with this camera and browser. Upload a LifeLink Face video instead.')
      setStatus('error')
    }
  }, [clearFinalizationWatchdog, clearStartWatchdog, discardActiveRecorder, releaseActiveCamera])

  const stopRecording = useCallback(() => {
    const recorder = recorderRef.current
    if (recorder?.state === 'recording') {
      const generation = recordingGenerationRef.current
      commitGenerationRef.current = generation
      setStatus('starting')
      finalizationWatchdogRef.current = window.setTimeout(() => {
        if (
          !mountedRef.current ||
          recordingGenerationRef.current !== generation ||
          commitGenerationRef.current !== generation ||
          recorderRef.current !== recorder
        ) return
        discardActiveRecorder()
        setRecordingStartedAtMs(null)
        setError('The video recorder did not finish in time. The incomplete recording was discarded.')
        setStatus('error')
      }, RECORDER_FINALIZE_TIMEOUT_MS)
      try {
        recorder.stop()
      } catch {
        recordingGenerationRef.current += 1
        clearFinalizationWatchdog()
        commitGenerationRef.current = null
        chunksRef.current = []
        setRecordingStartedAtMs(null)
        releaseActiveCamera()
        recorderRef.current = null
        setError('The browser could not finalize this recording. Please record again.')
        setStatus('error')
      }
    }
  }, [clearFinalizationWatchdog, discardActiveRecorder, releaseActiveCamera])

  const discardRecording = useCallback(() => {
    discardActiveRecorder()
    setRecordingFile(null)
    setRecordingStartedAtMs(null)
    setError(null)
    setStatus('idle')
  }, [discardActiveRecorder])

  const resetRecording = useCallback(() => {
    setRecordingFile(null)
    setRecordingStartedAtMs(null)
    setError(null)
    setStatus('idle')
  }, [])

  useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
      requestGenerationRef.current += 1
      discardActiveRecorder()
    }
  }, [discardActiveRecorder])

  return {
    status,
    error,
    recordingFile,
    recordingStartedAtMs,
    videoRef,
    enableCamera,
    startRecording,
    stopRecording,
    discardRecording,
    resetRecording,
    closeCamera,
  }
}
