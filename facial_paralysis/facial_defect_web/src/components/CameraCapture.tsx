import { useEffect, useRef, useState } from 'react'
import { LoaderCircle } from 'lucide-react'

type CameraCaptureProps = {
  readonly onCapture: (file: File) => Promise<void>
  readonly onCancel: () => void
}

type CameraPhase = 'starting' | 'ready' | 'capturing' | 'error'

function stopStream(stream: MediaStream | undefined): void {
  stream?.getTracks().forEach((track) => track.stop())
}

function cameraErrorMessage(error: unknown): string {
  if (error instanceof DOMException) {
    if (error.name === 'NotAllowedError') {
      return 'Camera access was not granted. Allow camera permission or use Upload photo.'
    }
    if (error.name === 'NotFoundError') {
      return 'No camera was found. Connect a camera or use Upload photo.'
    }
  }
  if (error instanceof Error && error.message) return error.message
  return 'The camera could not be started. Use Upload photo instead.'
}

function canvasBlob(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => {
        if (blob) resolve(blob)
        else reject(new Error('The camera frame could not be prepared.'))
      },
      'image/jpeg',
      0.92,
    )
  })
}

export function CameraCapture({
  onCapture,
  onCancel,
}: CameraCaptureProps) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const streamRef = useRef<MediaStream | undefined>(undefined)
  const [phase, setPhase] = useState<CameraPhase>('starting')
  const [error, setError] = useState<string>()

  useEffect(() => {
    let active = true

    const start = async () => {
      if (!navigator.mediaDevices?.getUserMedia) {
        setError(
          'Live camera is unavailable in this browser. Use Upload photo instead.',
        )
        setPhase('error')
        return
      }

      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: false,
          video: {
            facingMode: 'user',
            width: { ideal: 1280, min: 640 },
            height: { ideal: 960, min: 640 },
          },
        })
        if (!active) {
          stopStream(stream)
          return
        }
        streamRef.current = stream
        const video = videoRef.current
        if (!video) throw new Error('The camera preview is unavailable.')
        video.srcObject = stream
        await video.play()
        if (active) setPhase('ready')
      } catch (nextError) {
        stopStream(streamRef.current)
        streamRef.current = undefined
        if (active) {
          setError(cameraErrorMessage(nextError))
          setPhase('error')
        }
      }
    }

    void start()
    return () => {
      active = false
      stopStream(streamRef.current)
      streamRef.current = undefined
    }
  }, [])

  const captureFrame = async () => {
    const video = videoRef.current
    if (!video || phase !== 'ready') return

    setError(undefined)
    setPhase('capturing')
    try {
      if (video.videoWidth < 1 || video.videoHeight < 1) {
        throw new Error('Wait for the live camera preview, then try again.')
      }
      const canvas = document.createElement('canvas')
      canvas.width = video.videoWidth
      canvas.height = video.videoHeight
      const context = canvas.getContext('2d')
      if (!context) {
        throw new Error('The camera frame could not be prepared.')
      }
      context.drawImage(video, 0, 0, canvas.width, canvas.height)
      const blob = await canvasBlob(canvas)
      const file = new File([blob], `faceai-camera-${Date.now()}.jpg`, {
        type: 'image/jpeg',
        lastModified: Date.now(),
      })
      await onCapture(file)
    } catch (nextError) {
      setError(cameraErrorMessage(nextError))
      setPhase(streamRef.current ? 'ready' : 'error')
    }
  }

  return (
    <section
      className="camera-capture"
      aria-label="Camera preview"
      aria-busy={phase === 'starting' || phase === 'capturing'}
    >
      <div className="camera-capture__heading">
        <div>
          <h3>Camera preview</h3>
          <p>Center one face with a relaxed expression.</p>
        </div>
        <button type="button" onClick={onCancel}>
          Cancel camera
        </button>
      </div>

      <div className="camera-capture__viewport">
        <video
          ref={videoRef}
          aria-label="Live camera preview"
          autoPlay
          muted
          playsInline
        />
        {phase === 'starting' ? (
          <p className="camera-capture__loading" role="status">
            <LoaderCircle
              className="patient-loading-icon"
              aria-hidden="true"
            />
            Starting camera…
          </p>
        ) : null}
      </div>

      <p className="camera-capture__orientation">
        The live preview is mirrored for positioning. The saved photograph
        uses the non-mirrored clinical orientation.
      </p>

      {error ? (
        <p className="camera-capture__error" role="alert">
          {error}
        </p>
      ) : null}

      <button
        className="patient-primary-action"
        type="button"
        disabled={phase !== 'ready'}
        onClick={() => void captureFrame()}
      >
        {phase === 'capturing' ? 'Adding photo…' : 'Capture photo'}
      </button>
    </section>
  )
}
