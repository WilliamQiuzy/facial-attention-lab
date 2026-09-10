import {
  useRef,
  useState,
  type ChangeEvent,
} from 'react'
import { LoaderCircle } from 'lucide-react'
import {
  PatientWorkflowProviderError,
} from '../patientWorkflow/PatientWorkflowProvider'
import {
  CaptureFileValidationError,
} from '../patientWorkflow/captureFile'
import { CameraCapture } from './CameraCapture'

type CapturePanelProps = {
  readonly title: string
  readonly previewUrl?: string
  readonly previewWidth?: number
  readonly previewHeight?: number
  readonly previewDisclosure?: string
  readonly syntheticUnavailableReason?: string
  readonly onSelectFile: (
    file: File,
    source: 'camera' | 'upload',
  ) => Promise<unknown>
  readonly onUseSynthetic: () => Promise<unknown>
}

function safeCaptureError(error: unknown): string {
  if (
    error instanceof CaptureFileValidationError ||
    error instanceof PatientWorkflowProviderError
  ) {
    return error.message
  }
  return 'The photograph could not be added. Choose the file again or use the sample photo.'
}

export function CapturePanel({
  title,
  previewUrl,
  previewWidth,
  previewHeight,
  previewDisclosure,
  syntheticUnavailableReason,
  onSelectFile,
  onUseSynthetic,
}: CapturePanelProps) {
  const [busy, setBusy] = useState(false)
  const [cameraOpen, setCameraOpen] = useState(false)
  const [error, setError] = useState<string>()
  const busyRef = useRef(false)

  const addFile = async (
    event: ChangeEvent<HTMLInputElement>,
    source: 'camera' | 'upload',
  ) => {
    const input = event.currentTarget
    const file = input.files?.[0]
    if (!file || busyRef.current) return

    busyRef.current = true
    setBusy(true)
    setError(undefined)
    try {
      await onSelectFile(file, source)
    } catch (nextError) {
      setError(safeCaptureError(nextError))
    } finally {
      input.value = ''
      busyRef.current = false
      setBusy(false)
    }
  }

  const useSynthetic = async () => {
    if (busyRef.current) return
    busyRef.current = true
    setBusy(true)
    setError(undefined)
    try {
      await onUseSynthetic()
    } catch (nextError) {
      setError(safeCaptureError(nextError))
    } finally {
      busyRef.current = false
      setBusy(false)
    }
  }

  const captureCameraFile = async (file: File) => {
    if (busyRef.current) return
    busyRef.current = true
    setBusy(true)
    setError(undefined)
    try {
      await onSelectFile(file, 'camera')
      setCameraOpen(false)
    } catch (nextError) {
      throw new Error(safeCaptureError(nextError))
    } finally {
      busyRef.current = false
      setBusy(false)
    }
  }

  return (
    <section
      className={
        previewUrl
          ? 'capture-panel capture-panel--has-preview'
          : 'capture-panel'
      }
      aria-labelledby="capture-panel-title"
      aria-busy={busy}
    >
      <header className="capture-panel__header">
        <h2 id="capture-panel-title">{title}</h2>
        <p>
          Add one frontal, relaxed-expression photograph. JPEG, PNG, and
          WebP files up to 12 MiB are supported.
        </p>
      </header>

      {busy ? (
        <div
          className="capture-panel__busy"
          role="status"
          aria-label="Photograph preparation status"
          aria-live="polite"
          aria-atomic="true"
        >
          <LoaderCircle
            className="patient-loading-icon"
            aria-hidden="true"
          />
          <span>
            <strong>Preparing photograph…</strong>
            <small>
              Checking the image before it is added to this visit.
            </small>
          </span>
        </div>
      ) : null}

      {previewUrl ? (
        <figure className="capture-panel__preview">
          <img
            src={previewUrl}
            alt="Current frontal photograph"
            width={previewWidth}
            height={previewHeight}
            loading="eager"
            decoding="async"
          />
          <figcaption>
            <span>
              Current session photograph. Adding another photograph creates
              a new capture version.
            </span>
            {previewDisclosure ? (
              <strong>{previewDisclosure}</strong>
            ) : null}
          </figcaption>
        </figure>
      ) : null}

      {error ? (
        <p className="capture-panel__error" role="alert">
          {error}
        </p>
      ) : null}

      {cameraOpen ? (
        <CameraCapture
          onCapture={captureCameraFile}
          onCancel={() => setCameraOpen(false)}
        />
      ) : (
        <div className="capture-panel__actions">
          <button
            className="capture-panel__file-action"
            type="button"
            disabled={busy}
            onClick={() => setCameraOpen(true)}
          >
            Camera
          </button>

          <label className="capture-panel__file-action">
            <span>Upload photo</span>
            <input
              type="file"
              accept="image/jpeg,image/png,image/webp"
              aria-label="Upload photo"
              disabled={busy}
              onChange={(event) => void addFile(event, 'upload')}
            />
          </label>

          {!syntheticUnavailableReason ? (
            <button
              className="capture-panel__synthetic-action"
              type="button"
              disabled={busy}
              onClick={() => void useSynthetic()}
            >
              Sample photo
            </button>
          ) : null}
        </div>
      )}

      {syntheticUnavailableReason ? (
        <p className="capture-panel__synthetic-unavailable">
          {syntheticUnavailableReason}
        </p>
      ) : null}

      <p className="capture-panel__boundary">
        Sample or de-identified information only. A sample photo is used for
        interface review and can be replaced with Camera or Upload photo.
      </p>
    </section>
  )
}
