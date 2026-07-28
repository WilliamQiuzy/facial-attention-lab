import {
  useState,
  type ChangeEvent,
} from 'react'
import {
  PatientWorkflowProviderError,
} from '../patientWorkflow/PatientWorkflowProvider'
import {
  CaptureFileValidationError,
} from '../patientWorkflow/captureFile'

type CapturePanelProps = {
  readonly title: string
  readonly previewUrl?: string
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
  return 'The photograph could not be added. Choose the file again or use another synthetic/test image.'
}

export function CapturePanel({
  title,
  previewUrl,
  syntheticUnavailableReason,
  onSelectFile,
  onUseSynthetic,
}: CapturePanelProps) {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string>()

  const addFile = async (
    event: ChangeEvent<HTMLInputElement>,
    source: 'camera' | 'upload',
  ) => {
    const input = event.currentTarget
    const file = input.files?.[0]
    if (!file || busy) return

    setBusy(true)
    setError(undefined)
    try {
      await onSelectFile(file, source)
    } catch (nextError) {
      setError(safeCaptureError(nextError))
    } finally {
      input.value = ''
      setBusy(false)
    }
  }

  const useSynthetic = async () => {
    if (busy) return
    setBusy(true)
    setError(undefined)
    try {
      await onUseSynthetic()
    } catch (nextError) {
      setError(safeCaptureError(nextError))
    } finally {
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
    >
      <header className="capture-panel__header">
        <h2 id="capture-panel-title">{title}</h2>
        <p>
          Add one frontal, relaxed-expression photograph. JPEG, PNG, and
          WebP files up to 12 MiB are supported.
        </p>
      </header>

      {previewUrl ? (
        <figure className="capture-panel__preview">
          <img
            src={previewUrl}
            alt="Current frontal photograph"
            loading="eager"
            decoding="async"
          />
          <figcaption>
            Current session photograph. Adding another photograph creates a
            new capture version.
          </figcaption>
        </figure>
      ) : null}

      {error ? (
        <p className="capture-panel__error" role="alert">
          {error}
        </p>
      ) : null}

      <div className="capture-panel__actions">
        <label className="capture-panel__file-action">
          <span>Take photo</span>
          <input
            type="file"
            accept="image/*"
            capture="user"
            aria-label="Take photo"
            disabled={busy}
            onChange={(event) => void addFile(event, 'camera')}
          />
        </label>

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
            Use synthetic demo photo
          </button>
        ) : null}
      </div>

      {syntheticUnavailableReason ? (
        <p className="capture-panel__synthetic-unavailable">
          {syntheticUnavailableReason}
        </p>
      ) : null}

      <p className="capture-panel__boundary">
        Synthetic/test information only. The approved demo photograph is a
        standalone image and is not paired with another visit.
      </p>
    </section>
  )
}
