import { useEffect, useRef, type FormEvent } from 'react'
import type {
  CaptureQualityChecks,
} from '../patientWorkflow/types'

const QUALITY_CHECKS = [
  {
    key: 'faceVisibleAndCentered',
    label: 'Full face is visible and centered',
  },
  {
    key: 'focusLightingAndOcclusionAcceptable',
    label: 'Focus, lighting, and occlusion are acceptable',
  },
  {
    key: 'orientationConfirmed',
    label:
      'Patient left/right orientation is confirmed and the image is not mirrored',
  },
  {
    key: 'authorizationDocumented',
    label:
      'Photography authorization is documented for this visit',
  },
] as const satisfies readonly {
  readonly key: keyof CaptureQualityChecks
  readonly label: string
}[]

type CaptureQualityChecklistProps = {
  readonly checks: CaptureQualityChecks
  readonly onChange: (
    check: keyof CaptureQualityChecks,
    passed: boolean,
  ) => void
  readonly onRun: () => void
}

export function CaptureQualityChecklist({
  checks,
  onChange,
  onRun,
}: CaptureQualityChecklistProps) {
  const titleRef = useRef<HTMLHeadingElement>(null)
  const ready = QUALITY_CHECKS.every(({ key }) => checks[key])

  useEffect(() => {
    titleRef.current?.focus()
  }, [])

  const submit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    if (ready) onRun()
  }

  return (
    <form
      className="capture-quality"
      aria-labelledby="capture-quality-title"
      onSubmit={submit}
    >
      <h2
        ref={titleRef}
        id="capture-quality-title"
        tabIndex={-1}
      >
        Photo quality confirmation
      </h2>
      <fieldset>
        <legend>Confirm all four capture checks</legend>
        <p>
          Confirm capture consistency only. These checks are not a clinical
          assessment.
        </p>
        <div className="capture-quality__checks">
          {QUALITY_CHECKS.map(({ key, label }) => (
            <label key={key} className="capture-quality__check">
              <input
                type="checkbox"
                checked={checks[key]}
                onChange={(event) =>
                  onChange(key, event.currentTarget.checked)
                }
              />
              <span>{label}</span>
            </label>
          ))}
        </div>
      </fieldset>

      <button
        className="patient-primary-action"
        type="submit"
        disabled={!ready}
      >
        Run analysis
      </button>
      {!ready ? (
        <p className="capture-quality__blocked" role="status">
          Analysis remains blocked until all four confirmations are checked.
        </p>
      ) : null}
    </form>
  )
}
