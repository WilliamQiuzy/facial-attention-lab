import { useEffect, useRef } from 'react'
import { Check, LoaderCircle } from 'lucide-react'
import type { PatientRunStatus } from '../patientWorkflow/types'

type PatientJobProgressProps = {
  readonly status: Extract<
    PatientRunStatus,
    'queued' | 'running' | 'succeeded'
  >
  readonly previewUrl?: string
  readonly width?: number
  readonly height?: number
}

const phases = [
  'Photo received',
  'Quality confirmed',
  'Analysis',
  'Result prepared',
] as const

function completedPhaseCount(status: PatientJobProgressProps['status']) {
  if (status === 'queued') return 2
  if (status === 'running') return 2
  return 4
}

export function PatientJobProgress({
  status,
  previewUrl,
  width,
  height,
}: PatientJobProgressProps) {
  const progressRef = useRef<HTMLElement>(null)
  const headingRef = useRef<HTMLHeadingElement>(null)
  const completed = completedPhaseCount(status)
  const activeIndex =
    status === 'queued' || status === 'running' ? 2 : -1
  const announcement =
    status === 'queued'
      ? 'Analysis queued'
      : status === 'running'
        ? 'Analysis running'
        : 'Result prepared'
  const loadingMessage =
    status === 'queued'
      ? 'Waiting for analysis to begin…'
      : status === 'running'
        ? 'Matching the face and preparing images…'
        : 'Result prepared'
  const busy = status !== 'succeeded'
  const showPreview =
    Boolean(previewUrl) &&
    typeof width === 'number' &&
    width > 0 &&
    typeof height === 'number' &&
    height > 0

  useEffect(() => {
    const heading = headingRef.current
    heading?.focus({ preventScroll: true })
    heading?.scrollIntoView?.({
      behavior: 'auto',
      block: 'start',
    })
  }, [])

  return (
    <>
      <section
        ref={progressRef}
        className="patient-job-progress"
        aria-label="Analysis progress"
        aria-busy={busy}
        tabIndex={-1}
      >
      <header className="patient-job-progress__header">
        <span
          className="patient-job-progress__activity"
          aria-hidden="true"
        >
          {busy ? (
            <LoaderCircle className="patient-loading-icon" />
          ) : (
            <Check />
          )}
        </span>
        <div>
          <h2 ref={headingRef} tabIndex={-1}>
            Preparing result
          </h2>
          <p>{loadingMessage}</p>
        </div>
      </header>

      <progress
        className="patient-job-progress__bar"
        aria-label="Analysis completion"
        aria-valuetext={`${completed} of ${phases.length} steps complete`}
        max={phases.length}
        value={completed}
      >
        {completed} of {phases.length} steps complete
      </progress>

      <div className="patient-job-progress__layout">
        <ol className="patient-job-progress__phases">
          {phases.map((phase, index) => {
            const phaseStatus =
              index < completed
                ? 'complete'
                : index === activeIndex
                  ? 'current'
                  : 'pending'
            return (
              <li
                key={phase}
                className={`patient-job-progress__phase patient-job-progress__phase--${phaseStatus}`}
                aria-current={
                  phaseStatus === 'current' ? 'step' : undefined
                }
              >
                <span
                  className="patient-job-progress__marker"
                  data-step={index + 1}
                  data-state={phaseStatus}
                  aria-hidden="true"
                >
                  {phaseStatus === 'complete' ? (
                    <Check />
                  ) : (
                    index + 1
                  )}
                </span>
                <span className="patient-job-progress__phase-label">
                  {phase}
                </span>
                <small>
                  {phaseStatus === 'complete'
                    ? 'Complete'
                    : phaseStatus === 'current'
                      ? status === 'queued'
                        ? 'Queued'
                        : 'In progress'
                      : 'Waiting'}
                </small>
              </li>
            )
          })}
        </ol>

        {showPreview ? (
          <figure className="patient-job-progress__preview">
            <img
              src={previewUrl}
              alt="Photograph being analyzed"
              width={width}
              height={height}
              loading="eager"
              decoding="async"
            />
            <figcaption>Current photograph</figcaption>
          </figure>
        ) : null}
      </div>

      <p className="patient-job-progress__guidance">
        Keep this page open. No action is needed.
      </p>
      </section>
      <p
        className="patient-job-progress__announcement"
        role="status"
        aria-label="Analysis status announcement"
        aria-live="polite"
        aria-atomic="true"
      >
        {announcement}
      </p>
    </>
  )
}
