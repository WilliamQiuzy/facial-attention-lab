import type { PatientRunStatus } from '../patientWorkflow/types'

type PatientJobProgressProps = {
  readonly status: Extract<
    PatientRunStatus,
    'queued' | 'running' | 'succeeded'
  >
}

const phases = [
  'Photo received',
  'Quality confirmed',
  'Analysis running',
  'Result prepared',
] as const

function completedPhaseCount(status: PatientJobProgressProps['status']) {
  if (status === 'queued') return 2
  if (status === 'running') return 2
  return 4
}

export function PatientJobProgress({
  status,
}: PatientJobProgressProps) {
  const completed = completedPhaseCount(status)
  const activeIndex = status === 'running' ? 2 : -1
  const announcement =
    status === 'queued'
      ? 'Analysis queued'
      : status === 'running'
        ? 'Analysis running'
        : 'Result prepared'

  return (
    <section
      className="patient-job-progress"
      aria-label="Analysis progress"
    >
      <h2>Preparing result</h2>
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
              aria-current={phaseStatus === 'current' ? 'step' : undefined}
            >
              <span>{phase}</span>
              <small>
                {phaseStatus === 'complete'
                  ? 'Complete'
                  : phaseStatus === 'current'
                    ? 'In progress'
                    : 'Waiting'}
              </small>
            </li>
          )
        })}
      </ol>
      <p
        className="patient-job-progress__announcement"
        role="status"
        aria-live="polite"
        aria-atomic="true"
      >
        {announcement}
      </p>
    </section>
  )
}
