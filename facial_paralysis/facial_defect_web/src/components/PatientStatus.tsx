import type { PatientNextAction } from '../patientWorkflow/types'

const STATUS_LABELS: Readonly<Record<PatientNextAction, string>> = {
  start_visit: 'Visit needed',
  capture_photo: 'Photo needed',
  confirm_quality: 'Quality check needed',
  run_analysis: 'Ready for analysis',
  processing: 'Analysis in progress',
  retry_analysis: 'Analysis needs attention',
  review_result: 'Review needed',
  visit_complete: 'Complete',
  retake: 'Repeat photo needed',
}

type PatientStatusProps = {
  readonly action: PatientNextAction | undefined
}

export function PatientStatus({ action }: PatientStatusProps) {
  const status = action ? STATUS_LABELS[action] : 'Status unavailable'
  return (
    <span
      className={`patient-status patient-status--${action ?? 'unavailable'}`}
    >
      {status}
    </span>
  )
}
