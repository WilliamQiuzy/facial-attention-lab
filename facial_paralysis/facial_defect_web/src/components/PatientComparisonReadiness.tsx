import { Check } from 'lucide-react'
import { Link } from 'react-router-dom'
import {
  selectCurrentCapture,
  selectPatientVisits,
  selectVisitNextAction,
} from '../patientWorkflow/selectors'
import type {
  PatientComparisonState,
  PatientComparisonTimepoint,
  PatientVisit,
  PatientWorkflowState,
  VisitNextAction,
} from '../patientWorkflow/types'

type IncompleteComparisonState = Exclude<
  PatientComparisonState,
  { readonly phase: 'no_visits' | 'ready' }
>

type PatientComparisonReadinessProps = {
  readonly comparison: IncompleteComparisonState
  readonly patientId: string
  readonly workflowState: PatientWorkflowState
}

const TIMEPOINTS = ['preoperative', 'postoperative'] as const

const TIMEPOINT_LABELS: Readonly<
  Record<PatientComparisonTimepoint, string>
> = {
  preoperative: 'Preoperative',
  postoperative: 'Postoperative',
}

const STATUS_LABELS: Readonly<Record<VisitNextAction, string>> = {
  capture_photo: 'Photo needed',
  confirm_quality: 'Quality check needed',
  run_analysis: 'Ready for analysis',
  processing: 'Analysis in progress',
  retry_analysis: 'Analysis needs attention',
  review_result: 'Review needed',
  visit_complete: 'Ready for comparison',
  retake: 'New photo needed',
}

function visitFor(
  comparison: IncompleteComparisonState,
  workflowState: PatientWorkflowState,
  patientId: string,
  timepoint: PatientComparisonTimepoint,
): PatientVisit | undefined {
  if (comparison.phase === 'needs_photos') {
    return comparison.pair[timepoint]
  }
  if (comparison.phase === 'needs_results') {
    return comparison.pair[timepoint].visit
  }
  return selectPatientVisits(workflowState, patientId)
    .filter((visit) => visit.timepoint === timepoint)
    .at(-1)
}

function statusFor(
  workflowState: PatientWorkflowState,
  visit: PatientVisit | undefined,
): string {
  if (!visit) return 'Visit needed'
  if (!selectCurrentCapture(workflowState, visit.id)) {
    return 'Photo needed'
  }
  const action = selectVisitNextAction(workflowState, visit.id)
  return action ? STATUS_LABELS[action] : 'Action needed'
}

function primaryTimepoint(
  comparison: IncompleteComparisonState,
): PatientComparisonTimepoint {
  const missing =
    comparison.phase === 'missing_timepoint'
      ? comparison.missing
      : comparison.phase === 'needs_photos'
        ? comparison.missingPhotos
        : comparison.missingResults
  return missing[0] === 'preoperative'
    ? 'preoperative'
    : 'postoperative'
}

function nextRequiredTimepoint(
  comparison: IncompleteComparisonState,
  workflowState: PatientWorkflowState,
  patientId: string,
): PatientComparisonTimepoint {
  if (comparison.phase === 'missing_timepoint') {
    for (const timepoint of TIMEPOINTS) {
      const visit = visitFor(
        comparison,
        workflowState,
        patientId,
        timepoint,
      )
      if (!visit) continue
      const action = selectVisitNextAction(workflowState, visit.id)
      if (action === 'review_result' || action === 'retake') {
        return timepoint
      }
    }
  }
  return primaryTimepoint(comparison)
}

function actionLabel(
  timepoint: PatientComparisonTimepoint,
  action: VisitNextAction | undefined,
): string {
  const prefix = TIMEPOINT_LABELS[timepoint].toLowerCase()
  switch (action) {
    case 'confirm_quality':
      return `Confirm ${prefix} quality`
    case 'run_analysis':
      return `Run ${prefix} analysis`
    case 'processing':
      return `View ${prefix} progress`
    case 'retry_analysis':
      return `Retry ${prefix} analysis`
    case 'review_result':
      return `Review ${prefix} result`
    case 'visit_complete':
      return `Open ${prefix} result`
    case 'retake':
      return `Repeat ${prefix} photo`
    case 'capture_photo':
    default:
      return `Add ${prefix} photo`
  }
}

export function PatientComparisonReadiness({
  comparison,
  patientId,
  workflowState,
}: PatientComparisonReadinessProps) {
  const nextTimepoint = nextRequiredTimepoint(
    comparison,
    workflowState,
    patientId,
  )
  const nextVisit = visitFor(
    comparison,
    workflowState,
    patientId,
    nextTimepoint,
  )
  const isMissingVisit = !nextVisit
  const nextAction = nextVisit
    ? selectVisitNextAction(workflowState, nextVisit.id)
    : undefined
  const nextHref = nextVisit
    ? `/patients/${patientId}/visits/${nextVisit.id}`
    : `/patients/${patientId}/visits/new`
  const nextLabel = isMissingVisit
    ? `Add ${TIMEPOINT_LABELS[nextTimepoint].toLowerCase()} visit`
    : actionLabel(nextTimepoint, nextAction)

  return (
    <section
      className="patient-comparison-readiness"
      aria-label="Before and after readiness"
    >
      <div className="patient-comparison-readiness__heading">
        <div>
          <p className="patient-comparison-readiness__eyebrow">
            Before and after
          </p>
          <h2>Complete both visits to compare</h2>
        </div>
        <span className="patient-comparison-readiness__pending">
          Not ready
        </span>
      </div>
      <p className="patient-comparison-readiness__summary">
        The comparison will appear after the latest preoperative and
        postoperative visits each have a photo, an analysis result,
        and a saved review decision.
      </p>

      <ul className="patient-comparison-readiness__statuses">
        {TIMEPOINTS.map((timepoint) => {
          const itemVisit = visitFor(
            comparison,
            workflowState,
            patientId,
            timepoint,
          )
          const status = statusFor(workflowState, itemVisit)
          const complete = status === 'Ready for comparison'
          return (
            <li key={timepoint}>
              <span
                aria-hidden="true"
                className={
                  complete
                    ? 'patient-comparison-readiness__marker patient-comparison-readiness__marker--complete'
                    : 'patient-comparison-readiness__marker'
                }
              >
                {complete ? (
                  <Check size={17} strokeWidth={3} />
                ) : (
                  <span className="patient-comparison-readiness__marker-dot" />
                )}
              </span>
              <span>
                <strong>{TIMEPOINT_LABELS[timepoint]}</strong>
                <small>{status}</small>
              </span>
            </li>
          )
        })}
      </ul>

      <div className="patient-comparison-readiness__next">
        <p>{`${TIMEPOINT_LABELS[nextTimepoint]} ${statusFor(
          workflowState,
          nextVisit,
        ).toLowerCase()}`}</p>
        <Link className="patient-primary-action" to={nextHref}>
          {nextLabel}
        </Link>
      </div>
    </section>
  )
}
