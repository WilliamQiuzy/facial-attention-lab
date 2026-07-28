import { Link } from 'react-router-dom'
import {
  selectVisitNextAction,
} from '../patientWorkflow/selectors'
import type {
  PatientVisit,
  PatientWorkflowState,
  VisitNextAction,
} from '../patientWorkflow/types'
import { PatientStatus } from './PatientStatus'

const TIMEPOINT_LABELS: Readonly<
  Record<PatientVisit['timepoint'], string>
> = {
  preoperative: 'Preoperative',
  postoperative: 'Postoperative',
  follow_up: 'Follow-up',
}

const NEXT_ACTION_LABELS: Readonly<Record<VisitNextAction, string>> = {
  capture_photo: 'Add photo',
  confirm_quality: 'Confirm quality',
  run_analysis: 'Run analysis',
  processing: 'View progress',
  retry_analysis: 'Retry analysis',
  review_result: 'Review result',
  visit_complete: 'Open visit',
  retake: 'Repeat photo',
}

type VisitTimelineProps = {
  readonly state: PatientWorkflowState
  readonly visits: readonly PatientVisit[]
}

function formatDate(date: string): string {
  return new Intl.DateTimeFormat('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    timeZone: 'UTC',
  }).format(new Date(`${date}T00:00:00.000Z`))
}

export function VisitTimeline({
  state,
  visits,
}: VisitTimelineProps) {
  if (visits.length === 0) {
    return (
      <section
        className="visit-timeline"
        aria-label="Visit timeline"
      >
        <h2>Visits</h2>
        <div className="patient-empty-state">
          <h3>No photo visits yet</h3>
          <p>Add a visit when a synthetic or test photograph is ready.</p>
        </div>
      </section>
    )
  }

  return (
    <section
      className="visit-timeline"
      aria-label="Visit timeline"
    >
      <h2>Visits</h2>
      <ol className="visit-timeline__list">
        {visits.map((visit) => {
          const nextAction = selectVisitNextAction(state, visit.id)
          return (
            <li
              className="visit-timeline__item"
              key={visit.id}
            >
              <div className="visit-timeline__summary">
                <p className="visit-timeline__timepoint">
                  {TIMEPOINT_LABELS[visit.timepoint]}
                </p>
                <time dateTime={visit.visitDate}>
                  {formatDate(visit.visitDate)}
                </time>
              </div>
              <PatientStatus action={nextAction} />
              {nextAction ? (
                <Link
                  className="patient-link-action"
                  to={`/patients/${visit.patientId}/visits/${visit.id}`}
                >
                  {NEXT_ACTION_LABELS[nextAction]}
                </Link>
              ) : null}
            </li>
          )
        })}
      </ol>
    </section>
  )
}
