import { Link } from 'react-router-dom'
import { usePatientWorkflow } from '../patientWorkflow/PatientWorkflowProvider'
import {
  getOwnRecordValue,
  selectCurrentResult,
  selectCurrentReview,
} from '../patientWorkflow/selectors'
import type { PatientVisit } from '../patientWorkflow/types'

const TIMEPOINT_LABELS: Readonly<
  Record<PatientVisit['timepoint'], string>
> = {
  preoperative: 'Preoperative',
  postoperative: 'Postoperative',
  follow_up: 'Follow-up',
}

function formatDate(date: string): string {
  return new Intl.DateTimeFormat('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    timeZone: 'UTC',
  }).format(new Date(`${date}T00:00:00.000Z`))
}

export function ClinicalReviewQueuePage() {
  const { state } = usePatientWorkflow()
  const queue = state.visitOrder
    .flatMap((visitId) => {
      const visit = getOwnRecordValue(state.visitsById, visitId)
      const result = selectCurrentResult(state, visitId)
      const review = selectCurrentReview(state, visitId)
      if (!visit || !result || review) return []

      const patient = getOwnRecordValue(
        state.patientsById,
        visit.patientId,
      )
      return patient ? [{ patient, visit, result }] : []
    })
    .sort((first, second) =>
      second.result.createdAt.localeCompare(first.result.createdAt),
    )

  return (
    <div className="patient-workflow-page clinical-review-queue">
      <header className="patient-page-header page-shell">
        <h1>Reviews</h1>
        <p>
          Current simulated results waiting for a clinician decision.
        </p>
      </header>

      <section
        className="patient-page-content page-shell"
        aria-label="Clinical review queue"
      >
        {queue.length === 0 ? (
          <div
            className="patient-empty-state clinical-review-queue__empty"
            role="status"
            aria-label="Review queue status"
          >
            <h2>No results are waiting for review.</h2>
            <p>
              A visit appears here only after its current simulated result
              is prepared.
            </p>
            <Link className="patient-primary-action" to="/patients">
              View patients
            </Link>
          </div>
        ) : (
          <ul className="clinical-review-queue__list">
            {queue.map(({ patient, visit }) => (
              <li
                className="clinical-review-queue__item"
                key={visit.id}
              >
                <div className="clinical-review-queue__patient">
                  <strong>{patient.displayName}</strong>
                  <span>{patient.recordNumber}</span>
                </div>
                <div className="clinical-review-queue__visit">
                  <span>{TIMEPOINT_LABELS[visit.timepoint]}</span>
                  <time dateTime={visit.visitDate}>
                    {formatDate(visit.visitDate)}
                  </time>
                </div>
                <Link
                  className="patient-link-action"
                  to={`/patients/${patient.id}/visits/${visit.id}`}
                >
                  Review result
                </Link>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  )
}
