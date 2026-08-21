import { Link, useParams } from 'react-router-dom'
import { PatientComparisonReadiness } from '../components/PatientComparisonReadiness'
import { PatientIdentityHeader } from '../components/PatientIdentityHeader'
import { PatientLongitudinalComparison } from '../components/PatientLongitudinalComparison'
import { VisitTimeline } from '../components/VisitTimeline'
import { usePatientWorkflow } from '../patientWorkflow/PatientWorkflowProvider'
import {
  getOwnRecordValue,
  selectPatientComparisonState,
  selectPatientVisits,
} from '../patientWorkflow/selectors'

export function PatientDetailPage() {
  const { patientId = '' } = useParams()
  const { state, actions } = usePatientWorkflow()
  const patient = getOwnRecordValue(state.patientsById, patientId)

  if (!patient) {
    return (
      <section className="patient-workflow-page patient-page-content page-shell">
        <h1>Patient record unavailable</h1>
        <p>
          This patient record is not available in the current session.
        </p>
        <Link
          className="patient-link-action"
          to="/patients"
        >
          Back to patients
        </Link>
      </section>
    )
  }

  const visits = selectPatientVisits(state, patient.id)
  const comparison = selectPatientComparisonState(state, patient.id)
  let comparisonContent = null

  if (
    comparison.phase === 'missing_timepoint' ||
    comparison.phase === 'needs_photos' ||
    comparison.phase === 'needs_results'
  ) {
    comparisonContent = (
      <PatientComparisonReadiness
        comparison={comparison}
        patientId={patient.id}
        workflowState={state}
      />
    )
  } else if (comparison.phase === 'ready') {
    const preoperativePreviewUrl = actions.getCapturePreviewUrl(
      comparison.pair.preoperative.visit.id,
    )
    const postoperativePreviewUrl = actions.getCapturePreviewUrl(
      comparison.pair.postoperative.visit.id,
    )

    comparisonContent =
      preoperativePreviewUrl && postoperativePreviewUrl ? (
        <PatientLongitudinalComparison
          pair={{
            preoperative: {
              ...comparison.pair.preoperative,
              previewUrl: preoperativePreviewUrl,
            },
            postoperative: {
              ...comparison.pair.postoperative,
              previewUrl: postoperativePreviewUrl,
            },
          }}
        />
      ) : (
        <section
          className="patient-comparison-recovery"
          aria-label="Comparison media unavailable"
        >
          <p className="patient-comparison-recovery__eyebrow">
            Before and after
          </p>
          <h2>Current photos are unavailable</h2>
          <p>
            Both current photos are required before this comparison can
            be shown.
          </p>
          <div className="patient-comparison-recovery__actions">
            <Link
              className="patient-link-action"
              to={`/patients/${patient.id}/visits/${comparison.pair.preoperative.visit.id}`}
            >
              Open preoperative visit
            </Link>
            <Link
              className="patient-link-action"
              to={`/patients/${patient.id}/visits/${comparison.pair.postoperative.visit.id}`}
            >
              Open postoperative visit
            </Link>
          </div>
        </section>
      )
  }

  return (
    <div className="patient-workflow-page patient-detail-page">
      <header className="patient-page-header page-shell">
        <Link
          className="patient-back-link"
          to="/patients"
        >
          Back to patients
        </Link>
        <PatientIdentityHeader
          patient={patient}
          headingLevel={1}
        />
        {comparison.phase === 'no_visits' ? (
          <Link
            className="patient-primary-action"
            to={`/patients/${patient.id}/visits/new`}
          >
            Add photo visit
          </Link>
        ) : comparison.phase === 'ready' ? (
          <Link
            className="patient-secondary-action"
            to={`/patients/${patient.id}/visits/new`}
          >
            Add another visit
          </Link>
        ) : null}
      </header>

      <div className="patient-page-content page-shell">
        {comparisonContent}
        <VisitTimeline
          state={state}
          visits={visits}
          getPreviewUrl={actions.getCapturePreviewUrl}
        />
      </div>
    </div>
  )
}
