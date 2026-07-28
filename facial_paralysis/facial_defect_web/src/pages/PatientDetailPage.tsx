import { Link, useParams } from 'react-router-dom'
import { PatientIdentityHeader } from '../components/PatientIdentityHeader'
import { VisitTimeline } from '../components/VisitTimeline'
import { usePatientWorkflow } from '../patientWorkflow/PatientWorkflowProvider'
import {
  getOwnRecordValue,
  selectPatientVisits,
} from '../patientWorkflow/selectors'

export function PatientDetailPage() {
  const { patientId = '' } = useParams()
  const { state } = usePatientWorkflow()
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
        <Link
          className="patient-primary-action"
          to={`/patients/${patient.id}/visits/new`}
        >
          Add photo visit
        </Link>
      </header>

      <div className="patient-page-content page-shell">
        <VisitTimeline state={state} visits={visits} />
      </div>
    </div>
  )
}
