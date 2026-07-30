import { useMemo } from 'react'
import { Link } from 'react-router-dom'
import { PatientStatus } from '../components/PatientStatus'
import { usePatientWorkflow } from '../patientWorkflow/PatientWorkflowProvider'
import {
  getOwnRecordValue,
  selectPatientNextAction,
  selectPatientVisits,
} from '../patientWorkflow/selectors'
import type {
  PatientRecord,
  PatientVisit,
} from '../patientWorkflow/types'

const TIMEPOINT_LABELS: Readonly<
  Record<PatientVisit['timepoint'], string>
> = {
  preoperative: 'Preoperative',
  postoperative: 'Postoperative',
  follow_up: 'Follow-up',
}

function compactSearchValue(value: string): string {
  return value.toLocaleLowerCase().replace(/[^a-z0-9]/g, '')
}

function formatDate(date: string): string {
  return new Intl.DateTimeFormat('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    timeZone: 'UTC',
  }).format(new Date(`${date}T00:00:00.000Z`))
}

function latestVisitLabel(visit: PatientVisit | undefined): string {
  if (!visit) return 'No visits'
  return `${TIMEPOINT_LABELS[visit.timepoint]} · ${formatDate(
    visit.visitDate,
  )}`
}

export function PatientsPage() {
  const {
    state,
    patientListQuery: query,
    setPatientListQuery,
  } = usePatientWorkflow()
  const normalizedQuery = query.trim().toLocaleLowerCase()
  const compactQuery = compactSearchValue(query)
  const patients = state.patientOrder
    .map((patientId) =>
      getOwnRecordValue(state.patientsById, patientId),
    )
    .filter((patient): patient is PatientRecord => patient !== undefined)

  const matchingPatients = useMemo(
    () =>
      patients.filter((patient) => {
        if (!normalizedQuery) return true
        return (
          patient.displayName
            .toLocaleLowerCase()
            .includes(normalizedQuery) ||
          (compactQuery.length > 0 &&
            compactSearchValue(patient.recordNumber).includes(compactQuery))
        )
      }),
    [compactQuery, normalizedQuery, patients],
  )

  const updateQuery = (nextQuery: string) => {
    setPatientListQuery(nextQuery)
  }

  return (
    <div className="patient-workflow-page patients-page">
      <header className="patient-page-header page-shell">
        <div>
          <h1>Patients</h1>
          <p>
            Session-only synthetic and test records for the photo workflow.
          </p>
        </div>
        <Link
          className="patient-primary-action"
          to="/patients/new"
        >
          New patient
        </Link>
      </header>

      <div className="patient-page-content page-shell">
        <label className="patient-search">
          <span>Search patients</span>
          <input
            type="search"
            name="patientSearch"
            autoComplete="off"
            maxLength={128}
            value={query}
            onChange={(event) =>
              updateQuery(event.currentTarget.value)
            }
            placeholder="Name or record ID…"
          />
        </label>

        <div className="patient-list-heading">
          <h2>Patient records</h2>
          <output
            aria-label="Patient search results"
            aria-live="polite"
            aria-atomic="true"
          >
            {matchingPatients.length} of {patients.length} patients
          </output>
        </div>

        {matchingPatients.length > 0 ? (
          <ul className="patient-list">
            {matchingPatients.map((patient) => {
              const visits = selectPatientVisits(state, patient.id)
              const latestVisit = visits.at(-1)
              const nextAction = selectPatientNextAction(
                state,
                patient.id,
              )
              return (
                <li
                  className="patient-list__row"
                  aria-label={patient.displayName}
                  key={patient.id}
                >
                  <div className="patient-list__identity">
                    <h3>{patient.displayName}</h3>
                    <p>{patient.recordNumber}</p>
                    <span className="patient-record-kind">
                      {patient.recordKind === 'synthetic_demo'
                        ? 'Synthetic demo'
                        : 'Session test'}
                    </span>
                  </div>
                  <div className="patient-list__visit">
                    <span>Latest visit</span>
                    <p>{latestVisitLabel(latestVisit)}</p>
                  </div>
                  <PatientStatus action={nextAction} />
                  <Link
                    className="patient-link-action"
                    to={`/patients/${patient.id}`}
                  >
                    Open
                  </Link>
                </li>
              )
            })}
          </ul>
        ) : (
          <section
            className="patient-empty-state"
            aria-label="No matching patients"
          >
            <h3>
              {patients.length === 0
                ? 'No patient records in this session'
                : 'No patients match this search'}
            </h3>
            <p>
              {patients.length === 0
                ? 'Create a synthetic or test record to begin.'
                : 'Try a different name or record ID.'}
            </p>
            {patients.length > 0 ? (
              <button
                className="patient-secondary-action"
                type="button"
                onClick={() => updateQuery('')}
              >
                Clear search
              </button>
            ) : (
              <Link
                className="patient-secondary-action"
                to="/patients/new"
              >
                New patient
              </Link>
            )}
          </section>
        )}
      </div>
    </div>
  )
}
