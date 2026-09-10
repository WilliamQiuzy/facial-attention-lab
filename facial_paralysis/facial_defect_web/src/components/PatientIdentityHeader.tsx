import type { PatientRecord } from '../patientWorkflow/types'

type PatientIdentityHeaderProps = {
  readonly patient: PatientRecord
  readonly headingLevel?: 1 | 2
  readonly compact?: boolean
}

function formatDate(date: string): string {
  return new Intl.DateTimeFormat('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    timeZone: 'UTC',
  }).format(new Date(`${date}T00:00:00.000Z`))
}

export function PatientIdentityHeader({
  patient,
  headingLevel = 2,
  compact = false,
}: PatientIdentityHeaderProps) {
  const Heading = `h${headingLevel}` as const
  const recordKind =
    patient.recordKind === 'synthetic_demo'
      ? 'Sample record'
      : 'Session record'

  return (
    <section
      className={`patient-identity${compact ? ' patient-identity--compact' : ''}`}
      aria-label="Patient identity"
    >
      <div className="patient-identity__title">
        <Heading>{patient.displayName}</Heading>
        <span className="patient-record-kind">{recordKind}</span>
      </div>
      {compact ? (
        <>
          <dl className="patient-identity__details patient-identity__details--primary">
            <div>
              <dt>Record or study ID</dt>
              <dd>{patient.recordNumber}</dd>
            </div>
          </dl>
          <details className="patient-identity__more">
            <summary>More patient details</summary>
            <dl className="patient-identity__details patient-identity__details--secondary">
              <div>
                <dt>Date of birth</dt>
                <dd>{formatDate(patient.dateOfBirth)}</dd>
              </div>
              <div>
                <dt>Care pathway</dt>
                <dd>{patient.carePathway}</dd>
              </div>
            </dl>
          </details>
        </>
      ) : (
        <dl className="patient-identity__details">
          <div>
            <dt>Record or study ID</dt>
            <dd>{patient.recordNumber}</dd>
          </div>
          <div>
            <dt>Date of birth</dt>
            <dd>{formatDate(patient.dateOfBirth)}</dd>
          </div>
          <div>
            <dt>Care pathway</dt>
            <dd>{patient.carePathway}</dd>
          </div>
        </dl>
      )}
    </section>
  )
}
