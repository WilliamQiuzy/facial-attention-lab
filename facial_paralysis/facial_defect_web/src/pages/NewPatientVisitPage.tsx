import {
  useRef,
  useState,
  type FormEvent,
} from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { PatientIdentityHeader } from '../components/PatientIdentityHeader'
import {
  PatientWorkflowProviderError,
  usePatientWorkflow,
} from '../patientWorkflow/PatientWorkflowProvider'
import { getOwnRecordValue } from '../patientWorkflow/selectors'
import {
  validateVisitDraft,
  type PatientVisitDraftErrors,
} from '../patientWorkflow/validation'

type NewVisitErrors = PatientVisitDraftErrors & {
  readonly form?: string
}

function todayIso(): string {
  const now = new Date()
  const year = now.getFullYear()
  const month = String(now.getMonth() + 1).padStart(2, '0')
  const day = String(now.getDate()).padStart(2, '0')
  return `${year}-${month}-${day}`
}

function FieldError({
  id,
  message,
}: {
  readonly id: string
  readonly message: string | undefined
}) {
  return message ? (
    <p className="patient-field-error" id={id}>
      {message}
    </p>
  ) : null
}

export function NewPatientVisitPage() {
  const { patientId = '' } = useParams()
  const { state, actions } = usePatientWorkflow()
  const navigate = useNavigate()
  const patient = getOwnRecordValue(state.patientsById, patientId)
  const [timepoint, setTimepoint] = useState('')
  const [visitDate, setVisitDate] = useState(todayIso())
  const [errors, setErrors] = useState<NewVisitErrors>({})
  const submittingRef = useRef(false)
  const timepointRef = useRef<HTMLSelectElement>(null)
  const visitDateRef = useRef<HTMLInputElement>(null)

  const clearFieldError = (
    field: 'timepoint' | 'visitDate',
  ) => {
    setErrors((current) => {
      if (!current[field]) return current
      const next = { ...current }
      delete next[field]
      return next
    })
  }

  if (!patient) {
    return (
      <section className="patient-workflow-page patient-page-content page-shell">
        <h1>Patient record unavailable</h1>
        <p>
          This patient record is not available in the current session.
        </p>
        <Link className="patient-link-action" to="/patients">
          Back to patients
        </Link>
      </section>
    )
  }

  const focusFirstError = (nextErrors: NewVisitErrors) => {
    if (nextErrors.timepoint) {
      timepointRef.current?.focus()
      return
    }
    if (nextErrors.visitDate) visitDateRef.current?.focus()
  }

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    if (submittingRef.current) return

    const draft = {
      timepoint:
        timepoint === 'preoperative' ||
        timepoint === 'postoperative' ||
        timepoint === 'follow_up'
          ? timepoint
          : '',
      visitDate,
    } as const
    const validation = validateVisitDraft(draft, todayIso())
    if (!validation.ok) {
      setErrors(validation.errors)
      focusFirstError(validation.errors)
      return
    }

    submittingRef.current = true
    try {
      const visitId = actions.createVisit(patient.id, draft)
      navigate(`/patients/${patient.id}/visits/${visitId}`)
    } catch (error) {
      submittingRef.current = false
      if (error instanceof PatientWorkflowProviderError) {
        const providerErrors: NewVisitErrors =
          error.failure.field === 'timepoint' ||
          error.failure.field === 'visitDate'
            ? { [error.failure.field]: error.failure.message }
            : { form: error.failure.message }
        setErrors(providerErrors)
        focusFirstError(providerErrors)
        return
      }
      setErrors({
        form: 'The visit could not be created. Review the form and try again.',
      })
    }
  }

  return (
    <div className="patient-workflow-page patient-form-page">
      <header className="patient-page-header patient-visit-create-header page-shell">
        <Link
          className="patient-back-link"
          to={`/patients/${patient.id}`}
        >
          Back to patient record
        </Link>
        <h1>Add photo visit</h1>
        <PatientIdentityHeader patient={patient} compact />
      </header>

      <div className="patient-page-content page-shell">
        <form
          className="patient-form patient-form--visit"
          noValidate
          onSubmit={handleSubmit}
        >
          {Object.keys(errors).length > 0 ? (
            <div className="patient-form-alert" role="alert">
              <strong>Check the highlighted fields.</strong>
              {errors.form ? <p>{errors.form}</p> : null}
            </div>
          ) : null}

          <label className="patient-field">
            <span>Timepoint</span>
            <select
              ref={timepointRef}
              name="timepoint"
              value={timepoint}
              aria-invalid={Boolean(errors.timepoint)}
              aria-describedby={
                errors.timepoint ? 'new-visit-timepoint-error' : undefined
              }
              onChange={(event) => {
                setTimepoint(event.currentTarget.value)
                clearFieldError('timepoint')
              }}
            >
              <option value="">Select a timepoint</option>
              <option value="preoperative">Preoperative</option>
              <option value="postoperative">Postoperative</option>
              <option value="follow_up">Follow-up</option>
            </select>
            <FieldError
              id="new-visit-timepoint-error"
              message={errors.timepoint}
            />
          </label>

          <label className="patient-field">
            <span>Visit date</span>
            <input
              ref={visitDateRef}
              name="visitDate"
              type="date"
              autoComplete="off"
              max={todayIso()}
              value={visitDate}
              aria-invalid={Boolean(errors.visitDate)}
              aria-describedby={
                errors.visitDate ? 'new-visit-date-error' : undefined
              }
              onChange={(event) => {
                setVisitDate(event.currentTarget.value)
                clearFieldError('visitDate')
              }}
            />
            <FieldError
              id="new-visit-date-error"
              message={errors.visitDate}
            />
          </label>

          <p className="patient-form-note">
            The next screen adds one frontal, relaxed-expression
            patient or sample photograph.
          </p>

          <div className="patient-form-actions">
            <button
              className="patient-primary-action"
              type="submit"
            >
              Continue to photo
            </button>
            <Link
              className="patient-secondary-action"
              to={`/patients/${patient.id}`}
            >
              Cancel
            </Link>
          </div>
        </form>
      </div>
    </div>
  )
}
