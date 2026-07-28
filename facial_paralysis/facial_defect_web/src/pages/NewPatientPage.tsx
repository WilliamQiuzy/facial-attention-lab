import {
  useRef,
  useState,
  type FormEvent,
  type RefObject,
} from 'react'
import { Link, useNavigate } from 'react-router-dom'
import {
  PatientWorkflowProviderError,
  usePatientWorkflow,
} from '../patientWorkflow/PatientWorkflowProvider'
import {
  validatePatientDraft,
  validateVisitDraft,
  type PatientDraftErrors,
  type PatientVisitDraftErrors,
} from '../patientWorkflow/validation'

const CARE_PATHWAYS = [
  { value: 'facial_paralysis', label: 'Facial paralysis' },
  { value: 'facial_reconstruction', label: 'Facial reconstruction' },
  { value: 'follow_up_clinic', label: 'Follow-up clinic' },
  { value: 'other_test', label: 'Other synthetic/test pathway' },
] as const

type NewPatientErrors = PatientDraftErrors &
  PatientVisitDraftErrors & {
    readonly form?: string
  }

type PatientField =
  | 'displayName'
  | 'recordNumber'
  | 'dateOfBirth'
  | 'carePathway'
  | 'timepoint'
  | 'visitDate'
  | 'syntheticTestAttestation'

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

export function NewPatientPage() {
  const { state, actions } = usePatientWorkflow()
  const navigate = useNavigate()
  const [displayName, setDisplayName] = useState('')
  const [recordNumber, setRecordNumber] = useState('')
  const [dateOfBirth, setDateOfBirth] = useState('')
  const [carePathway, setCarePathway] = useState('')
  const [timepoint, setTimepoint] = useState('')
  const [visitDate, setVisitDate] = useState(todayIso())
  const [syntheticTestAttestation, setSyntheticTestAttestation] =
    useState(false)
  const [errors, setErrors] = useState<NewPatientErrors>({})
  const submittingRef = useRef(false)

  const displayNameRef = useRef<HTMLInputElement>(null)
  const recordNumberRef = useRef<HTMLInputElement>(null)
  const dateOfBirthRef = useRef<HTMLInputElement>(null)
  const carePathwayRef = useRef<HTMLSelectElement>(null)
  const timepointRef = useRef<HTMLSelectElement>(null)
  const visitDateRef = useRef<HTMLInputElement>(null)
  const attestationRef = useRef<HTMLInputElement>(null)
  const fieldRefs: Readonly<
    Record<PatientField, RefObject<HTMLElement | null>>
  > = {
    displayName: displayNameRef,
    recordNumber: recordNumberRef,
    dateOfBirth: dateOfBirthRef,
    carePathway: carePathwayRef,
    timepoint: timepointRef,
    visitDate: visitDateRef,
    syntheticTestAttestation: attestationRef,
  }

  const focusFirstError = (nextErrors: NewPatientErrors) => {
    const fieldOrder: readonly PatientField[] = [
      'displayName',
      'recordNumber',
      'dateOfBirth',
      'carePathway',
      'timepoint',
      'visitDate',
      'syntheticTestAttestation',
    ]
    const first = fieldOrder.find((field) => nextErrors[field])
    if (first) fieldRefs[first].current?.focus()
  }

  const selectedCarePathway =
    CARE_PATHWAYS.find((option) => option.value === carePathway)
      ?.label ?? ''

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    if (submittingRef.current) return

    const patientDraft = {
      displayName,
      recordNumber,
      dateOfBirth,
      carePathway: selectedCarePathway,
      syntheticTestAttestation,
    }
    const visitDraft = {
      timepoint:
        timepoint === 'preoperative' ||
        timepoint === 'postoperative' ||
        timepoint === 'follow_up'
          ? timepoint
          : '',
      visitDate,
    } as const
    const patientValidation = validatePatientDraft(
      patientDraft,
      state,
      todayIso(),
    )
    const visitValidation = validateVisitDraft(visitDraft, todayIso())
    const nextErrors: NewPatientErrors = {
      ...(patientValidation.ok ? {} : patientValidation.errors),
      ...(visitValidation.ok ? {} : visitValidation.errors),
    }
    if (Object.keys(nextErrors).length > 0) {
      setErrors(nextErrors)
      focusFirstError(nextErrors)
      return
    }

    submittingRef.current = true
    try {
      const { patientId, visitId } = actions.createPatient({
        ...patientDraft,
        initialVisit: visitDraft,
      })
      navigate(`/patients/${patientId}/visits/${visitId}`)
    } catch (error) {
      submittingRef.current = false
      if (error instanceof PatientWorkflowProviderError) {
        const field = error.failure.field
        const providerErrors: NewPatientErrors =
          field && field in fieldRefs
            ? { [field]: error.failure.message }
            : { form: error.failure.message }
        setErrors(providerErrors)
        focusFirstError(providerErrors)
        return
      }
      setErrors({
        form: 'The record could not be created. Review the form and try again.',
      })
    }
  }

  return (
    <div className="patient-workflow-page patient-form-page">
      <header className="patient-page-header page-shell">
        <div>
          <Link className="patient-back-link" to="/patients">
            Back to patients
          </Link>
          <h1>New patient</h1>
          <p>Create a session-only synthetic or test record.</p>
        </div>
      </header>

      <div className="patient-page-content page-shell">
        <form
          className="patient-form"
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
            <span>Display name</span>
            <input
              ref={displayNameRef}
              name="displayName"
              type="text"
              autoComplete="off"
              value={displayName}
              aria-invalid={Boolean(errors.displayName)}
              aria-describedby={
                errors.displayName ? 'displayName-error' : undefined
              }
              onChange={(event) =>
                setDisplayName(event.currentTarget.value)
              }
            />
            <FieldError
              id="displayName-error"
              message={errors.displayName}
            />
          </label>

          <label className="patient-field">
            <span>Record or study ID</span>
            <input
              ref={recordNumberRef}
              name="recordNumber"
              type="text"
              autoComplete="off"
              value={recordNumber}
              aria-invalid={Boolean(errors.recordNumber)}
              aria-describedby={
                errors.recordNumber ? 'recordNumber-error' : undefined
              }
              onChange={(event) =>
                setRecordNumber(event.currentTarget.value)
              }
            />
            <FieldError
              id="recordNumber-error"
              message={errors.recordNumber}
            />
          </label>

          <label className="patient-field">
            <span>Date of birth</span>
            <input
              ref={dateOfBirthRef}
              name="dateOfBirth"
              type="date"
              max={todayIso()}
              value={dateOfBirth}
              aria-invalid={Boolean(errors.dateOfBirth)}
              aria-describedby={
                errors.dateOfBirth ? 'dateOfBirth-error' : undefined
              }
              onChange={(event) =>
                setDateOfBirth(event.currentTarget.value)
              }
            />
            <FieldError
              id="dateOfBirth-error"
              message={errors.dateOfBirth}
            />
          </label>

          <label className="patient-field">
            <span>Care pathway</span>
            <select
              ref={carePathwayRef}
              name="carePathway"
              value={carePathway}
              aria-invalid={Boolean(errors.carePathway)}
              aria-describedby={
                errors.carePathway ? 'carePathway-error' : undefined
              }
              onChange={(event) =>
                setCarePathway(event.currentTarget.value)
              }
            >
              <option value="">Select a pathway</option>
              {CARE_PATHWAYS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
            <FieldError
              id="carePathway-error"
              message={errors.carePathway}
            />
          </label>

          <fieldset className="patient-form-section">
            <legend>First photo visit</legend>
            <p>
              Choose the planned timepoint before adding a photograph.
            </p>
            <label className="patient-field">
              <span>First visit timepoint</span>
              <select
                ref={timepointRef}
                name="timepoint"
                value={timepoint}
                aria-invalid={Boolean(errors.timepoint)}
                aria-describedby={
                  errors.timepoint ? 'timepoint-error' : undefined
                }
                onChange={(event) =>
                  setTimepoint(event.currentTarget.value)
                }
              >
                <option value="">Select a timepoint</option>
                <option value="preoperative">Preoperative</option>
                <option value="postoperative">Postoperative</option>
                <option value="follow_up">Follow-up</option>
              </select>
              <FieldError
                id="timepoint-error"
                message={errors.timepoint}
              />
            </label>

            <label className="patient-field">
              <span>First visit date</span>
              <input
                ref={visitDateRef}
                name="visitDate"
                type="date"
                max={todayIso()}
                value={visitDate}
                aria-invalid={Boolean(errors.visitDate)}
                aria-describedby={
                  errors.visitDate ? 'visitDate-error' : undefined
                }
                onChange={(event) =>
                  setVisitDate(event.currentTarget.value)
                }
              />
              <FieldError
                id="visitDate-error"
                message={errors.visitDate}
              />
            </label>
          </fieldset>

          <section
            className="patient-attestation"
            aria-labelledby="prototype-attestation-title"
          >
            <h2 id="prototype-attestation-title">
              Synthetic/test information only
            </h2>
            <p>
              Only synthetic or test information may be entered. Do not
              enter real patient information.
            </p>
            <label>
              <input
                ref={attestationRef}
                name="syntheticTestAttestation"
                type="checkbox"
                checked={syntheticTestAttestation}
                aria-invalid={Boolean(
                  errors.syntheticTestAttestation,
                )}
                aria-describedby={
                  errors.syntheticTestAttestation
                    ? 'syntheticTestAttestation-error'
                    : undefined
                }
                onChange={(event) =>
                  setSyntheticTestAttestation(
                    event.currentTarget.checked,
                  )
                }
              />
              <span>
                I confirm this record contains synthetic/test
                information only.
              </span>
            </label>
            <FieldError
              id="syntheticTestAttestation-error"
              message={errors.syntheticTestAttestation}
            />
          </section>

          <button
            className="patient-primary-action"
            type="submit"
          >
            Save and add photo
          </button>
        </form>
      </div>
    </div>
  )
}
