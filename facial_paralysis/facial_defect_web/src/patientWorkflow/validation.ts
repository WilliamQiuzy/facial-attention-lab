import type {
  AuthorizationSnapshotId,
  CaptureAssetId,
  PatientId,
  PatientResultId,
  PatientReviewId,
  PatientRunId,
  PatientTimepoint,
  PatientVisitId,
  PatientWorkflowState,
  SessionMediaHandle,
} from './types'

export type PatientDraft = {
  readonly displayName: string
  readonly recordNumber: string
  readonly dateOfBirth: string
  readonly carePathway: string
  readonly syntheticTestAttestation: boolean
}

export type NormalizedPatientDraft = Omit<
  PatientDraft,
  'syntheticTestAttestation'
>

export type PatientDraftErrors = Partial<
  Record<keyof PatientDraft, string>
>

export type PatientDraftValidation =
  | {
      readonly ok: true
      readonly value: NormalizedPatientDraft
    }
  | {
      readonly ok: false
      readonly errors: PatientDraftErrors
    }

export type PatientVisitDraft = {
  readonly timepoint: PatientTimepoint | ''
  readonly visitDate: string
}

export type PatientVisitDraftErrors = Partial<
  Record<keyof PatientVisitDraft, string>
>

export type PatientVisitDraftValidation =
  | {
      readonly ok: true
      readonly value: {
        readonly timepoint: PatientTimepoint
        readonly visitDate: string
      }
    }
  | {
      readonly ok: false
      readonly errors: PatientVisitDraftErrors
    }

const PATIENT_TIMEPOINTS: readonly PatientTimepoint[] = [
  'preoperative',
  'postoperative',
  'follow_up',
]

const SESSION_MEDIA_TOKEN_PATTERN = /^[A-Za-z0-9][A-Za-z0-9_-]{7,127}$/
const SESSION_MEDIA_HANDLE_PATTERN =
  /^session-media:[A-Za-z0-9][A-Za-z0-9_-]{7,127}$/
const ENTITY_ID_TOKEN_PATTERN = /^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/

function hasEntityIdFormat(
  value: unknown,
  prefix: string,
): value is string {
  if (typeof value !== 'string' || value !== value.trim()) return false
  if (!value.startsWith(prefix)) return false
  return ENTITY_ID_TOKEN_PATTERN.test(value.slice(prefix.length))
}

function createEntityId<T extends string>(
  value: string,
  prefix: string,
  label: string,
): T {
  if (!hasEntityIdFormat(value, prefix)) {
    throw new TypeError(`Invalid ${label} ID.`)
  }
  return value as T
}

export function isPatientId(value: unknown): value is PatientId {
  return hasEntityIdFormat(value, 'patient-')
}

export function createPatientId(value: string): PatientId {
  return createEntityId<PatientId>(value, 'patient-', 'patient')
}

export function isPatientVisitId(value: unknown): value is PatientVisitId {
  return hasEntityIdFormat(value, 'visit-')
}

export function createPatientVisitId(value: string): PatientVisitId {
  return createEntityId<PatientVisitId>(value, 'visit-', 'visit')
}

export function isAuthorizationSnapshotId(
  value: unknown,
): value is AuthorizationSnapshotId {
  return hasEntityIdFormat(value, 'authorization-')
}

export function createAuthorizationSnapshotId(
  value: string,
): AuthorizationSnapshotId {
  return createEntityId<AuthorizationSnapshotId>(
    value,
    'authorization-',
    'authorization',
  )
}

export function isCaptureAssetId(value: unknown): value is CaptureAssetId {
  return hasEntityIdFormat(value, 'capture-')
}

export function createCaptureAssetId(value: string): CaptureAssetId {
  return createEntityId<CaptureAssetId>(value, 'capture-', 'capture')
}

export function isPatientRunId(value: unknown): value is PatientRunId {
  return hasEntityIdFormat(value, 'run-')
}

export function createPatientRunId(value: string): PatientRunId {
  return createEntityId<PatientRunId>(value, 'run-', 'run')
}

export function isPatientResultId(value: unknown): value is PatientResultId {
  return hasEntityIdFormat(value, 'result-')
}

export function createPatientResultId(value: string): PatientResultId {
  return createEntityId<PatientResultId>(value, 'result-', 'result')
}

export function isPatientReviewId(value: unknown): value is PatientReviewId {
  return hasEntityIdFormat(value, 'review-')
}

export function createPatientReviewId(value: string): PatientReviewId {
  return createEntityId<PatientReviewId>(value, 'review-', 'review')
}

export function createSessionMediaHandle(
  token: string,
): SessionMediaHandle {
  if (
    token !== token.trim() ||
    !SESSION_MEDIA_TOKEN_PATTERN.test(token)
  ) {
    throw new TypeError('Invalid session media token.')
  }
  return `session-media:${token}` as SessionMediaHandle
}

export function isSessionMediaHandle(
  value: unknown,
): value is SessionMediaHandle {
  return (
    typeof value === 'string' &&
    SESSION_MEDIA_HANDLE_PATTERN.test(value)
  )
}

export function normalizeRecordNumber(value: string): string {
  return value
    .trim()
    .toUpperCase()
    .replace(/[\s_]+/g, '-')
    .replace(/[^A-Z0-9-]/g, '')
    .replace(/-+/g, '-')
    .replace(/^-|-$/g, '')
}

function recordNumberComparisonKey(value: string): string {
  return normalizeRecordNumber(value).replace(/-/g, '')
}

export function isValidIsoDate(value: string): boolean {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(value)
  if (!match) return false

  const year = Number(match[1])
  const month = Number(match[2])
  const day = Number(match[3])
  const date = new Date(Date.UTC(year, month - 1, day))

  return (
    date.getUTCFullYear() === year &&
    date.getUTCMonth() === month - 1 &&
    date.getUTCDate() === day
  )
}

export function validatePatientDraft(
  draft: PatientDraft,
  state: PatientWorkflowState,
  today: string,
): PatientDraftValidation {
  const value: NormalizedPatientDraft = {
    displayName: draft.displayName.trim(),
    recordNumber: normalizeRecordNumber(draft.recordNumber),
    dateOfBirth: draft.dateOfBirth.trim(),
    carePathway: draft.carePathway.trim(),
  }
  const errors: PatientDraftErrors = {}
  const trustedTodayIsValid = isValidIsoDate(today)

  if (!value.displayName) {
    errors.displayName = 'Display name is required.'
  }
  if (!value.recordNumber) {
    errors.recordNumber = 'Record number is required.'
  }
  if (!trustedTodayIsValid) {
    errors.dateOfBirth =
      'Date of birth cannot be validated because the trusted current date is invalid.'
  } else if (!value.dateOfBirth) {
    errors.dateOfBirth = 'Date of birth is required.'
  } else if (!isValidIsoDate(value.dateOfBirth)) {
    errors.dateOfBirth = 'Enter a valid date of birth.'
  } else if (value.dateOfBirth > today) {
    errors.dateOfBirth = 'Date of birth cannot be in the future.'
  }
  if (!value.carePathway) {
    errors.carePathway = 'Care pathway is required.'
  }
  if (draft.syntheticTestAttestation !== true) {
    errors.syntheticTestAttestation =
      'Confirm that only synthetic/test information is being entered.'
  }

  if (value.recordNumber) {
    const comparisonKey = recordNumberComparisonKey(value.recordNumber)
    const duplicate = Object.values(state.patientsById).some(
      (patient) =>
        patient !== undefined &&
        recordNumberComparisonKey(patient.recordNumber) === comparisonKey,
    )
    if (duplicate) {
      errors.recordNumber =
        'Record number is already in use in this session.'
    }
  }

  return Object.keys(errors).length > 0
    ? { ok: false, errors }
    : { ok: true, value }
}

export function validateVisitDraft(
  draft: PatientVisitDraft,
  today: string,
): PatientVisitDraftValidation {
  const visitDate = draft.visitDate.trim()
  const errors: PatientVisitDraftErrors = {}
  const trustedTodayIsValid = isValidIsoDate(today)

  if (
    !draft.timepoint ||
    !PATIENT_TIMEPOINTS.includes(draft.timepoint as PatientTimepoint)
  ) {
    errors.timepoint = 'Timepoint is required.'
  }
  if (!trustedTodayIsValid) {
    errors.visitDate =
      'Visit date cannot be validated because the trusted current date is invalid.'
  } else if (!visitDate) {
    errors.visitDate = 'Visit date is required.'
  } else if (!isValidIsoDate(visitDate)) {
    errors.visitDate = 'Enter a valid visit date.'
  } else if (visitDate > today) {
    errors.visitDate = 'Visit date cannot be in the future.'
  }

  return Object.keys(errors).length > 0
    ? { ok: false, errors }
    : {
        ok: true,
        value: {
          timepoint: draft.timepoint as PatientTimepoint,
          visitDate,
        },
      }
}

export function isSupportedPatientTimepoint(
  value: string,
): value is PatientTimepoint {
  return PATIENT_TIMEPOINTS.includes(value as PatientTimepoint)
}
