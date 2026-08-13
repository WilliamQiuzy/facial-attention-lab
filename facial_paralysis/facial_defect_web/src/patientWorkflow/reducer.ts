import {
  getOwnRecordValue,
  isCaptureQualityComplete,
  samePatientRunBinding,
  selectCurrentAuthorization,
  selectCurrentCapture,
  selectCurrentResult,
  selectCurrentRun,
} from './selectors'
import type {
  AuthorizationSnapshot,
  CaptureAsset,
  CaptureQualityChecks,
  PatientRecord,
  PatientNormalizedPoint,
  PatientResult,
  PatientReview,
  PatientRun,
  PatientRunBinding,
  PatientVisit,
  PatientWorkflowAction,
  PatientWorkflowFailure,
  PatientWorkflowFailureCode,
  PatientWorkflowState,
} from './types'
import {
  isAuthorizationSnapshotId,
  isCaptureAssetId,
  isPatientId,
  isPatientResultId,
  isPatientReviewId,
  isPatientRunId,
  isPatientVisitId,
  isSupportedPatientTimepoint,
  isSessionMediaHandle,
  isValidIsoDate,
  normalizeRecordNumber,
  validatePatientDraft,
  validateVisitDraft,
} from './validation'

const EMPTY_STATE: PatientWorkflowState = {
  patientsById: {},
  patientOrder: [],
  visitsById: {},
  visitOrder: [],
  authorizationsById: {},
  authorizationOrder: [],
  capturesById: {},
  captureOrder: [],
  runsById: {},
  runOrder: [],
  resultsById: {},
  resultOrder: [],
  reviewsById: {},
  reviewOrder: [],
}

function deepFreeze<T>(value: T): T {
  if (value === null || typeof value !== 'object' || Object.isFrozen(value)) {
    return value
  }

  Object.freeze(value)
  for (const nested of Object.values(value)) deepFreeze(nested)
  return value
}

function clearFailure(
  state: PatientWorkflowState,
): Omit<PatientWorkflowState, 'lastFailure'> {
  const { lastFailure: _lastFailure, ...rest } = state
  return rest
}

function failureState(
  state: PatientWorkflowState,
  code: PatientWorkflowFailureCode,
  message: string,
  field?: string,
): PatientWorkflowState {
  return {
    ...state,
    lastFailure: deepFreeze({
      code,
      message,
      ...(field ? { field } : {}),
    }),
  }
}

function timestampDate(timestamp: string): string | undefined {
  const date = timestamp.slice(0, 10)
  return isValidIsoDate(date) ? date : undefined
}

function validIdentifier(value: string): boolean {
  return value.trim().length > 0
}

function validQualityChecks(
  checks: CaptureQualityChecks,
): checks is CaptureQualityChecks {
  return (
    checks !== null &&
    typeof checks === 'object' &&
    typeof checks.faceVisibleAndCentered === 'boolean' &&
    typeof checks.focusLightingAndOcclusionAcceptable === 'boolean' &&
    typeof checks.orientationConfirmed === 'boolean' &&
    typeof checks.authorizationDocumented === 'boolean'
  )
}

function copyBinding(binding: PatientRunBinding): PatientRunBinding {
  return deepFreeze({
    patientId: binding.patientId,
    visitId: binding.visitId,
    captureId: binding.captureId,
    captureVersion: binding.captureVersion,
    captureSha256: binding.captureSha256,
    mediaHandle: binding.mediaHandle,
    authorizationRevision: binding.authorizationRevision,
    captureProtocol: binding.captureProtocol,
  })
}

function bindingMatchesCurrentState(
  state: PatientWorkflowState,
  binding: PatientRunBinding,
): boolean {
  const capture = selectCurrentCapture(state, binding.visitId)
  const authorization = selectCurrentAuthorization(state, binding.visitId)
  const visit = getOwnRecordValue(state.visitsById, binding.visitId)

  return (
    isPatientId(binding.patientId) &&
    isPatientVisitId(binding.visitId) &&
    isCaptureAssetId(binding.captureId) &&
    getOwnRecordValue(state.patientsById, binding.patientId) !== undefined &&
    visit?.patientId === binding.patientId &&
    capture !== undefined &&
    capture.patientId === binding.patientId &&
    capture.id === binding.captureId &&
    capture.version === binding.captureVersion &&
    capture.sha256 === binding.captureSha256 &&
    capture.mediaHandle === binding.mediaHandle &&
    isSessionMediaHandle(capture.mediaHandle) &&
    isSessionMediaHandle(binding.mediaHandle) &&
    capture.captureProtocol === binding.captureProtocol &&
    capture.qualityConfirmedAt !== undefined &&
    isCaptureQualityComplete(capture.qualityChecks) &&
    authorization?.patientId === binding.patientId &&
    authorization.status === 'documented' &&
    authorization.revision === binding.authorizationRevision
  )
}

function createPatient(
  state: PatientWorkflowState,
  patient: PatientRecord,
  trustedToday: string,
  syntheticTestAttestation: boolean,
): PatientWorkflowState {
  if (!isPatientId(patient.id)) {
    return failureState(
      state,
      'INVALID_PATIENT',
      'Patient ID is required.',
      'id',
    )
  }
  const patientId = patient.id
  if (getOwnRecordValue(state.patientsById, patientId)) {
    return failureState(
      state,
      'DUPLICATE_PATIENT_ID',
      `Patient ID ${patientId} already exists.`,
      'id',
    )
  }
  if (
    patient.recordKind !== 'synthetic_demo' &&
    patient.recordKind !== 'session_test'
  ) {
    return failureState(
      state,
      'INVALID_PATIENT',
      'Patient record kind is invalid.',
      'recordKind',
    )
  }

  const createdDate = timestampDate(patient.createdAt)
  if (
    !isValidIsoDate(trustedToday) ||
    !createdDate ||
    !Number.isFinite(Date.parse(patient.createdAt)) ||
    createdDate > trustedToday
  ) {
    return failureState(
      state,
      'INVALID_PATIENT',
      'Patient creation time is invalid.',
      'createdAt',
    )
  }

  const validation = validatePatientDraft(
    {
      displayName: patient.displayName,
      recordNumber: patient.recordNumber,
      dateOfBirth: patient.dateOfBirth,
      carePathway: patient.carePathway,
      syntheticTestAttestation,
    },
    state,
    trustedToday,
  )
  if (!validation.ok) {
    const [field, message] = Object.entries(validation.errors)[0] ?? [
      'patient',
      'Patient record is invalid.',
    ]
    return failureState(
      state,
      field === 'recordNumber'
        ? 'DUPLICATE_RECORD_NUMBER'
        : 'INVALID_PATIENT',
      message,
      field,
    )
  }

  const stored: PatientRecord = deepFreeze({
    id: patientId,
    ...validation.value,
    recordKind: patient.recordKind,
    createdAt: patient.createdAt,
  })

  return {
    ...clearFailure(state),
    patientsById: {
      ...state.patientsById,
      [stored.id]: stored,
    },
    patientOrder: [...state.patientOrder, stored.id],
  }
}

function createVisit(
  state: PatientWorkflowState,
  visit: PatientVisit,
  trustedToday: string,
): PatientWorkflowState {
  if (!isPatientVisitId(visit.id) || !isPatientId(visit.patientId)) {
    return failureState(
      state,
      'INVALID_VISIT',
      'Visit ID or patient ID is invalid.',
    )
  }
  if (!getOwnRecordValue(state.patientsById, visit.patientId)) {
    return failureState(
      state,
      'UNKNOWN_PATIENT',
      `Unknown patient: ${visit.patientId}.`,
      'patientId',
    )
  }
  const visitId = visit.id
  if (getOwnRecordValue(state.visitsById, visitId)) {
    return failureState(
      state,
      'DUPLICATE_VISIT_ID',
      `Visit ID ${visitId} already exists.`,
      'id',
    )
  }

  const createdDate = timestampDate(visit.createdAt)
  if (
    !isValidIsoDate(trustedToday) ||
    !createdDate ||
    !Number.isFinite(Date.parse(visit.createdAt)) ||
    createdDate > trustedToday ||
    !isSupportedPatientTimepoint(visit.timepoint)
  ) {
    return failureState(
      state,
      'INVALID_VISIT',
      'Visit timepoint or creation time is invalid.',
    )
  }
  const validation = validateVisitDraft(
    { timepoint: visit.timepoint, visitDate: visit.visitDate },
    trustedToday,
  )
  if (!validation.ok) {
    const [field, message] = Object.entries(validation.errors)[0] ?? [
      'visit',
      'Visit is invalid.',
    ]
    return failureState(state, 'INVALID_VISIT', message, field)
  }

  const stored: PatientVisit = deepFreeze({
    id: visitId,
    patientId: visit.patientId,
    ...validation.value,
    createdAt: visit.createdAt,
  })
  return {
    ...clearFailure(state),
    visitsById: {
      ...state.visitsById,
      [stored.id]: stored,
    },
    visitOrder: [...state.visitOrder, stored.id],
  }
}

function recordAuthorization(
  state: PatientWorkflowState,
  authorization: AuthorizationSnapshot,
): PatientWorkflowState {
  if (
    !isAuthorizationSnapshotId(authorization.id) ||
    !isPatientId(authorization.patientId) ||
    !isPatientVisitId(authorization.visitId)
  ) {
    return failureState(
      state,
      'INVALID_AUTHORIZATION',
      'Authorization identity is invalid.',
    )
  }
  const authorizationId = authorization.id
  const visit = getOwnRecordValue(
    state.visitsById,
    authorization.visitId,
  )
  if (!visit) {
    return failureState(
      state,
      'UNKNOWN_VISIT',
      `Unknown visit: ${authorization.visitId}.`,
      'visitId',
    )
  }
  if (
    !getOwnRecordValue(state.patientsById, authorization.patientId) ||
    visit.patientId !== authorization.patientId
  ) {
    return failureState(
      state,
      'VISIT_OWNERSHIP_MISMATCH',
      'Authorization patient does not own the visit.',
      'patientId',
    )
  }
  if (
    !validIdentifier(authorization.recordedAt) ||
    !['documented', 'withdrawn'].includes(authorization.status)
  ) {
    return failureState(
      state,
      'INVALID_AUTHORIZATION',
      'Authorization snapshot is invalid.',
    )
  }
  if (getOwnRecordValue(state.authorizationsById, authorizationId)) {
    return failureState(
      state,
      'DUPLICATE_AUTHORIZATION_ID',
      `Authorization ID ${authorizationId} already exists.`,
      'id',
    )
  }

  const current = selectCurrentAuthorization(state, visit.id)
  const expectedRevision = (current?.revision ?? 0) + 1
  if (authorization.revision !== expectedRevision) {
    return failureState(
      state,
      'INVALID_AUTHORIZATION_REVISION',
      `Authorization revision must be ${expectedRevision}.`,
      'revision',
    )
  }

  const stored: AuthorizationSnapshot = deepFreeze({
    id: authorizationId,
    patientId: authorization.patientId,
    visitId: authorization.visitId,
    revision: authorization.revision,
    status: authorization.status,
    recordedAt: authorization.recordedAt,
  })
  return {
    ...clearFailure(state),
    authorizationsById: {
      ...state.authorizationsById,
      [stored.id]: stored,
    },
    authorizationOrder: [...state.authorizationOrder, stored.id],
  }
}

function validCapture(capture: CaptureAsset): boolean {
  return (
    isCaptureAssetId(capture.id) &&
    isPatientId(capture.patientId) &&
    isPatientVisitId(capture.visitId) &&
    isSessionMediaHandle(capture.mediaHandle) &&
    /^[a-f0-9]{64}$/i.test(capture.sha256) &&
    ['camera', 'upload', 'synthetic_demo'].includes(capture.source) &&
    ['image/jpeg', 'image/png', 'image/webp'].includes(capture.mimeType) &&
    Number.isInteger(capture.sizeBytes) &&
    capture.sizeBytes > 0 &&
    Number.isInteger(capture.width) &&
    capture.width > 0 &&
    Number.isInteger(capture.height) &&
    capture.height > 0 &&
    capture.captureProtocol === 'frontal_relaxed_non_mirrored_v1' &&
    capture.status === 'current' &&
    validQualityChecks(capture.qualityChecks) &&
    validIdentifier(capture.capturedAt) &&
    (capture.source !== 'synthetic_demo' ||
      capture.syntheticSourceAssetId !== undefined)
  )
}

function copyCapture(capture: CaptureAsset): CaptureAsset {
  const complete = isCaptureQualityComplete(capture.qualityChecks)
  return deepFreeze({
    id: capture.id,
    patientId: capture.patientId,
    visitId: capture.visitId,
    version: capture.version,
    status: 'current',
    source: capture.source,
    mediaHandle: capture.mediaHandle,
    sha256: capture.sha256.toLowerCase(),
    mimeType: capture.mimeType,
    sizeBytes: capture.sizeBytes,
    width: capture.width,
    height: capture.height,
    captureProtocol: capture.captureProtocol,
    qualityChecks: {
      faceVisibleAndCentered: capture.qualityChecks.faceVisibleAndCentered,
      focusLightingAndOcclusionAcceptable:
        capture.qualityChecks.focusLightingAndOcclusionAcceptable,
      orientationConfirmed: capture.qualityChecks.orientationConfirmed,
      authorizationDocumented:
        capture.qualityChecks.authorizationDocumented,
    },
    capturedAt: capture.capturedAt,
    ...(complete && capture.qualityConfirmedAt
      ? { qualityConfirmedAt: capture.qualityConfirmedAt }
      : {}),
    ...(capture.syntheticSourceAssetId
      ? { syntheticSourceAssetId: capture.syntheticSourceAssetId }
      : {}),
  })
}

function addCapture(
  state: PatientWorkflowState,
  capture: CaptureAsset,
): PatientWorkflowState {
  const captureId = capture.id
  if (!validCapture(capture)) {
    return failureState(
      state,
      'INVALID_CAPTURE',
      'Capture metadata is invalid or includes no valid media handle.',
    )
  }
  const visit = getOwnRecordValue(state.visitsById, capture.visitId)
  if (!visit) {
    return failureState(
      state,
      'UNKNOWN_VISIT',
      `Unknown visit: ${capture.visitId}.`,
      'visitId',
    )
  }
  if (
    !getOwnRecordValue(state.patientsById, capture.patientId) ||
    visit.patientId !== capture.patientId
  ) {
    return failureState(
      state,
      'VISIT_OWNERSHIP_MISMATCH',
      'Capture patient does not own the visit.',
      'patientId',
    )
  }
  if (getOwnRecordValue(state.capturesById, captureId)) {
    return failureState(
      state,
      'DUPLICATE_CAPTURE_ID',
      `Capture ID ${captureId} already exists.`,
      'id',
    )
  }

  const current = selectCurrentCapture(state, visit.id)
  const expectedVersion = (current?.version ?? 0) + 1
  if (capture.version !== expectedVersion) {
    return failureState(
      state,
      'INVALID_CAPTURE_VERSION',
      `Capture version must be ${expectedVersion}.`,
      'version',
    )
  }

  const stored = copyCapture(capture)
  const capturesById: Partial<Record<string, CaptureAsset>> = {
    ...state.capturesById,
    [stored.id]: stored,
  }
  if (current) {
    capturesById[current.id] = deepFreeze({
      ...current,
      status: 'superseded',
      supersededByCaptureId: stored.id,
    })
  }

  const resultsById: Partial<Record<string, PatientResult>> = {
    ...state.resultsById,
  }
  for (const [resultId, result] of Object.entries(state.resultsById)) {
    if (
      result &&
      result.binding.visitId === visit.id &&
      result.freshness === 'current'
    ) {
      resultsById[resultId] = deepFreeze({
        ...result,
        freshness: 'stale',
      })
    }
  }

  return {
    ...clearFailure(state),
    capturesById,
    captureOrder: [...state.captureOrder, stored.id],
    resultsById,
  }
}

function setCaptureQuality(
  state: PatientWorkflowState,
  captureId: string,
  checks: CaptureQualityChecks,
  confirmedAt?: string,
): PatientWorkflowState {
  const capture = getOwnRecordValue(state.capturesById, captureId)
  if (!capture) {
    return failureState(
      state,
      'UNKNOWN_CAPTURE',
      `Unknown capture: ${captureId}.`,
      'captureId',
    )
  }
  if (
    capture.status !== 'current' ||
    selectCurrentCapture(state, capture.visitId)?.id !== capture.id
  ) {
    return failureState(
      state,
      'CAPTURE_NOT_CURRENT',
      'Quality may only be changed on the current capture.',
      'captureId',
    )
  }
  const qualityLocked = state.runOrder.some(
    (runId) =>
      getOwnRecordValue(state.runsById, runId)?.binding.captureId ===
      capture.id,
  )
  if (qualityLocked) {
    return failureState(
      state,
      'CAPTURE_QUALITY_LOCKED',
      'Capture quality is immutable after analysis has been queued.',
      'captureId',
    )
  }
  if (!validQualityChecks(checks)) {
    return failureState(
      state,
      'INVALID_CAPTURE',
      'Capture quality checks are invalid.',
      'checks',
    )
  }

  const complete = isCaptureQualityComplete(checks)
  const stored: CaptureAsset = deepFreeze({
    ...capture,
    qualityChecks: {
      faceVisibleAndCentered: checks.faceVisibleAndCentered,
      focusLightingAndOcclusionAcceptable:
        checks.focusLightingAndOcclusionAcceptable,
      orientationConfirmed: checks.orientationConfirmed,
      authorizationDocumented: checks.authorizationDocumented,
    },
    ...(complete && confirmedAt ? { qualityConfirmedAt: confirmedAt } : {}),
    ...(!complete ? { qualityConfirmedAt: undefined } : {}),
  })

  return {
    ...clearFailure(state),
    capturesById: {
      ...state.capturesById,
      [capture.id]: stored,
    },
  }
}

function createRun(
  state: PatientWorkflowState,
  run: PatientRun,
): PatientWorkflowState {
  if (
    !isPatientRunId(run.id) ||
    (run.retryOfRunId !== undefined &&
      !isPatientRunId(run.retryOfRunId)) ||
    run.status !== 'queued'
  ) {
    return failureState(
      state,
      'INVALID_RUN_BINDING',
      'A new run must have an ID and begin queued.',
    )
  }
  const runId = run.id
  if (getOwnRecordValue(state.runsById, runId)) {
    return failureState(
      state,
      'DUPLICATE_RUN_ID',
      `Run ID ${runId} already exists.`,
      'id',
    )
  }

  const current = selectCurrentRun(state, run.binding.visitId)
  if (run.retryOfRunId) {
    const prior = getOwnRecordValue(state.runsById, run.retryOfRunId)
    if (
      !prior ||
      prior.status !== 'failed' ||
      current?.id !== prior.id ||
      !samePatientRunBinding(prior.binding, run.binding)
    ) {
      return failureState(
        state,
        'INVALID_RETRY_BINDING',
        'A retry must preserve the exact binding of a failed run.',
        'retryOfRunId',
      )
    }
  } else {
    if (current) {
      return failureState(
        state,
        'INVALID_RUN_BINDING',
        'The current capture already has a run.',
      )
    }
  }

  if (!bindingMatchesCurrentState(state, run.binding)) {
    return failureState(
      state,
      'INVALID_RUN_BINDING',
      'Run binding does not match the current authorized capture.',
      'binding',
    )
  }

  const stored: PatientRun = deepFreeze({
    id: runId,
    status: 'queued',
    binding: copyBinding(run.binding),
    createdAt: run.createdAt,
    ...(run.retryOfRunId ? { retryOfRunId: run.retryOfRunId } : {}),
  })
  return {
    ...clearFailure(state),
    runsById: {
      ...state.runsById,
      [stored.id]: stored,
    },
    runOrder: [...state.runOrder, stored.id],
  }
}

function setRunStatus(
  state: PatientWorkflowState,
  action: Extract<
    PatientWorkflowAction,
    { readonly type: 'run/status/set' }
  >,
): PatientWorkflowState {
  const run = getOwnRecordValue(state.runsById, action.runId)
  if (!run) {
    return failureState(
      state,
      'UNKNOWN_RUN',
      `Unknown run: ${action.runId}.`,
      'runId',
    )
  }
  if (run.status === action.status) return clearFailure(state)

  if (
    run.status === 'queued' &&
    action.status === 'running' &&
    !bindingMatchesCurrentState(state, run.binding)
  ) {
    return failureState(
      state,
      'INVALID_RUN_BINDING',
      'Run binding is no longer current or launch-ready.',
      'binding',
    )
  }

  const legal =
    (run.status === 'queued' &&
      (action.status === 'running' || action.status === 'failed')) ||
    (run.status === 'running' &&
      (action.status === 'succeeded' || action.status === 'failed'))
  if (!legal) {
    return failureState(
      state,
      'INVALID_RUN_TRANSITION',
      `Run cannot move from ${run.status} to ${action.status}.`,
      'status',
    )
  }

  let storedFailure: PatientWorkflowFailure | undefined
  if (action.status === 'failed') {
    const supplied = action.failure
    if (
      supplied !== undefined &&
      (supplied === null ||
        typeof supplied !== 'object' ||
        supplied.code !== 'ANALYSIS_FAILED' ||
        typeof supplied.message !== 'string' ||
        supplied.message.trim().length === 0 ||
        supplied.message.length > 500 ||
        (supplied.field !== undefined &&
          (typeof supplied.field !== 'string' ||
            !/^[A-Za-z][A-Za-z0-9_.-]{0,63}$/.test(supplied.field))))
    ) {
      return failureState(
        state,
        'INVALID_RUN_FAILURE',
        'Supplied run failure is invalid.',
        'failure',
      )
    }
    storedFailure = {
      code: 'ANALYSIS_FAILED',
      message: supplied?.message.trim() ?? 'Analysis failed.',
      ...(supplied?.field ? { field: supplied.field } : {}),
    }
  }

  const stored: PatientRun = deepFreeze({
    id: run.id,
    status: action.status,
    binding: run.binding,
    createdAt: run.createdAt,
    ...(run.retryOfRunId ? { retryOfRunId: run.retryOfRunId } : {}),
    ...(storedFailure ? { failure: storedFailure } : {}),
  })

  return {
    ...clearFailure(state),
    runsById: {
      ...state.runsById,
      [run.id]: stored,
    },
  }
}

function validFaceRegistration(
  state: PatientWorkflowState,
  result: PatientResult,
): boolean {
  const capture = getOwnRecordValue(
    state.capturesById,
    result.binding.captureId,
  )
  const registration = result.faceRegistration
  const requiredFeatures = new Set([
    'face_oval',
    'left_eye',
    'right_eye',
    'left_eyebrow',
    'right_eyebrow',
    'lips',
  ])
  const features = new Set<string>()
  const ovalPoints =
    registration?.paths
      ?.filter((path) => path.feature === 'face_oval')
      .flatMap((path) => path.points) ?? []
  const ovalWidth =
    ovalPoints.length > 0
      ? Math.max(...ovalPoints.map((point) => point.x)) -
        Math.min(...ovalPoints.map((point) => point.x))
      : 0
  const ovalHeight =
    ovalPoints.length > 0
      ? Math.max(...ovalPoints.map((point) => point.y)) -
        Math.min(...ovalPoints.map((point) => point.y))
      : 0

  return (
    capture !== undefined &&
    registration?.schemaVersion === 'patient-face-registration/1' &&
    registration.source === 'on_device_face_landmarks' &&
    registration.coordinateSpace ===
      'decoded_image_normalized_v1' &&
    registration.captureSha256 === result.binding.captureSha256 &&
    registration.captureSha256 === capture.sha256 &&
    registration.sourceWidth === capture.width &&
    registration.sourceHeight === capture.height &&
    registration.captureProtocol ===
      result.binding.captureProtocol &&
    registration.captureProtocol === capture.captureProtocol &&
    registration.detectorId === 'mediapipe_face_landmarker' &&
    registration.detectorVersion ===
      'tasks-vision-1.0.0-model-float16-1' &&
    registration.faceCount === 1 &&
    Array.isArray(registration.paths) &&
    registration.paths.length >= requiredFeatures.size &&
    registration.paths.length <= 16 &&
    ovalWidth >= 0.08 &&
    ovalHeight >= 0.08 &&
    registration.paths.every((path) => {
      if (
        !requiredFeatures.has(path.feature) ||
        typeof path.closed !== 'boolean' ||
        !Array.isArray(path.points) ||
        path.points.length < 2 ||
        path.points.length > 128
      ) {
        return false
      }
      features.add(path.feature)
      return path.points.every(
        (point: PatientNormalizedPoint) =>
          Number.isFinite(point.x) &&
          point.x >= 0 &&
          point.x <= 1 &&
          Number.isFinite(point.y) &&
          point.y >= 0 &&
          point.y <= 1,
      )
    }) &&
    features.size === requiredFeatures.size
  )
}

function validResultOutput(
  state: PatientWorkflowState,
  result: PatientResult,
): boolean {
  return (
    validFaceRegistration(state, result) &&
    result.output?.origin === 'workflow_simulation' &&
    Array.isArray(result.output.points) &&
    result.output.points.every(
      (point) =>
        Number.isFinite(point.x) &&
        point.x >= 0 &&
        point.x <= 1 &&
        Number.isFinite(point.y) &&
        point.y >= 0 &&
        point.y <= 1 &&
        Number.isFinite(point.intensity) &&
        point.intensity >= 0 &&
        point.intensity <= 1 &&
        Number.isFinite(point.radius) &&
        point.radius > 0 &&
        point.radius <= 1,
    )
  )
}

function recordResult(
  state: PatientWorkflowState,
  result: PatientResult,
): PatientWorkflowState {
  if (
    !isPatientResultId(result.id) ||
    !isPatientRunId(result.runId)
  ) {
    return failureState(
      state,
      'INVALID_RESULT',
      'Result ID and run ID are required.',
    )
  }
  const resultId = result.id
  if (getOwnRecordValue(state.resultsById, resultId)) {
    return failureState(
      state,
      'DUPLICATE_RESULT_ID',
      `Result ID ${resultId} already exists.`,
      'id',
    )
  }
  const runAlreadyHasResult = Object.values(state.resultsById).some(
    (existing) => existing?.runId === result.runId,
  )
  if (runAlreadyHasResult) {
    return failureState(
      state,
      'INVALID_RESULT',
      'A run may have only one immutable result.',
      'runId',
    )
  }

  const run = getOwnRecordValue(state.runsById, result.runId)
  if (
    !run ||
    run.status !== 'succeeded' ||
    result.freshness !== 'current' ||
    !samePatientRunBinding(run.binding, result.binding) ||
    !bindingMatchesCurrentState(state, result.binding) ||
    !validResultOutput(state, result)
  ) {
    return failureState(
      state,
      'INVALID_RESULT',
      'Result does not match a succeeded, current, exact-bound run.',
    )
  }

  const points = result.output.points.map((point) =>
    deepFreeze({
      x: point.x,
      y: point.y,
      intensity: point.intensity,
      radius: point.radius,
    }),
  )
  const stored: PatientResult = deepFreeze({
    id: resultId,
    runId: result.runId,
    binding: copyBinding(result.binding),
    freshness: 'current',
    createdAt: result.createdAt,
    faceRegistration: {
      ...result.faceRegistration,
      paths: result.faceRegistration.paths.map((path) => ({
        feature: path.feature,
        closed: path.closed,
        points: path.points.map((point) => ({
          x: point.x,
          y: point.y,
        })),
      })),
    },
    output: {
      origin: 'workflow_simulation',
      points,
    },
  })
  const resultsById: Partial<Record<string, PatientResult>> = {
    ...state.resultsById,
  }
  for (const [resultId, existing] of Object.entries(state.resultsById)) {
    if (
      existing &&
      existing.binding.visitId === stored.binding.visitId &&
      existing.freshness === 'current'
    ) {
      resultsById[resultId] = deepFreeze({
        ...existing,
        freshness: 'stale',
      })
    }
  }
  resultsById[stored.id] = stored

  return {
    ...clearFailure(state),
    resultsById,
    resultOrder: [...state.resultOrder, stored.id],
  }
}

function recordReview(
  state: PatientWorkflowState,
  review: PatientReview,
): PatientWorkflowState {
  if (
    !isPatientReviewId(review.id) ||
    !isPatientId(review.patientId) ||
    !isPatientVisitId(review.visitId) ||
    !isPatientResultId(review.resultId) ||
    !isCaptureAssetId(review.captureId)
  ) {
    return failureState(
      state,
      'INVALID_REVIEW',
      'Review ID is required.',
      'id',
    )
  }
  const reviewId = review.id
  if (getOwnRecordValue(state.reviewsById, reviewId)) {
    return failureState(
      state,
      'DUPLICATE_REVIEW_ID',
      `Review ID ${reviewId} already exists.`,
      'id',
    )
  }
  if (
    review.decision !== 'reviewed' &&
    review.decision !== 'repeat_photo'
  ) {
    return failureState(
      state,
      'INVALID_REVIEW',
      'Review decision is invalid.',
      'decision',
    )
  }
  const note = review.note?.trim()
  if (review.decision === 'repeat_photo' && !note) {
    return failureState(
      state,
      'REPEAT_REASON_REQUIRED',
      'A reason is required when requesting a repeat photo.',
      'note',
    )
  }

  const result = getOwnRecordValue(state.resultsById, review.resultId)
  const visit = getOwnRecordValue(state.visitsById, review.visitId)
  const capture = selectCurrentCapture(state, review.visitId)
  const currentResult = selectCurrentResult(state, review.visitId)
  if (
    !result ||
    result.freshness !== 'current' ||
    currentResult?.id !== result.id ||
    !visit ||
    visit.patientId !== review.patientId ||
    result.binding.patientId !== review.patientId ||
    result.binding.visitId !== review.visitId ||
    result.binding.captureId !== review.captureId ||
    capture?.id !== review.captureId
  ) {
    return failureState(
      state,
      'INVALID_REVIEW',
      'Review does not match the current patient result.',
    )
  }
  const existing = Object.values(state.reviewsById).some(
    (entry) => entry?.resultId === result.id,
  )
  if (existing) {
    return failureState(
      state,
      'INVALID_REVIEW',
      'The current result has already been reviewed.',
      'resultId',
    )
  }

  const stored: PatientReview = deepFreeze({
    id: reviewId,
    patientId: review.patientId,
    visitId: review.visitId,
    resultId: review.resultId,
    captureId: review.captureId,
    decision: review.decision,
    ...(note ? { note } : {}),
    completedAt: review.completedAt,
  })
  return {
    ...clearFailure(state),
    reviewsById: {
      ...state.reviewsById,
      [stored.id]: stored,
    },
    reviewOrder: [...state.reviewOrder, stored.id],
  }
}

export function createInitialPatientWorkflowState(): PatientWorkflowState
export function createInitialPatientWorkflowState(
  seedPatients: readonly PatientRecord[],
  trustedToday: string,
): PatientWorkflowState
export function createInitialPatientWorkflowState(
  seedPatients: readonly PatientRecord[] = [],
  trustedToday?: string,
): PatientWorkflowState {
  let state: PatientWorkflowState = {
    ...EMPTY_STATE,
    patientsById: {},
    patientOrder: [],
    visitsById: {},
    visitOrder: [],
    authorizationsById: {},
    authorizationOrder: [],
    capturesById: {},
    captureOrder: [],
    runsById: {},
    runOrder: [],
    resultsById: {},
    resultOrder: [],
    reviewsById: {},
    reviewOrder: [],
  }

  for (const patient of seedPatients) {
    state = createPatient(state, patient, trustedToday ?? '', true)
  }
  return state
}

export function patientWorkflowReducer(
  state: PatientWorkflowState,
  action: PatientWorkflowAction,
): PatientWorkflowState {
  switch (action.type) {
    case 'patient/create':
      return createPatient(
        state,
        action.patient,
        action.trustedToday,
        action.syntheticTestAttestation,
      )
    case 'visit/create':
      return createVisit(state, action.visit, action.trustedToday)
    case 'authorization/record':
      return recordAuthorization(state, action.authorization)
    case 'capture/add':
      return addCapture(state, action.capture)
    case 'capture/quality/set':
      return setCaptureQuality(
        state,
        action.captureId,
        action.checks,
        action.confirmedAt,
      )
    case 'run/create':
      return createRun(state, action.run)
    case 'run/status/set':
      return setRunStatus(state, action)
    case 'result/record':
      return recordResult(state, action.result)
    case 'review/record':
      return recordReview(state, action.review)
  }
}

export type {
  PatientWorkflowAction,
  PatientWorkflowState,
} from './types'

export { normalizeRecordNumber }
