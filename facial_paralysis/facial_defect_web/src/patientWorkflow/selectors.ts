import type {
  AuthorizationSnapshot,
  CaptureAsset,
  CaptureQualityChecks,
  MissingRequiredTimepoints,
  PatientComparisonCaptureEntry,
  PatientComparisonResultEntry,
  PatientComparisonState,
  PatientNextAction,
  PatientResult,
  PatientReview,
  PatientRun,
  PatientRunBinding,
  PatientVisit,
  PatientWorkflowState,
  VisitNextAction,
} from './types'
import {
  isPatientId,
  isPatientVisitId,
  isSessionMediaHandle,
} from './validation'

export function getOwnRecordValue<T>(
  record: Readonly<Partial<Record<string, T>>>,
  key: string,
): T | undefined {
  return Object.hasOwn(record, key) ? record[key] : undefined
}

export function samePatientRunBinding(
  first: PatientRunBinding,
  second: PatientRunBinding,
): boolean {
  return (
    first.patientId === second.patientId &&
    first.visitId === second.visitId &&
    first.captureId === second.captureId &&
    first.captureVersion === second.captureVersion &&
    first.captureSha256 === second.captureSha256 &&
    first.mediaHandle === second.mediaHandle &&
    first.authorizationRevision === second.authorizationRevision &&
    first.captureProtocol === second.captureProtocol
  )
}

export function isCaptureQualityComplete(
  checks: CaptureQualityChecks,
): boolean {
  return (
    checks.faceVisibleAndCentered &&
    checks.focusLightingAndOcclusionAcceptable &&
    checks.orientationConfirmed &&
    checks.authorizationDocumented
  )
}

export function selectPatientVisits(
  state: PatientWorkflowState,
  patientId: string,
): readonly PatientVisit[] {
  if (!isPatientId(patientId)) return []
  return state.visitOrder
    .map((visitId) => getOwnRecordValue(state.visitsById, visitId))
    .filter(
      (visit): visit is PatientVisit =>
        visit !== undefined && visit.patientId === patientId,
    )
    .sort(
      (first, second) => {
        const visitDateOrder = first.visitDate.localeCompare(
          second.visitDate,
        )
        if (visitDateOrder !== 0) return visitDateOrder

        const firstCreatedAt = Date.parse(first.createdAt)
        const secondCreatedAt = Date.parse(second.createdAt)
        if (
          Number.isFinite(firstCreatedAt) &&
          Number.isFinite(secondCreatedAt)
        ) {
          const timestampOrder = firstCreatedAt - secondCreatedAt
          return timestampOrder !== 0
            ? timestampOrder
            : first.id.localeCompare(second.id)
        }

        return (
          first.createdAt.localeCompare(second.createdAt) ||
          first.id.localeCompare(second.id)
        )
      },
    )
}

function missingRequiredTimepoints(
  hasPreoperative: boolean,
  hasPostoperative: boolean,
): MissingRequiredTimepoints | undefined {
  if (!hasPreoperative && !hasPostoperative) {
    return ['preoperative', 'postoperative']
  }
  if (!hasPreoperative) return ['preoperative']
  if (!hasPostoperative) return ['postoperative']
  return undefined
}

export function selectPatientComparisonState(
  state: PatientWorkflowState,
  patientId: string,
): PatientComparisonState {
  const visits = selectPatientVisits(state, patientId)
  if (visits.length === 0) return { phase: 'no_visits' }

  const preoperative = visits
    .filter((visit) => visit.timepoint === 'preoperative')
    .at(-1)
  const postoperative = visits
    .filter((visit) => visit.timepoint === 'postoperative')
    .at(-1)
  const missing = missingRequiredTimepoints(
    preoperative !== undefined,
    postoperative !== undefined,
  )
  if (missing) return { phase: 'missing_timepoint', missing }

  // The checks above prove both visits exist while keeping the public union
  // free of optional fields and contradictory partial pairs.
  const visitPair = {
    preoperative: preoperative!,
    postoperative: postoperative!,
  }
  const selectedPreoperativeCapture = selectCurrentCapture(
    state,
    visitPair.preoperative.id,
  )
  const selectedPostoperativeCapture = selectCurrentCapture(
    state,
    visitPair.postoperative.id,
  )
  const preoperativeCapture =
    selectedPreoperativeCapture?.patientId === patientId &&
    selectedPreoperativeCapture.visitId === visitPair.preoperative.id
      ? selectedPreoperativeCapture
      : undefined
  const postoperativeCapture =
    selectedPostoperativeCapture?.patientId === patientId &&
    selectedPostoperativeCapture.visitId === visitPair.postoperative.id
      ? selectedPostoperativeCapture
      : undefined
  const missingPhotos = missingRequiredTimepoints(
    preoperativeCapture !== undefined,
    postoperativeCapture !== undefined,
  )
  if (missingPhotos) {
    return { phase: 'needs_photos', pair: visitPair, missingPhotos }
  }

  const capturePair = {
    preoperative: {
      visit: visitPair.preoperative,
      capture: preoperativeCapture!,
    } satisfies PatientComparisonCaptureEntry,
    postoperative: {
      visit: visitPair.postoperative,
      capture: postoperativeCapture!,
    } satisfies PatientComparisonCaptureEntry,
  }
  const preoperativeResult = selectComparisonResult(
    state,
    patientId,
    capturePair.preoperative,
  )
  const postoperativeResult = selectComparisonResult(
    state,
    patientId,
    capturePair.postoperative,
  )
  const missingResults = missingRequiredTimepoints(
    preoperativeResult !== undefined,
    postoperativeResult !== undefined,
  )
  if (missingResults) {
    return { phase: 'needs_results', pair: capturePair, missingResults }
  }

  return {
    phase: 'ready',
    pair: {
      preoperative: {
        ...capturePair.preoperative,
        result: preoperativeResult!,
      } satisfies PatientComparisonResultEntry,
      postoperative: {
        ...capturePair.postoperative,
        result: postoperativeResult!,
      } satisfies PatientComparisonResultEntry,
    },
  }
}

function selectComparisonResult(
  state: PatientWorkflowState,
  patientId: string,
  entry: PatientComparisonCaptureEntry,
): PatientResult | undefined {
  const { visit, capture } = entry
  const authorization = selectCurrentAuthorization(state, visit.id)
  if (
    authorization?.patientId !== patientId ||
    authorization.visitId !== visit.id ||
    authorization.status !== 'documented'
  ) {
    return undefined
  }

  const run = selectCurrentRun(state, visit.id)
  if (
    run?.binding.patientId !== patientId ||
    run.binding.visitId !== visit.id ||
    run.binding.captureId !== capture.id
  ) {
    return undefined
  }

  const result = selectCurrentResult(state, visit.id)
  if (
    result?.binding.patientId !== patientId ||
    result.binding.visitId !== visit.id ||
    result.binding.captureId !== capture.id
  ) {
    return undefined
  }
  return result
}

export function selectCurrentCapture(
  state: PatientWorkflowState,
  visitId: string,
): CaptureAsset | undefined {
  if (!isPatientVisitId(visitId)) return undefined
  let current: CaptureAsset | undefined

  for (const captureId of state.captureOrder) {
    const capture = getOwnRecordValue(state.capturesById, captureId)
    if (
      capture?.visitId === visitId &&
      capture.status === 'current' &&
      (!current || capture.version > current.version)
    ) {
      current = capture
    }
  }

  return current
}

export function selectCurrentAuthorization(
  state: PatientWorkflowState,
  visitId: string,
): AuthorizationSnapshot | undefined {
  if (!isPatientVisitId(visitId)) return undefined
  let current: AuthorizationSnapshot | undefined

  for (const authorizationId of state.authorizationOrder) {
    const authorization = getOwnRecordValue(
      state.authorizationsById,
      authorizationId,
    )
    if (
      authorization?.visitId === visitId &&
      (!current || authorization.revision > current.revision)
    ) {
      current = authorization
    }
  }

  return current
}

function bindingMatchesCapture(
  binding: PatientRunBinding,
  capture: CaptureAsset,
): boolean {
  return (
    binding.patientId === capture.patientId &&
    binding.visitId === capture.visitId &&
    binding.captureId === capture.id &&
    binding.captureVersion === capture.version &&
    binding.captureSha256 === capture.sha256 &&
    binding.mediaHandle === capture.mediaHandle &&
    isSessionMediaHandle(capture.mediaHandle) &&
    binding.captureProtocol === capture.captureProtocol
  )
}

export function selectCurrentRun(
  state: PatientWorkflowState,
  visitId: string,
): PatientRun | undefined {
  if (!isPatientVisitId(visitId)) return undefined
  const capture = selectCurrentCapture(state, visitId)
  const authorization = selectCurrentAuthorization(state, visitId)
  if (!capture || authorization?.status !== 'documented') return undefined

  let current: PatientRun | undefined
  for (const runId of state.runOrder) {
    const run = getOwnRecordValue(state.runsById, runId)
    if (
      run &&
      bindingMatchesCapture(run.binding, capture) &&
      run.binding.authorizationRevision === authorization.revision
    ) {
      current = run
    }
  }
  return current
}

export function selectCurrentResult(
  state: PatientWorkflowState,
  visitId: string,
): PatientResult | undefined {
  if (!isPatientVisitId(visitId)) return undefined
  const run = selectCurrentRun(state, visitId)
  if (!run) return undefined

  let current: PatientResult | undefined
  for (const resultId of state.resultOrder) {
    const result = getOwnRecordValue(state.resultsById, resultId)
    if (
      result?.runId === run.id &&
      result.freshness === 'current' &&
      samePatientRunBinding(result.binding, run.binding)
    ) {
      current = result
    }
  }
  return current
}

export function selectCurrentReview(
  state: PatientWorkflowState,
  visitId: string,
): PatientReview | undefined {
  if (!isPatientVisitId(visitId)) return undefined
  const result = selectCurrentResult(state, visitId)
  if (!result) return undefined

  let current: PatientReview | undefined
  for (const reviewId of state.reviewOrder) {
    const review = getOwnRecordValue(state.reviewsById, reviewId)
    if (review?.visitId === visitId && review.resultId === result.id) {
      current = review
    }
  }
  return current
}

export function selectVisitNextAction(
  state: PatientWorkflowState,
  visitId: string,
): VisitNextAction | undefined {
  if (!isPatientVisitId(visitId)) return undefined
  const visit = getOwnRecordValue(state.visitsById, visitId)
  if (!visit) return undefined

  const capture = selectCurrentCapture(state, visit.id)
  if (!capture) return 'capture_photo'

  const authorization = selectCurrentAuthorization(state, visit.id)
  if (
    !isCaptureQualityComplete(capture.qualityChecks) ||
    !capture.qualityConfirmedAt ||
    authorization?.status !== 'documented'
  ) {
    return 'confirm_quality'
  }

  const run = selectCurrentRun(state, visit.id)
  if (!run) return 'run_analysis'
  if (run.status === 'queued' || run.status === 'running') return 'processing'
  if (run.status === 'failed') return 'retry_analysis'

  const result = selectCurrentResult(state, visit.id)
  if (!result) return 'processing'

  const review = selectCurrentReview(state, visit.id)
  if (!review) return 'review_result'
  return review.decision === 'repeat_photo' ? 'retake' : 'visit_complete'
}

export function selectPatientNextAction(
  state: PatientWorkflowState,
  patientId: string,
): PatientNextAction | undefined {
  if (!isPatientId(patientId)) return undefined
  if (!getOwnRecordValue(state.patientsById, patientId)) return undefined

  const visits = selectPatientVisits(state, patientId)
  const latestVisit = visits.at(-1)
  return latestVisit
    ? selectVisitNextAction(state, latestVisit.id)
    : 'start_visit'
}
