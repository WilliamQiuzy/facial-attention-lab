import { getWorkbenchAsset, listWorkbenchAssets } from './catalog'
import {
  createCanonicalInferenceBindingSnapshot,
  validateNormalizedRoi,
} from './mockEngine'
import { validateInferenceOutputEnvelope } from './inferenceEnvelope'
import { selectExactResultTarget } from './reviewPolicy'
import { isVerifiedFullImageSourceBinding } from './sourceBinding'
import {
  WorkbenchError,
  type ApprovedRoiAnnotation,
  type AsyncRunActionBinding,
  type InferenceAttempt,
  type InferenceBinding,
  type InferenceOutput,
  type InferenceRun,
  type NormalizedRoi,
  type ResearchReviewNote,
  type ResultReview,
  type RoiAnnotation,
  type RoiStatus,
  type RunAttemptStatus,
  type WorkbenchFailure,
  type WorkbenchFailureReason,
  type WorkspaceAction,
  type WorkspaceState,
} from './types'

export type { WorkspaceAction, WorkspaceState } from './types'

const DEMO_AUTHOR = 'demo_author' as const
const DEMO_REVIEWER = 'demo_reviewer' as const

function demoGeometry(): NormalizedRoi {
  return Object.freeze({ x: 0, y: 0, width: 1, height: 1 })
}

function demoRoi(index: number): RoiAnnotation {
  const asset = listWorkbenchAssets()[index]
  const roi: ApprovedRoiAnnotation = {
    id: `roi-demo-${String(index + 1).padStart(2, '0')}`,
    caseId: asset.id,
    assetId: asset.id,
    version: index + 1,
    geometry: demoGeometry(),
    status: 'approved',
    authorId: DEMO_AUTHOR,
    reviewerId: DEMO_REVIEWER,
  }
  return Object.freeze(roi)
}

export function createInitialWorkspaceState(): WorkspaceState {
  const roisByCase: Record<string, RoiAnnotation> = {}
  for (const [index, asset] of listWorkbenchAssets().entries()) {
    roisByCase[asset.id] = demoRoi(index)
  }

  return {
    roisByCase,
    runsById: {},
    runOrder: [],
    attemptsById: {},
    reviewsById: {},
    reviewOrder: [],
  }
}

export function getCaseRoi(
  state: WorkspaceState,
  caseId: string,
): RoiAnnotation | undefined {
  return state.roisByCase[caseId as keyof typeof state.roisByCase]
}

export function getRun(
  state: WorkspaceState,
  runId: string,
): InferenceRun | undefined {
  return state.runsById[runId]
}

export function getAttempt(
  state: WorkspaceState,
  attemptId: string,
): InferenceAttempt | undefined {
  return state.attemptsById[attemptId]
}

export function getDisplayableResult(
  state: WorkspaceState,
  requestedId?: string,
): InferenceOutput | undefined {
  let attempt: InferenceAttempt | undefined

  if (requestedId) {
    attempt = state.attemptsById[requestedId]
    if (!attempt) {
      const requestedRun = state.runsById[requestedId]
      attempt = requestedRun?.activeAttemptId
        ? state.attemptsById[requestedRun.activeAttemptId]
        : undefined
    }
  } else if (state.activeRunId) {
    const activeRun = state.runsById[state.activeRunId]
    attempt = activeRun?.activeAttemptId
      ? state.attemptsById[activeRun.activeAttemptId]
      : undefined
  }

  const result = attempt?.result
  return attempt?.status === 'succeeded' &&
    attempt.binding !== undefined &&
    bindingMatchesCurrentRoi(state, attempt.binding) &&
    result?.freshness === 'current' &&
    result.output.qualityGates?.researchDisplayEligible === true &&
    result.output.qualityGates?.clinicalUseEligible === false
    ? result.output
    : undefined
}

function failureState(
  state: WorkspaceState,
  reason: WorkbenchFailureReason,
  message: string,
  field?: string,
): WorkspaceState {
  return {
    ...state,
    lastFailure: { reason, message, ...(field ? { field } : {}) },
  }
}

function clearFailure(state: WorkspaceState): Omit<WorkspaceState, 'lastFailure'> {
  const { lastFailure: _lastFailure, ...rest } = state
  return rest
}

function sameGeometry(first: NormalizedRoi, second: NormalizedRoi): boolean {
  return (
    first.x === second.x &&
    first.y === second.y &&
    first.width === second.width &&
    first.height === second.height
  )
}

function bindingMatchesCurrentRoi(
  state: WorkspaceState,
  binding: InferenceBinding,
): boolean {
  const asset = getWorkbenchAsset(binding.caseId)
  const roi = getCaseRoi(state, binding.caseId)
  return (
    isVerifiedFullImageSourceBinding(asset, roi) &&
    roi !== undefined &&
    roi.id === binding.roiId &&
    roi.version === binding.roiVersion &&
    roi.assetId === binding.assetId &&
    sameGeometry(roi.geometry, binding.roiGeometry) &&
    roi.authorId === DEMO_AUTHOR &&
    roi.reviewerId === DEMO_REVIEWER
  )
}

function canonicalBinding(
  binding: InferenceBinding,
): { readonly binding: InferenceBinding } | { readonly failure: WorkbenchFailure } {
  try {
    return { binding: createCanonicalInferenceBindingSnapshot(binding) }
  } catch (error) {
    if (error instanceof WorkbenchError) return { failure: error.failure }
    return {
      failure: {
        reason: 'BINDING_INTEGRITY_MISMATCH',
        message: 'The inference binding could not be reconstructed canonically.',
      },
    }
  }
}

function canonicalBindingKey(binding: InferenceBinding): string | undefined {
  const rebuilt = canonicalBinding(binding)
  if ('failure' in rebuilt) return undefined

  // Both operands are rebuilt by the same canonicalizer, so JSON serialization
  // compares the complete canonical binding rather than a hand-maintained field list.
  return JSON.stringify(rebuilt.binding)
}

function sameCanonicalBinding(
  first: InferenceBinding | undefined,
  second: InferenceBinding,
): boolean {
  if (!first) return false
  const firstKey = canonicalBindingKey(first)
  return firstKey !== undefined && firstKey === canonicalBindingKey(second)
}

function deepCloneAndFreeze<T>(value: T): T {
  if (Array.isArray(value)) {
    return Object.freeze(value.map((entry) => deepCloneAndFreeze(entry))) as T
  }
  if (value !== null && typeof value === 'object') {
    const clone = Object.fromEntries(
      Object.entries(value).map(([key, entry]) => [key, deepCloneAndFreeze(entry)]),
    )
    return Object.freeze(clone) as T
  }
  return value
}

function operationalIdExists(state: WorkspaceState, id: string): boolean {
  return Boolean(state.runsById[id] || state.attemptsById[id])
}

function attemptTokenExists(state: WorkspaceState, attemptToken: string): boolean {
  return Object.values(state.attemptsById).some(
    (attempt) => attempt.attemptToken === attemptToken,
  )
}

function staleResultsForCase(
  attemptsById: WorkspaceState['attemptsById'],
  caseId: string,
  exceptFingerprint?: string,
): WorkspaceState['attemptsById'] {
  let changed = false
  const next: Record<string, InferenceAttempt> = { ...attemptsById }

  for (const [attemptId, attempt] of Object.entries(attemptsById)) {
    if (
      attempt.binding?.caseId !== caseId ||
      !attempt.result ||
      attempt.result.freshness !== 'current' ||
      (exceptFingerprint !== undefined &&
        attempt.binding.inputFingerprint === exceptFingerprint)
    ) {
      continue
    }

    changed = true
    next[attemptId] = {
      ...attempt,
      result: { ...attempt.result, freshness: 'stale' },
    }
  }

  return changed ? next : attemptsById
}

function withChangedRoi(
  state: WorkspaceState,
  caseId: string,
  roi: RoiAnnotation,
): WorkspaceState {
  return {
    ...clearFailure(state),
    roisByCase: { ...state.roisByCase, [caseId]: roi },
    attemptsById: staleResultsForCase(state.attemptsById, caseId),
  }
}

function restoreFullImageSourceBinding(
  state: WorkspaceState,
  action: Extract<
    WorkspaceAction,
    { readonly type: 'sourceBinding/restoreFullImage' }
  >,
): WorkspaceState {
  const asset = getWorkbenchAsset(action.caseId)
  if (!asset) {
    return failureState(
      state,
      'UNKNOWN_CASE',
      `Unknown workbench case: ${action.caseId}.`,
      'caseId',
    )
  }

  const current = getCaseRoi(state, asset.id)
  if (isVerifiedFullImageSourceBinding(asset, current)) return state

  const version =
    current &&
    Number.isInteger(current.version) &&
    current.version > 0
      ? current.version + 1
      : 1
  const id =
    current && typeof current.id === 'string' && current.id.trim().length > 0
      ? current.id
      : `source-binding-${asset.id}`
  const restored: ApprovedRoiAnnotation = Object.freeze({
    id,
    caseId: asset.id,
    assetId: asset.id,
    version,
    geometry: demoGeometry(),
    status: 'approved',
    authorId: DEMO_AUTHOR,
    reviewerId: DEMO_REVIEWER,
  })

  return withChangedRoi(state, asset.id, restored)
}

function requireAuthor(
  state: WorkspaceState,
  actorId: string,
): WorkspaceState | undefined {
  return actorId === DEMO_AUTHOR
    ? undefined
    : failureState(
        state,
        'ACTOR_NOT_AUTHORIZED',
        'Only the declared demo author may perform this ROI action.',
        'actorId',
      )
}

function requireReviewer(
  state: WorkspaceState,
  actorId: string,
): WorkspaceState | undefined {
  if (actorId === DEMO_REVIEWER) return undefined
  return failureState(
    state,
    actorId === DEMO_AUTHOR ? 'SELF_REVIEW_FORBIDDEN' : 'ACTOR_NOT_AUTHORIZED',
    actorId === DEMO_AUTHOR
      ? 'The ROI author cannot review their own annotation.'
      : 'Only the declared demo reviewer may perform this ROI action.',
    'actorId',
  )
}

function requireRoi(
  state: WorkspaceState,
  caseId: string,
): RoiAnnotation | WorkspaceState {
  return (
    getCaseRoi(state, caseId) ??
    failureState(state, 'UNKNOWN_CASE', `Unknown workbench case: ${caseId}.`, 'caseId')
  )
}

function isFailureState(value: RoiAnnotation | WorkspaceState): value is WorkspaceState {
  return 'roisByCase' in value
}

function illegalRoiTransition(
  state: WorkspaceState,
  status: RoiStatus,
  actionType: string,
): WorkspaceState {
  return failureState(
    state,
    'ILLEGAL_TRANSITION',
    `ROI action ${actionType} is not legal from ${status}.`,
  )
}

function updateRoiGeometry(
  state: WorkspaceState,
  action: Extract<WorkspaceAction, { readonly type: 'roi/updateGeometry' }>,
): WorkspaceState {
  const roi = requireRoi(state, action.caseId)
  if (isFailureState(roi)) return roi

  const authorizationFailure = requireAuthor(state, action.actorId)
  if (authorizationFailure) return authorizationFailure
  if (roi.status !== 'draft' && roi.status !== 'changes_requested') {
    return illegalRoiTransition(state, roi.status, action.type)
  }
  if (!validateNormalizedRoi(action.geometry)) {
    return failureState(
      state,
      'INVALID_ROI_GEOMETRY',
      'ROI geometry must be finite, normalized, positive-area, and contained in the image.',
      'geometry',
    )
  }
  if (sameGeometry(roi.geometry, action.geometry)) return state

  const { reviewerId: _reviewerId, ...withoutReviewer } = roi
  return withChangedRoi(state, action.caseId, {
    ...withoutReviewer,
    geometry: { ...action.geometry },
    version: roi.version + 1,
  })
}

function transitionRoi(
  state: WorkspaceState,
  action: Exclude<
    Extract<WorkspaceAction, { readonly type: `roi/${string}` }>,
    { readonly type: 'roi/updateGeometry' }
  >,
): WorkspaceState {
  const roi = requireRoi(state, action.caseId)
  if (isFailureState(roi)) return roi

  switch (action.type) {
    case 'roi/submitReview': {
      const authorizationFailure = requireAuthor(state, action.actorId)
      if (authorizationFailure) return authorizationFailure
      if (roi.status !== 'draft' && roi.status !== 'changes_requested') {
        return illegalRoiTransition(state, roi.status, action.type)
      }
      const { reviewerId: _reviewerId, ...withoutReviewer } = roi
      return withChangedRoi(state, action.caseId, {
        ...withoutReviewer,
        status: 'in_review',
      })
    }
    case 'roi/approve': {
      const authorizationFailure = requireReviewer(state, action.actorId)
      if (authorizationFailure) return authorizationFailure
      if (roi.status !== 'in_review') {
        return illegalRoiTransition(state, roi.status, action.type)
      }
      const approved: ApprovedRoiAnnotation = {
        ...roi,
        status: 'approved',
        reviewerId: DEMO_REVIEWER,
      }
      return withChangedRoi(state, action.caseId, approved)
    }
    case 'roi/requestChanges': {
      const authorizationFailure = requireReviewer(state, action.actorId)
      if (authorizationFailure) return authorizationFailure
      if (roi.status !== 'in_review') {
        return illegalRoiTransition(state, roi.status, action.type)
      }
      return withChangedRoi(state, action.caseId, {
        ...roi,
        status: 'changes_requested',
        reviewerId: DEMO_REVIEWER,
      })
    }
    case 'roi/reopenDraft': {
      const authorizationFailure = requireAuthor(state, action.actorId)
      if (authorizationFailure) return authorizationFailure
      if (roi.status !== 'changes_requested') {
        return illegalRoiTransition(state, roi.status, action.type)
      }
      const { reviewerId: _reviewerId, ...withoutReviewer } = roi
      return withChangedRoi(state, action.caseId, {
        ...withoutReviewer,
        status: 'draft',
      })
    }
    case 'roi/supersede': {
      const authorizationFailure = requireAuthor(state, action.actorId)
      if (authorizationFailure) return authorizationFailure
      if (roi.status !== 'approved') {
        return illegalRoiTransition(state, roi.status, action.type)
      }
      return withChangedRoi(state, action.caseId, { ...roi, status: 'superseded' })
    }
  }
}

function createRun(
  state: WorkspaceState,
  action: Extract<WorkspaceAction, { readonly type: 'run/create' }>,
): WorkspaceState {
  const existingRun = state.runsById[action.runId]
  const existingAttempt = state.attemptsById[action.attemptId]
  if (existingRun || existingAttempt) {
    const exactReplay =
      existingRun?.attemptIds.includes(action.attemptId) === true &&
      existingAttempt?.clientRunId === action.runId &&
      existingAttempt.parentAttemptId === undefined &&
      sameCanonicalBinding(existingAttempt.binding, action.binding)
    return exactReplay
      ? state
      : failureState(
          state,
          'INVALID_OPERATIONAL_ID',
          'Conflicting run creation reuses an existing operational ID.',
        )
  }
  if (
    action.runId.trim().length === 0 ||
    action.attemptId.trim().length === 0 ||
    action.binding.attemptToken.trim().length === 0 ||
    action.runId === action.attemptId ||
    operationalIdExists(state, action.runId) ||
    operationalIdExists(state, action.attemptId) ||
    attemptTokenExists(state, action.binding.attemptToken)
  ) {
    return failureState(
      state,
      'INVALID_OPERATIONAL_ID',
      'Run IDs, attempt IDs, and attempt tokens must be non-empty and globally unique.',
    )
  }

  const rebuilt = canonicalBinding(action.binding)
  if ('failure' in rebuilt) {
    return { ...state, lastFailure: rebuilt.failure }
  }
  if (
    rebuilt.binding.clientRunId !== action.runId
  ) {
    return failureState(
      state,
      'ROI_BINDING_MISMATCH',
      'The run binding must match its action ID.',
      'binding',
    )
  }
  if (!bindingMatchesCurrentRoi(state, rebuilt.binding)) {
    return failureState(
      state,
      'FULL_IMAGE_SOURCE_BINDING_REQUIRED',
      'The run requires the current verified full-image source binding.',
      'binding',
    )
  }

  const attempt: InferenceAttempt = {
    id: action.attemptId,
    clientRunId: action.runId,
    attemptToken: rebuilt.binding.attemptToken,
    status: 'draft',
    binding: rebuilt.binding,
  }
  const run: InferenceRun = {
    clientRunId: action.runId,
    caseId: rebuilt.binding.caseId,
    assetId: rebuilt.binding.assetId,
    status: 'draft',
    attemptIds: [action.attemptId],
    activeAttemptId: action.attemptId,
  }
  const attemptsById = staleResultsForCase(
    state.attemptsById,
    rebuilt.binding.caseId,
    rebuilt.binding.inputFingerprint,
  )

  return {
    ...clearFailure(state),
    runsById: { ...state.runsById, [action.runId]: run },
    runOrder: [...state.runOrder, action.runId],
    attemptsById: { ...attemptsById, [action.attemptId]: attempt },
    activeRunId: action.runId,
  }
}

function getBoundActiveAttempt(
  state: WorkspaceState,
  action: AsyncRunActionBinding,
): { readonly run: InferenceRun; readonly attempt: InferenceAttempt } | undefined {
  const run = state.runsById[action.runId]
  if (!run || run.activeAttemptId !== action.attemptId) return undefined
  const attempt = state.attemptsById[action.attemptId]
  if (
    !attempt?.binding ||
    attempt.clientRunId !== action.runId ||
    run.status !== attempt.status ||
    attempt.attemptToken !== action.attemptToken ||
    attempt.binding.attemptToken !== action.attemptToken ||
    attempt.binding.inputFingerprint !== action.inputFingerprint
  ) {
    return undefined
  }
  return { run, attempt }
}

function replaceAttemptStatus(
  state: WorkspaceState,
  run: InferenceRun,
  attempt: InferenceAttempt,
  status: RunAttemptStatus,
  additions: Partial<InferenceAttempt> = {},
): WorkspaceState {
  const nextAttempt = { ...attempt, ...additions, status }
  return {
    ...clearFailure(state),
    runsById: {
      ...state.runsById,
      [run.clientRunId]: { ...run, status },
    },
    attemptsById: {
      ...state.attemptsById,
      [attempt.id]: nextAttempt,
    },
  }
}

function hasNewerDifferentInput(
  state: WorkspaceState,
  runId: string,
  binding: InferenceBinding,
): boolean {
  const runIndex = state.runOrder.indexOf(runId)
  if (runIndex < 0) return false

  return state.runOrder.slice(runIndex + 1).some((newerRunId) => {
    const newerRun = state.runsById[newerRunId]
    const newerAttempt = newerRun?.activeAttemptId
      ? state.attemptsById[newerRun.activeAttemptId]
      : undefined
    return (
      newerRun?.caseId === binding.caseId &&
      newerAttempt?.binding?.inputFingerprint !== binding.inputFingerprint
    )
  })
}

function transitionRun(
  state: WorkspaceState,
  action: Extract<WorkspaceAction, AsyncRunActionBinding>,
): WorkspaceState {
  const context = getBoundActiveAttempt(state, action)
  if (!context) return state
  const { run, attempt } = context

  switch (action.type) {
    case 'run/validate':
      return attempt.status === 'draft'
        ? replaceAttemptStatus(state, run, attempt, 'validating')
        : state
    case 'run/block':
      return attempt.status === 'validating'
        ? replaceAttemptStatus(state, run, attempt, 'blocked', {
            failure: { ...action.failure },
            result: undefined,
          })
        : state
    case 'run/queue':
      return attempt.status === 'validating'
        ? replaceAttemptStatus(state, run, attempt, 'queued')
        : state
    case 'run/start': {
      if (attempt.status !== 'queued') return state
      if (
        !attempt.binding ||
        !bindingMatchesCurrentRoi(state, attempt.binding)
      ) {
        return replaceAttemptStatus(state, run, attempt, 'blocked', {
          failure: {
            reason: 'FULL_IMAGE_SOURCE_BINDING_REQUIRED',
            message:
              'Launch blocked: restore the verified full-image source binding before inference.',
            field: 'sourceBinding',
          },
          result: undefined,
        })
      }
      return replaceAttemptStatus(state, run, attempt, 'running')
    }
    case 'run/succeed': {
      const attemptBinding = attempt.binding
      if (
        attempt.status !== 'running' ||
        !attemptBinding
      ) {
        return state
      }
      const envelope = validateInferenceOutputEnvelope(
        action.output,
        attemptBinding,
      )
      if (!envelope.valid) {
        return replaceAttemptStatus(state, run, attempt, 'failed', {
          failure: envelope.failure,
          result: undefined,
        })
      }
      const rebuiltOutputBinding = canonicalBinding(envelope.output.binding)
      if ('failure' in rebuiltOutputBinding) {
        return replaceAttemptStatus(state, run, attempt, 'failed', {
          failure: rebuiltOutputBinding.failure,
          result: undefined,
        })
      }
      const attemptBindingKey = canonicalBindingKey(attemptBinding)
      if (
        attemptBindingKey === undefined ||
        attemptBindingKey !== JSON.stringify(rebuiltOutputBinding.binding)
      ) {
        return replaceAttemptStatus(state, run, attempt, 'failed', {
          failure: {
            reason: 'IMMUTABLE_BINDING_MISMATCH',
            message: 'The resolved output does not match the active inference binding.',
          },
          result: undefined,
        })
      }
      const storedOutput = deepCloneAndFreeze({
        ...envelope.output,
        binding: rebuiltOutputBinding.binding,
      } as InferenceOutput)
      const freshness =
        bindingMatchesCurrentRoi(state, attemptBinding) &&
        !hasNewerDifferentInput(state, run.clientRunId, attemptBinding)
          ? 'current'
          : 'stale'
      return replaceAttemptStatus(state, run, attempt, 'succeeded', {
        failure: undefined,
        result: { output: storedOutput, freshness },
      })
    }
    case 'run/fail':
      return attempt.status === 'running'
        ? replaceAttemptStatus(state, run, attempt, 'failed', {
            failure: { ...action.failure },
            result: undefined,
          })
        : state
    case 'run/cancel':
      return attempt.status === 'queued' || attempt.status === 'running'
        ? replaceAttemptStatus(state, run, attempt, 'cancelled', {
            result: undefined,
          })
        : state
  }
}

function retryRun(
  state: WorkspaceState,
  action: Extract<WorkspaceAction, { readonly type: 'run/retry' }>,
): WorkspaceState {
  const run = state.runsById[action.runId]
  const parent = state.attemptsById[action.parentAttemptId]
  const existingAttempt = state.attemptsById[action.attemptId]
  if (existingAttempt) {
    const exactReplay =
      existingAttempt.clientRunId === action.runId &&
      existingAttempt.parentAttemptId === action.parentAttemptId &&
      run?.attemptIds.includes(action.attemptId) === true &&
      sameCanonicalBinding(existingAttempt.binding, action.binding)
    return exactReplay
      ? state
      : failureState(
          state,
          'INVALID_OPERATIONAL_ID',
          'Conflicting retry reuses an existing operational ID.',
        )
  }
  if (
    action.attemptId === action.runId ||
    state.runsById[action.attemptId] ||
    state.attemptsById[action.runId]
  ) {
    return failureState(
      state,
      'INVALID_OPERATIONAL_ID',
      'Retry attempt IDs must remain unique across runs and attempts.',
    )
  }
  if (!run || !parent?.binding || run.activeAttemptId !== parent.id) return state
  if (!['blocked', 'failed', 'cancelled'].includes(parent.status)) {
    return failureState(
      state,
      'ILLEGAL_TRANSITION',
      `Run retry is not legal from ${parent.status}.`,
    )
  }
  if (
    action.attemptId.trim().length === 0 ||
    action.binding.attemptToken.trim().length === 0 ||
    attemptTokenExists(state, action.binding.attemptToken)
  ) {
    return failureState(
      state,
      'INVALID_OPERATIONAL_ID',
      'Retry attempt IDs and tokens must be non-empty and unique.',
    )
  }

  const rebuilt = canonicalBinding(action.binding)
  if ('failure' in rebuilt) return { ...state, lastFailure: rebuilt.failure }
  if (
    rebuilt.binding.clientRunId !== action.runId ||
    rebuilt.binding.inputFingerprint !== parent.binding.inputFingerprint
  ) {
    return failureState(
      state,
      'IMMUTABLE_BINDING_MISMATCH',
      'A retry must preserve the parent scientific input and bind new operational IDs.',
      'binding',
    )
  }
  if (!bindingMatchesCurrentRoi(state, rebuilt.binding)) {
    return failureState(
      state,
      'FULL_IMAGE_SOURCE_BINDING_REQUIRED',
      'Retry requires the current verified full-image source binding.',
      'binding',
    )
  }

  const attempt: InferenceAttempt = {
    id: action.attemptId,
    clientRunId: action.runId,
    attemptToken: rebuilt.binding.attemptToken,
    parentAttemptId: parent.id,
    status: 'draft',
    binding: rebuilt.binding,
  }

  return {
    ...clearFailure(state),
    runsById: {
      ...state.runsById,
      [action.runId]: {
        ...run,
        status: 'draft',
        attemptIds: [...run.attemptIds, action.attemptId],
        activeAttemptId: action.attemptId,
      },
    },
    attemptsById: { ...state.attemptsById, [action.attemptId]: attempt },
    activeRunId: action.runId,
  }
}

function revokeResult(
  state: WorkspaceState,
  action: Extract<WorkspaceAction, { readonly type: 'result/revoke' }>,
): WorkspaceState {
  const run = state.runsById[action.runId]
  const attempt = state.attemptsById[action.attemptId]
  if (
    !run ||
    attempt?.clientRunId !== run.clientRunId ||
    attempt.status !== 'succeeded' ||
    !attempt.result ||
    attempt.result.freshness === 'revoked'
  ) {
    return state
  }

  return {
    ...clearFailure(state),
    attemptsById: {
      ...state.attemptsById,
      [attempt.id]: {
        ...attempt,
        result: { ...attempt.result, freshness: 'revoked' },
      },
    },
  }
}

function hasOwn(record: object, key: string): boolean {
  return Object.prototype.hasOwnProperty.call(record, key)
}

function normalizeReviewNote(
  note: ResearchReviewNote,
):
  | { readonly note: ResearchReviewNote }
  | { readonly field: keyof ResearchReviewNote } {
  const rationale = typeof note?.rationale === 'string' ? note.rationale.trim() : ''
  if (rationale.length === 0) return { field: 'rationale' }
  const limitations =
    typeof note?.limitations === 'string' ? note.limitations.trim() : ''
  if (limitations.length === 0) return { field: 'limitations' }
  return { note: { rationale, limitations } }
}

function reviewNoteFailure(
  state: WorkspaceState,
  field: keyof ResearchReviewNote,
): WorkspaceState {
  return failureState(
    state,
    'INVALID_REVIEW_NOTE',
    `Review ${field} must contain non-whitespace text.`,
    field,
  )
}

type ReviewTarget = {
  readonly run: InferenceRun
  readonly attempt: InferenceAttempt & {
    readonly binding: InferenceBinding
    readonly result: NonNullable<InferenceAttempt['result']>
  }
}

function exactReviewTarget(
  state: WorkspaceState,
  target: {
    readonly runId: string
    readonly attemptId: string
    readonly resultDigest: string
    readonly inputFingerprint: string
  },
  requireCurrent: boolean,
): { readonly target: ReviewTarget } | { readonly state: WorkspaceState } {
  const selected = selectExactResultTarget(state, target)
  if (!selected.ok) {
    const bindingIntegrityFailure = selected.blockers.some((entry) =>
      [
        'RESULT_BINDING_MISMATCH',
        'OUTPUT_ENVELOPE_INVALID',
        'DETERMINISTIC_OUTPUT_MISMATCH',
        'FULL_IMAGE_SOURCE_BINDING_REQUIRED',
        'ASSET_BINDING_FAILED',
      ].includes(entry.code),
    )
    return {
      state: failureState(
        state,
        bindingIntegrityFailure
          ? 'IMMUTABLE_BINDING_MISMATCH'
          : 'UNKNOWN_REVIEW_TARGET',
        selected.blockers.map((entry) => entry.message).join(' '),
      ),
    }
  }

  if (requireCurrent && selected.target.attempt.result.freshness !== 'current') {
    return {
      state: failureState(
        state,
        'UNKNOWN_REVIEW_TARGET',
        'A stale or revoked result cannot enter or advance research review.',
      ),
    }
  }

  return {
    target: {
      run: selected.target.run,
      attempt: selected.target.attempt,
    },
  }
}

function createResultReview(
  state: WorkspaceState,
  action: Extract<WorkspaceAction, { readonly type: 'review/create' }>,
): WorkspaceState {
  const authorizationFailure = requireAuthor(state, action.actorId)
  if (authorizationFailure) return authorizationFailure
  const normalized = normalizeReviewNote(action.note)
  if ('field' in normalized) return reviewNoteFailure(state, normalized.field)

  if (
    action.reviewId.trim().length === 0 ||
    hasOwn(state.reviewsById, action.reviewId) ||
    operationalIdExists(state, action.reviewId)
  ) {
    return failureState(
      state,
      'INVALID_OPERATIONAL_ID',
      'Review IDs must be non-empty and unique within the session.',
      'reviewId',
    )
  }

  const selected = exactReviewTarget(state, action, true)
  if ('state' in selected) return selected.state
  const targetAlreadyReviewed = Object.values(state.reviewsById).some(
    (review) =>
      review.runId === action.runId &&
      review.attemptId === action.attemptId &&
      review.resultDigest === action.resultDigest &&
      review.inputFingerprint === action.inputFingerprint,
  )
  if (targetAlreadyReviewed) {
    return failureState(
      state,
      'INVALID_OPERATIONAL_ID',
      'This exact run and attempt target already has a review in the current session.',
      'attemptId',
    )
  }

  const review: ResultReview = {
    id: action.reviewId,
    runId: action.runId,
    attemptId: action.attemptId,
    resultDigest: action.resultDigest,
    inputFingerprint: action.inputFingerprint,
    authorId: DEMO_AUTHOR,
    reviewerId: DEMO_REVIEWER,
    status: 'awaiting_review',
    decision: 'awaiting_review',
    events: [
      {
        sequence: 1,
        decision: 'awaiting_review',
        actorId: DEMO_AUTHOR,
        note: normalized.note,
      },
    ],
  }

  return {
    ...clearFailure(state),
    reviewsById: { ...state.reviewsById, [review.id]: review },
    reviewOrder: [...state.reviewOrder, review.id],
  }
}

function transitionResultReview(
  state: WorkspaceState,
  action: Extract<WorkspaceAction, { readonly type: `review/${string}` }>,
): WorkspaceState {
  if (!hasOwn(state.reviewsById, action.reviewId)) {
    return failureState(
      state,
      'UNKNOWN_REVIEW',
      'The exact review is unavailable in this session.',
      'reviewId',
    )
  }
  const review = state.reviewsById[action.reviewId]
  if (!review) {
    return failureState(state, 'UNKNOWN_REVIEW', 'The exact review is unavailable.')
  }
  const normalized = normalizeReviewNote(action.note)
  if ('field' in normalized) return reviewNoteFailure(state, normalized.field)

  const reviewerAction =
    action.type === 'review/approve' ||
    action.type === 'review/requestChanges' ||
    action.type === 'review/revoke'
  const authorizationFailure = reviewerAction
    ? requireReviewer(state, action.actorId)
    : requireAuthor(state, action.actorId)
  if (authorizationFailure) return authorizationFailure

  const expectedStatus =
    action.type === 'review/resubmit'
      ? 'changes_requested'
      : action.type === 'review/revoke'
        ? 'approved_for_research'
        : 'awaiting_review'
  if (review.status !== expectedStatus) {
    return failureState(
      state,
      'ILLEGAL_TRANSITION',
      `Review action ${action.type} is not legal from ${review.status}.`,
    )
  }

  if (action.type !== 'review/revoke') {
    const selected = exactReviewTarget(state, review, true)
    if ('state' in selected) return selected.state
  }

  const status =
    action.type === 'review/approve'
      ? 'approved_for_research'
      : action.type === 'review/requestChanges'
        ? 'changes_requested'
        : action.type === 'review/resubmit'
          ? 'awaiting_review'
          : 'revoked'
  const nextReview: ResultReview = {
    ...review,
    status,
    decision: status,
    events: [
      ...review.events,
      {
        sequence: review.events.length + 1,
        decision: status,
        actorId: reviewerAction ? DEMO_REVIEWER : DEMO_AUTHOR,
        note: normalized.note,
      },
    ],
  }

  return {
    ...clearFailure(state),
    reviewsById: { ...state.reviewsById, [review.id]: nextReview },
  }
}

export function workspaceReducer(
  state: WorkspaceState,
  action: WorkspaceAction,
): WorkspaceState {
  switch (action.type) {
    case 'session/reset':
      return createInitialWorkspaceState()
    case 'sourceBinding/restoreFullImage':
      return restoreFullImageSourceBinding(state, action)
    case 'roi/updateGeometry':
      return updateRoiGeometry(state, action)
    case 'roi/submitReview':
    case 'roi/approve':
    case 'roi/requestChanges':
    case 'roi/reopenDraft':
    case 'roi/supersede':
      return transitionRoi(state, action)
    case 'run/create':
      return createRun(state, action)
    case 'run/validate':
    case 'run/block':
    case 'run/queue':
    case 'run/start':
    case 'run/succeed':
    case 'run/fail':
    case 'run/cancel':
      return transitionRun(state, action)
    case 'run/retry':
      return retryRun(state, action)
    case 'result/revoke':
      return revokeResult(state, action)
    case 'review/create':
      return createResultReview(state, action)
    case 'review/approve':
    case 'review/requestChanges':
    case 'review/resubmit':
    case 'review/revoke':
      return transitionResultReview(state, action)
    default:
      return state
  }
}
