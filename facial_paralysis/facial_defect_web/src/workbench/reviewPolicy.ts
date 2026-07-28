import type { WorkbenchCatalogEntry } from './catalog'
import { getWorkbenchAsset } from './catalog'
import { validateInferenceOutputEnvelope } from './inferenceEnvelope'
import { runMockEngine } from './mockEngine'
import { isVerifiedFullImageSourceBinding } from './sourceBinding'
import type {
  InferenceAttempt,
  InferenceOutput,
  InferenceRun,
  ResearchReviewNote,
  ResultReview,
  RoiAnnotation,
  WorkspaceState,
} from './types'

export type ReviewPolicyBlockerCode =
  | 'UNKNOWN_RESULT_TARGET'
  | 'RESULT_NOT_SUCCEEDED'
  | 'RESULT_NOT_CURRENT'
  | 'RESULT_BINDING_MISMATCH'
  | 'REVIEW_REQUIRED'
  | 'REVIEW_NOT_FOUND'
  | 'REVIEW_NOT_APPROVED'
  | 'REVIEW_REVOKED'
  | 'REVIEW_NOTE_INVALID'
  | 'REVIEW_EVENT_INVALID'
  | 'REVIEW_IDENTITY_INVALID'
  | 'OUTPUT_ENVELOPE_INVALID'
  | 'DETERMINISTIC_OUTPUT_MISMATCH'
  | 'CONNECTED_OUTPUT_BLOCKED'
  | 'MOCK_ORIGIN_REQUIRED'
  | 'SIMULATED_CAPABILITY_REQUIRED'
  | 'QUALITY_GATE_FAILED'
  | 'PROVENANCE_GATE_FAILED'
  | 'FULL_IMAGE_SOURCE_BINDING_REQUIRED'
  | 'ASSET_BINDING_FAILED'

export type ReviewPolicyBlocker = {
  readonly code: ReviewPolicyBlockerCode
  readonly message: string
}

export type ExactResultTargetReference = {
  readonly runId: string
  readonly attemptId: string
  readonly resultDigest: string
  readonly inputFingerprint: string
}

export type ExactResultTarget = {
  readonly run: InferenceRun
  readonly attempt: InferenceAttempt & {
    readonly binding: NonNullable<InferenceAttempt['binding']>
    readonly result: NonNullable<InferenceAttempt['result']>
  }
  readonly output: InferenceOutput
}

export type ExactResultTargetSelection =
  | { readonly ok: true; readonly target: ExactResultTarget; readonly blockers: readonly [] }
  | { readonly ok: false; readonly blockers: readonly ReviewPolicyBlocker[] }

export type EligiblePatientReport = {
  readonly eligible: true
  readonly blockers: readonly []
  readonly review: ResultReview
  readonly run: InferenceRun
  readonly attempt: ExactResultTarget['attempt']
  readonly output: InferenceOutput & {
    readonly origin: 'mock_simulation'
    readonly capabilityStatus: 'simulated_ui_only'
  }
  readonly asset: WorkbenchCatalogEntry
  readonly roi: RoiAnnotation
}

export type BlockedPatientReport = {
  readonly eligible: false
  readonly blockers: readonly ReviewPolicyBlocker[]
  readonly review?: ResultReview
}

export type PatientReportEligibility =
  | EligiblePatientReport
  | BlockedPatientReport

export type ReviewQueueItem = {
  readonly runId: string
  readonly attemptId: string
  readonly resultDigest?: string
  readonly inputFingerprint?: string
  readonly reviewId?: string
  readonly status: InferenceAttempt['status']
  readonly canCreateReview: boolean
  readonly patientPreviewEligible: boolean
  readonly blockers: readonly ReviewPolicyBlocker[]
}

function hasOwn(record: object, key: string): boolean {
  return Object.prototype.hasOwnProperty.call(record, key)
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}

function blocker(
  code: ReviewPolicyBlockerCode,
  message: string,
): ReviewPolicyBlocker {
  return { code, message }
}

const RUN_KEYS = new Set([
  'clientRunId',
  'caseId',
  'assetId',
  'status',
  'attemptIds',
  'activeAttemptId',
])
const ATTEMPT_KEYS = new Set([
  'id',
  'clientRunId',
  'attemptToken',
  'parentAttemptId',
  'status',
  'binding',
  'result',
  'failure',
])
const ATTEMPT_STATUSES = new Set<InferenceAttempt['status']>([
  'draft',
  'validating',
  'blocked',
  'queued',
  'running',
  'succeeded',
  'failed',
  'cancelled',
])

type NormalizedReviewRuntime = {
  readonly roisByCase: Readonly<Record<string, RoiAnnotation | undefined>>
  readonly runsById: Readonly<Record<string, InferenceRun>>
  readonly runOrder: readonly string[]
  readonly attemptsById: Readonly<Record<string, InferenceAttempt>>
  readonly reviewsById: Readonly<Record<string, ResultReview>>
  readonly reviewOrder: readonly string[]
}

type ReviewRuntimeNormalization =
  | { readonly ok: true; readonly runtime: NormalizedReviewRuntime }
  | { readonly ok: false }

function isPlainRecord(value: unknown): value is Record<string, unknown> {
  if (!isRecord(value)) return false
  const prototype = Object.getPrototypeOf(value)
  return prototype === Object.prototype || prototype === null
}

function hasOnlyExpectedKeys(
  value: Record<string, unknown>,
  allowed: ReadonlySet<string>,
  required: readonly string[],
): boolean {
  const keys = Reflect.ownKeys(value)
  return (
    keys.every((key) => typeof key === 'string' && allowed.has(key)) &&
    required.every((key) => hasOwn(value, key))
  )
}

function normalizeStringArray(
  value: unknown,
  allowEmpty = false,
): readonly string[] | undefined {
  if (!Array.isArray(value) || Object.getPrototypeOf(value) !== Array.prototype) {
    return undefined
  }
  const keys = Reflect.ownKeys(value)
  if (
    keys.length !== value.length + 1 ||
    !hasOwn(value, 'length') ||
    (!allowEmpty && value.length === 0)
  ) {
    return undefined
  }
  const normalized: string[] = []
  const unique = new Set<string>()
  for (let index = 0; index < value.length; index += 1) {
    if (!hasOwn(value, String(index))) return undefined
    const entry: unknown = value[index]
    if (typeof entry !== 'string' || entry.length === 0 || unique.has(entry)) {
      return undefined
    }
    unique.add(entry)
    normalized.push(entry)
  }
  return normalized
}

function normalizeRun(value: unknown, mapKey: string): InferenceRun | undefined {
  if (
    !isPlainRecord(value) ||
    !hasOnlyExpectedKeys(value, RUN_KEYS, [
      'clientRunId',
      'caseId',
      'assetId',
      'status',
      'attemptIds',
      'activeAttemptId',
    ]) ||
    value.clientRunId !== mapKey ||
    typeof value.caseId !== 'string' ||
    value.caseId.length === 0 ||
    typeof value.assetId !== 'string' ||
    value.assetId.length === 0 ||
    !ATTEMPT_STATUSES.has(value.status as InferenceAttempt['status']) ||
    typeof value.activeAttemptId !== 'string' ||
    value.activeAttemptId.length === 0
  ) {
    return undefined
  }
  const attemptIds = normalizeStringArray(value.attemptIds)
  if (!attemptIds) return undefined
  return {
    clientRunId: value.clientRunId,
    caseId: value.caseId as InferenceRun['caseId'],
    assetId: value.assetId as InferenceRun['assetId'],
    status: value.status as InferenceRun['status'],
    attemptIds,
    activeAttemptId: value.activeAttemptId,
  }
}

function normalizeAttempt(
  value: unknown,
  mapKey: string,
): InferenceAttempt | undefined {
  if (
    !isPlainRecord(value) ||
    !hasOnlyExpectedKeys(value, ATTEMPT_KEYS, [
      'id',
      'clientRunId',
      'attemptToken',
      'status',
      'binding',
    ]) ||
    value.id !== mapKey ||
    typeof value.clientRunId !== 'string' ||
    value.clientRunId.length === 0 ||
    typeof value.attemptToken !== 'string' ||
    value.attemptToken.length === 0 ||
    !ATTEMPT_STATUSES.has(value.status as InferenceAttempt['status']) ||
    !isPlainRecord(value.binding) ||
    (hasOwn(value, 'parentAttemptId') &&
      value.parentAttemptId !== undefined &&
      (typeof value.parentAttemptId !== 'string' || value.parentAttemptId.length === 0)) ||
    (hasOwn(value, 'result') &&
      value.result !== undefined &&
      !isPlainRecord(value.result)) ||
    (hasOwn(value, 'failure') &&
      value.failure !== undefined &&
      !isPlainRecord(value.failure))
  ) {
    return undefined
  }
  if (hasOwn(value, 'result') && value.result !== undefined) {
    const result = value.result as Record<string, unknown>
    if (
      !hasOnlyExpectedKeys(result, new Set(['output', 'freshness']), [
        'output',
        'freshness',
      ]) ||
      !isPlainRecord(result.output) ||
      (result.freshness !== 'current' &&
        result.freshness !== 'stale' &&
        result.freshness !== 'revoked')
    ) {
      return undefined
    }
  }
  return value as unknown as InferenceAttempt
}

function normalizeReviewRuntimeState(
  state: WorkspaceState,
): ReviewRuntimeNormalization {
  try {
    const rawState: unknown = state
    if (!isPlainRecord(rawState)) return { ok: false }
    const rawRuns = rawState.runsById
    const rawAttempts = rawState.attemptsById
    const rawReviews = rawState.reviewsById
    const rawRois = rawState.roisByCase
    const runOrder = normalizeStringArray(rawState.runOrder)
    const reviewOrder = normalizeStringArray(rawState.reviewOrder, true)
    if (
      !isPlainRecord(rawRuns) ||
      !isPlainRecord(rawAttempts) ||
      !isPlainRecord(rawReviews) ||
      !isPlainRecord(rawRois) ||
      !reviewOrder
    ) {
      return { ok: false }
    }
    if (
      [rawRuns, rawAttempts, rawReviews, rawRois].some((record) =>
        Reflect.ownKeys(record).some((key) => typeof key !== 'string'),
      )
    ) {
      return { ok: false }
    }

    const runKeys = Object.keys(rawRuns)
    const attemptKeys = Object.keys(rawAttempts)
    const reviewKeys = Object.keys(rawReviews)
    if (reviewOrder.length !== reviewKeys.length) return { ok: false }
    const orderedReviews = new Set(reviewOrder)
    if (orderedReviews.size !== reviewKeys.length) return { ok: false }
    for (const reviewId of reviewKeys) {
      if (!orderedReviews.has(reviewId)) return { ok: false }
    }
    if (runKeys.length === 0 && attemptKeys.length === 0) {
      const emptyOrder = Array.isArray(rawState.runOrder) && rawState.runOrder.length === 0
      return emptyOrder
        ? {
            ok: true,
            runtime: {
              roisByCase: rawRois as Readonly<Record<string, RoiAnnotation | undefined>>,
              runsById: {},
              runOrder: [],
              attemptsById: {},
              reviewsById: rawReviews as Readonly<Record<string, ResultReview>>,
              reviewOrder,
            },
          }
        : { ok: false }
    }
    if (!runOrder || runOrder.length !== runKeys.length) return { ok: false }

    const runsById: Record<string, InferenceRun> = {}
    const attemptsById: Record<string, InferenceAttempt> = {}
    for (const runId of runKeys) {
      const run = normalizeRun(rawRuns[runId], runId)
      if (!run) return { ok: false }
      runsById[runId] = run
    }
    for (const attemptId of attemptKeys) {
      const attempt = normalizeAttempt(rawAttempts[attemptId], attemptId)
      if (!attempt) return { ok: false }
      attemptsById[attemptId] = attempt
    }

    const orderedRuns = new Set(runOrder)
    if (orderedRuns.size !== runKeys.length) return { ok: false }
    for (const runId of runKeys) {
      if (!orderedRuns.has(runId)) return { ok: false }
    }

    const referencedAttempts = new Set<string>()
    for (const run of Object.values(runsById)) {
      const runAttempts = new Set(run.attemptIds)
      if (!run.activeAttemptId || !runAttempts.has(run.activeAttemptId)) {
        return { ok: false }
      }
      for (const attemptId of run.attemptIds) {
        if (referencedAttempts.has(attemptId) || !hasOwn(attemptsById, attemptId)) {
          return { ok: false }
        }
        const attempt = attemptsById[attemptId]
        if (!attempt || attempt.clientRunId !== run.clientRunId) {
          return { ok: false }
        }
        referencedAttempts.add(attemptId)
      }
      if (attemptsById[run.activeAttemptId]?.status !== run.status) {
        return { ok: false }
      }
    }
    if (referencedAttempts.size !== attemptKeys.length) return { ok: false }

    return {
      ok: true,
      runtime: {
        roisByCase: rawRois as Readonly<Record<string, RoiAnnotation | undefined>>,
        runsById,
        runOrder,
        attemptsById,
        reviewsById: rawReviews as Readonly<Record<string, ResultReview>>,
        reviewOrder,
      },
    }
  } catch {
    return { ok: false }
  }
}

function malformedRuntimeBlocker(): ReviewPolicyBlocker {
  return blocker(
    'UNKNOWN_RESULT_TARGET',
    'The run and attempt session state is malformed. Review actions are blocked.',
  )
}

export function reviewRuntimeStateBlockers(
  state: WorkspaceState,
): readonly ReviewPolicyBlocker[] {
  return normalizeReviewRuntimeState(state).ok ? [] : [malformedRuntimeBlocker()]
}

function geometryMatches(
  left: unknown,
  right: unknown,
): boolean {
  return (
    isRecord(left) &&
    isRecord(right) &&
    left.x === right.x &&
    left.y === right.y &&
    left.width === right.width &&
    left.height === right.height
  )
}

function currentScientificBindingBlockers(
  roisByCase: Readonly<Record<string, RoiAnnotation | undefined>>,
  run: InferenceRun,
  attempt: ExactResultTarget['attempt'],
): readonly ReviewPolicyBlocker[] {
  const { binding } = attempt
  const asset = getWorkbenchAsset(binding.assetId)
  const roi = hasOwn(roisByCase, binding.caseId)
    ? roisByCase[binding.caseId]
    : undefined

  if (
    !asset ||
    !roi ||
    asset.id !== run.caseId ||
    asset.id !== run.assetId ||
    binding.caseId !== run.caseId ||
    binding.assetId !== run.assetId ||
    binding.caseId !== binding.assetId ||
    asset.sha256 !== binding.assetSha256
  ) {
    return [
      blocker(
        'ASSET_BINDING_FAILED',
        'The canonical asset identity must remain current.',
      ),
    ]
  }

  if (!isVerifiedFullImageSourceBinding(asset, roi)) {
    return [
      blocker(
        'FULL_IMAGE_SOURCE_BINDING_REQUIRED',
        'Restore the verified full-image source binding before review or patient preview.',
      ),
    ]
  }

  if (
    roi.id !== binding.roiId ||
    roi.caseId !== binding.caseId ||
    roi.assetId !== binding.assetId ||
    roi.version !== binding.roiVersion ||
    !geometryMatches(roi.geometry, binding.roiGeometry)
  ) {
    return [
      blocker(
        'ASSET_BINDING_FAILED',
        'The current full-image source binding must match the immutable result binding.',
      ),
    ]
  }

  return []
}

export function selectExactResultTarget(
  state: WorkspaceState,
  reference: ExactResultTargetReference,
): ExactResultTargetSelection {
  const normalized = normalizeReviewRuntimeState(state)
  if (!normalized.ok) {
    return { ok: false, blockers: [malformedRuntimeBlocker()] }
  }
  const { runsById, attemptsById, roisByCase } = normalized.runtime
  if (
    !hasOwn(runsById, reference.runId) ||
    !hasOwn(attemptsById, reference.attemptId)
  ) {
    return {
      ok: false,
      blockers: [
        blocker(
          'UNKNOWN_RESULT_TARGET',
          'The exact run and attempt are unavailable in this memory-only session.',
        ),
      ],
    }
  }

  const run = runsById[reference.runId]
  const attempt = attemptsById[reference.attemptId]
  if (
    !run ||
    !attempt ||
    attempt.status !== 'succeeded' ||
    run.status !== 'succeeded' ||
    run.activeAttemptId !== reference.attemptId ||
    attempt.clientRunId !== reference.runId ||
    !run.attemptIds.includes(reference.attemptId) ||
    !attempt.binding ||
    !attempt.result
  ) {
    return {
      ok: false,
      blockers: [
        blocker(
          'RESULT_NOT_SUCCEEDED',
          'Only the active succeeded attempt can be selected for review.',
        ),
      ],
    }
  }

  if (attempt.result.freshness !== 'current') {
    return {
      ok: false,
      blockers: [
        blocker(
          'RESULT_NOT_CURRENT',
          `The result is ${attempt.result.freshness}; a current result is required.`,
        ),
      ],
    }
  }

  const rawOutput: unknown = attempt.result.output
  if (!isRecord(rawOutput) || !isRecord(rawOutput.binding)) {
    return {
      ok: false,
      blockers: [
        blocker(
          'RESULT_BINDING_MISMATCH',
          'The stored output or its immutable binding is structurally invalid.',
        ),
      ],
    }
  }
  const output = rawOutput as unknown as InferenceOutput
  const outputBinding = rawOutput.binding
  if (
    output.resultDigest !== reference.resultDigest ||
    attempt.binding.inputFingerprint !== reference.inputFingerprint ||
    outputBinding.inputFingerprint !== reference.inputFingerprint ||
    outputBinding.clientRunId !== reference.runId ||
    outputBinding.attemptToken !== attempt.attemptToken ||
    outputBinding.caseId !== run.caseId ||
    outputBinding.assetId !== run.assetId
  ) {
    return {
      ok: false,
      blockers: [
        blocker(
          'RESULT_BINDING_MISMATCH',
          'Run, attempt, result digest, and scientific fingerprint must match exactly.',
        ),
      ],
    }
  }

  const exactAttempt = attempt as ExactResultTarget['attempt']
  const integrityBlockers: ReviewPolicyBlocker[] = [
    ...currentScientificBindingBlockers(roisByCase, run, exactAttempt),
  ]
  const expectedMode = rawOutput.origin === 'mock_simulation'
    ? 'mock'
    : rawOutput.origin === 'model_prediction'
      ? 'connected'
      : undefined
  const envelope = validateInferenceOutputEnvelope(
    rawOutput,
    exactAttempt.binding,
    expectedMode,
  )
  if (!envelope.valid) {
    integrityBlockers.push(
      blocker(
        'OUTPUT_ENVELOPE_INVALID',
        `${envelope.failure.reason}: ${envelope.failure.message}`,
      ),
    )
  } else if (
    envelope.output.origin === 'mock_simulation' &&
    canonicalJson(runMockEngine(exactAttempt.binding)) !== canonicalJson(envelope.output)
  ) {
    integrityBlockers.push(
      blocker(
        'DETERMINISTIC_OUTPUT_MISMATCH',
        'The stored mock result does not match the deterministic engine output.',
      ),
    )
  }

  if (integrityBlockers.length > 0) {
    return { ok: false, blockers: integrityBlockers }
  }

  return {
    ok: true,
    target: {
      run,
      attempt: exactAttempt,
      output,
    },
    blockers: [],
  }
}

function validNote(note: unknown): note is ResearchReviewNote {
  return (
    isRecord(note) &&
    typeof note.rationale === 'string' &&
    note.rationale.trim().length > 0 &&
    typeof note.limitations === 'string' &&
    note.limitations.trim().length > 0
  )
}

function reviewEventsAreCoherent(review: ResultReview): boolean {
  if (!Array.isArray(review.events) || review.events.length === 0) return false
  if (review.events.some((event) => !isRecord(event))) return false
  const events = review.events as readonly ResultReview['events'][number][]
  const first = events[0]
  if (
    first?.sequence !== 1 ||
    first.decision !== 'awaiting_review' ||
    first.actorId !== 'demo_author'
  ) {
    return false
  }
  const legalActor = (event: ResultReview['events'][number]) =>
    event.decision === 'awaiting_review'
      ? event.actorId === 'demo_author'
      : event.actorId === 'demo_reviewer'
  const legalTransition = (
    previous: ResultReview['events'][number],
    next: ResultReview['events'][number],
  ) =>
    (previous.decision === 'awaiting_review' &&
      (next.decision === 'approved_for_research' ||
        next.decision === 'changes_requested')) ||
    (previous.decision === 'changes_requested' &&
      next.decision === 'awaiting_review') ||
    (previous.decision === 'approved_for_research' &&
      next.decision === 'revoked')
  return (
    events.every(
      (event, index) =>
        event.sequence === index + 1 &&
        validNote(event.note) &&
        legalActor(event) &&
        (index === 0 || legalTransition(events[index - 1]!, event)),
    ) &&
    events[events.length - 1]?.decision === review.status &&
    review.decision === review.status
  )
}

function canonicalJson(value: unknown): string {
  if (Array.isArray(value)) {
    return `[${value.map((entry) => canonicalJson(entry)).join(',')}]`
  }
  if (isRecord(value)) {
    return `{${Object.keys(value)
      .sort()
      .map((key) => `${JSON.stringify(key)}:${canonicalJson(value[key])}`)
      .join(',')}}`
  }
  return JSON.stringify(value) ?? 'null'
}

function outputPolicyBlockers(
  output: InferenceOutput,
): readonly ReviewPolicyBlocker[] {
  const blockers: ReviewPolicyBlocker[] = []
  if (output.origin === 'model_prediction') {
    blockers.push(
      blocker(
        'CONNECTED_OUTPUT_BLOCKED',
        'Connected model output is always blocked from patient preview and export.',
      ),
    )
  } else if (output.origin !== 'mock_simulation') {
    blockers.push(blocker('MOCK_ORIGIN_REQUIRED', 'Mock simulation origin is required.'))
  }
  if (output.capabilityStatus !== 'simulated_ui_only') {
    blockers.push(
      blocker(
        'SIMULATED_CAPABILITY_REQUIRED',
        'Only simulated-ui capability may enter a patient-facing preview.',
      ),
    )
  }
  const quality = isRecord(output.qualityGates) ? output.qualityGates : undefined
  if (
    !quality ||
    quality.bindingIntegrity !== 'passed' ||
    quality.sourceBindingIntegrity !== 'passed' ||
    quality.finiteValues !== 'passed' ||
    quality.normalizedBounds !== 'passed' ||
    quality.researchDisplayEligible !== true ||
    quality.clinicalUseEligible !== false
  ) {
    blockers.push(
      blocker(
        'QUALITY_GATE_FAILED',
        'All research display gates must pass while clinical use remains blocked.',
      ),
    )
  }
  const provenance = isRecord(output.provenance) ? output.provenance : undefined
  if (
    output.origin !== 'mock_simulation' ||
    !provenance ||
    provenance.engine !== 'deterministic_mock_engine' ||
    provenance.canonicalSyntheticAsset !== true ||
    provenance.deterministic !== true ||
    provenance.networkAccessed !== false ||
    provenance.storageAccessed !== false ||
    provenance.humanGazeData !== false
  ) {
    blockers.push(
      blocker(
        'PROVENANCE_GATE_FAILED',
        'Patient preview requires canonical, deterministic, offline mock provenance.',
      ),
    )
  }
  return blockers
}

export function evaluatePatientReportEligibility(
  state: WorkspaceState,
  reviewId: string,
): PatientReportEligibility {
  const normalized = normalizeReviewRuntimeState(state)
  if (!normalized.ok) {
    return { eligible: false, blockers: [malformedRuntimeBlocker()] }
  }
  const { reviewsById, reviewOrder, roisByCase } = normalized.runtime
  if (!hasOwn(reviewsById, reviewId)) {
    return {
      eligible: false,
      blockers: [
        blocker(
          'REVIEW_NOT_FOUND',
          'The exact review is unavailable in this memory-only session.',
        ),
      ],
    }
  }
  const review = reviewsById[reviewId]
  if (!review) {
    return {
      eligible: false,
      blockers: [blocker('REVIEW_NOT_FOUND', 'The exact review is unavailable.')],
    }
  }

  const blockers: ReviewPolicyBlocker[] = []
  const reviewOrderOccurrences = reviewOrder.filter(
    (candidateId) => candidateId === reviewId,
  ).length
  if (
    review.id !== reviewId ||
    review.authorId !== 'demo_author' ||
    review.reviewerId !== 'demo_reviewer' ||
    reviewOrderOccurrences !== 1
  ) {
    blockers.push(
      blocker(
        'REVIEW_IDENTITY_INVALID',
        'Review key, identity, actors, and canonical session order must agree exactly.',
      ),
    )
  }
  if (review.status === 'revoked') {
    blockers.push(blocker('REVIEW_REVOKED', 'Research approval has been revoked.'))
  } else if (review.status !== 'approved_for_research') {
    blockers.push(
      blocker(
        'REVIEW_NOT_APPROVED',
        'Independent approval for research display is required.',
      ),
    )
  }
  const reviewEvents = Array.isArray(review.events) ? review.events : []
  if (
    reviewEvents.length === 0 ||
    reviewEvents.some((event) => !isRecord(event) || !validNote(event.note))
  ) {
    blockers.push(
      blocker(
        'REVIEW_NOTE_INVALID',
        'Every review event requires non-empty rationale and limitations.',
      ),
    )
  }
  if (!reviewEventsAreCoherent(review)) {
    blockers.push(
      blocker(
        'REVIEW_EVENT_INVALID',
        'Review events must be ordered, role-separated, and match the current decision.',
      ),
    )
  }

  const selected = selectExactResultTarget(state, review)
  if (!selected.ok) {
    return { eligible: false, review, blockers: [...blockers, ...selected.blockers] }
  }
  blockers.push(...outputPolicyBlockers(selected.target.output))

  const asset = getWorkbenchAsset(selected.target.output.binding.assetId)
  const roi = hasOwn(roisByCase, selected.target.output.binding.caseId)
    ? roisByCase[selected.target.output.binding.caseId]
    : undefined
  if (!asset || !roi) {
    blockers.push(
      blocker(
        'ASSET_BINDING_FAILED',
        'The canonical asset and independently approved ROI must remain current.',
      ),
    )
  }

  if (
    blockers.length > 0 ||
    !asset ||
    !roi ||
    selected.target.output.origin !== 'mock_simulation' ||
    selected.target.output.capabilityStatus !== 'simulated_ui_only'
  ) {
    return { eligible: false, review, blockers }
  }

  return {
    eligible: true,
    blockers: [],
    review,
    run: selected.target.run,
    attempt: selected.target.attempt,
    output: selected.target.output,
    asset,
    roi,
  }
}

function reviewForExactTarget(
  runtime: Pick<NormalizedReviewRuntime, 'reviewsById' | 'reviewOrder'>,
  reference: ExactResultTargetReference,
): ResultReview | undefined {
  const matches = runtime.reviewOrder.flatMap((reviewId) => {
    if (!hasOwn(runtime.reviewsById, reviewId)) return []
    const review = runtime.reviewsById[reviewId]
    return review &&
      review.runId === reference.runId &&
      review.attemptId === reference.attemptId &&
      review.resultDigest === reference.resultDigest &&
      review.inputFingerprint === reference.inputFingerprint
      ? [review]
      : []
  })
  return matches.length === 1 ? matches[0] : undefined
}

export function listReviewQueueItems(
  state: WorkspaceState,
): readonly ReviewQueueItem[] {
  const normalized = normalizeReviewRuntimeState(state)
  if (!normalized.ok) return []
  const { runsById, runOrder, attemptsById } = normalized.runtime
  const items: ReviewQueueItem[] = []
  for (const runId of [...runOrder].reverse()) {
    if (!hasOwn(runsById, runId)) continue
    const run = runsById[runId]
    const attemptId = run?.activeAttemptId
    if (!run || !attemptId || !hasOwn(attemptsById, attemptId)) continue
    const attempt = attemptsById[attemptId]
    const binding = attempt?.binding
    const output = attempt?.result?.output
    if (!attempt) continue
    if (!binding || !output) {
      items.push({
        runId,
        attemptId,
        resultDigest: undefined,
        inputFingerprint: undefined,
        reviewId: undefined,
        status: attempt.status,
        canCreateReview: false,
        patientPreviewEligible: false,
        blockers: [
          blocker(
            'RESULT_NOT_SUCCEEDED',
            `The active attempt is ${attempt.status} and has no reviewable result.`,
          ),
        ],
      })
      continue
    }
    const reference: ExactResultTargetReference = {
      runId,
      attemptId,
      resultDigest: output.resultDigest,
      inputFingerprint: binding.inputFingerprint,
    }
    const selected = selectExactResultTarget(state, reference)
    const review = reviewForExactTarget(normalized.runtime, reference)
    const patient = review
      ? evaluatePatientReportEligibility(state, review.id)
      : undefined
    const blockers = review
      ? patient?.blockers ?? []
      : !selected.ok
        ? selected.blockers
        : [
            blocker('REVIEW_REQUIRED', 'A structured independent review is required.'),
            ...outputPolicyBlockers(selected.target.output),
          ]
    items.push({
      ...reference,
      reviewId: review?.id,
      status: attempt.status,
      canCreateReview: !review && selected.ok,
      patientPreviewEligible: patient?.eligible === true,
      blockers,
    })
  }
  return items
}
