import { describe, expect, it, vi } from 'vitest'
import { listWorkbenchAssets } from './catalog'
import { createInferenceBinding, runMockEngine, validateNormalizedRoi } from './mockEngine'
import {
  createInitialWorkspaceState,
  getAttempt,
  getCaseRoi,
  getDisplayableResult,
  getRun,
  workspaceReducer,
} from './reducer'
import type {
  ApprovedRoiAnnotation,
  InferenceBinding,
  InferenceOutput,
  NormalizedRoi,
  WorkspaceAction,
  WorkspaceState,
} from './types'

function getApprovedFixture(state: WorkspaceState, catalogIndex = 2) {
  const asset = listWorkbenchAssets()[catalogIndex]
  const roi = getCaseRoi(state, asset.id)

  expect(roi?.status).toBe('approved')
  return { asset, roi: roi as ApprovedRoiAnnotation }
}

function makeBinding(
  state: WorkspaceState,
  overrides: {
    readonly catalogIndex?: number
    readonly runId?: string
    readonly attemptToken?: string
    readonly threshold?: number
  } = {},
): InferenceBinding {
  const { asset, roi } = getApprovedFixture(state, overrides.catalogIndex)

  return createInferenceBinding({
    clientRunId: overrides.runId ?? 'run-001',
    attemptToken: overrides.attemptToken ?? 'token-001',
    caseId: asset.id,
    assetId: asset.id,
    assetSha256: asset.sha256,
    roi,
    modelVersion: 'mock-salience-v0.3',
    modelMode: 'mock_only',
    config: { threshold: overrides.threshold ?? 0.42, smoothing: 0.27 },
  })
}

function reduce(state: WorkspaceState, ...actions: readonly WorkspaceAction[]) {
  return actions.reduce(workspaceReducer, state)
}

function asyncActionBinding(binding: InferenceBinding, attemptId = 'attempt-001') {
  return {
    runId: binding.clientRunId,
    attemptId,
    attemptToken: binding.attemptToken,
    inputFingerprint: binding.inputFingerprint,
  }
}

function createRunningRun(
  state: WorkspaceState,
  binding = makeBinding(state),
  attemptId = 'attempt-001',
) {
  const asyncBinding = asyncActionBinding(binding, attemptId)
  return {
    binding,
    asyncBinding,
    state: reduce(
      state,
      { type: 'run/create', runId: binding.clientRunId, attemptId, binding },
      { type: 'run/validate', ...asyncBinding },
      { type: 'run/queue', ...asyncBinding },
      { type: 'run/start', ...asyncBinding },
    ),
  }
}

function createSuccessfulRun(
  state: WorkspaceState,
  binding = makeBinding(state),
  attemptId = 'attempt-001',
) {
  const running = createRunningRun(state, binding, attemptId)
  const output = runMockEngine(binding)
  return {
    ...running,
    output,
    state: workspaceReducer(running.state, {
      type: 'run/succeed',
      ...running.asyncBinding,
      output,
    }),
  }
}

function deepFreeze<T>(value: T): T {
  if (!value || typeof value !== 'object' || Object.isFrozen(value)) return value
  Object.freeze(value)
  for (const nested of Object.values(value)) deepFreeze(nested)
  return value
}

function makeMutableOutput(binding: InferenceBinding) {
  const output = runMockEngine(binding)
  return {
    ...output,
    binding: {
      ...output.binding,
      config: { ...output.binding.config },
      roiGeometry: { ...output.binding.roiGeometry },
    },
    heatmap: output.heatmap.map((point) => ({ ...point })),
    metrics: { ...output.metrics },
    qualityGates: { ...output.qualityGates },
    provenance: { ...output.provenance },
  } satisfies InferenceOutput
}

describe('workspace reducer initial state', () => {
  it('seeds deterministic independent ROI workflow state without storage or network access', () => {
    const storageSpy = vi.spyOn(Storage.prototype, 'getItem')
    const fetchSpy = vi.spyOn(globalThis, 'fetch')

    const first = createInitialWorkspaceState()
    const second = createInitialWorkspaceState()
    const catalog = listWorkbenchAssets()
    const rois = catalog.map((entry) => getCaseRoi(first, entry.id))

    expect(catalog).toHaveLength(10)
    expect(rois.map((roi) => roi?.status)).toEqual([
      'draft',
      'in_review',
      'approved',
      'approved',
      'approved',
      'approved',
      'approved',
      'approved',
      'approved',
      'approved',
    ])
    expect(rois.every((roi) => roi && validateNormalizedRoi(roi.geometry))).toBe(true)
    expect(rois.slice(2).every((roi) => roi?.authorId === 'demo_author')).toBe(true)
    expect(rois.slice(2).every((roi) => roi?.reviewerId === 'demo_reviewer')).toBe(true)
    expect(first).toEqual(second)
    expect(first).not.toBe(second)
    expect(first.roisByCase).not.toBe(second.roisByCase)
    expect(getCaseRoi(first, catalog[0].id)).not.toBe(
      getCaseRoi(second, catalog[0].id),
    )
    expect(getCaseRoi(first, catalog[0].id)?.geometry).not.toBe(
      getCaseRoi(second, catalog[0].id)?.geometry,
    )
    expect(first).toMatchObject({
      runsById: {},
      runOrder: [],
      attemptsById: {},
      batchesById: {},
      batchOrder: [],
      reviewsByDigest: {},
    })
    expect(first.activeRunId).toBeUndefined()
    expect(first.lastFailure).toBeUndefined()
    expect(storageSpy).not.toHaveBeenCalled()
    expect(fetchSpy).not.toHaveBeenCalled()

    storageSpy.mockRestore()
    fetchSpy.mockRestore()
  })
})

describe('workspace reducer ROI lifecycle', () => {
  it('supports every legal author-reviewer transition and geometry edit', () => {
    const initial = createInitialWorkspaceState()
    const caseId = listWorkbenchAssets()[0].id
    const original = getCaseRoi(initial, caseId)!
    const geometry: NormalizedRoi = { x: 0.2, y: 0.18, width: 0.42, height: 0.37 }

    const edited = workspaceReducer(initial, {
      type: 'roi/updateGeometry',
      caseId,
      actorId: 'demo_author',
      geometry,
    })
    expect(getCaseRoi(edited, caseId)).toMatchObject({
      status: 'draft',
      version: original.version + 1,
      geometry,
    })

    const inReview = workspaceReducer(edited, {
      type: 'roi/submitReview',
      caseId,
      actorId: 'demo_author',
    })
    expect(getCaseRoi(inReview, caseId)?.status).toBe('in_review')

    const changesRequested = workspaceReducer(inReview, {
      type: 'roi/requestChanges',
      caseId,
      actorId: 'demo_reviewer',
    })
    expect(getCaseRoi(changesRequested, caseId)).toMatchObject({
      status: 'changes_requested',
      reviewerId: 'demo_reviewer',
    })

    const reopened = workspaceReducer(changesRequested, {
      type: 'roi/reopenDraft',
      caseId,
      actorId: 'demo_author',
    })
    expect(getCaseRoi(reopened, caseId)?.status).toBe('draft')

    const approved = reduce(
      reopened,
      { type: 'roi/submitReview', caseId, actorId: 'demo_author' },
      { type: 'roi/approve', caseId, actorId: 'demo_reviewer' },
    )
    expect(getCaseRoi(approved, caseId)).toMatchObject({
      status: 'approved',
      authorId: 'demo_author',
      reviewerId: 'demo_reviewer',
    })

    const superseded = workspaceReducer(approved, {
      type: 'roi/supersede',
      caseId,
      actorId: 'demo_author',
    })
    expect(getCaseRoi(superseded, caseId)?.status).toBe('superseded')
  })

  it('rejects invalid geometry without changing ROI workflow state', () => {
    const initial = createInitialWorkspaceState()
    const caseId = listWorkbenchAssets()[0].id
    const roiBefore = getCaseRoi(initial, caseId)

    const next = workspaceReducer(initial, {
      type: 'roi/updateGeometry',
      caseId,
      actorId: 'demo_author',
      geometry: { x: 0.8, y: 0.2, width: 0.3, height: 0.4 },
    })

    expect(getCaseRoi(next, caseId)).toBe(roiBefore)
    expect(next.lastFailure?.reason).toBe('INVALID_ROI_GEOMETRY')
  })

  it('enforces author-reviewer separation and records stable workflow failures', () => {
    const initial = createInitialWorkspaceState()
    const caseId = listWorkbenchAssets()[1].id
    const roiBefore = getCaseRoi(initial, caseId)

    const selfReview = workspaceReducer(initial, {
      type: 'roi/approve',
      caseId,
      actorId: 'demo_author',
    })
    expect(getCaseRoi(selfReview, caseId)).toBe(roiBefore)
    expect(selfReview.lastFailure?.reason).toBe('SELF_REVIEW_FORBIDDEN')

    const unauthorized = workspaceReducer(initial, {
      type: 'roi/approve',
      caseId,
      actorId: 'not_a_workbench_actor',
    })
    expect(getCaseRoi(unauthorized, caseId)).toBe(roiBefore)
    expect(unauthorized.lastFailure?.reason).toBe('ACTOR_NOT_AUTHORIZED')

    const illegal = workspaceReducer(initial, {
      type: 'roi/supersede',
      caseId,
      actorId: 'demo_author',
    })
    expect(getCaseRoi(illegal, caseId)).toBe(roiBefore)
    expect(illegal.lastFailure?.reason).toBe('ILLEGAL_TRANSITION')
  })
})

describe('workspace reducer run lifecycle', () => {
  it('moves through the legal happy path and exposes only the current success', () => {
    const initial = createInitialWorkspaceState()
    const binding = makeBinding(initial)
    const asyncBinding = asyncActionBinding(binding)
    const output = runMockEngine(binding)

    const created = workspaceReducer(initial, {
      type: 'run/create',
      runId: binding.clientRunId,
      attemptId: asyncBinding.attemptId,
      binding,
    })
    expect(getRun(created, binding.clientRunId)).toMatchObject({
      clientRunId: binding.clientRunId,
      caseId: binding.caseId,
      assetId: binding.assetId,
      status: 'draft',
      attemptIds: [asyncBinding.attemptId],
      activeAttemptId: asyncBinding.attemptId,
    })
    expect(getAttempt(created, asyncBinding.attemptId)).toMatchObject({
      id: asyncBinding.attemptId,
      clientRunId: binding.clientRunId,
      attemptToken: binding.attemptToken,
      status: 'draft',
      binding,
    })
    expect(created.activeRunId).toBe(binding.clientRunId)
    expect(getDisplayableResult(created)).toBeUndefined()

    const validating = workspaceReducer(created, {
      type: 'run/validate',
      ...asyncBinding,
    })
    expect(getAttempt(validating, asyncBinding.attemptId)?.status).toBe('validating')

    const queued = workspaceReducer(validating, {
      type: 'run/queue',
      ...asyncBinding,
    })
    expect(getAttempt(queued, asyncBinding.attemptId)?.status).toBe('queued')

    const running = workspaceReducer(queued, {
      type: 'run/start',
      ...asyncBinding,
    })
    expect(getAttempt(running, asyncBinding.attemptId)?.status).toBe('running')

    const succeeded = workspaceReducer(running, {
      type: 'run/succeed',
      ...asyncBinding,
      output,
    })
    expect(getRun(succeeded, binding.clientRunId)?.status).toBe('succeeded')
    expect(getAttempt(succeeded, asyncBinding.attemptId)).toMatchObject({
      status: 'succeeded',
      result: { output, freshness: 'current' },
    })
    const storedOutput = getAttempt(succeeded, asyncBinding.attemptId)?.result?.output
    expect(storedOutput).toEqual(output)
    expect(storedOutput).not.toBe(output)
    expect(getDisplayableResult(succeeded)).toBe(storedOutput)
    expect(getDisplayableResult(succeeded, asyncBinding.attemptId)).toBe(storedOutput)
    expect(getDisplayableResult(succeeded, binding.clientRunId)).toBe(storedOutput)
  })

  it('hides current successes unless research display is eligible and clinical use stays blocked', () => {
    const running = createRunningRun(createInitialWorkspaceState())
    const output = makeMutableOutput(running.binding)
    output.qualityGates.researchDisplayEligible = false
    expect(output.qualityGates.clinicalUseEligible).toBe(false)

    const succeeded = workspaceReducer(running.state, {
      type: 'run/succeed',
      ...running.asyncBinding,
      output,
    })
    expect(getAttempt(succeeded, running.asyncBinding.attemptId)).toMatchObject({
      status: 'succeeded',
      result: {
        freshness: 'current',
        output: {
          qualityGates: {
            researchDisplayEligible: false,
            clinicalUseEligible: false,
          },
        },
      },
    })
    expect(getDisplayableResult(succeeded)).toBeUndefined()

    const attempt = getAttempt(succeeded, running.asyncBinding.attemptId)!
    const clinicallyEligibleState: WorkspaceState = {
      ...succeeded,
      attemptsById: {
        ...succeeded.attemptsById,
        [attempt.id]: {
          ...attempt,
          result: {
            ...attempt.result!,
            output: {
              ...attempt.result!.output,
              qualityGates: {
                ...attempt.result!.output.qualityGates,
                researchDisplayEligible: true,
                clinicalUseEligible: true,
              },
            } as unknown as InferenceOutput,
          },
        },
      },
    }
    expect(getDisplayableResult(clinicallyEligibleState)).toBeUndefined()
  })

  it('enforces one global run-attempt ID namespace and unique initial tokens', () => {
    const initial = createInitialWorkspaceState()
    const binding = makeBinding(initial)
    const createAction = {
      type: 'run/create',
      runId: binding.clientRunId,
      attemptId: 'attempt-001',
      binding,
    } as const satisfies WorkspaceAction
    const created = workspaceReducer(initial, createAction)
    expect(workspaceReducer(created, createAction)).toBe(created)

    const conflictingBinding = makeBinding(created, {
      runId: binding.clientRunId,
      attemptToken: 'token-conflicting-create',
    })
    const conflictingReplay = workspaceReducer(created, {
      ...createAction,
      binding: conflictingBinding,
    })
    expect(conflictingReplay).not.toBe(created)
    expect(conflictingReplay.runsById).toBe(created.runsById)
    expect(conflictingReplay.attemptsById).toBe(created.attemptsById)
    expect(conflictingReplay.lastFailure?.reason).toBe('INVALID_OPERATIONAL_ID')

    const collisionCases = [
      {
        runId: 'attempt-001',
        attemptId: 'attempt-run-id-collision',
        attemptToken: 'token-run-id-collision',
      },
      {
        runId: 'run-attempt-id-collision',
        attemptId: binding.clientRunId,
        attemptToken: 'token-attempt-id-collision',
      },
      {
        runId: 'same-operational-id',
        attemptId: 'same-operational-id',
        attemptToken: 'token-same-operational-id',
      },
      {
        runId: 'run-token-collision',
        attemptId: 'attempt-token-collision',
        attemptToken: binding.attemptToken,
      },
    ] as const

    for (const collision of collisionCases) {
      const collisionBinding = makeBinding(created, collision)
      const rejected = workspaceReducer(created, {
        type: 'run/create',
        runId: collision.runId,
        attemptId: collision.attemptId,
        binding: collisionBinding,
      })
      expect(rejected.runsById).toBe(created.runsById)
      expect(rejected.attemptsById).toBe(created.attemptsById)
      expect(rejected.lastFailure?.reason).toBe('INVALID_OPERATIONAL_ID')
    }
  })

  it('ignores direct success and every stale async binding mismatch by identity', () => {
    const initial = createInitialWorkspaceState()
    const binding = makeBinding(initial)
    const asyncBinding = asyncActionBinding(binding)
    const created = workspaceReducer(initial, {
      type: 'run/create',
      runId: binding.clientRunId,
      attemptId: asyncBinding.attemptId,
      binding,
    })
    const output = runMockEngine(binding)

    expect(
      workspaceReducer(created, {
        type: 'run/succeed',
        ...asyncBinding,
        output,
      }),
    ).toBe(created)

    const mismatches: WorkspaceAction[] = [
      { type: 'run/validate', ...asyncBinding, runId: 'stale-run' },
      { type: 'run/validate', ...asyncBinding, attemptId: 'stale-attempt' },
      { type: 'run/validate', ...asyncBinding, attemptToken: 'stale-token' },
      { type: 'run/validate', ...asyncBinding, inputFingerprint: 'stale-input' },
    ]
    for (const action of mismatches) {
      expect(workspaceReducer(created, action)).toBe(created)
    }

    const running = createRunningRun(initial, binding).state
    const otherBinding = makeBinding(initial, {
      runId: binding.clientRunId,
      attemptToken: binding.attemptToken,
      threshold: 0.67,
    })
    const mismatchedOutput = runMockEngine(otherBinding)
    expect(
      workspaceReducer(running, {
        type: 'run/succeed',
        ...asyncBinding,
        output: mismatchedOutput,
      }),
    ).toBe(running)

    const inconsistentStatus: WorkspaceState = {
      ...running,
      runsById: {
        ...running.runsById,
        [binding.clientRunId]: {
          ...getRun(running, binding.clientRunId)!,
          status: 'draft',
        },
      },
    }
    expect(
      workspaceReducer(inconsistentStatus, {
        type: 'run/succeed',
        ...asyncBinding,
        output,
      }),
    ).toBe(inconsistentStatus)
  })

  it('keeps cancelled and failed attempts result-free and ignores late success', () => {
    const initial = createInitialWorkspaceState()

    const queuedBinding = makeBinding(initial, {
      runId: 'run-cancelled',
      attemptToken: 'token-cancelled',
    })
    const queuedAsync = asyncActionBinding(queuedBinding, 'attempt-cancelled')
    const queued = reduce(
      initial,
      {
        type: 'run/create',
        runId: queuedBinding.clientRunId,
        attemptId: queuedAsync.attemptId,
        binding: queuedBinding,
      },
      { type: 'run/validate', ...queuedAsync },
      { type: 'run/queue', ...queuedAsync },
    )
    const cancelled = workspaceReducer(queued, {
      type: 'run/cancel',
      ...queuedAsync,
    })
    expect(getAttempt(cancelled, queuedAsync.attemptId)).toMatchObject({
      status: 'cancelled',
    })
    expect(getAttempt(cancelled, queuedAsync.attemptId)?.result).toBeUndefined()
    expect(getDisplayableResult(cancelled)).toBeUndefined()
    expect(
      workspaceReducer(cancelled, {
        type: 'run/succeed',
        ...queuedAsync,
        output: runMockEngine(queuedBinding),
      }),
    ).toBe(cancelled)

    const failedBinding = makeBinding(initial, {
      runId: 'run-failed',
      attemptToken: 'token-failed',
    })
    const failedAsync = asyncActionBinding(failedBinding, 'attempt-failed')
    const running = createRunningRun(initial, failedBinding, failedAsync.attemptId).state
    const failure = {
      reason: 'NETWORK_ERROR',
      message: 'Synthetic failure fixture.',
    } as const
    const failed = workspaceReducer(running, {
      type: 'run/fail',
      ...failedAsync,
      failure,
    })
    expect(getAttempt(failed, failedAsync.attemptId)).toMatchObject({
      status: 'failed',
      failure,
    })
    expect(getAttempt(failed, failedAsync.attemptId)?.result).toBeUndefined()
    expect(getDisplayableResult(failed)).toBeUndefined()
    expect(
      workspaceReducer(failed, {
        type: 'run/succeed',
        ...failedAsync,
        output: runMockEngine(failedBinding),
      }),
    ).toBe(failed)
  })

  it('stales prior current results when a different input or ROI revision supersedes them', () => {
    const initial = createInitialWorkspaceState()
    const first = createSuccessfulRun(initial)
    const firstAttemptId = first.asyncBinding.attemptId

    const changedBinding = makeBinding(first.state, {
      runId: 'run-different-input',
      attemptToken: 'token-different-input',
      threshold: 0.63,
    })
    expect(changedBinding.inputFingerprint).not.toBe(first.binding.inputFingerprint)
    const changedRun = workspaceReducer(first.state, {
      type: 'run/create',
      runId: changedBinding.clientRunId,
      attemptId: 'attempt-different-input',
      binding: changedBinding,
    })
    expect(getAttempt(changedRun, firstAttemptId)?.result?.freshness).toBe('stale')
    expect(getDisplayableResult(changedRun, firstAttemptId)).toBeUndefined()

    const secondInitial = createInitialWorkspaceState()
    const successful = createSuccessfulRun(secondInitial)
    const superseded = workspaceReducer(successful.state, {
      type: 'roi/supersede',
      caseId: successful.binding.caseId,
      actorId: 'demo_author',
    })
    expect(getAttempt(superseded, successful.asyncBinding.attemptId)?.result?.freshness).toBe(
      'stale',
    )
    expect(getDisplayableResult(superseded, successful.asyncBinding.attemptId)).toBeUndefined()

    const editableState: WorkspaceState = {
      ...successful.state,
      roisByCase: {
        ...successful.state.roisByCase,
        [successful.binding.caseId]: {
          ...getCaseRoi(successful.state, successful.binding.caseId)!,
          status: 'changes_requested',
        },
      },
    }
    const roi = getCaseRoi(editableState, successful.binding.caseId)!
    const edited = workspaceReducer(editableState, {
      type: 'roi/updateGeometry',
      caseId: successful.binding.caseId,
      actorId: 'demo_author',
      geometry: { ...roi.geometry, x: roi.geometry.x + 0.01 },
    })
    expect(getCaseRoi(edited, successful.binding.caseId)?.version).toBe(roi.version + 1)
    expect(getAttempt(edited, successful.asyncBinding.attemptId)?.result?.freshness).toBe(
      'stale',
    )
  })

  it('appends retries with parent lineage and ignores late parent completion', () => {
    const initial = createInitialWorkspaceState()
    const binding = makeBinding(initial)
    const parent = createRunningRun(initial, binding)
    const failure = {
      reason: 'REQUEST_TIMEOUT',
      message: 'Synthetic timeout fixture.',
    } as const
    const failed = workspaceReducer(parent.state, {
      type: 'run/fail',
      ...parent.asyncBinding,
      failure,
    })
    const parentSnapshot = getAttempt(failed, parent.asyncBinding.attemptId)
    const retryBinding = makeBinding(failed, {
      runId: binding.clientRunId,
      attemptToken: 'token-retry-002',
    })
    expect(retryBinding.inputFingerprint).toBe(binding.inputFingerprint)
    const retried = workspaceReducer(failed, {
      type: 'run/retry',
      runId: binding.clientRunId,
      attemptId: 'attempt-retry-002',
      parentAttemptId: parent.asyncBinding.attemptId,
      binding: retryBinding,
    })

    expect(getAttempt(retried, parent.asyncBinding.attemptId)).toBe(parentSnapshot)
    expect(getAttempt(retried, parent.asyncBinding.attemptId)).toMatchObject({
      status: 'failed',
      failure,
    })
    expect(getAttempt(retried, 'attempt-retry-002')).toMatchObject({
      status: 'draft',
      parentAttemptId: parent.asyncBinding.attemptId,
      binding: retryBinding,
    })
    expect(getRun(retried, binding.clientRunId)).toMatchObject({
      status: 'draft',
      attemptIds: [parent.asyncBinding.attemptId, 'attempt-retry-002'],
      activeAttemptId: 'attempt-retry-002',
    })

    expect(
      workspaceReducer(retried, {
        type: 'run/succeed',
        ...parent.asyncBinding,
        output: runMockEngine(binding),
      }),
    ).toBe(retried)

    const retryAsync = asyncActionBinding(retryBinding, 'attempt-retry-002')
    const retrySucceeded = reduce(
      retried,
      { type: 'run/validate', ...retryAsync },
      { type: 'run/queue', ...retryAsync },
      { type: 'run/start', ...retryAsync },
      {
        type: 'run/succeed',
        ...retryAsync,
        output: runMockEngine(retryBinding),
      },
    )
    expect(getAttempt(retrySucceeded, parent.asyncBinding.attemptId)).toBe(parentSnapshot)
    expect(getDisplayableResult(retrySucceeded)).toEqual(runMockEngine(retryBinding))
    expect(getDisplayableResult(retrySucceeded, parent.asyncBinding.attemptId)).toBeUndefined()
  })

  it('rejects retry ID collisions and conflicting replays while preserving exact replay identity', () => {
    const initial = createInitialWorkspaceState()
    const binding = makeBinding(initial)
    const parent = createRunningRun(initial, binding)
    const failed = workspaceReducer(parent.state, {
      type: 'run/fail',
      ...parent.asyncBinding,
      failure: { reason: 'REQUEST_TIMEOUT', message: 'Synthetic timeout fixture.' },
    })
    const retryBinding = makeBinding(failed, {
      runId: binding.clientRunId,
      attemptToken: 'token-retry-global-id',
    })
    const idCollision = workspaceReducer(failed, {
      type: 'run/retry',
      runId: binding.clientRunId,
      attemptId: binding.clientRunId,
      parentAttemptId: parent.asyncBinding.attemptId,
      binding: retryBinding,
    })
    expect(idCollision.runsById).toBe(failed.runsById)
    expect(idCollision.attemptsById).toBe(failed.attemptsById)
    expect(idCollision.lastFailure?.reason).toBe('INVALID_OPERATIONAL_ID')

    const retryAction = {
      type: 'run/retry',
      runId: binding.clientRunId,
      attemptId: 'attempt-retry-global-id',
      parentAttemptId: parent.asyncBinding.attemptId,
      binding: retryBinding,
    } as const satisfies WorkspaceAction
    const retried = workspaceReducer(failed, retryAction)
    expect(workspaceReducer(retried, retryAction)).toBe(retried)

    const conflictingBinding = makeBinding(failed, {
      runId: binding.clientRunId,
      attemptToken: 'token-retry-conflict',
    })
    const conflictingReplay = workspaceReducer(retried, {
      ...retryAction,
      binding: conflictingBinding,
    })
    expect(conflictingReplay).not.toBe(retried)
    expect(conflictingReplay.runsById).toBe(retried.runsById)
    expect(conflictingReplay.attemptsById).toBe(retried.attemptsById)
    expect(conflictingReplay.lastFailure?.reason).toBe('INVALID_OPERATIONAL_ID')
  })

  it('rejects a retry that reuses an existing operational attempt token', () => {
    const initial = createInitialWorkspaceState()
    const binding = makeBinding(initial)
    const parent = createRunningRun(initial, binding)
    const failed = workspaceReducer(parent.state, {
      type: 'run/fail',
      ...parent.asyncBinding,
      failure: { reason: 'REQUEST_TIMEOUT', message: 'Synthetic timeout fixture.' },
    })
    const reusedTokenBinding = makeBinding(failed, {
      runId: binding.clientRunId,
      attemptToken: binding.attemptToken,
    })

    const rejected = workspaceReducer(failed, {
      type: 'run/retry',
      runId: binding.clientRunId,
      attemptId: 'attempt-retry-reused-token',
      parentAttemptId: parent.asyncBinding.attemptId,
      binding: reusedTokenBinding,
    })

    expect(getRun(rejected, binding.clientRunId)?.attemptIds).toEqual([
      parent.asyncBinding.attemptId,
    ])
    expect(getAttempt(rejected, 'attempt-retry-reused-token')).toBeUndefined()
    expect(rejected.lastFailure?.reason).toBe('INVALID_OPERATIONAL_ID')
  })

  it('revokes a successful result without exposing it again', () => {
    const successful = createSuccessfulRun(createInitialWorkspaceState())
    const revoked = workspaceReducer(successful.state, {
      type: 'result/revoke',
      runId: successful.binding.clientRunId,
      attemptId: successful.asyncBinding.attemptId,
    })

    expect(getAttempt(revoked, successful.asyncBinding.attemptId)?.result?.freshness).toBe(
      'revoked',
    )
    expect(getDisplayableResult(revoked)).toBeUndefined()
    expect(
      workspaceReducer(revoked, {
        type: 'result/revoke',
        runId: successful.binding.clientRunId,
        attemptId: successful.asyncBinding.attemptId,
      }),
    ).toBe(revoked)
  })

  it('makes duplicate events exact no-ops without mutating frozen prior state or payloads', () => {
    const initial = createInitialWorkspaceState()
    const canonical = makeBinding(initial)
    const callerBinding = {
      ...canonical,
      config: { ...canonical.config },
      roiGeometry: { ...canonical.roiGeometry },
    }
    const createAction = {
      type: 'run/create',
      runId: canonical.clientRunId,
      attemptId: 'attempt-001',
      binding: callerBinding,
    } as const satisfies WorkspaceAction
    const frozenInitial = deepFreeze(initial)
    const created = workspaceReducer(frozenInitial, createAction)
    expect(workspaceReducer(created, createAction)).toBe(created)
    expect(getAttempt(created, 'attempt-001')?.binding).not.toBe(callerBinding)

    callerBinding.config.threshold = 0.9
    callerBinding.roiGeometry.x = 0.7
    expect(getAttempt(created, 'attempt-001')?.binding).toEqual(canonical)

    const asyncBinding = asyncActionBinding(canonical)
    const running = reduce(
      created,
      { type: 'run/validate', ...asyncBinding },
      { type: 'run/queue', ...asyncBinding },
      { type: 'run/start', ...asyncBinding },
    )
    const frozenRunning = deepFreeze(running)
    expect(
      workspaceReducer(frozenRunning, { type: 'run/start', ...asyncBinding }),
    ).toBe(frozenRunning)

    const output = makeMutableOutput(canonical)
    expect(Object.isFrozen(output)).toBe(false)
    const succeeded = workspaceReducer(frozenRunning, {
      type: 'run/succeed',
      ...asyncBinding,
      output,
    })
    expect(
      workspaceReducer(succeeded, {
        type: 'run/succeed',
        ...asyncBinding,
        output,
      }),
    ).toBe(succeeded)
    const storedOutput = getAttempt(succeeded, 'attempt-001')?.result?.output
    expect(storedOutput).toEqual(output)
    expect(storedOutput).not.toBe(output)
    expect(Object.isFrozen(storedOutput)).toBe(true)
    expect(Object.isFrozen(storedOutput?.binding)).toBe(true)
    expect(Object.isFrozen(storedOutput?.binding.config)).toBe(true)
    expect(Object.isFrozen(storedOutput?.binding.roiGeometry)).toBe(true)
    expect(Object.isFrozen(storedOutput?.heatmap)).toBe(true)
    expect(Object.isFrozen(storedOutput?.heatmap[0])).toBe(true)
    expect(Object.isFrozen(storedOutput?.metrics)).toBe(true)
    expect(Object.isFrozen(storedOutput?.qualityGates)).toBe(true)
    expect(Object.isFrozen(storedOutput?.provenance)).toBe(true)

    output.resultDigest = 'caller-mutated-digest'
    output.binding.config.threshold = 0.99
    output.binding.roiGeometry.x = 0.91
    output.heatmap[0].x = 0.93
    output.metrics.focusScore = 0
    output.qualityGates.researchDisplayEligible = false
    expect(storedOutput).not.toEqual(output)
    expect(storedOutput?.resultDigest).not.toBe(output.resultDigest)
    expect(storedOutput?.binding.config.threshold).toBe(canonical.config.threshold)
    expect(storedOutput?.binding.roiGeometry.x).toBe(canonical.roiGeometry.x)
    expect(storedOutput?.heatmap[0].x).not.toBe(output.heatmap[0].x)
    expect(storedOutput?.metrics.focusScore).not.toBe(output.metrics.focusScore)
    expect(storedOutput?.qualityGates.researchDisplayEligible).toBe(true)
    expect(getDisplayableResult(succeeded)).toBe(storedOutput)
    expect(getAttempt(frozenRunning, 'attempt-001')?.status).toBe('running')
    expect(getDisplayableResult(frozenRunning)).toBeUndefined()
  })
})
