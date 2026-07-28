import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useLayoutEffect,
  useMemo,
  useReducer,
  useRef,
  type ReactNode,
} from 'react'
import { getWorkbenchAsset } from './catalog'
import { auditBatchManifest, type BatchManifest } from './batchManifest'
import {
  batchSessionReducer,
  createInitialBatchSessionState,
  manifestMatchesBatchDraft,
  type BatchSessionAction,
  type BatchSessionState,
} from './batchSession'
import { createInferenceBinding } from './mockEngine'
import { isVerifiedFullImageSourceBinding } from './sourceBinding'
import { validateInferenceOutputEnvelope } from './inferenceEnvelope'
import { selectExactResultTarget } from './reviewPolicy'
import {
  createInitialWorkspaceState,
  getCaseRoi,
  workspaceReducer,
} from './reducer'
import type { WorkbenchGateway, WorkbenchGatewayMode } from './WorkbenchGateway'
import {
  WorkbenchError,
  type AsyncRunActionBinding,
  type InferenceConfiguration,
  type InferenceOutput,
  type MockModelVersion,
  type NormalizedRoi,
  type ResearchReviewNote,
  type WorkbenchFailure,
  type WorkspaceAction,
  type WorkspaceState,
} from './types'

export type WorkspaceRuntimeIdKind = 'run' | 'attempt' | 'token' | 'review'

export type WorkspaceRuntime = {
  readonly nextId: (kind: WorkspaceRuntimeIdKind) => string
  readonly reset?: () => void
}

export type StartRunInput = {
  readonly caseId: string
  readonly modelVersion: MockModelVersion
  readonly config: InferenceConfiguration
}

export type RunIdentifiers = {
  readonly runId: string
  readonly attemptId: string
}

export type CreateReviewInput = {
  readonly runId: string
  readonly attemptId: string
  readonly note: ResearchReviewNote
}

export type ReviewIdentifiers = {
  readonly reviewId: string
}

export type WorkspaceActions = {
  readonly resetSession: () => void
  readonly updateRoi: (caseId: string, geometry: NormalizedRoi) => void
  readonly submitRoi: (caseId: string) => void
  readonly approveRoi: (caseId: string) => void
  readonly requestRoiChanges: (caseId: string) => void
  readonly reopenRoi: (caseId: string) => void
  readonly supersedeRoi: (caseId: string) => void
  readonly restoreFullImageSourceBinding: (caseId: string) => void
  readonly startRun: (input: StartRunInput) => RunIdentifiers
  readonly cancelRun: (runId: string) => void
  readonly retryRun: (runId: string) => RunIdentifiers
  readonly revokeResult: (runId: string, attemptId: string) => void
  readonly createReview: (input: CreateReviewInput) => ReviewIdentifiers
  readonly approveReview: (reviewId: string, note: ResearchReviewNote) => void
  readonly requestReviewChanges: (
    reviewId: string,
    note: ResearchReviewNote,
  ) => void
  readonly resubmitReview: (reviewId: string, note: ResearchReviewNote) => void
  readonly revokeReview: (reviewId: string, note: ResearchReviewNote) => void
}

export type BatchWorkspaceActions = {
  readonly toggleCase: (caseId: string) => void
  readonly selectAllCases: (caseIds: readonly string[]) => void
  readonly clearSelection: () => void
  readonly updateConfig: (
    field: keyof InferenceConfiguration,
    value: number,
  ) => void
  readonly updateModel: (modelVersion: MockModelVersion) => void
  readonly setManifest: (manifest: BatchManifest) => void
  readonly startBatch: () => string | undefined
}

export type WorkspaceContextValue = {
  readonly state: WorkspaceState
  readonly actions: WorkspaceActions
  readonly batchState: BatchSessionState
  readonly batchActions: BatchWorkspaceActions
  readonly gatewayMode: WorkbenchGatewayMode
  readonly persistence: 'memory_only'
}

export type WorkspaceProviderProps = {
  readonly children: ReactNode
  readonly gateway: WorkbenchGateway
  readonly runtime?: WorkspaceRuntime
  readonly initialState?: WorkspaceState
  readonly queueDelayMs?: number
}

function createCounterRuntime(): WorkspaceRuntime {
  let counter = 0
  return {
    nextId(kind) {
      counter += 1
      return `${kind}-${counter}`
    },
  }
}

function cloneValue<T>(value: T): T {
  if (Array.isArray(value)) return value.map((entry) => cloneValue(entry)) as T
  if (value !== null && typeof value === 'object') {
    return Object.fromEntries(
      Object.entries(value).map(([key, entry]) => [key, cloneValue(entry)]),
    ) as T
  }
  return value
}

function initializeWorkspaceState(initialState: WorkspaceState | undefined): WorkspaceState {
  return initialState ? cloneValue(initialState) : createInitialWorkspaceState()
}

function unknownCase(caseId: string): WorkbenchError {
  return new WorkbenchError({
    reason: 'UNKNOWN_CASE',
    message: `Unknown workbench case: ${caseId}.`,
    field: 'caseId',
  })
}

function invalidOperationalId(message: string): WorkbenchError {
  return new WorkbenchError({
    reason: 'INVALID_OPERATIONAL_ID',
    message,
  })
}

function fullImageSourceBindingRequired(): WorkbenchError {
  return new WorkbenchError({
    reason: 'FULL_IMAGE_SOURCE_BINDING_REQUIRED',
    message:
      'Run blocked: restore the verified full-image source binding before inference.',
    field: 'sourceBinding',
  })
}

function operationalIdExists(state: WorkspaceState, id: string): boolean {
  return Boolean(state.runsById[id] || state.attemptsById[id])
}

function attemptTokenExists(state: WorkspaceState, token: string): boolean {
  return Object.values(state.attemptsById).some(
    (attempt) => attempt.attemptToken === token,
  )
}

function assertNewRunOperationalIds(
  state: WorkspaceState,
  runId: string,
  attemptId: string,
  attemptToken: string,
): void {
  if (
    typeof runId !== 'string' ||
    typeof attemptId !== 'string' ||
    typeof attemptToken !== 'string' ||
    runId.trim().length === 0 ||
    attemptId.trim().length === 0 ||
    attemptToken.trim().length === 0 ||
    runId === attemptId ||
    operationalIdExists(state, runId) ||
    operationalIdExists(state, attemptId) ||
    attemptTokenExists(state, attemptToken)
  ) {
    throw invalidOperationalId(
      'Run IDs, attempt IDs, and attempt tokens must be non-empty and unique.',
    )
  }
}

function assertRetryOperationalIds(
  state: WorkspaceState,
  runId: string,
  attemptId: string,
  attemptToken: string,
): void {
  if (
    typeof attemptId !== 'string' ||
    typeof attemptToken !== 'string' ||
    attemptId.trim().length === 0 ||
    attemptToken.trim().length === 0 ||
    attemptId === runId ||
    operationalIdExists(state, attemptId) ||
    attemptTokenExists(state, attemptToken)
  ) {
    throw invalidOperationalId(
      'Retry attempt IDs and tokens must be non-empty and unique.',
    )
  }
}

function gatewayFailure(error: unknown): WorkbenchFailure {
  if (error instanceof WorkbenchError) return error.failure
  return {
    reason: 'NETWORK_ERROR',
    message:
      error instanceof Error && error.message.trim().length > 0
        ? error.message
        : 'The inference request failed.',
  }
}

const WorkspaceContext = createContext<WorkspaceContextValue | null>(null)

export function WorkspaceProvider({
  children,
  gateway,
  runtime,
  initialState,
  queueDelayMs = 0,
}: WorkspaceProviderProps) {
  const defaultRuntimeRef = useRef<WorkspaceRuntime | null>(null)
  if (defaultRuntimeRef.current === null) {
    defaultRuntimeRef.current = createCounterRuntime()
  }
  const activeRuntime = runtime ?? defaultRuntimeRef.current
  const [state, dispatch] = useReducer(
    workspaceReducer,
    initialState,
    initializeWorkspaceState,
  )
  const [batchState, batchDispatch] = useReducer(
    batchSessionReducer,
    undefined,
    createInitialBatchSessionState,
  )
  const stateRef = useRef(state)
  const batchStateRef = useRef(batchState)
  const batchJobSequenceRef = useRef(0)
  const controllersRef = useRef(new Map<string, AbortController>())
  const pendingStartsRef = useRef(
    new Map<string, ReturnType<typeof setTimeout>>(),
  )
  const mountedRef = useRef(false)

  useLayoutEffect(() => {
    stateRef.current = state
  }, [state])

  useLayoutEffect(() => {
    batchStateRef.current = batchState
  }, [batchState])

  const dispatchTracked = useCallback((action: WorkspaceAction): WorkspaceState => {
    const nextState = workspaceReducer(stateRef.current, action)
    stateRef.current = nextState
    dispatch(action)
    return nextState
  }, [])

  const dispatchBatchTracked = useCallback(
    (action: BatchSessionAction): BatchSessionState => {
      const nextState = batchSessionReducer(batchStateRef.current, action)
      batchStateRef.current = nextState
      batchDispatch(action)
      return nextState
    },
    [],
  )

  useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
      const pendingStarts = [...pendingStartsRef.current.values()]
      pendingStartsRef.current.clear()
      for (const timer of pendingStarts) clearTimeout(timer)
      const controllers = [...controllersRef.current.values()]
      controllersRef.current.clear()
      for (const controller of controllers) controller.abort()
    }
  }, [])

  const abortAllRequests = useCallback(() => {
    const controllers = [...controllersRef.current.values()]
    controllersRef.current.clear()
    for (const controller of controllers) controller.abort()
  }, [])

  const cancelAllPendingStarts = useCallback(() => {
    const pendingStarts = [...pendingStartsRef.current.values()]
    pendingStartsRef.current.clear()
    for (const timer of pendingStarts) clearTimeout(timer)
  }, [])

  const resetSession = useCallback(() => {
    cancelAllPendingStarts()
    abortAllRequests()
    dispatchTracked({ type: 'session/reset' })
    dispatchBatchTracked({ type: 'session/reset' })
  }, [abortAllRequests, cancelAllPendingStarts, dispatchBatchTracked, dispatchTracked])

  const updateRoi = useCallback(
    (caseId: string, geometry: NormalizedRoi) => {
      dispatchTracked({
        type: 'roi/updateGeometry',
        caseId,
        actorId: 'demo_author',
        geometry,
      })
    },
    [dispatchTracked],
  )

  const submitRoi = useCallback(
    (caseId: string) => {
      dispatchTracked({ type: 'roi/submitReview', caseId, actorId: 'demo_author' })
    },
    [dispatchTracked],
  )

  const approveRoi = useCallback(
    (caseId: string) => {
      dispatchTracked({ type: 'roi/approve', caseId, actorId: 'demo_reviewer' })
    },
    [dispatchTracked],
  )

  const requestRoiChanges = useCallback(
    (caseId: string) => {
      dispatchTracked({
        type: 'roi/requestChanges',
        caseId,
        actorId: 'demo_reviewer',
      })
    },
    [dispatchTracked],
  )

  const reopenRoi = useCallback(
    (caseId: string) => {
      dispatchTracked({ type: 'roi/reopenDraft', caseId, actorId: 'demo_author' })
    },
    [dispatchTracked],
  )

  const supersedeRoi = useCallback(
    (caseId: string) => {
      dispatchTracked({ type: 'roi/supersede', caseId, actorId: 'demo_author' })
    },
    [dispatchTracked],
  )

  const restoreFullImageSourceBinding = useCallback(
    (caseId: string) => {
      dispatchTracked({ type: 'sourceBinding/restoreFullImage', caseId })
    },
    [dispatchTracked],
  )

  const launchRequest = useCallback(
    (
      binding: Parameters<WorkbenchGateway['runInference']>[0],
      attemptId: string,
    ) => {
      const controller = new AbortController()
      controllersRef.current.set(attemptId, controller)
      const asyncBinding: AsyncRunActionBinding = {
        runId: binding.clientRunId,
        attemptId,
        attemptToken: binding.attemptToken,
        inputFingerprint: binding.inputFingerprint,
      }

      let request: Promise<InferenceOutput>
      try {
        request = gateway.runInference(binding, { signal: controller.signal })
      } catch (error) {
        request = Promise.reject(error)
      }

      void Promise.resolve(request)
        .then((output) => {
          if (
            mountedRef.current &&
            controllersRef.current.get(attemptId) === controller &&
            !controller.signal.aborted
          ) {
            const envelope = validateInferenceOutputEnvelope(
              output,
              binding,
              gateway.mode,
            )
            if (!envelope.valid) {
              dispatchTracked({
                type: 'run/fail',
                ...asyncBinding,
                failure: envelope.failure,
              })
            } else {
              dispatchTracked({
                type: 'run/succeed',
                ...asyncBinding,
                output: envelope.output,
              })
            }
          }
        })
        .catch((error: unknown) => {
          if (
            mountedRef.current &&
            controllersRef.current.get(attemptId) === controller &&
            !controller.signal.aborted
          ) {
            dispatchTracked({
              type: 'run/fail',
              ...asyncBinding,
              failure: gatewayFailure(error),
            })
          }
        })
        .finally(() => {
          if (controllersRef.current.get(attemptId) === controller) {
            controllersRef.current.delete(attemptId)
          }
        })
    },
    [dispatchTracked, gateway],
  )

  const scheduleLaunch = useCallback(
    (
      binding: Parameters<WorkbenchGateway['runInference']>[0],
      attemptId: string,
    ) => {
      const asyncBinding: AsyncRunActionBinding = {
        runId: binding.clientRunId,
        attemptId,
        attemptToken: binding.attemptToken,
        inputFingerprint: binding.inputFingerprint,
      }
      const delay = Number.isFinite(queueDelayMs) && queueDelayMs >= 0
        ? queueDelayMs
        : 0
      const timer = setTimeout(() => {
        pendingStartsRef.current.delete(attemptId)
        if (!mountedRef.current) return
        const nextState = dispatchTracked({ type: 'run/start', ...asyncBinding })
        if (nextState.attemptsById[attemptId]?.status === 'running') {
          launchRequest(binding, attemptId)
        }
      }, delay)
      pendingStartsRef.current.set(attemptId, timer)
    },
    [dispatchTracked, launchRequest, queueDelayMs],
  )

  const startRun = useCallback(
    (input: StartRunInput): RunIdentifiers => {
      const asset = getWorkbenchAsset(input.caseId)
      if (!asset) throw unknownCase(input.caseId)
      const roi = getCaseRoi(stateRef.current, asset.id)
      if (!roi || !isVerifiedFullImageSourceBinding(asset, roi)) {
        throw fullImageSourceBindingRequired()
      }

      const runId = activeRuntime.nextId('run')
      const attemptId = activeRuntime.nextId('attempt')
      const attemptToken = activeRuntime.nextId('token')
      assertNewRunOperationalIds(
        stateRef.current,
        runId,
        attemptId,
        attemptToken,
      )
      const binding = createInferenceBinding({
        clientRunId: runId,
        attemptToken,
        caseId: asset.id,
        assetId: asset.id,
        assetSha256: asset.sha256,
        roi,
        modelVersion: input.modelVersion,
        modelMode: 'mock_only',
        config: input.config,
      })
      const asyncBinding: AsyncRunActionBinding = {
        runId,
        attemptId,
        attemptToken,
        inputFingerprint: binding.inputFingerprint,
      }

      dispatchTracked({ type: 'run/create', runId, attemptId, binding })
      dispatchTracked({ type: 'run/validate', ...asyncBinding })
      dispatchTracked({ type: 'run/queue', ...asyncBinding })
      scheduleLaunch(binding, attemptId)

      return { runId, attemptId }
    },
    [activeRuntime, dispatchTracked, scheduleLaunch],
  )

  const cancelRun = useCallback(
    (runId: string) => {
      const run = stateRef.current.runsById[runId]
      const attemptId = run?.activeAttemptId
      const attempt = attemptId ? stateRef.current.attemptsById[attemptId] : undefined
      const binding = attempt?.binding
      if (!run || !attemptId || !attempt || !binding) return

      const pendingStart = pendingStartsRef.current.get(attemptId)
      if (pendingStart) {
        pendingStartsRef.current.delete(attemptId)
        clearTimeout(pendingStart)
      }
      const controller = controllersRef.current.get(attemptId)
      if (controller) {
        controllersRef.current.delete(attemptId)
        controller.abort()
      }
      dispatchTracked({
        type: 'run/cancel',
        runId,
        attemptId,
        attemptToken: attempt.attemptToken,
        inputFingerprint: binding.inputFingerprint,
      })
    },
    [dispatchTracked],
  )

  const retryRun = useCallback(
    (runId: string): RunIdentifiers => {
      const run = stateRef.current.runsById[runId]
      const parentAttemptId = run?.activeAttemptId
      const parent = parentAttemptId
        ? stateRef.current.attemptsById[parentAttemptId]
        : undefined
      const parentBinding = parent?.binding
      if (!run || !parentAttemptId || !parent || !parentBinding) {
        throw new WorkbenchError({
          reason: 'INVALID_OPERATIONAL_ID',
          message: `Unknown active run: ${runId}.`,
          field: 'runId',
        })
      }
      if (!['blocked', 'failed', 'cancelled'].includes(parent.status)) {
        throw new WorkbenchError({
          reason: 'ILLEGAL_TRANSITION',
          message: `Run retry is not legal from ${parent.status}.`,
        })
      }

      const asset = getWorkbenchAsset(parentBinding.caseId)
      const roi = getCaseRoi(stateRef.current, parentBinding.caseId)
      if (!asset) throw unknownCase(parentBinding.caseId)
      if (!roi || !isVerifiedFullImageSourceBinding(asset, roi)) {
        throw fullImageSourceBindingRequired()
      }

      const attemptId = activeRuntime.nextId('attempt')
      const attemptToken = activeRuntime.nextId('token')
      assertRetryOperationalIds(
        stateRef.current,
        runId,
        attemptId,
        attemptToken,
      )
      const binding = createInferenceBinding({
        clientRunId: runId,
        attemptToken,
        caseId: asset.id,
        assetId: asset.id,
        assetSha256: asset.sha256,
        roi,
        modelVersion: parentBinding.modelVersion,
        modelMode: 'mock_only',
        config: parentBinding.config,
      })
      if (binding.inputFingerprint !== parentBinding.inputFingerprint) {
        throw new WorkbenchError({
          reason: 'IMMUTABLE_BINDING_MISMATCH',
          message: 'A retry must preserve the parent scientific input.',
          field: 'binding',
        })
      }
      const asyncBinding: AsyncRunActionBinding = {
        runId,
        attemptId,
        attemptToken,
        inputFingerprint: binding.inputFingerprint,
      }

      dispatchTracked({
        type: 'run/retry',
        runId,
        attemptId,
        parentAttemptId,
        binding,
      })
      dispatchTracked({ type: 'run/validate', ...asyncBinding })
      dispatchTracked({ type: 'run/queue', ...asyncBinding })
      scheduleLaunch(binding, attemptId)

      return { runId, attemptId }
    },
    [activeRuntime, dispatchTracked, scheduleLaunch],
  )

  const revokeResult = useCallback(
    (runId: string, attemptId: string) => {
      dispatchTracked({ type: 'result/revoke', runId, attemptId })
    },
    [dispatchTracked],
  )

  const createReview = useCallback(
    (input: CreateReviewInput): ReviewIdentifiers => {
      const current = stateRef.current
      if (
        !Object.prototype.hasOwnProperty.call(current.runsById, input.runId) ||
        !Object.prototype.hasOwnProperty.call(current.attemptsById, input.attemptId)
      ) {
        throw new WorkbenchError({
          reason: 'UNKNOWN_REVIEW_TARGET',
          message: 'The exact run and attempt are unavailable in this session.',
        })
      }
      const attempt = current.attemptsById[input.attemptId]
      const output = attempt?.result?.output
      const binding = attempt?.binding
      if (!attempt || !output || !binding) {
        throw new WorkbenchError({
          reason: 'UNKNOWN_REVIEW_TARGET',
          message: 'A succeeded result is required before review can be created.',
        })
      }
      const reference = {
        runId: input.runId,
        attemptId: input.attemptId,
        resultDigest: output.resultDigest,
        inputFingerprint: binding.inputFingerprint,
      }
      const selected = selectExactResultTarget(current, reference)
      if (!selected.ok) {
        throw new WorkbenchError({
          reason: 'UNKNOWN_REVIEW_TARGET',
          message: selected.blockers.map((entry) => entry.message).join(' '),
        })
      }

      const reviewId = activeRuntime.nextId('review')
      const next = dispatchTracked({
        type: 'review/create',
        reviewId,
        ...reference,
        actorId: 'demo_author',
        note: input.note,
      })
      if (
        !Object.prototype.hasOwnProperty.call(next.reviewsById, reviewId) ||
        next.reviewsById[reviewId]?.id !== reviewId
      ) {
        throw new WorkbenchError(
          next.lastFailure ?? {
            reason: 'INVALID_OPERATIONAL_ID',
            message: 'The provider could not create the requested review.',
          },
        )
      }
      return { reviewId }
    },
    [activeRuntime, dispatchTracked],
  )

  const approveReview = useCallback(
    (reviewId: string, note: ResearchReviewNote) => {
      dispatchTracked({
        type: 'review/approve',
        reviewId,
        actorId: 'demo_reviewer',
        note,
      })
    },
    [dispatchTracked],
  )

  const requestReviewChanges = useCallback(
    (reviewId: string, note: ResearchReviewNote) => {
      dispatchTracked({
        type: 'review/requestChanges',
        reviewId,
        actorId: 'demo_reviewer',
        note,
      })
    },
    [dispatchTracked],
  )

  const resubmitReview = useCallback(
    (reviewId: string, note: ResearchReviewNote) => {
      dispatchTracked({
        type: 'review/resubmit',
        reviewId,
        actorId: 'demo_author',
        note,
      })
    },
    [dispatchTracked],
  )

  const revokeReview = useCallback(
    (reviewId: string, note: ResearchReviewNote) => {
      dispatchTracked({
        type: 'review/revoke',
        reviewId,
        actorId: 'demo_reviewer',
        note,
      })
    },
    [dispatchTracked],
  )

  const toggleBatchCase = useCallback(
    (caseId: string) => {
      dispatchBatchTracked({ type: 'selection/toggle', caseId })
    },
    [dispatchBatchTracked],
  )

  const selectAllBatchCases = useCallback(
    (caseIds: readonly string[]) => {
      dispatchBatchTracked({ type: 'selection/selectAll', caseIds })
    },
    [dispatchBatchTracked],
  )

  const clearBatchSelection = useCallback(() => {
    dispatchBatchTracked({ type: 'selection/clear' })
  }, [dispatchBatchTracked])

  const updateBatchConfig = useCallback(
    (field: keyof InferenceConfiguration, value: number) => {
      dispatchBatchTracked({ type: 'config/update', field, value })
    },
    [dispatchBatchTracked],
  )

  const updateBatchModel = useCallback(
    (modelVersion: MockModelVersion) => {
      dispatchBatchTracked({ type: 'model/update', modelVersion })
    },
    [dispatchBatchTracked],
  )

  const setBatchManifest = useCallback(
    (manifest: BatchManifest) => {
      dispatchBatchTracked({ type: 'manifest/set', manifest })
    },
    [dispatchBatchTracked],
  )

  const startBatch = useCallback((): string | undefined => {
    const currentBatch = batchStateRef.current
    if (currentBatch.job) return currentBatch.job.id
    const manifest = currentBatch.manifest
    if (!manifest) return undefined
    const manifestAudit = auditBatchManifest(manifest, stateRef.current)
    if (!manifestAudit.valid) return undefined

    try {
      if (!manifestMatchesBatchDraft(currentBatch)) return undefined
      const readyItems = manifest.items.filter(
        (item) => item.preflight === 'ready',
      )
      if (readyItems.length === 0) return undefined

      let draftWorkspace = stateRef.current
      const actionsToCommit: WorkspaceAction[] = []
      const launches: Array<{
        readonly binding: Parameters<WorkbenchGateway['runInference']>[0]
        readonly attemptId: string
      }> = []
      const runIdsByCase: Record<string, string> = {}
      const allocatedIds = new Set([
        ...Object.keys(draftWorkspace.runsById),
        ...Object.keys(draftWorkspace.attemptsById),
        ...Object.values(draftWorkspace.attemptsById).map(
          (attempt) => attempt.attemptToken,
        ),
      ])

      for (const item of readyItems) {
        const asset = getWorkbenchAsset(item.caseId)
        const roi = asset ? getCaseRoi(draftWorkspace, asset.id) : undefined
        if (!asset || !roi) return undefined

        const runId = activeRuntime.nextId('run')
        const attemptId = activeRuntime.nextId('attempt')
        const attemptToken = activeRuntime.nextId('token')
        const nextIds = [runId, attemptId, attemptToken]
        if (
          nextIds.some((id) => typeof id !== 'string' || id.trim().length === 0) ||
          new Set(nextIds).size !== nextIds.length ||
          nextIds.some((id) => allocatedIds.has(id))
        ) {
          return undefined
        }
        nextIds.forEach((id) => allocatedIds.add(id))
        assertNewRunOperationalIds(
          draftWorkspace,
          runId,
          attemptId,
          attemptToken,
        )
        const binding = createInferenceBinding({
          clientRunId: runId,
          attemptToken,
          caseId: asset.id,
          assetId: asset.id,
          assetSha256: asset.sha256,
          roi,
          modelVersion: manifest.modelVersion,
          modelMode: 'mock_only',
          config: manifest.config,
        })
        const asyncBinding: AsyncRunActionBinding = {
          runId,
          attemptId,
          attemptToken,
          inputFingerprint: binding.inputFingerprint,
        }
        const runActions: WorkspaceAction[] = [
          { type: 'run/create', runId, attemptId, binding },
          { type: 'run/validate', ...asyncBinding },
          { type: 'run/queue', ...asyncBinding },
        ]
        for (const action of runActions) {
          draftWorkspace = workspaceReducer(draftWorkspace, action)
        }
        if (
          draftWorkspace.runsById[runId]?.activeAttemptId !== attemptId ||
          draftWorkspace.attemptsById[attemptId]?.status !== 'queued'
        ) {
          return undefined
        }
        actionsToCommit.push(...runActions)
        launches.push({ binding, attemptId })
        runIdsByCase[item.caseId] = runId
      }

      const nextSequence = batchJobSequenceRef.current + 1
      const jobId = `batch-job-${nextSequence}`
      if (allocatedIds.has(jobId)) return undefined
      const submitAction: BatchSessionAction = {
        type: 'job/submit',
        jobId,
        runIdsByCase,
      }
      const nextBatch = batchSessionReducer(currentBatch, submitAction)
      if (nextBatch === currentBatch || nextBatch.job?.id !== jobId) {
        return undefined
      }

      stateRef.current = draftWorkspace
      for (const action of actionsToCommit) dispatch(action)
      batchJobSequenceRef.current = nextSequence
      batchStateRef.current = nextBatch
      batchDispatch(submitAction)
      for (const launch of launches) {
        scheduleLaunch(launch.binding, launch.attemptId)
      }
      return jobId
    } catch {
      return undefined
    }
  }, [activeRuntime, scheduleLaunch])

  const actions = useMemo<WorkspaceActions>(
    () => ({
      resetSession,
      updateRoi,
      submitRoi,
      approveRoi,
      requestRoiChanges,
      reopenRoi,
      supersedeRoi,
      restoreFullImageSourceBinding,
      startRun,
      cancelRun,
      retryRun,
      revokeResult,
      createReview,
      approveReview,
      requestReviewChanges,
      resubmitReview,
      revokeReview,
    }),
    [
      approveRoi,
      approveReview,
      cancelRun,
      createReview,
      reopenRoi,
      requestRoiChanges,
      requestReviewChanges,
      resetSession,
      restoreFullImageSourceBinding,
      resubmitReview,
      retryRun,
      revokeResult,
      revokeReview,
      startRun,
      submitRoi,
      supersedeRoi,
      updateRoi,
    ],
  )
  const batchActions = useMemo<BatchWorkspaceActions>(
    () => ({
      toggleCase: toggleBatchCase,
      selectAllCases: selectAllBatchCases,
      clearSelection: clearBatchSelection,
      updateConfig: updateBatchConfig,
      updateModel: updateBatchModel,
      setManifest: setBatchManifest,
      startBatch,
    }),
    [
      clearBatchSelection,
      selectAllBatchCases,
      setBatchManifest,
      startBatch,
      toggleBatchCase,
      updateBatchConfig,
      updateBatchModel,
    ],
  )
  const value = useMemo<WorkspaceContextValue>(
    () => ({
      state,
      actions,
      batchState,
      batchActions,
      gatewayMode: gateway.mode,
      persistence: 'memory_only',
    }),
    [actions, batchActions, batchState, gateway.mode, state],
  )

  return <WorkspaceContext.Provider value={value}>{children}</WorkspaceContext.Provider>
}

export function useWorkspace(): WorkspaceContextValue {
  const workspace = useContext(WorkspaceContext)
  if (workspace === null) {
    throw new Error('useWorkspace must be used within a WorkspaceProvider.')
  }
  return workspace
}
