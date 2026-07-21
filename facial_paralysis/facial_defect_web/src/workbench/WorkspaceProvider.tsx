import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useReducer,
  useRef,
  type ReactNode,
} from 'react'
import { getWorkbenchAsset } from './catalog'
import { createInferenceBinding } from './mockEngine'
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
  type WorkbenchFailure,
  type WorkspaceAction,
  type WorkspaceState,
} from './types'

export type WorkspaceRuntimeIdKind = 'run' | 'attempt' | 'token'

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

export type WorkspaceActions = {
  readonly resetSession: () => void
  readonly updateRoi: (caseId: string, geometry: NormalizedRoi) => void
  readonly submitRoi: (caseId: string) => void
  readonly approveRoi: (caseId: string) => void
  readonly requestRoiChanges: (caseId: string) => void
  readonly reopenRoi: (caseId: string) => void
  readonly supersedeRoi: (caseId: string) => void
  readonly startRun: (input: StartRunInput) => RunIdentifiers
  readonly cancelRun: (runId: string) => void
  readonly retryRun: (runId: string) => RunIdentifiers
  readonly revokeResult: (runId: string, attemptId: string) => void
}

export type WorkspaceContextValue = {
  readonly state: WorkspaceState
  readonly actions: WorkspaceActions
  readonly gatewayMode: WorkbenchGatewayMode
  readonly persistence: 'memory_only'
}

export type WorkspaceProviderProps = {
  readonly children: ReactNode
  readonly gateway: WorkbenchGateway
  readonly runtime?: WorkspaceRuntime
  readonly initialState?: WorkspaceState
}

function createCounterRuntime(): WorkspaceRuntime {
  let counter = 0
  return {
    nextId(kind) {
      counter += 1
      return `${kind}-${counter}`
    },
    reset() {
      counter = 0
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
  const stateRef = useRef(state)
  const controllersRef = useRef(new Map<string, AbortController>())
  const mountedRef = useRef(false)
  stateRef.current = state

  const dispatchTracked = useCallback((action: WorkspaceAction) => {
    stateRef.current = workspaceReducer(stateRef.current, action)
    dispatch(action)
  }, [])

  useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
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

  const resetSession = useCallback(() => {
    abortAllRequests()
    activeRuntime.reset?.()
    dispatchTracked({ type: 'session/reset' })
  }, [abortAllRequests, activeRuntime, dispatchTracked])

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
            dispatchTracked({
              type: 'run/succeed',
              ...asyncBinding,
              output,
            })
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

  const startRun = useCallback(
    (input: StartRunInput): RunIdentifiers => {
      const asset = getWorkbenchAsset(input.caseId)
      if (!asset) throw unknownCase(input.caseId)
      const roi = getCaseRoi(stateRef.current, asset.id)
      if (!roi) throw unknownCase(input.caseId)

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
      dispatchTracked({ type: 'run/start', ...asyncBinding })
      launchRequest(binding, attemptId)

      return { runId, attemptId }
    },
    [activeRuntime, dispatchTracked, launchRequest],
  )

  const cancelRun = useCallback(
    (runId: string) => {
      const run = stateRef.current.runsById[runId]
      const attemptId = run?.activeAttemptId
      const attempt = attemptId ? stateRef.current.attemptsById[attemptId] : undefined
      const binding = attempt?.binding
      if (!run || !attemptId || !attempt || !binding) return

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
      if (!asset || !roi) throw unknownCase(parentBinding.caseId)

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
      dispatchTracked({ type: 'run/start', ...asyncBinding })
      launchRequest(binding, attemptId)

      return { runId, attemptId }
    },
    [activeRuntime, dispatchTracked, launchRequest],
  )

  const revokeResult = useCallback(
    (runId: string, attemptId: string) => {
      dispatchTracked({ type: 'result/revoke', runId, attemptId })
    },
    [dispatchTracked],
  )

  const actions = useMemo<WorkspaceActions>(
    () => ({
      resetSession,
      updateRoi,
      submitRoi,
      approveRoi,
      requestRoiChanges,
      reopenRoi,
      supersedeRoi,
      startRun,
      cancelRun,
      retryRun,
      revokeResult,
    }),
    [
      approveRoi,
      cancelRun,
      reopenRoi,
      requestRoiChanges,
      resetSession,
      retryRun,
      revokeResult,
      startRun,
      submitRoi,
      supersedeRoi,
      updateRoi,
    ],
  )
  const value = useMemo<WorkspaceContextValue>(
    () => ({ state, actions, gatewayMode: gateway.mode, persistence: 'memory_only' }),
    [actions, gateway.mode, state],
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
