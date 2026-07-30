import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react'
import {
  approvedAssets,
  type ApprovedAsset,
} from '../data/approvedAssetManifest'
import type { WorkbenchAssetId } from '../data/workbenchAssetDefinitions'
import { SessionMediaVault } from './SessionMediaVault'
import {
  CaptureFileValidationError,
  validateCaptureFile,
  type CaptureFileValidationResult,
} from './captureFile'
import { createDemoPatientResult } from './demoPatientInference'
import {
  createInitialPatientWorkflowState,
  patientWorkflowReducer,
} from './reducer'
import {
  getOwnRecordValue,
  isCaptureQualityComplete,
  samePatientRunBinding,
  selectCurrentAuthorization,
  selectCurrentCapture,
  selectCurrentResult,
  selectCurrentReview,
  selectCurrentRun,
} from './selectors'
import type {
  AuthorizationSnapshot,
  CaptureAsset,
  CaptureQualityChecks,
  CaptureSource,
  PatientFaceFeature,
  PatientFaceRegistration,
  PatientId,
  PatientRecord,
  PatientResult,
  PatientReviewDecision,
  PatientRun,
  PatientRunBinding,
  PatientRunId,
  PatientSimulationOutput,
  PatientVisit,
  PatientVisitId,
  PatientWorkflowAction,
  PatientWorkflowFailure,
  PatientWorkflowFailureCode,
  PatientWorkflowState,
} from './types'
import {
  createAuthorizationSnapshotId,
  createCaptureAssetId,
  createPatientId,
  createPatientResultId,
  createPatientReviewId,
  createPatientRunId,
  createPatientVisitId,
  createSessionMediaHandle,
  isPatientId,
  isPatientVisitId,
  validatePatientDraft,
  validateVisitDraft,
  type PatientDraft,
  type PatientVisitDraft,
} from './validation'

export type PatientWorkflowRuntimeIdKind =
  | 'patient'
  | 'visit'
  | 'authorization'
  | 'capture'
  | 'media'
  | 'run'
  | 'result'
  | 'review'

export type PatientWorkflowRuntime = {
  readonly nextIdToken: (kind: PatientWorkflowRuntimeIdKind) => string
  readonly now: () => string
  readonly today: () => string
  readonly reset?: () => void
}

export type PatientCapturePreparer = (
  media: Blob,
) => Promise<CaptureFileValidationResult>

export type SyntheticPatientMediaLoader = (
  asset: ApprovedAsset,
) => Promise<Blob>

export type PatientSimulationRunner = (
  binding: Readonly<PatientRunBinding>,
  faceRegistration: Readonly<PatientFaceRegistration>,
) => PatientSimulationOutput | Promise<PatientSimulationOutput>

export type PatientFaceRegistrationInput = Readonly<{
  readonly media: Blob
  readonly captureSha256: string
  readonly sourceWidth: number
  readonly sourceHeight: number
  readonly captureProtocol: CaptureAsset['captureProtocol']
}>

export type PatientFaceRegistrationRunner = (
  input: PatientFaceRegistrationInput,
) => PatientFaceRegistration | Promise<PatientFaceRegistration>

const runDefaultFaceRegistration: PatientFaceRegistrationRunner =
  async (input) => {
    const { detectPatientFaceRegistration } = await import(
      './onDeviceFaceRegistration'
    )
    return detectPatientFaceRegistration(input)
  }

export type CreatePatientInput = PatientDraft & {
  readonly initialVisit: PatientVisitDraft
  readonly recordKind?: PatientRecord['recordKind']
}

export type CreatePatientIdentifiers = {
  readonly patientId: PatientId
  readonly visitId: PatientVisitId
}

export type PatientWorkflowActions = {
  readonly createPatient: (
    input: CreatePatientInput,
  ) => CreatePatientIdentifiers
  readonly createVisit: (
    patientId: string,
    input: PatientVisitDraft,
  ) => PatientVisitId
  readonly attachSessionCapture: (
    visitId: string,
    media: File | Blob,
    source?: Exclude<CaptureSource, 'synthetic_demo'>,
  ) => Promise<ReturnType<typeof createCaptureAssetId>>
  readonly attachSyntheticCapture: (
    visitId: string,
    assetId: string,
  ) => Promise<ReturnType<typeof createCaptureAssetId>>
  readonly setQualityCheck: (
    visitId: string,
    check: keyof CaptureQualityChecks,
    passed: boolean,
  ) => void
  readonly submitAnalysis: (visitId: string) => PatientRunId
  readonly retryAnalysis: (visitId: string) => PatientRunId
  readonly completeReview: (
    visitId: string,
    decision: PatientReviewDecision,
    note?: string,
  ) => void
  readonly requestRetake: (visitId: string) => void
  readonly resetSession: () => void
  readonly getCapturePreviewUrl: (
    visitId: string,
  ) => string | undefined
}

export type PatientWorkflowContextValue = {
  readonly state: PatientWorkflowState
  readonly actions: PatientWorkflowActions
  readonly patientListQuery: string
  readonly setPatientListQuery: (query: string) => void
  readonly mode: 'simulation_only'
  readonly persistence: 'memory_only'
}

export type PatientWorkflowProviderProps = {
  readonly children: ReactNode
  readonly runtime?: PatientWorkflowRuntime
  readonly mediaVault?: SessionMediaVault
  readonly prepareCapture?: PatientCapturePreparer
  readonly loadSyntheticMedia?: SyntheticPatientMediaLoader
  readonly simulationRunner?: PatientSimulationRunner
  readonly faceRegistrationRunner?: PatientFaceRegistrationRunner
  readonly queueDelayMs?: number
  readonly analysisDelayMs?: number
  readonly initialState?: PatientWorkflowState
}

export class PatientWorkflowProviderError extends Error {
  readonly name = 'PatientWorkflowProviderError'

  constructor(readonly failure: PatientWorkflowFailure) {
    super(failure.message)
  }
}

const PatientWorkflowContext =
  createContext<PatientWorkflowContextValue | null>(null)

const QUALITY_CHECK_KEYS = new Set<keyof CaptureQualityChecks>([
  'faceVisibleAndCentered',
  'focusLightingAndOcclusionAcceptable',
  'orientationConfirmed',
  'authorizationDocumented',
])

const EMPTY_QUALITY: CaptureQualityChecks = {
  faceVisibleAndCentered: false,
  focusLightingAndOcclusionAcceptable: false,
  orientationConfirmed: false,
  authorizationDocumented: false,
}

function providerFailure(
  code: PatientWorkflowFailureCode,
  message: string,
  field?: string,
): PatientWorkflowFailure {
  return Object.freeze({
    code,
    message,
    ...(field ? { field } : {}),
  })
}

function throwProviderFailure(
  code: PatientWorkflowFailureCode,
  message: string,
  field?: string,
): never {
  throw new PatientWorkflowProviderError(
    providerFailure(code, message, field),
  )
}

function createDefaultRuntime(): PatientWorkflowRuntime {
  const counters = new Map<PatientWorkflowRuntimeIdKind, number>()
  return {
    nextIdToken(kind) {
      const next = (counters.get(kind) ?? 0) + 1
      counters.set(kind, next)
      return `${kind}_${String(next).padStart(8, '0')}`
    },
    now: () => formatLocalTimestamp(new Date()),
    today: () => formatLocalCalendarDate(new Date()),
    reset: () => counters.clear(),
  }
}

function padCalendarPart(value: number, length = 2): string {
  return String(value).padStart(length, '0')
}

function formatLocalCalendarDate(date: Date): string {
  return [
    padCalendarPart(date.getFullYear(), 4),
    padCalendarPart(date.getMonth() + 1),
    padCalendarPart(date.getDate()),
  ].join('-')
}

function formatLocalTimestamp(date: Date): string {
  const offsetMinutes = date.getTimezoneOffset()
  const offsetSign = offsetMinutes <= 0 ? '+' : '-'
  const absoluteOffset = Math.abs(offsetMinutes)
  const offsetHours = Math.floor(absoluteOffset / 60)
  const offsetRemainder = absoluteOffset % 60

  return `${formatLocalCalendarDate(date)}T${padCalendarPart(
    date.getHours(),
  )}:${padCalendarPart(date.getMinutes())}:${padCalendarPart(
    date.getSeconds(),
  )}.${padCalendarPart(date.getMilliseconds(), 3)}${offsetSign}${padCalendarPart(
    offsetHours,
  )}:${padCalendarPart(offsetRemainder)}`
}

async function loadSyntheticMediaWithFetch(
  asset: ApprovedAsset,
): Promise<Blob> {
  const response = await fetch(asset.url)
  if (!response.ok) {
    throw new Error('The approved synthetic image could not be loaded.')
  }
  return response.blob()
}

function clonePlain<T>(value: T): T {
  if (Array.isArray(value)) {
    return value.map((entry) => clonePlain(entry)) as T
  }
  if (value !== null && typeof value === 'object') {
    return Object.fromEntries(
      Object.entries(value).map(([key, entry]) => [
        key,
        clonePlain(entry),
      ]),
    ) as T
  }
  return value
}

function boundedDelay(value: number | undefined): number {
  return typeof value === 'number' &&
    Number.isFinite(value) &&
    value >= 0
    ? value
    : 0
}

function reducerResult(
  state: PatientWorkflowState,
  action: PatientWorkflowAction,
): PatientWorkflowState {
  return patientWorkflowReducer(state, action)
}

function reduceOrThrow(
  state: PatientWorkflowState,
  action: PatientWorkflowAction,
): PatientWorkflowState {
  const next = reducerResult(state, action)
  if (next.lastFailure) {
    throw new PatientWorkflowProviderError(next.lastFailure)
  }
  return next
}

function entityId(
  runtime: PatientWorkflowRuntime,
  kind: Exclude<PatientWorkflowRuntimeIdKind, 'media'>,
): string {
  const token = runtime.nextIdToken(kind)
  const value = `${kind}-${token}`
  switch (kind) {
    case 'patient':
      return createPatientId(value)
    case 'visit':
      return createPatientVisitId(value)
    case 'authorization':
      return createAuthorizationSnapshotId(value)
    case 'capture':
      return createCaptureAssetId(value)
    case 'run':
      return createPatientRunId(value)
    case 'result':
      return createPatientResultId(value)
    case 'review':
      return createPatientReviewId(value)
  }
}

function mediaHandle(runtime: PatientWorkflowRuntime) {
  return createSessionMediaHandle(runtime.nextIdToken('media'))
}

function hasSyntheticCaptureOnAnotherVisit(
  state: PatientWorkflowState,
  patientId: PatientId,
  visitId: PatientVisitId,
): boolean {
  return state.captureOrder.some((captureId) => {
    const capture = getOwnRecordValue(state.capturesById, captureId)
    return (
      capture?.patientId === patientId &&
      capture.visitId !== visitId &&
      capture.source === 'synthetic_demo'
    )
  })
}

function copyBinding(
  binding: PatientRunBinding,
): PatientRunBinding {
  return Object.freeze({
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

function exactBindingForCurrentVisit(
  state: PatientWorkflowState,
  visitId: string,
): PatientRunBinding {
  if (!isPatientVisitId(visitId)) {
    return throwProviderFailure(
      'UNKNOWN_VISIT',
      `Unknown visit: ${visitId}.`,
      'visitId',
    )
  }
  const visit = getOwnRecordValue(state.visitsById, visitId)
  if (!visit) {
    return throwProviderFailure(
      'UNKNOWN_VISIT',
      `Unknown visit: ${visitId}.`,
      'visitId',
    )
  }
  const capture = selectCurrentCapture(state, visit.id)
  if (!capture) {
    return throwProviderFailure(
      'UNKNOWN_CAPTURE',
      'A current capture is required before analysis.',
      'captureId',
    )
  }
  if (
    !isCaptureQualityComplete(capture.qualityChecks) ||
    !capture.qualityConfirmedAt
  ) {
    return throwProviderFailure(
      'CAPTURE_QUALITY_INCOMPLETE',
      'All four quality checks must be confirmed before analysis.',
      'qualityChecks',
    )
  }
  const authorization = selectCurrentAuthorization(state, visit.id)
  if (
    authorization?.patientId !== visit.patientId ||
    authorization.status !== 'documented'
  ) {
    return throwProviderFailure(
      'INVALID_AUTHORIZATION',
      'Current documented authorization is required before analysis.',
      'authorization',
    )
  }
  return copyBinding({
    patientId: visit.patientId,
    visitId: visit.id,
    captureId: capture.id,
    captureVersion: capture.version,
    captureSha256: capture.sha256,
    mediaHandle: capture.mediaHandle,
    authorizationRevision: authorization.revision,
    captureProtocol: capture.captureProtocol,
  })
}

type LaunchValidation =
  | {
      readonly ok: true
      readonly capture: CaptureAsset
      readonly blob: Blob
    }
  | {
      readonly ok: false
      readonly field: string
    }

function validateLaunchState(
  state: PatientWorkflowState,
  binding: PatientRunBinding,
  vault: SessionMediaVault,
): LaunchValidation {
  const patient = getOwnRecordValue(
    state.patientsById,
    binding.patientId,
  )
  const visit = getOwnRecordValue(state.visitsById, binding.visitId)
  const capture = selectCurrentCapture(state, binding.visitId)
  const authorization = selectCurrentAuthorization(
    state,
    binding.visitId,
  )

  if (!patient || visit?.patientId !== binding.patientId) {
    return { ok: false, field: 'binding' }
  }
  if (
    !capture ||
    capture.status !== 'current' ||
    capture.patientId !== binding.patientId ||
    capture.id !== binding.captureId ||
    capture.version !== binding.captureVersion ||
    capture.sha256 !== binding.captureSha256 ||
    capture.mediaHandle !== binding.mediaHandle ||
    capture.captureProtocol !== binding.captureProtocol
  ) {
    return { ok: false, field: 'binding' }
  }
  if (
    !isCaptureQualityComplete(capture.qualityChecks) ||
    !capture.qualityConfirmedAt
  ) {
    return { ok: false, field: 'binding' }
  }
  if (
    authorization?.patientId !== binding.patientId ||
    authorization.status !== 'documented' ||
    authorization.revision !== binding.authorizationRevision
  ) {
    return { ok: false, field: 'binding' }
  }
  const entry = vault.get(binding.mediaHandle)
  if (!entry) return { ok: false, field: 'mediaHandle' }
  return { ok: true, capture, blob: entry.blob }
}

function preparedMetadataMatchesCapture(
  result: CaptureFileValidationResult,
  capture: CaptureAsset,
): boolean {
  return (
    result.ok &&
    result.value.metadata.sha256 === capture.sha256 &&
    result.value.metadata.mimeType === capture.mimeType &&
    result.value.metadata.sizeBytes === capture.sizeBytes &&
    result.value.metadata.width === capture.width &&
    result.value.metadata.height === capture.height
  )
}

function normalizeSimulationOutput(
  output: unknown,
): PatientSimulationOutput | undefined {
  if (
    output === null ||
    typeof output !== 'object' ||
    !Array.isArray((output as PatientSimulationOutput).points) ||
    (output as PatientSimulationOutput).origin !==
      'workflow_simulation'
  ) {
    return undefined
  }
  if (
    Object.keys(output).sort().join(',') !== 'origin,points'
  ) {
    return undefined
  }

  const points = (output as PatientSimulationOutput).points
  const copied = []
  for (const point of points) {
    if (
      point === null ||
      typeof point !== 'object' ||
      Object.keys(point).sort().join(',') !==
        'intensity,radius,x,y' ||
      !Number.isFinite(point.x) ||
      point.x < 0 ||
      point.x > 1 ||
      !Number.isFinite(point.y) ||
      point.y < 0 ||
      point.y > 1 ||
      !Number.isFinite(point.intensity) ||
      point.intensity < 0 ||
      point.intensity > 1 ||
      !Number.isFinite(point.radius) ||
      point.radius <= 0 ||
      point.radius > 1
    ) {
      return undefined
    }
    copied.push(
      Object.freeze({
        x: point.x,
        y: point.y,
        intensity: point.intensity,
        radius: point.radius,
      }),
    )
  }
  return Object.freeze({
    origin: 'workflow_simulation',
    points: Object.freeze(copied),
  })
}

const REQUIRED_FACE_FEATURES = new Set<PatientFaceFeature>([
  'face_oval',
  'left_eye',
  'right_eye',
  'left_eyebrow',
  'right_eyebrow',
  'lips',
])

function normalizeFaceRegistration(
  candidate: unknown,
  input: PatientFaceRegistrationInput,
): PatientFaceRegistration | undefined {
  if (candidate === null || typeof candidate !== 'object') {
    return undefined
  }
  const registration = candidate as PatientFaceRegistration
  if (
    Object.keys(registration).sort().join(',') !==
      [
        'captureProtocol',
        'captureSha256',
        'coordinateSpace',
        'detectorId',
        'detectorVersion',
        'faceCount',
        'paths',
        'schemaVersion',
        'source',
        'sourceHeight',
        'sourceWidth',
      ]
        .sort()
        .join(',') ||
    registration.schemaVersion !== 'patient-face-registration/1' ||
    registration.source !== 'on_device_face_landmarks' ||
    registration.coordinateSpace !==
      'decoded_image_normalized_v1' ||
    registration.captureSha256 !== input.captureSha256 ||
    registration.sourceWidth !== input.sourceWidth ||
    registration.sourceHeight !== input.sourceHeight ||
    registration.captureProtocol !== input.captureProtocol ||
    registration.detectorId !== 'mediapipe_face_landmarker' ||
    registration.detectorVersion !==
      'tasks-vision-1.0.0-model-float16-1' ||
    registration.faceCount !== 1 ||
    !Array.isArray(registration.paths)
  ) {
    return undefined
  }

  const features = new Set<PatientFaceFeature>()
  const paths = []
  for (const path of registration.paths) {
    if (
      path === null ||
      typeof path !== 'object' ||
      Object.keys(path).sort().join(',') !==
        'closed,feature,points' ||
      !REQUIRED_FACE_FEATURES.has(path.feature) ||
      typeof path.closed !== 'boolean' ||
      !Array.isArray(path.points) ||
      path.points.length < 2 ||
      path.points.length > 128
    ) {
      return undefined
    }
    const points = []
    for (const point of path.points) {
      if (
        point === null ||
        typeof point !== 'object' ||
        Object.keys(point).sort().join(',') !== 'x,y' ||
        !Number.isFinite(point.x) ||
        point.x < 0 ||
        point.x > 1 ||
        !Number.isFinite(point.y) ||
        point.y < 0 ||
        point.y > 1
      ) {
        return undefined
      }
      points.push(Object.freeze({ x: point.x, y: point.y }))
    }
    features.add(path.feature)
    paths.push(
      Object.freeze({
        feature: path.feature,
        closed: path.closed,
        points: Object.freeze(points),
      }),
    )
  }
  if (
    registration.paths.length < REQUIRED_FACE_FEATURES.size ||
    registration.paths.length > 16 ||
    features.size !== REQUIRED_FACE_FEATURES.size ||
    [...REQUIRED_FACE_FEATURES].some(
      (feature) => !features.has(feature),
    )
  ) {
    return undefined
  }
  const ovalPoints = paths
    .filter((path) => path.feature === 'face_oval')
    .flatMap((path) => path.points)
  const ovalWidth =
    Math.max(...ovalPoints.map((point) => point.x)) -
    Math.min(...ovalPoints.map((point) => point.x))
  const ovalHeight =
    Math.max(...ovalPoints.map((point) => point.y)) -
    Math.min(...ovalPoints.map((point) => point.y))
  if (ovalWidth < 0.08 || ovalHeight < 0.08) {
    return undefined
  }

  return Object.freeze({
    schemaVersion: 'patient-face-registration/1',
    source: 'on_device_face_landmarks',
    coordinateSpace: 'decoded_image_normalized_v1',
    captureSha256: registration.captureSha256,
    sourceWidth: registration.sourceWidth,
    sourceHeight: registration.sourceHeight,
    captureProtocol: registration.captureProtocol,
    detectorId: 'mediapipe_face_landmarker',
    detectorVersion: 'tasks-vision-1.0.0-model-float16-1',
    faceCount: 1,
    paths: Object.freeze(paths),
  })
}

export function PatientWorkflowProvider({
  children,
  runtime,
  mediaVault,
  prepareCapture = validateCaptureFile,
  loadSyntheticMedia = loadSyntheticMediaWithFetch,
  simulationRunner = createDemoPatientResult,
  faceRegistrationRunner = runDefaultFaceRegistration,
  queueDelayMs,
  analysisDelayMs,
  initialState,
}: PatientWorkflowProviderProps) {
  const defaultRuntimeRef = useRef<PatientWorkflowRuntime | null>(null)
  if (defaultRuntimeRef.current === null) {
    defaultRuntimeRef.current = createDefaultRuntime()
  }
  const activeRuntime = runtime ?? defaultRuntimeRef.current

  const defaultVaultRef = useRef<SessionMediaVault | null>(null)
  if (defaultVaultRef.current === null) {
    defaultVaultRef.current = new SessionMediaVault()
  }
  const vault = mediaVault ?? defaultVaultRef.current

  const initialStateRef = useRef<PatientWorkflowState | null>(null)
  if (initialStateRef.current === null) {
    initialStateRef.current = initialState
      ? clonePlain(initialState)
      : createInitialPatientWorkflowState()
  }
  const [state, setState] = useState<PatientWorkflowState>(() =>
    clonePlain(initialStateRef.current!),
  )
  const [patientListQuery, setPatientListQueryState] = useState('')
  const stateRef = useRef(state)
  const mountedRef = useRef(false)
  const lifecycleRef = useRef(0)
  const queueTimersRef = useRef(
    new Map<PatientRunId, ReturnType<typeof setTimeout>>(),
  )
  const analysisTimersRef = useRef(
    new Map<PatientRunId, ReturnType<typeof setTimeout>>(),
  )

  const commitState = useCallback((next: PatientWorkflowState) => {
    stateRef.current = next
    setState(next)
  }, [])
  const setPatientListQuery = useCallback((query: string) => {
    setPatientListQueryState(query.slice(0, 128))
  }, [])

  const clearTimers = useCallback(() => {
    for (const timer of queueTimersRef.current.values()) {
      clearTimeout(timer)
    }
    for (const timer of analysisTimersRef.current.values()) {
      clearTimeout(timer)
    }
    queueTimersRef.current.clear()
    analysisTimersRef.current.clear()
  }, [])

  useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
      lifecycleRef.current += 1
      clearTimers()
      vault.clear()
    }
  }, [clearTimers, vault])

  const failRun = useCallback(
    (runId: PatientRunId, field: string) => {
      const run = getOwnRecordValue(stateRef.current.runsById, runId)
      if (!run || !['queued', 'running'].includes(run.status)) return
      const next = reducerResult(stateRef.current, {
        type: 'run/status/set',
        runId,
        status: 'failed',
        failure: providerFailure(
          'ANALYSIS_FAILED',
          'The simulated analysis could not be completed safely.',
          field,
        ),
      })
      if (next.runsById[runId]?.status === 'failed') {
        commitState(next)
      }
    },
    [commitState],
  )

  const completeRun = useCallback(
    (
      runId: PatientRunId,
      binding: PatientRunBinding,
      output: PatientSimulationOutput,
      faceRegistration: PatientFaceRegistration,
      lifecycle: number,
    ) => {
      if (
        !mountedRef.current ||
        lifecycleRef.current !== lifecycle
      ) {
        return
      }
      const run = getOwnRecordValue(stateRef.current.runsById, runId)
      const validation = validateLaunchState(
        stateRef.current,
        binding,
        vault,
      )
      if (
        !run ||
        run.status !== 'running' ||
        !samePatientRunBinding(run.binding, binding) ||
        !validation.ok
      ) {
        failRun(runId, validation.ok ? 'binding' : validation.field)
        return
      }

      try {
        const resultId = entityId(
          activeRuntime,
          'result',
        ) as PatientResult['id']
        let next = reduceOrThrow(stateRef.current, {
          type: 'run/status/set',
          runId,
          status: 'succeeded',
        })
        next = reduceOrThrow(next, {
          type: 'result/record',
          result: {
            id: resultId,
            runId,
            binding: copyBinding(binding),
            freshness: 'current',
            createdAt: activeRuntime.now(),
            faceRegistration,
            output,
          },
        })
        commitState(next)
      } catch {
        failRun(runId, 'output')
      }
    },
    [activeRuntime, commitState, failRun, vault],
  )

  const launchRun = useCallback(
    async (runId: PatientRunId, lifecycle: number) => {
      if (
        !mountedRef.current ||
        lifecycleRef.current !== lifecycle
      ) {
        return
      }
      const queued = getOwnRecordValue(stateRef.current.runsById, runId)
      if (!queued || queued.status !== 'queued') return
      const binding = copyBinding(queued.binding)
      const firstValidation = validateLaunchState(
        stateRef.current,
        binding,
        vault,
      )
      if (!firstValidation.ok) {
        failRun(runId, firstValidation.field)
        return
      }

      let prepared: CaptureFileValidationResult
      try {
        prepared = await prepareCapture(firstValidation.blob)
      } catch {
        failRun(runId, 'captureSha256')
        return
      }
      if (
        !mountedRef.current ||
        lifecycleRef.current !== lifecycle
      ) {
        return
      }

      const secondValidation = validateLaunchState(
        stateRef.current,
        binding,
        vault,
      )
      if (!secondValidation.ok) {
        failRun(runId, secondValidation.field)
        return
      }
      if (
        secondValidation.blob !== firstValidation.blob ||
        !preparedMetadataMatchesCapture(
          prepared,
          secondValidation.capture,
        )
      ) {
        failRun(runId, 'captureSha256')
        return
      }

      const started = reducerResult(stateRef.current, {
        type: 'run/status/set',
        runId,
        status: 'running',
      })
      if (started.runsById[runId]?.status !== 'running') {
        commitState(started)
        failRun(runId, 'binding')
        return
      }
      commitState(started)

      let candidate: unknown
      const registrationInput: PatientFaceRegistrationInput = {
        media: secondValidation.blob,
        captureSha256: secondValidation.capture.sha256,
        sourceWidth: secondValidation.capture.width,
        sourceHeight: secondValidation.capture.height,
        captureProtocol: secondValidation.capture.captureProtocol,
      }
      let registrationCandidate: unknown
      try {
        registrationCandidate =
          await faceRegistrationRunner(registrationInput)
      } catch {
        if (
          mountedRef.current &&
          lifecycleRef.current === lifecycle
        ) {
          failRun(runId, 'faceRegistration')
        }
        return
      }
      const faceRegistration = normalizeFaceRegistration(
        registrationCandidate,
        registrationInput,
      )
      if (!faceRegistration) {
        failRun(runId, 'faceRegistration.binding')
        return
      }
      try {
        candidate = await simulationRunner(binding, faceRegistration)
      } catch {
        if (
          mountedRef.current &&
          lifecycleRef.current === lifecycle
        ) {
          failRun(runId, 'simulation')
        }
        return
      }
      if (
        !mountedRef.current ||
        lifecycleRef.current !== lifecycle
      ) {
        return
      }
      const output = normalizeSimulationOutput(candidate)
      if (!output) {
        failRun(runId, 'output')
        return
      }

      const delay = boundedDelay(analysisDelayMs)
      if (delay === 0) {
        completeRun(
          runId,
          binding,
          output,
          faceRegistration,
          lifecycle,
        )
        return
      }
      const timer = setTimeout(() => {
        analysisTimersRef.current.delete(runId)
        completeRun(
          runId,
          binding,
          output,
          faceRegistration,
          lifecycle,
        )
      }, delay)
      analysisTimersRef.current.set(runId, timer)
    },
    [
      analysisDelayMs,
      commitState,
      completeRun,
      failRun,
      prepareCapture,
      faceRegistrationRunner,
      simulationRunner,
      vault,
    ],
  )

  const scheduleRun = useCallback(
    (runId: PatientRunId) => {
      const lifecycle = lifecycleRef.current
      const timer = setTimeout(() => {
        queueTimersRef.current.delete(runId)
        void launchRun(runId, lifecycle)
      }, boundedDelay(queueDelayMs))
      queueTimersRef.current.set(runId, timer)
    },
    [launchRun, queueDelayMs],
  )

  const createPatient = useCallback(
    (input: CreatePatientInput): CreatePatientIdentifiers => {
      const patientValidation = validatePatientDraft(
        input,
        stateRef.current,
        activeRuntime.today(),
      )
      if (!patientValidation.ok) {
        const [field, message] = Object.entries(
          patientValidation.errors,
        )[0] ?? ['patient', 'Patient record is invalid.']
        throwProviderFailure(
          field === 'recordNumber'
            ? 'DUPLICATE_RECORD_NUMBER'
            : 'INVALID_PATIENT',
          message,
          field,
        )
      }
      const visitValidation = validateVisitDraft(
        input.initialVisit,
        activeRuntime.today(),
      )
      if (!visitValidation.ok) {
        const [field, message] = Object.entries(
          visitValidation.errors,
        )[0] ?? ['visit', 'Visit is invalid.']
        throwProviderFailure('INVALID_VISIT', message, field)
      }

      try {
        const patientId = entityId(
          activeRuntime,
          'patient',
        ) as PatientId
        const visitId = entityId(
          activeRuntime,
          'visit',
        ) as PatientVisitId
        const createdAt = activeRuntime.now()
        const patient: PatientRecord = {
          id: patientId,
          ...patientValidation.value,
          recordKind: input.recordKind ?? 'session_test',
          createdAt,
        }
        const visit: PatientVisit = {
          id: visitId,
          patientId,
          ...visitValidation.value,
          createdAt,
        }
        let next = reduceOrThrow(stateRef.current, {
          type: 'patient/create',
          patient,
          trustedToday: activeRuntime.today(),
          syntheticTestAttestation: input.syntheticTestAttestation,
        })
        next = reduceOrThrow(next, {
          type: 'visit/create',
          visit,
          trustedToday: activeRuntime.today(),
        })
        commitState(next)
        return { patientId, visitId }
      } catch (error) {
        if (error instanceof PatientWorkflowProviderError) throw error
        return throwProviderFailure(
          'INVALID_PATIENT',
          'The patient and initial visit could not be created.',
        )
      }
    },
    [activeRuntime, commitState],
  )

  const createVisit = useCallback(
    (patientId: string, input: PatientVisitDraft): PatientVisitId => {
      if (
        !isPatientId(patientId) ||
        !getOwnRecordValue(stateRef.current.patientsById, patientId)
      ) {
        return throwProviderFailure(
          'UNKNOWN_PATIENT',
          `Unknown patient: ${patientId}.`,
          'patientId',
        )
      }
      const validation = validateVisitDraft(
        input,
        activeRuntime.today(),
      )
      if (!validation.ok) {
        const [field, message] = Object.entries(validation.errors)[0] ?? [
          'visit',
          'Visit is invalid.',
        ]
        return throwProviderFailure('INVALID_VISIT', message, field)
      }
      try {
        const visitId = entityId(
          activeRuntime,
          'visit',
        ) as PatientVisitId
        const next = reduceOrThrow(stateRef.current, {
          type: 'visit/create',
          visit: {
            id: visitId,
            patientId,
            ...validation.value,
            createdAt: activeRuntime.now(),
          },
          trustedToday: activeRuntime.today(),
        })
        commitState(next)
        return visitId
      } catch (error) {
        if (error instanceof PatientWorkflowProviderError) throw error
        return throwProviderFailure(
          'INVALID_VISIT',
          'The visit could not be created.',
        )
      }
    },
    [activeRuntime, commitState],
  )

  const attachPreparedCapture = useCallback(
    async (
      visitId: string,
      media: Blob,
      source: CaptureSource,
      syntheticSourceAssetId?: WorkbenchAssetId,
      expectedSha256?: string,
    ) => {
      if (
        !isPatientVisitId(visitId) ||
        !getOwnRecordValue(stateRef.current.visitsById, visitId)
      ) {
        return throwProviderFailure(
          'UNKNOWN_VISIT',
          `Unknown visit: ${visitId}.`,
          'visitId',
        )
      }
      const lifecycle = lifecycleRef.current
      let prepared: CaptureFileValidationResult
      try {
        prepared = await prepareCapture(media)
      } catch {
        throw new CaptureFileValidationError(
          'PROCESSING_FAILED',
          'The image could not be prepared. Choose the file again or select another image.',
        )
      }
      if (!prepared.ok) throw prepared.error
      if (
        expectedSha256 !== undefined &&
        prepared.value.metadata.sha256 !== expectedSha256
      ) {
        return throwProviderFailure(
          'INVALID_CAPTURE',
          'The loaded synthetic image does not match the approved asset.',
          'sha256',
        )
      }
      if (
        !mountedRef.current ||
        lifecycleRef.current !== lifecycle
      ) {
        return throwProviderFailure(
          'INVALID_CAPTURE',
          'The capture session changed before the image was ready.',
        )
      }

      const currentState = stateRef.current
      const visit = getOwnRecordValue(currentState.visitsById, visitId)
      if (!visit) {
        return throwProviderFailure(
          'UNKNOWN_VISIT',
          `Unknown visit: ${visitId}.`,
          'visitId',
        )
      }
      if (
        source === 'synthetic_demo' &&
        hasSyntheticCaptureOnAnotherVisit(
          currentState,
          visit.patientId,
          visit.id,
        )
      ) {
        return throwProviderFailure(
          'INVALID_CAPTURE',
          'A standalone catalog demo is already used by another visit in this record.',
          'assetId',
        )
      }
      const previous = selectCurrentCapture(currentState, visit.id)
      if (
        source === 'synthetic_demo' &&
        previous?.source === 'synthetic_demo' &&
        previous.syntheticSourceAssetId === syntheticSourceAssetId &&
        previous.sha256 === prepared.value.metadata.sha256
      ) {
        return throwProviderFailure(
          'INVALID_CAPTURE',
          'This visit already uses the current catalog demo.',
          'assetId',
        )
      }
      try {
        const captureId = entityId(
          activeRuntime,
          'capture',
        ) as CaptureAsset['id']
        const handle = mediaHandle(activeRuntime)
        const handleAlreadyOwned = Object.values(
          currentState.capturesById,
        ).some((capture) => capture?.mediaHandle === handle)
        if (handleAlreadyOwned) {
          return throwProviderFailure(
            'INVALID_CAPTURE',
            'The session media handle is already in use.',
            'mediaHandle',
          )
        }
        const capture: CaptureAsset = {
          id: captureId,
          patientId: visit.patientId,
          visitId: visit.id,
          version: (previous?.version ?? 0) + 1,
          status: 'current',
          source,
          mediaHandle: handle,
          ...prepared.value.metadata,
          captureProtocol: 'frontal_relaxed_non_mirrored_v1',
          qualityChecks: EMPTY_QUALITY,
          capturedAt: activeRuntime.now(),
          ...(syntheticSourceAssetId
            ? { syntheticSourceAssetId }
            : {}),
        }

        vault.set(handle, prepared.value.vaultMedia)
        const next = reducerResult(currentState, {
          type: 'capture/add',
          capture,
        })
        if (next.lastFailure) {
          vault.delete(handle)
          throw new PatientWorkflowProviderError(next.lastFailure)
        }
        commitState(next)
        if (
          previous &&
          previous.mediaHandle !== capture.mediaHandle
        ) {
          vault.delete(previous.mediaHandle)
        }
        return captureId
      } catch (error) {
        if (
          error instanceof PatientWorkflowProviderError ||
          error instanceof CaptureFileValidationError
        ) {
          throw error
        }
        return throwProviderFailure(
          'INVALID_CAPTURE',
          'The prepared capture could not be stored safely.',
        )
      }
    },
    [activeRuntime, commitState, prepareCapture, vault],
  )

  const attachSessionCapture = useCallback(
    (
      visitId: string,
      media: File | Blob,
      source: Exclude<CaptureSource, 'synthetic_demo'> = 'upload',
    ) => {
      if (source !== 'camera' && source !== 'upload') {
        return Promise.reject(
          new PatientWorkflowProviderError(
            providerFailure(
              'INVALID_CAPTURE',
              'Capture source must be camera or upload.',
              'source',
            ),
          ),
        )
      }
      return attachPreparedCapture(visitId, media, source)
    },
    [attachPreparedCapture],
  )

  const attachSyntheticCapture = useCallback(
    async (visitId: string, assetId: string) => {
      const lifecycle = lifecycleRef.current
      if (!isPatientVisitId(visitId)) {
        return throwProviderFailure(
          'UNKNOWN_VISIT',
          `Unknown visit: ${visitId}.`,
          'visitId',
        )
      }
      const visit = getOwnRecordValue(
        stateRef.current.visitsById,
        visitId,
      )
      if (!visit) {
        return throwProviderFailure(
          'UNKNOWN_VISIT',
          `Unknown visit: ${visitId}.`,
          'visitId',
        )
      }
      if (
        hasSyntheticCaptureOnAnotherVisit(
          stateRef.current,
          visit.patientId,
          visit.id,
        )
      ) {
        return throwProviderFailure(
          'INVALID_CAPTURE',
          'A standalone catalog demo is already used by another visit in this record.',
          'assetId',
        )
      }
      const asset = approvedAssets.find(
        (candidate) => candidate.id === assetId,
      )
      if (!asset) {
        return throwProviderFailure(
          'INVALID_CAPTURE',
          'Choose an approved standalone synthetic demo image.',
          'assetId',
        )
      }
      const currentCapture = selectCurrentCapture(
        stateRef.current,
        visit.id,
      )
      if (
        currentCapture?.source === 'synthetic_demo' &&
        currentCapture.syntheticSourceAssetId === asset.id &&
        currentCapture.sha256 === asset.sha256
      ) {
        return throwProviderFailure(
          'INVALID_CAPTURE',
          'This visit already uses the current catalog demo.',
          'assetId',
        )
      }
      let media: Blob
      try {
        media = await loadSyntheticMedia(asset)
      } catch {
        return throwProviderFailure(
          'INVALID_CAPTURE',
          'The approved synthetic image could not be loaded.',
          'assetId',
        )
      }
      if (!(media instanceof Blob)) {
        return throwProviderFailure(
          'INVALID_CAPTURE',
          'The approved synthetic loader did not return image media.',
          'assetId',
        )
      }
      if (
        !mountedRef.current ||
        lifecycleRef.current !== lifecycle
      ) {
        return throwProviderFailure(
          'INVALID_CAPTURE',
          'The capture session changed before the synthetic image was ready.',
        )
      }
      return attachPreparedCapture(
        visitId,
        media,
        'synthetic_demo',
        asset.id,
        asset.sha256,
      )
    },
    [attachPreparedCapture, loadSyntheticMedia],
  )

  const setQualityCheck = useCallback(
    (
      visitId: string,
      check: keyof CaptureQualityChecks,
      passed: boolean,
    ) => {
      if (
        !isPatientVisitId(visitId) ||
        !getOwnRecordValue(stateRef.current.visitsById, visitId)
      ) {
        return throwProviderFailure(
          'UNKNOWN_VISIT',
          `Unknown visit: ${visitId}.`,
          'visitId',
        )
      }
      if (!QUALITY_CHECK_KEYS.has(check)) {
        return throwProviderFailure(
          'INVALID_CAPTURE',
          'Unknown capture quality check.',
          'check',
        )
      }
      const capture = selectCurrentCapture(stateRef.current, visitId)
      if (!capture) {
        return throwProviderFailure(
          'UNKNOWN_CAPTURE',
          'A current capture is required before confirming quality.',
          'captureId',
        )
      }
      if (capture.qualityChecks[check] === passed) return

      const checks = {
        ...capture.qualityChecks,
        [check]: passed,
      }
      let draft = stateRef.current
      const currentAuthorization = selectCurrentAuthorization(
        draft,
        visitId,
      )
      const shouldDocument = isCaptureQualityComplete(checks)
      const shouldWithdraw =
        check === 'authorizationDocumented' &&
        !passed &&
        currentAuthorization?.status === 'documented'
      let authorization: AuthorizationSnapshot | undefined

      if (shouldDocument || shouldWithdraw) {
        try {
          authorization = {
            id: entityId(
              activeRuntime,
              'authorization',
            ) as AuthorizationSnapshot['id'],
            patientId: capture.patientId,
            visitId: capture.visitId,
            revision: (currentAuthorization?.revision ?? 0) + 1,
            status: shouldDocument ? 'documented' : 'withdrawn',
            recordedAt: activeRuntime.now(),
          }
          draft = reduceOrThrow(draft, {
            type: 'authorization/record',
            authorization,
          })
        } catch (error) {
          if (error instanceof PatientWorkflowProviderError) throw error
          return throwProviderFailure(
            'INVALID_AUTHORIZATION',
            'Authorization could not be recorded.',
          )
        }
      }

      const next = reducerResult(draft, {
        type: 'capture/quality/set',
        captureId: capture.id,
        checks,
        ...(shouldDocument
          ? { confirmedAt: activeRuntime.now() }
          : {}),
      })
      if (next.lastFailure) {
        if (shouldWithdraw && authorization) {
          commitState(draft)
          return
        }
        throw new PatientWorkflowProviderError(next.lastFailure)
      }
      commitState(next)
    },
    [activeRuntime, commitState],
  )

  const submitAnalysis = useCallback(
    (visitId: string): PatientRunId => {
      const binding = exactBindingForCurrentVisit(
        stateRef.current,
        visitId,
      )
      if (!vault.has(binding.mediaHandle)) {
        return throwProviderFailure(
          'INVALID_CAPTURE',
          'The current session image is unavailable.',
          'mediaHandle',
        )
      }
      try {
        const runId = entityId(
          activeRuntime,
          'run',
        ) as PatientRunId
        const run: PatientRun = {
          id: runId,
          status: 'queued',
          binding,
          createdAt: activeRuntime.now(),
        }
        const next = reduceOrThrow(stateRef.current, {
          type: 'run/create',
          run,
        })
        commitState(next)
        scheduleRun(runId)
        return runId
      } catch (error) {
        if (error instanceof PatientWorkflowProviderError) throw error
        return throwProviderFailure(
          'INVALID_RUN_BINDING',
          'The analysis run could not be created.',
        )
      }
    },
    [activeRuntime, commitState, scheduleRun, vault],
  )

  const retryAnalysis = useCallback(
    (visitId: string): PatientRunId => {
      if (!isPatientVisitId(visitId)) {
        return throwProviderFailure(
          'UNKNOWN_VISIT',
          `Unknown visit: ${visitId}.`,
          'visitId',
        )
      }
      const failed = selectCurrentRun(stateRef.current, visitId)
      if (!failed || failed.status !== 'failed') {
        return throwProviderFailure(
          'INVALID_RETRY_BINDING',
          'Only the current exact-bound failed run may be retried.',
          'runId',
        )
      }
      try {
        const runId = entityId(
          activeRuntime,
          'run',
        ) as PatientRunId
        const next = reduceOrThrow(stateRef.current, {
          type: 'run/create',
          run: {
            id: runId,
            status: 'queued',
            binding: copyBinding(failed.binding),
            createdAt: activeRuntime.now(),
            retryOfRunId: failed.id,
          },
        })
        commitState(next)
        scheduleRun(runId)
        return runId
      } catch (error) {
        if (error instanceof PatientWorkflowProviderError) throw error
        return throwProviderFailure(
          'INVALID_RETRY_BINDING',
          'The exact failed analysis binding could not be retried.',
        )
      }
    },
    [activeRuntime, commitState, scheduleRun],
  )

  const completeReview = useCallback(
    (
      visitId: string,
      decision: PatientReviewDecision,
      note?: string,
    ) => {
      if (!isPatientVisitId(visitId)) {
        return throwProviderFailure(
          'UNKNOWN_VISIT',
          `Unknown visit: ${visitId}.`,
          'visitId',
        )
      }
      const result = selectCurrentResult(stateRef.current, visitId)
      const capture = selectCurrentCapture(stateRef.current, visitId)
      if (!result || !capture) {
        return throwProviderFailure(
          'INVALID_REVIEW',
          'A current result is required before review.',
          'resultId',
        )
      }
      try {
        const reviewId = entityId(
          activeRuntime,
          'review',
        ) as ReturnType<typeof createPatientReviewId>
        const next = reduceOrThrow(stateRef.current, {
          type: 'review/record',
          review: {
            id: reviewId,
            patientId: result.binding.patientId,
            visitId: result.binding.visitId,
            resultId: result.id,
            captureId: capture.id,
            decision,
            ...(note !== undefined ? { note } : {}),
            completedAt: activeRuntime.now(),
          },
        })
        commitState(next)
      } catch (error) {
        if (error instanceof PatientWorkflowProviderError) throw error
        return throwProviderFailure(
          'INVALID_REVIEW',
          'The review could not be completed.',
        )
      }
    },
    [activeRuntime, commitState],
  )

  const requestRetake = useCallback(
    (visitId: string) => {
      if (!isPatientVisitId(visitId)) {
        return throwProviderFailure(
          'UNKNOWN_VISIT',
          `Unknown visit: ${visitId}.`,
          'visitId',
        )
      }
      const review = selectCurrentReview(stateRef.current, visitId)
      const capture = selectCurrentCapture(stateRef.current, visitId)
      if (review?.decision !== 'repeat_photo' || !capture) {
        return throwProviderFailure(
          'INVALID_REVIEW',
          'Complete a Repeat photo review before requesting a retake.',
          'decision',
        )
      }
      vault.delete(capture.mediaHandle)
    },
    [vault],
  )

  const resetSession = useCallback(() => {
    activeRuntime.reset?.()
    lifecycleRef.current += 1
    clearTimers()
    vault.clear()
    setPatientListQueryState('')
    commitState(clonePlain(initialStateRef.current!))
  }, [activeRuntime, clearTimers, commitState, vault])

  const getCapturePreviewUrl = useCallback(
    (visitId: string) => {
      const capture = selectCurrentCapture(stateRef.current, visitId)
      return capture
        ? vault.get(capture.mediaHandle)?.previewUrl
        : undefined
    },
    [vault],
  )

  const actions = useMemo<PatientWorkflowActions>(
    () => ({
      createPatient,
      createVisit,
      attachSessionCapture,
      attachSyntheticCapture,
      setQualityCheck,
      submitAnalysis,
      retryAnalysis,
      completeReview,
      requestRetake,
      resetSession,
      getCapturePreviewUrl,
    }),
    [
      attachSessionCapture,
      attachSyntheticCapture,
      completeReview,
      createPatient,
      createVisit,
      getCapturePreviewUrl,
      requestRetake,
      resetSession,
      retryAnalysis,
      setQualityCheck,
      submitAnalysis,
    ],
  )
  const value = useMemo<PatientWorkflowContextValue>(
    () => ({
      state,
      actions,
      patientListQuery,
      setPatientListQuery,
      mode: 'simulation_only',
      persistence: 'memory_only',
    }),
    [actions, patientListQuery, setPatientListQuery, state],
  )

  return (
    <PatientWorkflowContext.Provider value={value}>
      {children}
    </PatientWorkflowContext.Provider>
  )
}

export function usePatientWorkflow(): PatientWorkflowContextValue {
  const workflow = useContext(PatientWorkflowContext)
  if (!workflow) {
    throw new Error(
      'usePatientWorkflow must be used within a PatientWorkflowProvider.',
    )
  }
  return workflow
}
