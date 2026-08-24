import { FACES_PROTOCOL, FACES_PROTOCOL_VERSION, type FacesActionId } from '../protocol/facesProtocol'

export const INFERENCE_SCHEMA_VERSION = 'facial-paralysis-shared-v9-inference/v1' as const
export const MANIFEST_SCHEMA_VERSION = 'faces-v9-capture-manifest/v1' as const
export const TIMELINE_SCHEMA_VERSION = 'faces-action-timeline/v1' as const
export const SCRIPT_VERSION = 'faces-script/24-004956-v1' as const
export const EXPECTED_MODEL_ID = 'broad_literature_shared_v9_blv9_009_ensemble' as const
export const EXPECTED_CANDIDATE_ID = 'BLV9-009' as const
export const EXPECTED_RELEASE_MANIFEST_SHA256 =
  'c4fdaf054f3076a2e31b0e1ae93d1e91a45212817eb39d1c4a53620a4007b18f' as const
export const EXPECTED_FACE_LANDMARKER_SHA256 =
  '64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff' as const
export const EXPECTED_PREPROCESSING_VERSION = 'faces-to-shared-v9/v1' as const

export type RecordingSource = 'livelink-upload' | 'browser-camera'

export interface CaptureActionTimingDraft {
  readonly id: FacesActionId
  readonly promptStartMs: number
  readonly holdStartMs: number
  readonly holdEndMs: number
  readonly completionMs: number
}

export interface CaptureTimelineDraft {
  readonly recordingDurationMs: number
  readonly actions: readonly CaptureActionTimingDraft[]
  readonly sourceRecordingSha256?: string
  readonly sourceSidecar?: string
}

export interface ResearchInferenceResult {
  readonly mode: 'research-inference'
  readonly model: {
    readonly modelId: typeof EXPECTED_MODEL_ID
    readonly candidateId: typeof EXPECTED_CANDIDATE_ID
    readonly releaseManifestSha256: typeof EXPECTED_RELEASE_MANIFEST_SHA256
    readonly ensembleMembers: 3
  }
  readonly preprocessing: {
    readonly version: typeof EXPECTED_PREPROCESSING_VERSION
    readonly faceLandmarkerSha256: typeof EXPECTED_FACE_LANDMARKER_SHA256
    readonly mirrorMethod: 'horizontal_flip_and_redetect'
    readonly protocol: 'cue_aligned_action'
    readonly timingSource: 'capture_event_log'
  }
  readonly quality: {
    readonly eligible: true
    readonly actionsUsed: 7
    readonly actions: readonly {
      readonly id: Exclude<FacesActionId, 'repose'>
      readonly v9Action: string
      readonly holdStartMs: number
      readonly holdEndMs: number
      readonly validSamples: number
    }[]
  }
  readonly prediction: {
    readonly probability: number
    readonly memberProbabilities: readonly [number, number, number]
    readonly predictedClass: 0 | 1
    readonly threshold: 0.5
    readonly interpretation: 'research_score_only'
  }
  readonly clinicalUseEligible: false
}

export class InferenceContractError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'InferenceContractError'
  }
}

type JsonRecord = Record<string, unknown>

function contractError(path: string, message: string): never {
  throw new InferenceContractError(`${path}: ${message}`)
}

function recordAt(value: unknown, path: string): JsonRecord {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    contractError(path, 'expected an object')
  }
  return value as JsonRecord
}

function exactKeys(record: JsonRecord, expected: readonly string[], path: string): void {
  const expectedSet = new Set(expected)
  const unknown = Object.keys(record).filter((key) => !expectedSet.has(key))
  const missing = expected.filter((key) => !(key in record))
  if (unknown.length > 0) contractError(path, `unknown field ${unknown.join(', ')}`)
  if (missing.length > 0) contractError(path, `missing field ${missing.join(', ')}`)
}

function exact<T>(value: unknown, expected: T, path: string): T {
  if (value !== expected) contractError(path, `expected ${String(expected)}`)
  return expected
}

function integer(value: unknown, path: string, min = 0): number {
  if (typeof value !== 'number' || !Number.isInteger(value) || value < min) {
    contractError(path, `expected an integer at least ${min}`)
  }
  return value
}

function probability(value: unknown, path: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0 || value > 1) {
    contractError(path, 'expected a probability')
  }
  return value
}

const ACTIVE_IDS = FACES_PROTOCOL.slice(1).map((step) => step.id)
const V9_ACTIONS = [
  'BROW_RAISE', 'EYE_GENTLE', 'EYE_FORCEFUL', 'SMILE_GENTLE',
  'LIP_PUCKER', 'SHOW_BOTTOM_TEETH', 'SMILE_FULL',
] as const

export function parseInferenceResponse(value: unknown): ResearchInferenceResult {
  const root = recordAt(value, 'response')
  exactKeys(root, ['schema_version', 'model', 'preprocessing', 'quality', 'prediction', 'clinical_use_eligible'], 'response')
  exact(root.schema_version, INFERENCE_SCHEMA_VERSION, 'response.schema_version')
  exact(root.clinical_use_eligible, false, 'response.clinical_use_eligible')

  const model = recordAt(root.model, 'model')
  exactKeys(model, ['model_id', 'candidate_id', 'release_manifest_sha256', 'ensemble_members'], 'model')
  exact(model.model_id, EXPECTED_MODEL_ID, 'model.model_id')
  exact(model.candidate_id, EXPECTED_CANDIDATE_ID, 'model.candidate_id')
  exact(model.release_manifest_sha256, EXPECTED_RELEASE_MANIFEST_SHA256, 'model.release_manifest_sha256')
  exact(model.ensemble_members, 3, 'model.ensemble_members')

  const preprocessing = recordAt(root.preprocessing, 'preprocessing')
  exactKeys(preprocessing, ['version', 'face_landmarker_sha256', 'mirror_method', 'protocol', 'timing_source'], 'preprocessing')
  exact(preprocessing.version, EXPECTED_PREPROCESSING_VERSION, 'preprocessing.version')
  exact(preprocessing.face_landmarker_sha256, EXPECTED_FACE_LANDMARKER_SHA256, 'preprocessing.face_landmarker_sha256')
  exact(preprocessing.mirror_method, 'horizontal_flip_and_redetect', 'preprocessing.mirror_method')
  exact(preprocessing.protocol, 'cue_aligned_action', 'preprocessing.protocol')
  exact(preprocessing.timing_source, 'capture_event_log', 'preprocessing.timing_source')

  const quality = recordAt(root.quality, 'quality')
  exactKeys(quality, ['eligible', 'actions_used', 'actions'], 'quality')
  exact(quality.eligible, true, 'quality.eligible')
  exact(quality.actions_used, 7, 'quality.actions_used')
  if (!Array.isArray(quality.actions) || quality.actions.length !== 7) {
    contractError('quality.actions', 'expected seven active actions')
  }
  let previousHoldEnd = -1
  const actions = quality.actions.map((raw, index) => {
    const path = `quality.actions[${index}]`
    const row = recordAt(raw, path)
    exactKeys(row, ['id', 'v9_action', 'hold_start_ms', 'hold_end_ms', 'valid_samples'], path)
    exact(row.id, ACTIVE_IDS[index], `${path}.id`)
    exact(row.v9_action, V9_ACTIONS[index], `${path}.v9_action`)
    const holdStartMs = integer(row.hold_start_ms, `${path}.hold_start_ms`)
    const holdEndMs = integer(row.hold_end_ms, `${path}.hold_end_ms`, 1)
    if (holdEndMs - holdStartMs !== 3_000 || holdStartMs < previousHoldEnd) {
      contractError(path, 'hold intervals must be ordered three-second windows')
    }
    previousHoldEnd = holdEndMs
    const validSamples = integer(row.valid_samples, `${path}.valid_samples`, 26)
    if (validSamples > 32) contractError(`${path}.valid_samples`, 'cannot exceed 32')
    return {
      id: ACTIVE_IDS[index] as Exclude<FacesActionId, 'repose'>,
      v9Action: V9_ACTIONS[index], holdStartMs, holdEndMs, validSamples,
    }
  })

  const prediction = recordAt(root.prediction, 'prediction')
  exactKeys(prediction, ['probability', 'member_probabilities', 'predicted_class', 'threshold', 'interpretation'], 'prediction')
  const aggregate = probability(prediction.probability, 'prediction.probability')
  if (!Array.isArray(prediction.member_probabilities) || prediction.member_probabilities.length !== 3) {
    contractError('prediction.member_probabilities', 'expected three ensemble members')
  }
  const members = prediction.member_probabilities.map((item, index) => probability(item, `prediction.member_probabilities[${index}]`)) as [number, number, number]
  if (Math.abs(aggregate - members.reduce((sum, item) => sum + item, 0) / 3) > 1e-7) {
    contractError('prediction.probability', 'does not equal the ensemble mean')
  }
  const predictedClass = integer(prediction.predicted_class, 'prediction.predicted_class')
  if ((predictedClass !== 0 && predictedClass !== 1) || Number(aggregate >= 0.5) !== predictedClass) {
    contractError('prediction.predicted_class', 'does not match the frozen threshold')
  }
  exact(prediction.threshold, 0.5, 'prediction.threshold')
  exact(prediction.interpretation, 'research_score_only', 'prediction.interpretation')

  return {
    mode: 'research-inference',
    model: {
      modelId: EXPECTED_MODEL_ID,
      candidateId: EXPECTED_CANDIDATE_ID,
      releaseManifestSha256: EXPECTED_RELEASE_MANIFEST_SHA256,
      ensembleMembers: 3,
    },
    preprocessing: {
      version: EXPECTED_PREPROCESSING_VERSION,
      faceLandmarkerSha256: EXPECTED_FACE_LANDMARKER_SHA256,
      mirrorMethod: 'horizontal_flip_and_redetect',
      protocol: 'cue_aligned_action',
      timingSource: 'capture_event_log',
    },
    quality: { eligible: true, actionsUsed: 7, actions },
    prediction: {
      probability: aggregate,
      memberProbabilities: members,
      predictedClass: predictedClass as 0 | 1,
      threshold: 0.5,
      interpretation: 'research_score_only',
    },
    clinicalUseEligible: false,
  }
}

interface AnalyzeRecordingOptions {
  readonly endpoint: string
  readonly recordingSource: RecordingSource
  readonly reanimatedSmileApplicable: boolean
  readonly timeline: CaptureTimelineDraft
  readonly fetcher?: typeof fetch
}

function validateEndpoint(endpoint: string): string {
  if (endpoint.startsWith('/') && !endpoint.startsWith('//') && !endpoint.includes('?') && !endpoint.includes('#')) {
    return endpoint
  }
  let url: URL
  try {
    url = new URL(endpoint)
  } catch {
    throw new Error('Research endpoint must be a same-origin path or valid URL.')
  }
  const local = url.hostname === '127.0.0.1' || url.hostname === 'localhost'
  if (url.protocol !== 'https:' && !(local && url.protocol === 'http:')) {
    throw new Error('Research endpoint must use HTTPS (localhost HTTP is allowed for development).')
  }
  return url.toString()
}

function validateTimeline(timeline: CaptureTimelineDraft): CaptureTimelineDraft {
  if (!timeline || !Array.isArray(timeline.actions) || timeline.actions.length !== 8) {
    throw new InferenceContractError('timeline: all eight externally timed FACES actions are required')
  }
  const duration = integer(timeline.recordingDurationMs, 'timeline.recordingDurationMs', 1)
  let previousCompletion = -1
  timeline.actions.forEach((row, index) => {
    if (row.id !== FACES_PROTOCOL[index].id) contractError(`timeline.actions[${index}].id`, 'protocol order drifted')
    const prompt = integer(row.promptStartMs, `timeline.actions[${index}].promptStartMs`)
    const holdStart = integer(row.holdStartMs, `timeline.actions[${index}].holdStartMs`)
    const holdEnd = integer(row.holdEndMs, `timeline.actions[${index}].holdEndMs`)
    const completion = integer(row.completionMs, `timeline.actions[${index}].completionMs`)
    if (prompt < previousCompletion || holdStart < prompt || holdEnd - holdStart !== 3_000 || completion < holdEnd || completion > duration) {
      contractError(`timeline.actions[${index}]`, 'timestamps must be ordered with one exact three-second hold')
    }
    previousCompletion = completion
  })
  return timeline
}

export function parseCaptureTimelineSidecar(source: string): CaptureTimelineDraft {
  let raw: unknown
  try {
    raw = JSON.parse(source)
  } catch {
    throw new InferenceContractError('timeline sidecar: invalid JSON')
  }
  const root = recordAt(raw, 'timeline sidecar')
  exactKeys(root, ['schema_version', 'script_version', 'recording_sha256', 'timing_source', 'recording_duration_ms', 'actions'], 'timeline sidecar')
  exact(root.schema_version, TIMELINE_SCHEMA_VERSION, 'timeline sidecar.schema_version')
  exact(root.script_version, SCRIPT_VERSION, 'timeline sidecar.script_version')
  exact(root.timing_source, 'capture_event_log', 'timeline sidecar.timing_source')
  if (typeof root.recording_sha256 !== 'string' || !/^[0-9a-f]{64}$/.test(root.recording_sha256)) {
    contractError('timeline sidecar.recording_sha256', 'expected lowercase SHA-256')
  }
  if (!Array.isArray(root.actions) || root.actions.length !== 8) {
    contractError('timeline sidecar.actions', 'expected eight actions')
  }
  const actions = root.actions.map((rawAction, index): CaptureActionTimingDraft => {
    const path = `timeline sidecar.actions[${index}]`
    const row = recordAt(rawAction, path)
    exactKeys(row, ['action', 'status', 'prompt_start_ms', 'hold_start_ms', 'hold_end_ms', 'completion_ms'], path)
    exact(row.action, index === 0 ? 'neutral_repose' : FACES_PROTOCOL[index].id, `${path}.action`)
    exact(row.status, 'completed', `${path}.status`)
    return {
      id: FACES_PROTOCOL[index].id,
      promptStartMs: integer(row.prompt_start_ms, `${path}.prompt_start_ms`),
      holdStartMs: integer(row.hold_start_ms, `${path}.hold_start_ms`),
      holdEndMs: integer(row.hold_end_ms, `${path}.hold_end_ms`),
      completionMs: integer(row.completion_ms, `${path}.completion_ms`),
    }
  })
  return validateTimeline({
    recordingDurationMs: integer(root.recording_duration_ms, 'timeline sidecar.recording_duration_ms', 1),
    actions,
    sourceRecordingSha256: root.recording_sha256,
    sourceSidecar: source,
  })
}

async function sha256(file: File): Promise<string> {
  if (!globalThis.crypto?.subtle) throw new Error('This browser cannot hash the recording safely.')
  const digest = await globalThis.crypto.subtle.digest('SHA-256', await file.arrayBuffer())
  return Array.from(new Uint8Array(digest), (value) => value.toString(16).padStart(2, '0')).join('')
}

export async function analyzeRecording(file: File, options: AnalyzeRecordingOptions): Promise<ResearchInferenceResult> {
  if (!options.reanimatedSmileApplicable) {
    throw new InferenceContractError('Shared V9 requires seven active movements; Step 8 cannot be imputed.')
  }
  const timeline = validateTimeline(options.timeline)
  const digest = await sha256(file)
  if (timeline.sourceRecordingSha256 && timeline.sourceRecordingSha256 !== digest) {
    throw new InferenceContractError('timeline sidecar: recording SHA-256 differs from the selected video')
  }
  const body = new FormData()
  body.append('video', file)
  body.append('manifest', JSON.stringify({
    schema_version: MANIFEST_SCHEMA_VERSION,
    protocol_version: FACES_PROTOCOL_VERSION,
    recording_source: options.recordingSource,
    video_sha256: digest,
    reanimated_smile_applicable: true,
  }))
  const generatedTimeline = JSON.stringify({
    schema_version: TIMELINE_SCHEMA_VERSION,
    script_version: SCRIPT_VERSION,
    recording_sha256: digest,
    timing_source: 'capture_event_log',
    recording_duration_ms: timeline.recordingDurationMs,
    actions: timeline.actions.map((row) => ({
      action: row.id === 'repose' ? 'neutral_repose' : row.id,
      status: 'completed',
      prompt_start_ms: row.promptStartMs,
      hold_start_ms: row.holdStartMs,
      hold_end_ms: row.holdEndMs,
      completion_ms: row.completionMs,
    })),
  })
  body.append('timeline', timeline.sourceSidecar ?? generatedTimeline)
  const fetcher = options.fetcher ?? fetch
  const response = await fetcher(validateEndpoint(options.endpoint), {
    method: 'POST', body, credentials: 'same-origin', cache: 'no-store', redirect: 'error',
  })
  if (!response.ok) throw new Error(`Research endpoint returned HTTP ${response.status}. No result was accepted.`)
  return parseInferenceResponse(await response.json())
}
