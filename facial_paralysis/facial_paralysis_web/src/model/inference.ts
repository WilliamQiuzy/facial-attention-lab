import { FACES_PROTOCOL, FACES_PROTOCOL_VERSION, type FacesActionId } from '../protocol/facesProtocol'

export const INFERENCE_SCHEMA_VERSION = 'facial-paralysis-shared-v9-inference/v2' as const
export const MANIFEST_SCHEMA_VERSION = 'faces-v9-capture-manifest/v1' as const
export const TIMELINE_SCHEMA_VERSION = 'faces-action-timeline/v1' as const
export const SCRIPT_VERSION = 'faces-script/24-004956-v1' as const
export const EXPECTED_MODEL_ID = 'broad_literature_shared_v9_blv9_009_ensemble' as const
export const EXPECTED_CANDIDATE_ID = 'BLV9-009' as const
export const EXPECTED_RELEASE_MANIFEST_SHA256 =
  '81e396954090a0da6b99519909c1af15b6df5d1585ba27a642539352fe0a0c64' as const
export const EXPECTED_FACE_LANDMARKER_SHA256 =
  '64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff' as const
export const EXPECTED_PREPROCESSING_VERSION = 'faces-to-shared-v9/v1' as const
export const MAX_VIDEO_BYTES = 512 * 1024 * 1024
export const READINESS_TIMEOUT_MS = 10_000
export const INFERENCE_TIMEOUT_MS = 300_000

export type RecordingSource = 'livelink-upload' | 'browser-camera'
const TIMING_SOURCES = ['capture_event_log', 'audio_forced_alignment', 'blinded_manual'] as const
export type CaptureTimingSource = typeof TIMING_SOURCES[number]

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
  readonly timingSource?: CaptureTimingSource
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
    readonly timingSource: CaptureTimingSource
  }
  readonly quality: {
    readonly eligible: true
    readonly actionsUsed: 6 | 7
    readonly optionalActionsUnavailable: readonly 'reanimated_smile'[]
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
    readonly interpretation: 'class_1_research_score_only'
    readonly endpointSemantics: 'meei_facial_palsy_vs_healthy_control_development_head'
    readonly class0Label: 'meei_healthy_control'
    readonly class1Label: 'meei_facial_palsy'
  }
  readonly reportEvidence: {
    readonly normalization: 'original_view_centered_eye_axis_aligned_interocular_scaled'
    readonly interpretation: 'measured_movement_observation_not_causal_or_severity'
    readonly contextFrameMethod: 'registered_hold_midpoint_not_model_selected'
    readonly actions: readonly {
      readonly id: Exclude<FacesActionId, 'repose'>
      readonly contextFrameMs: number
      readonly observations: readonly {
        readonly metric: string
        readonly value: number
        readonly unit: 'interocular_distance'
      }[]
    }[]
  }
  readonly clinicalUseEligible: false
}

export class InferenceContractError extends Error {
  readonly retryable: boolean

  constructor(message: string, retryable = false) {
    super(message)
    this.name = 'InferenceContractError'
    this.retryable = retryable
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

function nonnegativeFinite(value: unknown, path: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) {
    contractError(path, 'expected a finite nonnegative measurement')
  }
  return value
}

function captureTimingSource(value: unknown, path: string): CaptureTimingSource {
  if (typeof value !== 'string' || !TIMING_SOURCES.includes(value as CaptureTimingSource)) {
    contractError(path, 'expected capture_event_log, audio_forced_alignment, or blinded_manual')
  }
  return value as CaptureTimingSource
}

const ACTIVE_IDS = FACES_PROTOCOL.slice(1).map((step) => step.id)
const V9_ACTIONS = [
  'BROW_RAISE', 'EYE_GENTLE', 'EYE_FORCEFUL', 'SMILE_GENTLE',
  'LIP_PUCKER', 'SHOW_BOTTOM_TEETH', 'SMILE_FULL',
] as const
const ACTION_EVIDENCE_METRICS = [
  ['brow_height_asymmetry_iod', 'brow_height_change_from_rest_iod'],
  ['eye_aperture_asymmetry_iod', 'residual_eye_aperture_iod', 'eye_closure_change_from_rest_iod'],
  ['eye_aperture_asymmetry_iod', 'residual_eye_aperture_iod', 'eye_closure_change_from_rest_iod'],
  ['mouth_corner_vertical_asymmetry_iod', 'mouth_corner_vertical_change_from_rest_iod'],
  ['mouth_corner_horizontal_asymmetry_iod', 'mouth_width_change_from_rest_iod'],
  ['mouth_corner_vertical_asymmetry_iod', 'lower_lip_change_from_rest_iod', 'mouth_open_change_from_rest_iod'],
  ['mouth_corner_vertical_asymmetry_iod', 'mouth_corner_vertical_change_from_rest_iod'],
] as const

export function parseInferenceResponse(value: unknown): ResearchInferenceResult {
  const root = recordAt(value, 'response')
  exactKeys(root, ['schema_version', 'model', 'preprocessing', 'quality', 'prediction', 'report_evidence', 'clinical_use_eligible'], 'response')
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
  const timingSource = captureTimingSource(preprocessing.timing_source, 'preprocessing.timing_source')

  const quality = recordAt(root.quality, 'quality')
  exactKeys(quality, ['eligible', 'actions_used', 'optional_actions_unavailable', 'actions'], 'quality')
  exact(quality.eligible, true, 'quality.eligible')
  if (!Array.isArray(quality.optional_actions_unavailable)) {
    contractError('quality.optional_actions_unavailable', 'expected an array')
  }
  const optionalActionsUnavailable = quality.optional_actions_unavailable
  if (
    optionalActionsUnavailable.length > 1
    || (optionalActionsUnavailable.length === 1 && optionalActionsUnavailable[0] !== 'reanimated_smile')
  ) {
    contractError('quality.optional_actions_unavailable', 'only reanimated smile may be unavailable')
  }
  const actionCount = optionalActionsUnavailable.length === 0 ? 7 : 6
  exact(quality.actions_used, actionCount, 'quality.actions_used')
  if (!Array.isArray(quality.actions) || quality.actions.length !== actionCount) {
    contractError('quality.actions', `expected ${actionCount} active actions`)
  }
  let previousHoldEnd = -1
  const actions = quality.actions.map((raw, index) => {
    const path = `quality.actions[${index}]`
    const row = recordAt(raw, path)
    exactKeys(row, ['id', 'v9_action', 'hold_start_ms', 'hold_end_ms', 'valid_samples'], path)
    exact(row.id, ACTIVE_IDS.slice(0, actionCount)[index], `${path}.id`)
    exact(row.v9_action, V9_ACTIONS.slice(0, actionCount)[index], `${path}.v9_action`)
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
  exactKeys(prediction, [
    'probability', 'member_probabilities', 'predicted_class', 'threshold',
    'interpretation', 'endpoint_semantics', 'class_0_label', 'class_1_label',
  ], 'prediction')
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
  exact(prediction.interpretation, 'class_1_research_score_only', 'prediction.interpretation')
  exact(
    prediction.endpoint_semantics,
    'meei_facial_palsy_vs_healthy_control_development_head',
    'prediction.endpoint_semantics',
  )
  exact(prediction.class_0_label, 'meei_healthy_control', 'prediction.class_0_label')
  exact(prediction.class_1_label, 'meei_facial_palsy', 'prediction.class_1_label')

  const reportEvidence = recordAt(root.report_evidence, 'report_evidence')
  exactKeys(
    reportEvidence,
    ['normalization', 'interpretation', 'context_frame_method', 'actions'],
    'report_evidence',
  )
  exact(
    reportEvidence.normalization,
    'original_view_centered_eye_axis_aligned_interocular_scaled',
    'report_evidence.normalization',
  )
  exact(
    reportEvidence.interpretation,
    'measured_movement_observation_not_causal_or_severity',
    'report_evidence.interpretation',
  )
  exact(
    reportEvidence.context_frame_method,
    'registered_hold_midpoint_not_model_selected',
    'report_evidence.context_frame_method',
  )
  if (!Array.isArray(reportEvidence.actions) || reportEvidence.actions.length !== actionCount) {
    contractError('report_evidence.actions', `expected ${actionCount} action evidence rows`)
  }
  const evidenceActions = reportEvidence.actions.map((raw, index) => {
    const path = `report_evidence.actions[${index}]`
    const row = recordAt(raw, path)
    exactKeys(row, ['id', 'context_frame_ms', 'observations'], path)
    exact(row.id, actions[index].id, `${path}.id`)
    const contextFrameMs = integer(row.context_frame_ms, `${path}.context_frame_ms`)
    const expectedMidpoint = (actions[index].holdStartMs + actions[index].holdEndMs) / 2
    if (contextFrameMs !== expectedMidpoint) {
      contractError(`${path}.context_frame_ms`, 'expected the registered hold midpoint')
    }
    const expectedMetrics = ACTION_EVIDENCE_METRICS[index]
    if (!Array.isArray(row.observations) || row.observations.length !== expectedMetrics.length) {
      contractError(`${path}.observations`, 'metric count differs from the action contract')
    }
    const observations = row.observations.map((rawObservation, observationIndex) => {
      const observationPath = `${path}.observations[${observationIndex}]`
      const observation = recordAt(rawObservation, observationPath)
      exactKeys(observation, ['metric', 'value', 'unit'], observationPath)
      exact(observation.metric, expectedMetrics[observationIndex], `${observationPath}.metric`)
      exact(observation.unit, 'interocular_distance', `${observationPath}.unit`)
      return {
        metric: expectedMetrics[observationIndex],
        value: nonnegativeFinite(observation.value, `${observationPath}.value`),
        unit: 'interocular_distance' as const,
      }
    })
    return { id: actions[index].id, contextFrameMs, observations }
  })

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
      timingSource,
    },
    quality: {
      eligible: true,
      actionsUsed: actionCount as 6 | 7,
      optionalActionsUnavailable: optionalActionsUnavailable as 'reanimated_smile'[],
      actions,
    },
    prediction: {
      probability: aggregate,
      memberProbabilities: members,
      predictedClass: predictedClass as 0 | 1,
      threshold: 0.5,
      interpretation: 'class_1_research_score_only',
      endpointSemantics: 'meei_facial_palsy_vs_healthy_control_development_head',
      class0Label: 'meei_healthy_control',
      class1Label: 'meei_facial_palsy',
    },
    reportEvidence: {
      normalization: 'original_view_centered_eye_axis_aligned_interocular_scaled',
      interpretation: 'measured_movement_observation_not_causal_or_severity',
      contextFrameMethod: 'registered_hold_midpoint_not_model_selected',
      actions: evidenceActions,
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

const ENDPOINT_FAILURE_GUIDANCE = Object.freeze({
  invalid_capture_request: {
    status: 400,
    message: 'The recording request was incomplete. The same recording is still available; review its action timeline and retry.',
  },
  video_required: {
    status: 400,
    message: 'No video data was received. The same recording is still available; retry the upload.',
  },
  video_too_large: {
    status: 413,
    message: 'The recording is larger than 512 MB and cannot be uploaded. Use the guided browser recording or an approved compressed copy.',
  },
  multipart_required: {
    status: 415,
    message: 'The research upload format was not accepted. Keep the same recording and retry from this page.',
  },
  idempotency_key_required: {
    status: 428,
    message: 'The research request identity was missing. Keep the recording in this browser and contact the research system administrator.',
  },
  idempotency_key_conflict: {
    status: 409,
    message: 'The research request identity did not match the recording evidence. Do not resubmit this recording until the research system is checked.',
  },
  capture_evidence_invalid: {
    status: 422,
    message: 'The video and action timeline did not match. Clear this recording and complete the guided sequence again without refreshing or switching tabs.',
  },
  video_format_unsupported: {
    status: 422,
    message: 'This video format is not supported. Re-record in this browser, or upload a MOV, MP4, M4V, AVI, or WebM file.',
  },
  video_frame_rate_too_low: {
    status: 422,
    message: 'The video frame rate is too low to measure every action reliably. Re-record in this browser with other camera applications closed.',
  },
  video_dimensions_unsupported: {
    status: 422,
    message: 'The video dimensions are not supported. Re-record with the camera unobstructed, or upload the original-resolution recording.',
  },
  video_timing_mismatch: {
    status: 422,
    message: 'The recorded video timing did not match the guided action timeline. Re-record without pausing, switching cameras, locking the screen, or moving this tab to the background.',
  },
  video_decode_failed: {
    status: 422,
    message: 'The browser video could not be decoded reliably. Re-record in this browser, or upload a supported MOV, MP4, M4V, AVI, or WebM file.',
  },
  face_geometry_invalid: {
    status: 422,
    message: 'Facial geometry could not be measured reliably. Re-record with the face centered and upright, the full face visible, and steady front lighting.',
  },
  preprocessing_failed: {
    status: 422,
    message: 'The recording did not pass the preprocessing checks. Re-record the complete guided sequence with the face centered and this tab kept active.',
  },
  inference_unavailable: {
    status: 502,
    message: 'The model service could not complete this request. The same recording is still available; wait briefly and retry.',
  },
  model_not_ready: {
    status: 503,
    message: 'The model service is not ready. Keep this page open and retry before recording again.',
  },
  gateway_unavailable: {
    status: 500,
    message: 'The video processing service could not complete this request. The same recording is still available; wait briefly and retry.',
  },
} as const)

const TRACKING_ACTION_LABELS: Readonly<Record<string, string>> = Object.freeze({
  neutral_repose: 'Neutral Expression',
  eyebrow_raise: 'Eyebrow Raise',
  gentle_eye_closure: 'Gentle Eye Closure',
  tight_eye_squeeze: 'Tight Eye Squeeze',
  relaxed_smile: 'Relaxed Smile',
  lip_pucker: 'Lip Pucker',
  lower_teeth_show: 'Show Bottom Teeth',
  reanimated_smile: 'Reanimated Smile',
})

function exactObjectKeys(value: Record<string, unknown>, keys: readonly string[]): boolean {
  return Object.keys(value).length === keys.length && keys.every((key) => Object.hasOwn(value, key))
}

async function endpointFailure(response: Response): Promise<Error> {
  try {
    const source = await response.text()
    if (source.length < 1 || source.length > 4_096) throw new Error('error body outside bound')
    const payload: unknown = JSON.parse(source)
    if (typeof payload !== 'object' || payload === null || Array.isArray(payload)) throw new Error('error body is not an object')
    const root = payload as Record<string, unknown>
    if (!exactObjectKeys(root, ['detail'])) throw new Error('error body fields drifted')
    const detailValue = root.detail
    if (typeof detailValue !== 'object' || detailValue === null || Array.isArray(detailValue)) throw new Error('error detail is not an object')
    const detail = detailValue as Record<string, unknown>
    const code = detail.code
    if (code === 'face_tracking_insufficient') {
      if (
        response.status !== 422
        || !exactObjectKeys(detail, ['code', 'action', 'valid_samples', 'required_samples'])
        || typeof detail.action !== 'string'
        || !Object.hasOwn(TRACKING_ACTION_LABELS, detail.action)
        || typeof detail.valid_samples !== 'number'
        || !Number.isInteger(detail.valid_samples)
        || detail.valid_samples < 0
        || detail.valid_samples >= 26
        || detail.required_samples !== 26
      ) throw new Error('tracking detail differs from contract')
      return new InferenceContractError(
        `Face tracking was insufficient during ${TRACKING_ACTION_LABELS[detail.action]}: ${detail.valid_samples} of 26 required samples were usable. Re-record and keep the full face and neck visible, the camera at eye level, and steady front lighting.`,
        false,
      )
    }
    if (
      typeof code === 'string'
      && exactObjectKeys(detail, ['code'])
      && Object.hasOwn(ENDPOINT_FAILURE_GUIDANCE, code)
    ) {
      const guidance = ENDPOINT_FAILURE_GUIDANCE[code as keyof typeof ENDPOINT_FAILURE_GUIDANCE]
      if (response.status === guidance.status) {
        return new InferenceContractError(guidance.message, response.status >= 500)
      }
    }
  } catch {
    // Preserve the generic fail-closed message for malformed or unknown errors.
  }
  return new Error(`Research endpoint returned HTTP ${response.status}. No result was accepted.`)
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

function readinessEndpoint(endpoint: string): string {
  const validated = validateEndpoint(endpoint)
  if (!validated.endsWith('/infer')) {
    throw new InferenceContractError('Research endpoint must end with the frozen /infer route.')
  }
  return `${validated.slice(0, -'/infer'.length)}/ready`
}

export async function checkResearchEndpoint(
  endpoint: string,
  fetcher: typeof fetch = fetch,
): Promise<void> {
  const controller = new AbortController()
  let timedOut = false
  const timeout = setTimeout(() => {
    timedOut = true
    controller.abort()
  }, READINESS_TIMEOUT_MS)
  let response: Response
  try {
    response = await fetcher(readinessEndpoint(endpoint), {
      method: 'GET', credentials: 'same-origin', cache: 'no-store', redirect: 'error', signal: controller.signal,
    })
  } catch {
    throw new InferenceContractError(
      timedOut
        ? 'The research endpoint readiness check timed out. Retry the endpoint check before recording.'
        : 'The research endpoint could not be reached. Retry the endpoint check before recording.',
    )
  } finally {
    clearTimeout(timeout)
  }
  if (!response.ok) {
    const failure = await endpointFailure(response)
    throw failure instanceof InferenceContractError
      ? failure
      : new InferenceContractError('The research endpoint is not ready. Retry the endpoint check before recording.')
  }
  let payload: unknown
  try {
    payload = await response.json()
  } catch {
    throw new InferenceContractError('The research endpoint readiness response was not accepted.')
  }
  if (typeof payload !== 'object' || payload === null || Array.isArray(payload)) {
    throw new InferenceContractError('The research endpoint readiness response was not accepted.')
  }
  const ready = payload as Record<string, unknown>
  if (
    !exactObjectKeys(ready, ['status', 'model_id', 'candidate_id', 'ensemble_members', 'preprocessing'])
    || ready.status !== 'ready'
    || ready.model_id !== EXPECTED_MODEL_ID
    || ready.candidate_id !== EXPECTED_CANDIDATE_ID
    || ready.ensemble_members !== 3
    || ready.preprocessing !== EXPECTED_PREPROCESSING_VERSION
  ) {
    throw new InferenceContractError('The analysis endpoint does not match the required deployment.')
  }
}

function validateTimeline(
  timeline: CaptureTimelineDraft,
  expectedActionCount?: 7 | 8,
): CaptureTimelineDraft {
  if (
    !timeline
    || !Array.isArray(timeline.actions)
    || ![7, 8].includes(timeline.actions.length)
    || (expectedActionCount !== undefined && timeline.actions.length !== expectedActionCount)
  ) {
    throw new InferenceContractError('timeline: expected the exact seven- or eight-step FACES script')
  }
  const duration = integer(timeline.recordingDurationMs, 'timeline.recordingDurationMs', 1)
  if (timeline.timingSource !== undefined) {
    captureTimingSource(timeline.timingSource, 'timeline.timingSource')
  }
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
  const timingSource = captureTimingSource(root.timing_source, 'timeline sidecar.timing_source')
  if (typeof root.recording_sha256 !== 'string' || !/^[0-9a-f]{64}$/.test(root.recording_sha256)) {
    contractError('timeline sidecar.recording_sha256', 'expected lowercase SHA-256')
  }
  if (!Array.isArray(root.actions) || ![7, 8].includes(root.actions.length)) {
    contractError('timeline sidecar.actions', 'expected seven or eight actions')
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
    timingSource,
    sourceRecordingSha256: root.recording_sha256,
    sourceSidecar: source,
  })
}

async function sha256(file: File): Promise<string> {
  if (!globalThis.crypto?.subtle) throw new Error('This browser cannot hash the recording safely.')
  const digest = await globalThis.crypto.subtle.digest('SHA-256', await file.arrayBuffer())
  return Array.from(new Uint8Array(digest), (value) => value.toString(16).padStart(2, '0')).join('')
}

async function sha256Bytes(value: Uint8Array): Promise<string> {
  if (!globalThis.crypto?.subtle) throw new Error('This browser cannot hash the request safely.')
  const digest = await globalThis.crypto.subtle.digest('SHA-256', Uint8Array.from(value).buffer)
  return Array.from(new Uint8Array(digest), (item) => item.toString(16).padStart(2, '0')).join('')
}

async function idempotencyKey(
  videoSha256: string,
  manifestSource: string,
  timelineSource: string,
): Promise<string> {
  const encoder = new TextEncoder()
  const manifestSha256 = await sha256Bytes(encoder.encode(manifestSource))
  const timelineSha256 = await sha256Bytes(encoder.encode(timelineSource))
  const components = [
    videoSha256,
    manifestSha256,
    timelineSha256,
    EXPECTED_MODEL_ID,
    EXPECTED_CANDIDATE_ID,
    EXPECTED_PREPROCESSING_VERSION,
  ].join('\n')
  return sha256Bytes(encoder.encode(`facial-process-shared-v9-idempotency/v1\n${components}`))
}

export async function analyzeRecording(file: File, options: AnalyzeRecordingOptions): Promise<ResearchInferenceResult> {
  if (file.size < 1) {
    throw new InferenceContractError('The recording is empty. Record the complete guided sequence again.')
  }
  if (file.size > MAX_VIDEO_BYTES) {
    throw new InferenceContractError('The recording is larger than 512 MB and cannot be uploaded.')
  }
  const timeline = validateTimeline(
    options.timeline,
    options.reanimatedSmileApplicable ? 8 : 7,
  )
  const digest = await sha256(file)
  if (timeline.sourceRecordingSha256 && timeline.sourceRecordingSha256 !== digest) {
    throw new InferenceContractError('timeline sidecar: recording SHA-256 differs from the selected video')
  }
  const manifestSource = JSON.stringify({
    schema_version: MANIFEST_SCHEMA_VERSION,
    protocol_version: FACES_PROTOCOL_VERSION,
    recording_source: options.recordingSource,
    video_sha256: digest,
    reanimated_smile_applicable: options.reanimatedSmileApplicable,
  })
  const generatedTimeline = JSON.stringify({
    schema_version: TIMELINE_SCHEMA_VERSION,
    script_version: SCRIPT_VERSION,
    recording_sha256: digest,
    timing_source: timeline.timingSource ?? 'capture_event_log',
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
  const timelineSource = timeline.sourceSidecar ?? generatedTimeline
  const requestKey = await idempotencyKey(digest, manifestSource, timelineSource)
  const body = new FormData()
  body.append('video', file)
  body.append('manifest', manifestSource)
  body.append('timeline', timelineSource)
  const fetcher = options.fetcher ?? fetch
  const controller = new AbortController()
  let timedOut = false
  const timeout = setTimeout(() => {
    timedOut = true
    controller.abort()
  }, INFERENCE_TIMEOUT_MS)
  let response: Response
  try {
    response = await fetcher(validateEndpoint(options.endpoint), {
      method: 'POST',
      body,
      headers: { 'Idempotency-Key': requestKey },
      credentials: 'same-origin',
      cache: 'no-store',
      redirect: 'error',
      signal: controller.signal,
    })
  } catch {
    throw new InferenceContractError(
      timedOut
        ? 'The research analysis timed out. The same recording is still available; wait briefly and retry.'
        : 'Could not reach the research endpoint. The same recording is still available; check the connection and retry.',
      true,
    )
  } finally {
    clearTimeout(timeout)
  }
  if (!response.ok) throw await endpointFailure(response as Response)
  return parseInferenceResponse(await response.json())
}
