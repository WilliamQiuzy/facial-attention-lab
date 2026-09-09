import {
  FACES_PROTOCOL,
  FACES_PROTOCOL_VERSION,
  type FacesActionId,
} from '../protocol/facesProtocol'

export const INFERENCE_SCHEMA_VERSION = 'facial-palsy-research-inference/v1' as const
export const MANIFEST_SCHEMA_VERSION = 'faces-capture-manifest/v1' as const
export const EXPECTED_MODEL_FILE = 'warmstart_v4_expanded.pt' as const
export const EXPECTED_MODEL_SHA256 =
  '6310052121ed8a9a9e746716cb9c0d178eb252b438b6de7d33160eb555f6417b' as const
export const EXPECTED_PREPROCESSING_VERSION = 'predict-pipeline/v1' as const
export const EXPECTED_SEGMENTATION_VERSION = 'faces-segmentation/v1' as const

export type RecordingSource = 'livelink-upload' | 'browser-camera'
export type SeverityLabel = 'Normal' | 'Slight' | 'Strong'
export type SegmentStatus = 'completed' | 'not_applicable'

export interface RegionalSeverity {
  readonly level: 0 | 1 | 2
  readonly expected: number
  readonly pGt: readonly [number, number]
  readonly label: SeverityLabel
}

export interface ActionSegment {
  readonly id: FacesActionId
  readonly status: SegmentStatus
  readonly startMs: number | null
  readonly endMs: number | null
}

export interface ResearchInferenceResult {
  readonly mode: 'research-inference'
  readonly provenance: {
    readonly modelFile: typeof EXPECTED_MODEL_FILE
    readonly modelSha256: typeof EXPECTED_MODEL_SHA256
    readonly preprocessingVersion: typeof EXPECTED_PREPROCESSING_VERSION
    readonly segmentationVersion: typeof EXPECTED_SEGMENTATION_VERSION
  }
  readonly segmentation: {
    readonly durationMs: number
    readonly actions: readonly ActionSegment[]
  }
  readonly scores: {
    readonly palsyProbability: number
    readonly eyes: RegionalSeverity
    readonly mouth: RegionalSeverity
  }
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

function finiteNumber(value: unknown, path: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    contractError(path, 'expected a finite number')
  }
  return value
}

function numberInRange(value: unknown, min: number, max: number, path: string): number {
  const parsed = finiteNumber(value, path)
  if (parsed < min || parsed > max) contractError(path, `must be between ${min} and ${max}`)
  return parsed
}

function exactString<T extends string>(value: unknown, expected: T, path: string): T {
  if (value !== expected) contractError(path, `expected ${expected}`)
  return expected
}

const SEVERITY_LABELS = ['Normal', 'Slight', 'Strong'] as const

function parseRegion(value: unknown, path: string): RegionalSeverity {
  const record = recordAt(value, path)
  exactKeys(record, ['level', 'expected', 'p_gt', 'label'], path)

  const level = finiteNumber(record.level, `${path}.level`)
  if (!Number.isInteger(level) || level < 0 || level > 2) {
    contractError(`${path}.level`, 'must be an integer from 0 to 2')
  }
  const expected = numberInRange(record.expected, 0, 2, `${path}.expected`)
  if (!Array.isArray(record.p_gt) || record.p_gt.length !== 2) {
    contractError(`${path}.p_gt`, 'ordinal threshold array must contain exactly two values')
  }
  const first = numberInRange(record.p_gt[0], 0, 1, `${path}.p_gt[0]`)
  const second = numberInRange(record.p_gt[1], 0, 1, `${path}.p_gt[1]`)
  if (first < second) {
    contractError(`${path}.p_gt`, 'ordinal threshold probabilities must be non-increasing')
  }
  const expectedLabel = SEVERITY_LABELS[level as 0 | 1 | 2]
  if (record.label !== expectedLabel) {
    contractError(`${path}.label`, `must match level ${level} (${expectedLabel})`)
  }

  return {
    level: level as 0 | 1 | 2,
    expected,
    pGt: [first, second],
    label: expectedLabel,
  }
}

function parseSegmentation(value: unknown): ResearchInferenceResult['segmentation'] {
  const record = recordAt(value, 'segmentation')
  exactKeys(record, ['duration_ms', 'actions'], 'segmentation')
  const durationMs = finiteNumber(record.duration_ms, 'segmentation.duration_ms')
  if (!Number.isInteger(durationMs) || durationMs <= 0) {
    contractError('segmentation.duration_ms', 'must be a positive integer')
  }
  if (!Array.isArray(record.actions) || record.actions.length !== FACES_PROTOCOL.length) {
    contractError('segmentation.actions', 'must cover all eight protocol actions')
  }

  let previousEnd = -1
  const actions = record.actions.map((rawAction, index): ActionSegment => {
    const path = `segmentation.actions[${index}]`
    const action = recordAt(rawAction, path)
    exactKeys(action, ['id', 'status', 'start_ms', 'end_ms'], path)
    const expectedStep = FACES_PROTOCOL[index]
    exactString(action.id, expectedStep.id, `${path}.id`)

    const status = action.status
    if (index < FACES_PROTOCOL.length - 1 && status !== 'completed') {
      contractError(`${path}.status`, 'required protocol segment must be completed')
    }
    if (index === FACES_PROTOCOL.length - 1 && status !== 'completed' && status !== 'not_applicable') {
      contractError(`${path}.status`, 'conditional segment must be completed or not_applicable')
    }

    if (status === 'not_applicable') {
      if (action.start_ms !== null || action.end_ms !== null) {
        contractError(path, 'not_applicable segment timestamps must be null')
      }
      return {
        id: expectedStep.id,
        status: 'not_applicable',
        startMs: null,
        endMs: null,
      }
    }

    const startMs = finiteNumber(action.start_ms, `${path}.start_ms`)
    const endMs = finiteNumber(action.end_ms, `${path}.end_ms`)
    if (!Number.isInteger(startMs) || !Number.isInteger(endMs)) {
      contractError(path, 'segment timestamps must be integer milliseconds')
    }
    if (startMs < 0 || endMs <= startMs || endMs > durationMs || startMs < previousEnd) {
      contractError(path, 'segment timestamps must be positive, ordered, non-overlapping, and in bounds')
    }
    previousEnd = endMs
    return {
      id: expectedStep.id,
      status: 'completed',
      startMs,
      endMs,
    }
  })

  return { durationMs, actions }
}

export function parseInferenceResponse(value: unknown): ResearchInferenceResult {
  const root = recordAt(value, 'response')
  exactKeys(root, ['schema_version', 'provenance', 'segmentation', 'scores'], 'response')
  exactString(root.schema_version, INFERENCE_SCHEMA_VERSION, 'response.schema_version')

  const provenance = recordAt(root.provenance, 'provenance')
  exactKeys(
    provenance,
    ['model_file', 'model_sha256', 'preprocessing_version', 'segmentation_version'],
    'provenance',
  )
  exactString(provenance.model_file, EXPECTED_MODEL_FILE, 'provenance.model_file')
  exactString(provenance.model_sha256, EXPECTED_MODEL_SHA256, 'provenance.model_sha256')
  exactString(
    provenance.preprocessing_version,
    EXPECTED_PREPROCESSING_VERSION,
    'provenance.preprocessing_version',
  )
  exactString(
    provenance.segmentation_version,
    EXPECTED_SEGMENTATION_VERSION,
    'provenance.segmentation_version',
  )

  const scores = recordAt(root.scores, 'scores')
  exactKeys(scores, ['palsy_probability', 'eyes', 'mouth'], 'scores')

  return {
    mode: 'research-inference',
    provenance: {
      modelFile: EXPECTED_MODEL_FILE,
      modelSha256: EXPECTED_MODEL_SHA256,
      preprocessingVersion: EXPECTED_PREPROCESSING_VERSION,
      segmentationVersion: EXPECTED_SEGMENTATION_VERSION,
    },
    segmentation: parseSegmentation(root.segmentation),
    scores: {
      palsyProbability: numberInRange(
        scores.palsy_probability,
        0,
        1,
        'scores.palsy_probability',
      ),
      eyes: parseRegion(scores.eyes, 'scores.eyes'),
      mouth: parseRegion(scores.mouth, 'scores.mouth'),
    },
  }
}

interface AnalyzeRecordingOptions {
  readonly endpoint: string
  readonly recordingSource: RecordingSource
  readonly reanimatedSmileApplicable: boolean
  readonly fetcher?: typeof fetch
}

function validateEndpoint(endpoint: string): string {
  let url: URL
  try {
    url = new URL(endpoint)
  } catch {
    throw new Error('Research endpoint must be a valid URL.')
  }
  const local = url.hostname === '127.0.0.1' || url.hostname === 'localhost'
  if (url.protocol !== 'https:' && !(local && url.protocol === 'http:')) {
    throw new Error('Research endpoint must use HTTPS (localhost HTTP is allowed for development).')
  }
  return url.toString()
}

export async function analyzeRecording(
  file: File,
  options: AnalyzeRecordingOptions,
): Promise<ResearchInferenceResult> {
  const body = new FormData()
  body.append('video', file)
  body.append(
    'manifest',
    JSON.stringify({
      schema_version: MANIFEST_SCHEMA_VERSION,
      protocol_version: FACES_PROTOCOL_VERSION,
      recording_source: options.recordingSource,
      reanimated_smile_applicable: options.reanimatedSmileApplicable,
    }),
  )

  const fetcher = options.fetcher ?? fetch
  const response = await fetcher(validateEndpoint(options.endpoint), {
    method: 'POST',
    body,
    credentials: 'omit',
    cache: 'no-store',
    redirect: 'error',
  })
  if (!response.ok) {
    throw new Error(`Research endpoint returned HTTP ${response.status}. No result was accepted.`)
  }
  const accepted = parseInferenceResponse(await response.json())
  const reanimatedSmile = accepted.segmentation.actions.at(-1)
  const expectedStatus = options.reanimatedSmileApplicable ? 'completed' : 'not_applicable'
  if (reanimatedSmile?.status !== expectedStatus) {
    throw new InferenceContractError(
      'segmentation.actions[7].status: reanimated-smile status contradicts the clinician choice',
    )
  }
  return accepted
}
