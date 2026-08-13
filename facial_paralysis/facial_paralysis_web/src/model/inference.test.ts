import { describe, expect, it, vi } from 'vitest'

import {
  EXPECTED_MODEL_SHA256,
  InferenceContractError,
  analyzeRecording,
  parseInferenceResponse,
} from './inference'

const validResponse = () => ({
  schema_version: 'facial-palsy-research-inference/v1',
  provenance: {
    model_file: 'warmstart_v4_expanded.pt',
    model_sha256: EXPECTED_MODEL_SHA256,
    preprocessing_version: 'predict-pipeline/v1',
    segmentation_version: 'faces-segmentation/v1',
  },
  segmentation: {
    duration_ms: 42_000,
    actions: [
      { id: 'repose', status: 'completed', start_ms: 0, end_ms: 3_000 },
      { id: 'eyebrow_raise', status: 'completed', start_ms: 5_000, end_ms: 8_000 },
      { id: 'gentle_eye_closure', status: 'completed', start_ms: 10_000, end_ms: 13_000 },
      { id: 'tight_eye_squeeze', status: 'completed', start_ms: 15_000, end_ms: 18_000 },
      { id: 'relaxed_smile', status: 'completed', start_ms: 20_000, end_ms: 23_000 },
      { id: 'lip_pucker', status: 'completed', start_ms: 25_000, end_ms: 28_000 },
      { id: 'lower_teeth_show', status: 'completed', start_ms: 30_000, end_ms: 33_000 },
      { id: 'reanimated_smile', status: 'not_applicable', start_ms: null, end_ms: null },
    ],
  },
  scores: {
    palsy_probability: 0.73,
    eyes: { level: 1, expected: 1.2, p_gt: [0.82, 0.38], label: 'Slight' },
    mouth: { level: 2, expected: 1.7, p_gt: [0.91, 0.79], label: 'Strong' },
  },
})

describe('parseInferenceResponse', () => {
  it('accepts the exact versioned v4 response contract', () => {
    const parsed = parseInferenceResponse(validResponse())
    expect(parsed.scores.eyes.label).toBe('Slight')
    expect(parsed.segmentation.actions).toHaveLength(8)
  })

  it('accepts a completed optional reanimated-smile segment', () => {
    const response = validResponse()
    response.segmentation.actions[7] = {
      id: 'reanimated_smile',
      status: 'completed',
      start_ms: 35_000,
      end_ms: 38_000,
    }
    expect(parseInferenceResponse(response).segmentation.actions[7].status).toBe('completed')
  })

  it('rejects incomplete required segmentation', () => {
    const response = validResponse()
    response.segmentation.actions[4].status = 'skipped'
    expect(() => parseInferenceResponse(response)).toThrow(InferenceContractError)
  })

  it('rejects a skipped conditional segment until applicability is resolved', () => {
    const response = validResponse()
    response.segmentation.actions[7].status = 'skipped'
    expect(() => parseInferenceResponse(response)).toThrow(/segmentation/i)
  })

  it('rejects overlapping or out-of-bounds segment timestamps', () => {
    const overlapping = validResponse()
    overlapping.segmentation.actions[2].start_ms = 7_000
    expect(() => parseInferenceResponse(overlapping)).toThrow(/segment/i)

    const outOfBounds = validResponse()
    outOfBounds.segmentation.actions[6].end_ms = 50_000
    expect(() => parseInferenceResponse(outOfBounds)).toThrow(/segment/i)
  })

  it('rejects probabilities outside their supported ranges', () => {
    const response = validResponse()
    response.scores.palsy_probability = 1.01
    expect(() => parseInferenceResponse(response)).toThrow(/probability/i)
  })

  it('rejects ordinal thresholds with the wrong length or order', () => {
    const wrongLength = validResponse()
    wrongLength.scores.eyes.p_gt = [0.8]
    expect(() => parseInferenceResponse(wrongLength)).toThrow(/threshold/i)

    const wrongOrder = validResponse()
    wrongOrder.scores.mouth.p_gt = [0.3, 0.8]
    expect(() => parseInferenceResponse(wrongOrder)).toThrow(/threshold/i)
  })

  it('rejects mismatched ordinal labels and levels', () => {
    const response = validResponse()
    response.scores.eyes.label = 'Strong'
    expect(() => parseInferenceResponse(response)).toThrow(/label/i)
  })

  it('rejects unpinned provenance', () => {
    const response = validResponse()
    const provenance: { model_sha256: string } = response.provenance
    provenance.model_sha256 = '0'.repeat(64)
    expect(() => parseInferenceResponse(response)).toThrow(/provenance/i)
  })

  it('rejects unsupported clinical or spatial fields and any unknown field', () => {
    for (const field of ['hb_grade', 'heatmap', 'coarse3', 'extra']) {
      const response = validResponse() as Record<string, unknown>
      response[field] = field
      expect(() => parseInferenceResponse(response)).toThrow(/unknown/i)
    }
  })
})

describe('analyzeRecording', () => {
  it('posts video and a versioned protocol manifest to the configured endpoint', async () => {
    const fetcher = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => validResponse(),
    })
    const file = new File(['synthetic'], 'faces-protocol.webm', { type: 'video/webm' })

    await analyzeRecording(file, {
      endpoint: 'https://research.example.test/infer',
      recordingSource: 'livelink-upload',
      reanimatedSmileApplicable: false,
      fetcher,
    })

    const [, init] = fetcher.mock.calls[0]
    const body = init.body as FormData
    expect(body.get('video')).toBe(file)
    expect(JSON.parse(String(body.get('manifest')))).toEqual({
      schema_version: 'faces-capture-manifest/v1',
      protocol_version: 'FACES-v0.01',
      recording_source: 'livelink-upload',
      reanimated_smile_applicable: false,
    })
  })

  it('rejects segmentation that contradicts the clinician step-8 choice', async () => {
    const file = new File(['synthetic'], 'faces-protocol.webm', { type: 'video/webm' })
    const notApplicableResponse = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => validResponse(),
    })
    await expect(
      analyzeRecording(file, {
        endpoint: 'https://research.example.test/infer',
        recordingSource: 'livelink-upload',
        reanimatedSmileApplicable: true,
        fetcher: notApplicableResponse,
      }),
    ).rejects.toThrow(/reanimated.*clinician/i)

    const completed = validResponse()
    completed.segmentation.actions[7] = {
      id: 'reanimated_smile',
      status: 'completed',
      start_ms: 35_000,
      end_ms: 38_000,
    }
    const completedResponse = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => completed,
    })
    await expect(
      analyzeRecording(file, {
        endpoint: 'https://research.example.test/infer',
        recordingSource: 'livelink-upload',
        reanimatedSmileApplicable: false,
        fetcher: completedResponse,
      }),
    ).rejects.toThrow(/reanimated.*clinician/i)
  })

  it('propagates network failure without returning demonstration data', async () => {
    const fetcher = vi.fn().mockRejectedValue(new Error('offline'))
    const file = new File(['synthetic'], 'faces-protocol.webm', { type: 'video/webm' })

    await expect(
      analyzeRecording(file, {
        endpoint: 'https://research.example.test/infer',
        recordingSource: 'browser-camera',
        reanimatedSmileApplicable: true,
        fetcher,
      }),
    ).rejects.toThrow(/offline/i)
  })
})
