import { describe, expect, it, vi } from 'vitest'

import {
  EXPECTED_MODEL_ID,
  EXPECTED_RELEASE_MANIFEST_SHA256,
  InferenceContractError,
  analyzeRecording,
  parseInferenceResponse,
  type CaptureTimelineDraft,
} from './inference'

const actionIds = [
  'repose', 'eyebrow_raise', 'gentle_eye_closure', 'tight_eye_squeeze',
  'relaxed_smile', 'lip_pucker', 'lower_teeth_show', 'reanimated_smile',
] as const

const timeline = (): CaptureTimelineDraft => ({
  recordingDurationMs: 32_000,
  actions: actionIds.map((id, index) => ({
    id,
    promptStartMs: index * 4_000,
    holdStartMs: index * 4_000 + 500,
    holdEndMs: index * 4_000 + 3_500,
    completionMs: index * 4_000 + 3_750,
  })),
})

const response = () => ({
  schema_version: 'facial-paralysis-shared-v9-inference/v1',
  model: {
    model_id: EXPECTED_MODEL_ID,
    candidate_id: 'BLV9-009',
    release_manifest_sha256: EXPECTED_RELEASE_MANIFEST_SHA256,
    ensemble_members: 3,
  },
  preprocessing: {
    version: 'faces-to-shared-v9/v1',
    face_landmarker_sha256: '64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff',
    mirror_method: 'horizontal_flip_and_redetect',
    protocol: 'cue_aligned_action',
    timing_source: 'capture_event_log',
  },
  quality: {
    eligible: true,
    actions_used: 7,
    actions: actionIds.slice(1).map((id, index) => ({
      id,
      v9_action: ['BROW_RAISE', 'EYE_GENTLE', 'EYE_FORCEFUL', 'SMILE_GENTLE', 'LIP_PUCKER', 'SHOW_BOTTOM_TEETH', 'SMILE_FULL'][index],
      hold_start_ms: index * 4_000 + 4_500,
      hold_end_ms: index * 4_000 + 7_500,
      valid_samples: 32,
    })),
  },
  prediction: {
    probability: 0.73,
    member_probabilities: [0.71, 0.74, 0.74],
    predicted_class: 1,
    threshold: 0.5,
    interpretation: 'research_score_only',
  },
  clinical_use_eligible: false,
})

describe('Shared V9 inference contract', () => {
  it('accepts only the pinned Shared V9 binary research result', () => {
    const parsed = parseInferenceResponse(response())
    expect(parsed.model.candidateId).toBe('BLV9-009')
    expect(parsed.prediction.probability).toBe(0.73)
    expect(parsed.quality.actions).toHaveLength(7)
    expect(parsed.clinicalUseEligible).toBe(false)
  })

  it('rejects identity drift, extra clinical fields, and inconsistent ensemble means', () => {
    const wrongModel = response()
    ;(wrongModel.model as { model_id: string }).model_id = 'wrong'
    expect(() => parseInferenceResponse(wrongModel)).toThrow(InferenceContractError)

    const extra = response() as Record<string, unknown>
    extra.hb_grade = 3
    expect(() => parseInferenceResponse(extra)).toThrow(/unknown/i)

    const inconsistent = response()
    inconsistent.prediction.probability = 0.9
    expect(() => parseInferenceResponse(inconsistent)).toThrow(/ensemble/i)
  })

  it('hashes exact video bytes and posts the manifest plus external timeline', async () => {
    const fetcher = vi.fn().mockResolvedValue({ ok: true, json: async () => response() })
    const file = new File(['synthetic'], 'faces.webm', { type: 'video/webm' })
    await analyzeRecording(file, {
      endpoint: '/api/v1/facial-paralysis/infer',
      recordingSource: 'browser-camera',
      reanimatedSmileApplicable: true,
      timeline: timeline(),
      fetcher,
    })
    const [url, init] = fetcher.mock.calls[0]
    expect(url).toBe('/api/v1/facial-paralysis/infer')
    expect(init.credentials).toBe('same-origin')
    const body = init.body as FormData
    const manifest = JSON.parse(String(body.get('manifest')))
    const postedTimeline = JSON.parse(String(body.get('timeline')))
    expect(manifest.schema_version).toBe('faces-v9-capture-manifest/v1')
    expect(manifest.video_sha256).toMatch(/^[0-9a-f]{64}$/)
    expect(postedTimeline.recording_sha256).toBe(manifest.video_sha256)
    expect(postedTimeline.actions[0].action).toBe('neutral_repose')
    expect(postedTimeline.actions[7].action).toBe('reanimated_smile')
  })

  it('fails before upload when the eighth movement or exact timeline is absent', async () => {
    const file = new File(['synthetic'], 'faces.webm', { type: 'video/webm' })
    const fetcher = vi.fn()
    await expect(analyzeRecording(file, {
      endpoint: '/api/v1/facial-paralysis/infer',
      recordingSource: 'browser-camera',
      reanimatedSmileApplicable: false,
      timeline: timeline(),
      fetcher,
    })).rejects.toThrow(/seven active movements/i)
    expect(fetcher).not.toHaveBeenCalled()
  })
})
