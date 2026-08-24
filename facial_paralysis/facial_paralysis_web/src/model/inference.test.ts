import { describe, expect, it, vi } from 'vitest'

import {
  EXPECTED_MODEL_ID,
  EXPECTED_RELEASE_MANIFEST_SHA256,
  InferenceContractError,
  analyzeRecording,
  parseCaptureTimelineSidecar,
  parseInferenceResponse,
  type CaptureTimelineDraft,
} from './inference'

const actionIds = [
  'repose', 'eyebrow_raise', 'gentle_eye_closure', 'tight_eye_squeeze',
  'relaxed_smile', 'lip_pucker', 'lower_teeth_show', 'reanimated_smile',
] as const

const timeline = (includeOptional = true): CaptureTimelineDraft => ({
  recordingDurationMs: (includeOptional ? 8 : 7) * 4_000,
  actions: actionIds.slice(0, includeOptional ? 8 : 7).map((id, index) => ({
    id,
    promptStartMs: index * 4_000,
    holdStartMs: index * 4_000 + 500,
    holdEndMs: index * 4_000 + 3_500,
    completionMs: index * 4_000 + 3_750,
  })),
})

const response = (includeOptional = true) => ({
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
    actions_used: includeOptional ? 7 : 6,
    optional_actions_unavailable: includeOptional ? [] : ['reanimated_smile'],
    actions: actionIds.slice(1, includeOptional ? 8 : 7).map((id, index) => ({
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

  it('posts the medically valid seven-step script without imputing step eight', async () => {
    const file = new File(['synthetic'], 'faces.webm', { type: 'video/webm' })
    const fetcher = vi.fn().mockResolvedValue({ ok: true, json: async () => response(false) })
    const parsed = await analyzeRecording(file, {
      endpoint: '/api/v1/facial-paralysis/infer',
      recordingSource: 'browser-camera',
      reanimatedSmileApplicable: false,
      timeline: timeline(false),
      fetcher,
    })
    expect(parsed.quality.actionsUsed).toBe(6)
    expect(parsed.quality.optionalActionsUnavailable).toEqual(['reanimated_smile'])
    const body = fetcher.mock.calls[0][1].body as FormData
    const manifest = JSON.parse(String(body.get('manifest')))
    const postedTimeline = JSON.parse(String(body.get('timeline')))
    expect(manifest.reanimated_smile_applicable).toBe(false)
    expect(postedTimeline.actions).toHaveLength(7)
    expect(postedTimeline.actions.at(-1).action).toBe('lower_teeth_show')
  })

  it('preserves an audited Mayo audio-aligned timing source instead of relabelling it', async () => {
    const file = new File(['mayo-audio-aligned-fixture'], 'faces.mp4', { type: 'video/mp4' })
    const digestBytes = await crypto.subtle.digest('SHA-256', await file.arrayBuffer())
    const digest = Array.from(new Uint8Array(digestBytes), (value) => value.toString(16).padStart(2, '0')).join('')
    const draft = timeline(false)
    const source = JSON.stringify({
      schema_version: 'faces-action-timeline/v1',
      script_version: 'faces-script/24-004956-v1',
      recording_sha256: digest,
      timing_source: 'audio_forced_alignment',
      recording_duration_ms: draft.recordingDurationMs,
      actions: draft.actions.map((row) => ({
        action: row.id === 'repose' ? 'neutral_repose' : row.id,
        status: 'completed',
        prompt_start_ms: row.promptStartMs,
        hold_start_ms: row.holdStartMs,
        hold_end_ms: row.holdEndMs,
        completion_ms: row.completionMs,
      })),
    })
    const acceptedResponse = response(false)
    acceptedResponse.preprocessing.timing_source = 'audio_forced_alignment'
    const fetcher = vi.fn().mockResolvedValue({ ok: true, json: async () => acceptedResponse })
    await analyzeRecording(file, {
      endpoint: '/api/v1/facial-paralysis/infer',
      recordingSource: 'livelink-upload',
      reanimatedSmileApplicable: false,
      timeline: parseCaptureTimelineSidecar(source),
      fetcher,
    })
    const body = fetcher.mock.calls[0][1].body as FormData
    expect(String(body.get('timeline'))).toBe(source)
  })
})
