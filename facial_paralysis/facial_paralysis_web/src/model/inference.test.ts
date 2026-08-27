import { describe, expect, it, vi } from 'vitest'

import {
  EXPECTED_MODEL_ID,
  EXPECTED_RELEASE_MANIFEST_SHA256,
  INFERENCE_TIMEOUT_MS,
  InferenceContractError,
  analyzeRecording,
  checkResearchEndpoint,
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
  schema_version: 'facial-paralysis-shared-v9-inference/v3',
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
    interpretation: 'class_1_research_score_only',
    endpoint_semantics: 'meei_facial_palsy_vs_healthy_control_development_head',
    class_0_label: 'meei_healthy_control',
    class_1_label: 'meei_facial_palsy',
  },
  report_evidence: {
    normalization: 'original_view_centered_eye_axis_aligned_interocular_scaled',
    interpretation: 'measured_movement_observation_not_causal_or_severity',
    context_frame_method: 'registered_hold_midpoint_not_model_selected',
    attribution: {
      method: 'integrated_gradients_shared_action_tokens',
      baseline: 'within_recording_neutral_clinical_zero_dense_response',
      scope: 'action_region_model_influence_not_landmark_causality',
      integration_steps: 32,
      max_completeness_error: 0.005,
    },
    actions: actionIds.slice(1, includeOptional ? 8 : 7).map((id, index) => ({
      id,
      region: index === 0 ? 'brow' : index < 3 ? 'eye' : 'mouth',
      context_frame_ms: index * 4_000 + 6_000,
      observations: ([
        ['brow_height_asymmetry_iod', 'brow_height_change_from_rest_iod'],
        ['eye_aperture_asymmetry_iod', 'residual_eye_aperture_iod', 'eye_closure_change_from_rest_iod'],
        ['eye_aperture_asymmetry_iod', 'residual_eye_aperture_iod', 'eye_closure_change_from_rest_iod'],
        ['mouth_corner_vertical_asymmetry_iod', 'mouth_corner_vertical_change_from_rest_iod'],
        ['mouth_corner_horizontal_asymmetry_iod', 'mouth_width_change_from_rest_iod'],
        ['mouth_corner_vertical_asymmetry_iod', 'lower_lip_change_from_rest_iod', 'mouth_open_change_from_rest_iod'],
        ['mouth_corner_vertical_asymmetry_iod', 'mouth_corner_vertical_change_from_rest_iod'],
      ][index] as string[]).map((metric, metricIndex) => ({
        metric,
        value: 0.01 * (metricIndex + 1),
        unit: 'interocular_distance',
      })),
      model_influence: index === 6 ? {
        status: 'unavailable',
        reason: 'stability_gate_failed',
      } : {
        status: 'stable',
        direction: index === 2 ? 'toward_class_0' : 'toward_class_1',
        strength: index < 2 ? 'strong' : index < 4 ? 'moderate' : 'smaller',
        relative_magnitude: Math.max(0.1, 1 - index * 0.15),
      },
      stability: {
        ensemble_sign_agreement: index === 6 ? 2 : 3,
        mirror_consistent: index !== 6,
        temporal_checks_passed: index === 6 ? 1 : 2,
      },
    })),
  },
  clinical_use_eligible: false,
})

describe('Shared V9 inference contract', () => {
  it('accepts only the exact pinned Shared V9 readiness response', async () => {
    const fetcher = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({
        status: 'ready',
        model_id: EXPECTED_MODEL_ID,
        candidate_id: 'BLV9-009',
        ensemble_members: 3,
        preprocessing: 'faces-to-shared-v9/v1',
      }),
    })
    await expect(checkResearchEndpoint('/api/v1/facial-paralysis/infer', fetcher)).resolves.toBeUndefined()
    expect(fetcher).toHaveBeenCalledWith(
      '/api/v1/facial-paralysis/ready',
      expect.objectContaining({
        method: 'GET', credentials: 'same-origin', cache: 'no-store', redirect: 'error',
        signal: expect.any(AbortSignal),
      }),
    )

    fetcher.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => ({
        status: 'ready', model_id: 'wrong-model', candidate_id: 'BLV9-009', ensemble_members: 3,
        preprocessing: 'faces-to-shared-v9/v1',
      }),
    })
    await expect(checkResearchEndpoint('/api/v1/facial-paralysis/infer', fetcher)).rejects.toThrow(/does not match the required deployment/i)
  })

  it('reports readiness transport and HTTP failures without exposing server text', async () => {
    const fetcher = vi.fn().mockRejectedValueOnce(new Error('/private/patient connection refused'))
    await expect(checkResearchEndpoint('/api/v1/facial-paralysis/infer', fetcher)).rejects.toThrow(/could not be reached/i)

    fetcher.mockResolvedValueOnce({
      ok: false,
      status: 503,
      text: async () => JSON.stringify({ detail: { code: 'model_not_ready' } }),
    })
    await expect(checkResearchEndpoint('/api/v1/facial-paralysis/infer', fetcher)).rejects.toThrow(/not ready/i)
  })

  it('accepts only the pinned Shared V9 binary research result', () => {
    const parsed = parseInferenceResponse(response())
    expect(parsed.model.candidateId).toBe('BLV9-009')
    expect(parsed.prediction.probability).toBe(0.73)
    expect(parsed.quality.actions).toHaveLength(7)
    expect(parsed.reportEvidence.actions[0].observations[1].metric).toBe('brow_height_change_from_rest_iod')
    expect(parsed.reportEvidence.actions[0].modelInfluence).toMatchObject({
      status: 'stable', direction: 'toward_class_1', strength: 'strong',
    })
    expect(parsed.reportEvidence.actions[6].modelInfluence).toEqual({
      status: 'unavailable', reason: 'stability_gate_failed',
    })
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

    const causal = response()
    ;(causal.report_evidence.actions[0] as Record<string, unknown>).caused_score = true
    expect(() => parseInferenceResponse(causal)).toThrow(/unknown/i)
  })

  it.each([
    [0, 0], [0.499, 0], [0.5, 1], [1, 1],
  ] as const)('accepts class-1 research score %s at the frozen threshold', (score, predictedClass) => {
    const candidate = response()
    candidate.prediction.probability = score
    candidate.prediction.member_probabilities = [score, score, score]
    candidate.prediction.predicted_class = predictedClass
    expect(parseInferenceResponse(candidate).prediction).toMatchObject({
      probability: score,
      predictedClass,
      interpretation: 'class_1_research_score_only',
    })
  })

  it('rejects nonfinite, negative, misaligned, and expanded descriptive evidence', () => {
    const negative = response()
    negative.report_evidence.actions[0].observations[0].value = -0.1
    expect(() => parseInferenceResponse(negative)).toThrow(/nonnegative/i)

    const misaligned = response()
    misaligned.report_evidence.actions[0].context_frame_ms += 1
    expect(() => parseInferenceResponse(misaligned)).toThrow(/midpoint/i)

    const wrongMetric = response()
    wrongMetric.report_evidence.actions[0].observations[0].metric = 'affected_side'
    expect(() => parseInferenceResponse(wrongMetric)).toThrow(/metric/i)

    const fabricatedStable = response()
    fabricatedStable.report_evidence.actions[0].stability.ensemble_sign_agreement = 2
    expect(() => parseInferenceResponse(fabricatedStable)).toThrow(/stable/i)

    const rawLogit = response()
    ;(rawLogit.report_evidence.actions[0].model_influence as Record<string, unknown>).raw_logit = 1.5
    expect(() => parseInferenceResponse(rawLogit)).toThrow(/unknown/i)
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
    expect(init.headers).toEqual(expect.objectContaining({
      'Idempotency-Key': expect.stringMatching(/^[0-9a-f]{64}$/),
    }))
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

  it('turns a safe preprocessing gate code into actionable recording guidance', async () => {
    const file = new File(['synthetic'], 'faces.webm', { type: 'video/webm' })
    const fetcher = vi.fn().mockResolvedValue({
      ok: false,
      status: 422,
      text: async () => JSON.stringify({
        detail: {
          code: 'face_tracking_insufficient',
          action: 'lower_teeth_show',
          valid_samples: 25,
          required_samples: 26,
        },
      }),
    })

    await expect(analyzeRecording(file, {
      endpoint: '/api/v1/facial-paralysis/infer',
      recordingSource: 'browser-camera',
      reanimatedSmileApplicable: false,
      timeline: timeline(false),
      fetcher,
    })).rejects.toThrow(/show bottom teeth.*25 of 26.*keep the full face and neck visible/i)
  })

  it('does not surface an unrecognized server error payload', async () => {
    const file = new File(['synthetic'], 'faces.webm', { type: 'video/webm' })
    const fetcher = vi.fn().mockResolvedValue({
      ok: false,
      status: 422,
      text: async () => JSON.stringify({ detail: { code: 'private_path_/patient/name' } }),
    })

    await expect(analyzeRecording(file, {
      endpoint: '/api/v1/facial-paralysis/infer',
      recordingSource: 'browser-camera',
      reanimatedSmileApplicable: false,
      timeline: timeline(false),
      fetcher,
    })).rejects.toThrow('Research endpoint returned HTTP 422. No result was accepted.')
  })

  it.each([
    [400, 'invalid_capture_request', /recording request was incomplete.*same recording/i],
    [400, 'video_required', /no video data was received.*same recording/i],
    [413, 'video_too_large', /larger than 512 MB/i],
    [415, 'multipart_required', /upload format was not accepted/i],
    [428, 'idempotency_key_required', /request identity was missing/i],
    [409, 'idempotency_key_conflict', /request identity did not match/i],
    [422, 'capture_evidence_invalid', /video and action timeline did not match/i],
    [422, 'video_format_unsupported', /video format is not supported/i],
    [422, 'video_frame_rate_too_low', /frame rate is too low/i],
    [422, 'video_dimensions_unsupported', /video dimensions are not supported/i],
    [422, 'video_timing_mismatch', /timing did not match.*without pausing/i],
    [422, 'video_decode_failed', /could not be decoded reliably/i],
    [422, 'face_geometry_invalid', /facial geometry could not be measured reliably/i],
    [422, 'preprocessing_failed', /did not pass the preprocessing checks/i],
    [502, 'inference_unavailable', /model service could not complete.*same recording/i],
    [503, 'model_not_ready', /model service is not ready.*before recording again/i],
    [500, 'gateway_unavailable', /processing service could not complete.*same recording/i],
  ] as const)('maps HTTP %s code %s to closed recovery guidance', async (status, code, message) => {
    const file = new File(['synthetic'], 'faces.webm', { type: 'video/webm' })
    const fetcher = vi.fn().mockResolvedValue({
      ok: false,
      status,
      text: async () => JSON.stringify({ detail: { code } }),
    })
    await expect(analyzeRecording(file, {
      endpoint: '/api/v1/facial-paralysis/infer',
      recordingSource: 'browser-camera',
      reanimatedSmileApplicable: false,
      timeline: timeline(false),
      fetcher,
    })).rejects.toThrow(message)
  })

  it('bounds and strictly rejects malformed or expanded server error bodies', async () => {
    const file = new File(['synthetic'], 'faces.webm', { type: 'video/webm' })
    for (const source of [
      '{bad json',
      JSON.stringify({ detail: { code: 'video_decode_failed', leaked_path: '/private/patient.mov' } }),
      'x'.repeat(4_097),
    ]) {
      const fetcher = vi.fn().mockResolvedValue({
        ok: false,
        status: 422,
        text: async () => source,
      })
      await expect(analyzeRecording(file, {
        endpoint: '/api/v1/facial-paralysis/infer',
        recordingSource: 'browser-camera',
        reanimatedSmileApplicable: false,
        timeline: timeline(false),
        fetcher,
      })).rejects.toThrow('Research endpoint returned HTTP 422. No result was accepted.')
    }
  })

  it('turns a network failure into retry guidance without echoing the exception', async () => {
    const file = new File(['synthetic'], 'faces.webm', { type: 'video/webm' })
    const fetcher = vi.fn().mockRejectedValue(new Error('/private/patient-name connection reset'))
    await expect(analyzeRecording(file, {
      endpoint: '/api/v1/facial-paralysis/infer',
      recordingSource: 'browser-camera',
      reanimatedSmileApplicable: false,
      timeline: timeline(false),
      fetcher,
    })).rejects.toThrow(/could not reach the research endpoint.*same recording/i)
  })

  it('bounds a hung inference request and preserves retry guidance', async () => {
    vi.useFakeTimers()
    try {
      const file = new File(['synthetic'], 'faces.webm', { type: 'video/webm' })
      const fetcher = vi.fn((_url: RequestInfo | URL, init?: RequestInit) => new Promise<Response>((_resolve, reject) => {
        init?.signal?.addEventListener('abort', () => reject(new DOMException('aborted private request', 'AbortError')))
      }))
      const pending = analyzeRecording(file, {
        endpoint: '/api/v1/facial-paralysis/infer',
        recordingSource: 'browser-camera',
        reanimatedSmileApplicable: false,
        timeline: timeline(false),
        fetcher,
      })
      await vi.waitFor(() => expect(fetcher).toHaveBeenCalledTimes(1))
      const rejection = expect(pending).rejects.toThrow(/timed out.*same recording.*retry/i)
      await vi.advanceTimersByTimeAsync(INFERENCE_TIMEOUT_MS)
      await rejection
    } finally {
      vi.useRealTimers()
    }
  })

  it.each([
    [0, /recording is empty/i],
    [512 * 1024 * 1024 + 1, /larger than 512 MB/i],
  ] as const)('refuses camera or upload files with unusable size %s before fetch', async (size, message) => {
    const file = new File(['synthetic'], 'faces.webm', { type: 'video/webm' })
    Object.defineProperty(file, 'size', { configurable: true, value: size })
    const fetcher = vi.fn()
    await expect(analyzeRecording(file, {
      endpoint: '/api/v1/facial-paralysis/infer',
      recordingSource: 'browser-camera',
      reanimatedSmileApplicable: false,
      timeline: timeline(false),
      fetcher,
    })).rejects.toThrow(message)
    expect(fetcher).not.toHaveBeenCalled()
  })
})
