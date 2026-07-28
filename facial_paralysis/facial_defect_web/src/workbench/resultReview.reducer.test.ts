import { describe, expect, it } from 'vitest'
import { listWorkbenchAssets } from './catalog'
import { CONNECTED_INFERENCE_WATERMARK } from './inferenceEnvelope'
import { createInferenceBinding, runMockEngine } from './mockEngine'
import {
  createInitialWorkspaceState,
  getCaseRoi,
  workspaceReducer,
} from './reducer'
import type {
  ApprovedRoiAnnotation,
  ConnectedInferenceOutput,
  InferenceBinding,
  InferenceOutput,
  WorkspaceAction,
  WorkspaceState,
} from './types'

function connectedOutput(binding: InferenceBinding): ConnectedInferenceOutput {
  const mock = runMockEngine(binding)
  return {
    ...mock,
    attentionSemantics: {
      ...mock.attentionSemantics,
      clinicalAoi: {
        ...mock.attentionSemantics.clinicalAoi,
        registration: 'registration_geometry_unavailable_v1',
      },
    },
    origin: 'model_prediction',
    capabilityStatus: 'research_unvalidated',
    watermark: CONNECTED_INFERENCE_WATERMARK,
    resultDigest: `connected_${mock.resultDigest}`,
    modelIdentity: {
      modelId: 'observer-attention-test',
      modelVersion: 'test-v1',
      artifactSha256: 'a'.repeat(64),
      preprocessingVersion: 'preprocess-v1',
      calibrationVersion: 'calibration-v1',
      displayScaleId: 'display-scale-v1',
    },
    provenance: {
      engine: 'connected_model_gateway',
      engineVersion: 'result-review-test',
      canonicalSyntheticAsset: true,
      deterministic: false,
      networkAccessed: true,
      storageAccessed: false,
      observedGazePayloadIncluded: false,
      trainingDataProvenance: 'not_disclosed',
    },
  }
}

function reduce(state: WorkspaceState, actions: readonly WorkspaceAction[]) {
  return actions.reduce(workspaceReducer, state)
}

function succeededState() {
  let state = createInitialWorkspaceState()
  const asset = listWorkbenchAssets()[2]
  const roi = getCaseRoi(state, asset.id) as ApprovedRoiAnnotation
  const binding = createInferenceBinding({
    clientRunId: 'run-review-001',
    attemptToken: 'token-review-001',
    caseId: asset.id,
    assetId: asset.id,
    assetSha256: asset.sha256,
    roi,
    modelVersion: 'mock-salience-v0.3',
    modelMode: 'mock_only',
    config: { threshold: 0.42, smoothing: 0.27 },
  })
  const asyncBinding = {
    runId: binding.clientRunId,
    attemptId: 'attempt-review-001',
    attemptToken: binding.attemptToken,
    inputFingerprint: binding.inputFingerprint,
  }
  state = reduce(state, [
    {
      type: 'run/create',
      runId: binding.clientRunId,
      attemptId: asyncBinding.attemptId,
      binding,
    },
    { type: 'run/validate', ...asyncBinding },
    { type: 'run/queue', ...asyncBinding },
    { type: 'run/start', ...asyncBinding },
    {
      type: 'run/succeed',
      ...asyncBinding,
      output: runMockEngine(binding),
    },
  ])
  return { state, binding, attemptId: asyncBinding.attemptId }
}

function createReviewAction(
  binding: InferenceBinding,
  attemptId: string,
  overrides: Record<string, unknown> = {},
) {
  return {
    type: 'review/create',
    reviewId: 'review-1',
    runId: binding.clientRunId,
    attemptId,
    resultDigest: runMockEngine(binding).resultDigest,
    inputFingerprint: binding.inputFingerprint,
    actorId: 'demo_author',
    note: {
      rationale: '  Suitable for a synthetic research discussion.  ',
      limitations: '  Simulated attention only; no clinical inference.  ',
    },
    ...overrides,
  } as WorkspaceAction
}

describe('result review reducer', () => {
  it('creates an awaiting review bound to the exact immutable result and trims its note', () => {
    const { state, binding, attemptId } = succeededState()

    const reviewed = workspaceReducer(state, createReviewAction(binding, attemptId))

    expect(reviewed.reviewOrder).toEqual(['review-1'])
    expect(reviewed.reviewsById['review-1']).toMatchObject({
      id: 'review-1',
      runId: binding.clientRunId,
      attemptId,
      resultDigest: runMockEngine(binding).resultDigest,
      inputFingerprint: binding.inputFingerprint,
      status: 'awaiting_review',
      authorId: 'demo_author',
      reviewerId: 'demo_reviewer',
      events: [
        {
          sequence: 1,
          decision: 'awaiting_review',
          actorId: 'demo_author',
          note: {
            rationale: 'Suitable for a synthetic research discussion.',
            limitations: 'Simulated attention only; no clinical inference.',
          },
        },
      ],
    })
  })

  it.each([
    [{ rationale: '   ', limitations: 'Known limitation.' }, 'rationale'],
    [{ rationale: 'Research rationale.', limitations: '\n\t' }, 'limitations'],
  ])('rejects an empty structured review note field', (note, field) => {
    const { state, binding, attemptId } = succeededState()

    const reviewed = workspaceReducer(
      state,
      createReviewAction(binding, attemptId, { note }),
    )

    expect(reviewed.reviewOrder).toEqual([])
    expect(reviewed.lastFailure).toMatchObject({
      reason: 'INVALID_REVIEW_NOTE',
      field,
    })
  })

  it('fails closed when any exact result target field is wrong or a prototype key', () => {
    const { state, binding, attemptId } = succeededState()
    const digestMismatch = workspaceReducer(
      state,
      createReviewAction(binding, attemptId, { resultDigest: 'result_wrong' }),
    )
    const fingerprintMismatch = workspaceReducer(
      state,
      createReviewAction(binding, attemptId, { inputFingerprint: 'input_wrong' }),
    )
    const prototypeAttempt = workspaceReducer(
      state,
      createReviewAction(binding, 'toString'),
    )

    expect(digestMismatch.reviewOrder).toEqual([])
    expect(digestMismatch.lastFailure?.reason).toBe('IMMUTABLE_BINDING_MISMATCH')
    expect(fingerprintMismatch.reviewOrder).toEqual([])
    expect(fingerprintMismatch.lastFailure?.reason).toBe('IMMUTABLE_BINDING_MISMATCH')
    expect(prototypeAttempt.reviewOrder).toEqual([])
    expect(prototypeAttempt.lastFailure?.reason).toBe('UNKNOWN_REVIEW_TARGET')
  })

  it.each([
    ['canonical asset SHA', ({ state, attemptId }: ReturnType<typeof succeededState>) => {
      const attempt = state.attemptsById[attemptId]!
      const corruptBinding = { ...attempt.binding!, assetSha256: 'sha256_corrupt' }
      return {
        ...state,
        attemptsById: {
          ...state.attemptsById,
          [attemptId]: {
            ...attempt,
            binding: corruptBinding,
            result: {
              ...attempt.result!,
              output: {
                ...attempt.result!.output,
                binding: corruptBinding,
              } as InferenceOutput,
            },
          },
        },
      }
    }],
    ['current ROI geometry', ({ state, binding }: ReturnType<typeof succeededState>) => ({
      ...state,
      roisByCase: {
        ...state.roisByCase,
        [binding.caseId]: {
          ...state.roisByCase[binding.caseId]!,
          geometry: {
            ...state.roisByCase[binding.caseId]!.geometry,
            x: state.roisByCase[binding.caseId]!.geometry.x + 0.01,
          },
        },
      },
    })],
    ['output envelope', ({ state, attemptId }: ReturnType<typeof succeededState>) => {
      const attempt = state.attemptsById[attemptId]!
      return {
        ...state,
        attemptsById: {
          ...state.attemptsById,
          [attemptId]: {
            ...attempt,
            result: {
              ...attempt.result!,
              output: {
                ...attempt.result!.output,
                watermark: 'unsafe',
              } as unknown as InferenceOutput,
            },
          },
        },
      }
    }],
    ['deterministic mock output', ({ state, attemptId }: ReturnType<typeof succeededState>) => {
      const attempt = state.attemptsById[attemptId]!
      const firstPoint = attempt.result!.output.heatmap[0]!
      return {
        ...state,
        attemptsById: {
          ...state.attemptsById,
          [attemptId]: {
            ...attempt,
            result: {
              ...attempt.result!,
              output: {
                ...attempt.result!.output,
                heatmap: [
                  { ...firstPoint, intensity: firstPoint.intensity === 1 ? 0.99 : 1 },
                  ...attempt.result!.output.heatmap.slice(1),
                ],
              },
            },
          },
        },
      }
    }],
  ] as const)(
    'rejects direct review creation when %s integrity is corrupt',
    (_field, corrupt) => {
      const seeded = succeededState()
      const corrupted = corrupt(seeded) as WorkspaceState

      const reviewed = workspaceReducer(
        corrupted,
        createReviewAction(seeded.binding, seeded.attemptId),
      )

      expect(reviewed.reviewOrder).toEqual([])
      expect(reviewed.lastFailure?.reason).toBe('IMMUTABLE_BINDING_MISMATCH')
    },
  )

  it('allows a valid connected result to enter and advance internal research review', () => {
    const seeded = succeededState()
    const attempt = seeded.state.attemptsById[seeded.attemptId]!
    const output = connectedOutput(seeded.binding)
    const connectedState: WorkspaceState = {
      ...seeded.state,
      attemptsById: {
        ...seeded.state.attemptsById,
        [attempt.id]: {
          ...attempt,
          result: { ...attempt.result!, output },
        },
      },
    }
    const awaiting = workspaceReducer(
      connectedState,
      createReviewAction(seeded.binding, seeded.attemptId, {
        resultDigest: output.resultDigest,
      }),
    )
    const approved = workspaceReducer(awaiting, {
      type: 'review/approve',
      reviewId: 'review-1',
      actorId: 'demo_reviewer',
      note: {
        rationale: 'Connected output reviewed for internal research use.',
        limitations: 'Patient preview and clinical use remain blocked.',
      },
    } as WorkspaceAction)

    expect(awaiting.reviewsById['review-1']?.status).toBe('awaiting_review')
    expect(approved.reviewsById['review-1']?.status).toBe('approved_for_research')
  })

  it('enforces independent reviewer approval and appends, rather than mutates, notes', () => {
    const { state, binding, attemptId } = succeededState()
    const awaiting = workspaceReducer(state, createReviewAction(binding, attemptId))
    const selfReview = workspaceReducer(awaiting, {
      type: 'review/approve',
      reviewId: 'review-1',
      actorId: 'demo_author',
      note: { rationale: 'I approve.', limitations: 'Still simulated.' },
    } as WorkspaceAction)
    const approved = workspaceReducer(awaiting, {
      type: 'review/approve',
      reviewId: 'review-1',
      actorId: 'demo_reviewer',
      note: {
        rationale: 'Independent research review completed.',
        limitations: 'Not validated for patient or clinical use.',
      },
    } as WorkspaceAction)

    expect(selfReview.reviewsById['review-1']?.status).toBe('awaiting_review')
    expect(selfReview.lastFailure?.reason).toBe('SELF_REVIEW_FORBIDDEN')
    expect(approved.reviewsById['review-1']?.status).toBe('approved_for_research')
    expect(approved.reviewsById['review-1']?.events).toHaveLength(2)
    expect(approved.reviewsById['review-1']?.events[0]).toBe(
      awaiting.reviewsById['review-1']?.events[0],
    )
  })

  it('supports request changes, author resubmission, reviewer approval, then reviewer revocation', () => {
    const { state, binding, attemptId } = succeededState()
    const awaiting = workspaceReducer(state, createReviewAction(binding, attemptId))
    const changes = workspaceReducer(awaiting, {
      type: 'review/requestChanges',
      reviewId: 'review-1',
      actorId: 'demo_reviewer',
      note: { rationale: 'Needs clearer scope.', limitations: 'Scope is ambiguous.' },
    } as WorkspaceAction)
    const resubmitted = workspaceReducer(changes, {
      type: 'review/resubmit',
      reviewId: 'review-1',
      actorId: 'demo_author',
      note: { rationale: 'Scope clarified.', limitations: 'Simulation remains limited.' },
    } as WorkspaceAction)
    const approved = workspaceReducer(resubmitted, {
      type: 'review/approve',
      reviewId: 'review-1',
      actorId: 'demo_reviewer',
      note: { rationale: 'Ready for research display.', limitations: 'Not clinical.' },
    } as WorkspaceAction)
    const revoked = workspaceReducer(approved, {
      type: 'review/revoke',
      reviewId: 'review-1',
      actorId: 'demo_reviewer',
      note: { rationale: 'Research display withdrawn.', limitations: 'Do not export.' },
    } as WorkspaceAction)

    expect(changes.reviewsById['review-1']?.status).toBe('changes_requested')
    expect(resubmitted.reviewsById['review-1']?.status).toBe('awaiting_review')
    expect(approved.reviewsById['review-1']?.status).toBe('approved_for_research')
    expect(revoked.reviewsById['review-1']?.status).toBe('revoked')
    expect(revoked.reviewsById['review-1']?.events.map((event) => event.sequence)).toEqual([
      1, 2, 3, 4, 5,
    ])
  })

  it('rejects duplicate review IDs and illegal lifecycle shortcuts', () => {
    const { state, binding, attemptId } = succeededState()
    const awaiting = workspaceReducer(state, createReviewAction(binding, attemptId))
    const collision = workspaceReducer(
      awaiting,
      createReviewAction(binding, attemptId, {
        note: { rationale: 'Different rationale.', limitations: 'Different limit.' },
      }),
    )
    const revokeAwaiting = workspaceReducer(awaiting, {
      type: 'review/revoke',
      reviewId: 'review-1',
      actorId: 'demo_reviewer',
      note: { rationale: 'Skip approval.', limitations: 'Illegal shortcut.' },
    } as WorkspaceAction)

    expect(collision.lastFailure?.reason).toBe('INVALID_OPERATIONAL_ID')
    expect(collision.reviewsById['review-1']?.events).toHaveLength(1)
    expect(revokeAwaiting.lastFailure?.reason).toBe('ILLEGAL_TRANSITION')
    expect(revokeAwaiting.reviewsById['review-1']?.status).toBe('awaiting_review')
  })
})
