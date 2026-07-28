import { describe, expect, it } from 'vitest'
import {
  evaluatePatientReportEligibility,
  listReviewQueueItems,
  selectExactResultTarget,
} from './reviewPolicy'
import { CONNECTED_INFERENCE_WATERMARK } from './inferenceEnvelope'
import {
  createApprovedReviewState,
  createSucceededReviewTarget,
  targetReference,
} from './reviewTestFixtures'
import { createInferenceBinding, runMockEngine } from './mockEngine'
import { workspaceReducer } from './reducer'
import type {
  ConnectedInferenceOutput,
  InferenceBinding,
  InferenceOutput,
  WorkspaceState,
} from './types'

function connectedOutput(output: ReturnType<typeof runMockEngine>): ConnectedInferenceOutput {
  return {
    ...output,
    attentionSemantics: {
      ...output.attentionSemantics,
      clinicalAoi: {
        ...output.attentionSemantics.clinicalAoi,
        registration: 'registration_geometry_unavailable_v1',
      },
    },
    origin: 'model_prediction',
    capabilityStatus: 'research_unvalidated',
    watermark: CONNECTED_INFERENCE_WATERMARK,
    resultDigest: `connected_${output.resultDigest}`,
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
      engineVersion: 'review-policy-test',
      canonicalSyntheticAsset: true,
      deterministic: false,
      networkAccessed: true,
      storageAccessed: false,
      observedGazePayloadIncluded: false,
      trainingDataProvenance: 'not_disclosed',
    },
  }
}

type RuntimeStateCorruptor = (
  state: WorkspaceState,
  runId: string,
  attemptId: string,
) => WorkspaceState

const malformedRuntimeStateCases: readonly [
  label: string,
  corrupt: RuntimeStateCorruptor,
][] = [
  [
    'null runsById map',
    (state) => ({ ...state, runsById: null }) as unknown as WorkspaceState,
  ],
  [
    'null attemptsById map',
    (state) => ({ ...state, attemptsById: null }) as unknown as WorkspaceState,
  ],
  [
    'null reviewsById map',
    (state) => ({ ...state, reviewsById: null }) as unknown as WorkspaceState,
  ],
  [
    'null reviewOrder',
    (state) => ({ ...state, reviewOrder: null }) as unknown as WorkspaceState,
  ],
  [
    'null roisByCase map',
    (state) => ({ ...state, roisByCase: null }) as unknown as WorkspaceState,
  ],
  [
    'object-shaped attemptIds',
    (state, runId, attemptId) => ({
      ...state,
      runsById: {
        ...state.runsById,
        [runId]: {
          ...state.runsById[runId],
          attemptIds: { 0: attemptId, length: 1 },
        },
      },
    }) as unknown as WorkspaceState,
  ],
  [
    'run missing attemptIds',
    (state, runId) => ({
      ...state,
      runsById: {
        ...state.runsById,
        [runId]: Object.fromEntries(
          Object.entries(state.runsById[runId]!).filter(([key]) => key !== 'attemptIds'),
        ),
      },
    }) as unknown as WorkspaceState,
  ],
  [
    'run with an extra key',
    (state, runId) => ({
      ...state,
      runsById: {
        ...state.runsById,
        [runId]: { ...state.runsById[runId], unexpected: true },
      },
    }) as unknown as WorkspaceState,
  ],
  [
    'attempt missing attemptToken',
    (state, _runId, attemptId) => ({
      ...state,
      attemptsById: {
        ...state.attemptsById,
        [attemptId]: Object.fromEntries(
          Object.entries(state.attemptsById[attemptId]!).filter(
            ([key]) => key !== 'attemptToken',
          ),
        ),
      },
    }) as unknown as WorkspaceState,
  ],
  [
    'attempt with an extra key',
    (state, _runId, attemptId) => ({
      ...state,
      attemptsById: {
        ...state.attemptsById,
        [attemptId]: { ...state.attemptsById[attemptId], unexpected: true },
      },
    }) as unknown as WorkspaceState,
  ],
  [
    'activeAttemptId outside the run attempt list',
    (state, runId) => ({
      ...state,
      runsById: {
        ...state.runsById,
        [runId]: {
          ...state.runsById[runId],
          activeAttemptId: 'attempt-not-in-this-run',
        },
      },
    }) as unknown as WorkspaceState,
  ],
]

describe('review eligibility policy', () => {
  it.each(malformedRuntimeStateCases)(
    'fails closed without throwing across exact selection, queue, and detail policy for %s',
    (_label, corrupt) => {
      const approved = createApprovedReviewState()
      const state = corrupt(
        approved.state,
        approved.binding.clientRunId,
        approved.attemptId,
      )

      expect(() => selectExactResultTarget(state, approved.reference)).not.toThrow()
      const selected = selectExactResultTarget(state, approved.reference)
      expect(selected).toMatchObject({ ok: false })
      expect(selected.blockers.length).toBeGreaterThan(0)

      expect(() => listReviewQueueItems(state)).not.toThrow()
      expect(
        listReviewQueueItems(state).every(
          (item) => !item.canCreateReview && !item.patientPreviewEligible,
        ),
      ).toBe(true)

      expect(() =>
        evaluatePatientReportEligibility(state, approved.reviewId),
      ).not.toThrow()
      const patient = evaluatePatientReportEligibility(state, approved.reviewId)
      expect(patient.eligible).toBe(false)
      expect(patient.blockers.length).toBeGreaterThan(0)
    },
  )

  it('selects only an own-property, active, succeeded, current immutable target', () => {
    const { state, binding, attemptId } = createSucceededReviewTarget()
    const reference = targetReference(binding, attemptId)

    expect(selectExactResultTarget(state, reference).ok).toBe(true)
    expect(
      selectExactResultTarget(state, { ...reference, attemptId: 'toString' }).ok,
    ).toBe(false)
    expect(
      selectExactResultTarget(state, {
        ...reference,
        resultDigest: 'result_different',
      }).ok,
    ).toBe(false)
  })

  it.each([
    [
      'asset SHA',
      (output: InferenceOutput) => ({
        ...output,
        binding: { ...output.binding, assetSha256: 'sha256_corrupt' },
      }),
    ],
    [
      'ROI identity',
      (output: InferenceOutput) => ({
        ...output,
        binding: { ...output.binding, roiId: 'roi-corrupt' },
      }),
    ],
    [
      'ROI version',
      (output: InferenceOutput) => ({
        ...output,
        binding: { ...output.binding, roiVersion: output.binding.roiVersion + 1 },
      }),
    ],
    [
      'ROI geometry',
      (output: InferenceOutput) => ({
        ...output,
        binding: {
          ...output.binding,
          roiGeometry: {
            ...output.binding.roiGeometry,
            x: output.binding.roiGeometry.x + 0.01,
          },
        },
      }),
    ],
    [
      'envelope watermark',
      (output: InferenceOutput) => ({ ...output, watermark: 'unsafe' }),
    ],
  ] as const)(
    'rejects exact-result selection when the stored %s is corrupted',
    (_field, corrupt) => {
      const seeded = createSucceededReviewTarget()
      const attempt = seeded.state.attemptsById[seeded.attemptId]!
      const state: WorkspaceState = {
        ...seeded.state,
        attemptsById: {
          ...seeded.state.attemptsById,
          [attempt.id]: {
            ...attempt,
            result: {
              ...attempt.result!,
              output: corrupt(attempt.result!.output) as InferenceOutput,
            },
          },
        },
      }

      expect(selectExactResultTarget(
        state,
        targetReference(seeded.binding, seeded.attemptId),
      ).ok).toBe(false)
      expect(listReviewQueueItems(state)[0]).toMatchObject({
        canCreateReview: false,
      })
    },
  )

  it.each([
    ['asset SHA', (state: WorkspaceState, _caseId: InferenceBinding['caseId']) => ({
      ...state,
      attemptsById: {
        ...state.attemptsById,
        [state.runsById['run-policy-001']!.activeAttemptId!]: {
          ...state.attemptsById[state.runsById['run-policy-001']!.activeAttemptId!]!,
          binding: {
            ...state.attemptsById[state.runsById['run-policy-001']!.activeAttemptId!]!.binding!,
            assetSha256: 'sha256_corrupt',
          },
        },
      },
    })],
    ['ROI identity', (state: WorkspaceState, caseId: InferenceBinding['caseId']) => ({
      ...state,
      roisByCase: {
        ...state.roisByCase,
        [caseId]: { ...state.roisByCase[caseId]!, id: 'roi-corrupt' },
      },
    })],
    ['ROI version', (state: WorkspaceState, caseId: InferenceBinding['caseId']) => ({
      ...state,
      roisByCase: {
        ...state.roisByCase,
        [caseId]: {
          ...state.roisByCase[caseId]!,
          version: state.roisByCase[caseId]!.version + 1,
        },
      },
    })],
    ['ROI geometry', (state: WorkspaceState, caseId: InferenceBinding['caseId']) => ({
      ...state,
      roisByCase: {
        ...state.roisByCase,
        [caseId]: {
          ...state.roisByCase[caseId]!,
          geometry: {
            ...state.roisByCase[caseId]!.geometry,
            x: state.roisByCase[caseId]!.geometry.x + 0.01,
          },
        },
      },
    })],
  ] as const)(
    'rejects exact-result selection when the current canonical %s no longer matches',
    (_field, corrupt) => {
      const seeded = createSucceededReviewTarget()
      const state = corrupt(seeded.state, seeded.binding.caseId) as WorkspaceState

      expect(selectExactResultTarget(
        state,
        targetReference(seeded.binding, seeded.attemptId),
      ).ok).toBe(false)
      expect(listReviewQueueItems(state)[0]?.canCreateReview).toBe(false)
    },
  )

  it('rejects valid-shaped deterministic mock tampering before review creation', () => {
    const seeded = createSucceededReviewTarget()
    const attempt = seeded.state.attemptsById[seeded.attemptId]!
    const firstPoint = attempt.result!.output.heatmap[0]!
    const state: WorkspaceState = {
      ...seeded.state,
      attemptsById: {
        ...seeded.state.attemptsById,
        [attempt.id]: {
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

    expect(selectExactResultTarget(
      state,
      targetReference(seeded.binding, seeded.attemptId),
    )).toMatchObject({
      ok: false,
      blockers: expect.arrayContaining([
        expect.objectContaining({ code: 'DETERMINISTIC_OUTPUT_MISMATCH' }),
      ]),
    })
    expect(listReviewQueueItems(state)[0]?.canCreateReview).toBe(false)
  })

  it('selects a complete connected envelope for internal review but blocks patient preview', () => {
    const approved = createApprovedReviewState()
    const attempt = approved.state.attemptsById[approved.attemptId]!
    const connected = connectedOutput(runMockEngine(approved.binding))
    const state: WorkspaceState = {
      ...approved.state,
      attemptsById: {
        ...approved.state.attemptsById,
        [attempt.id]: {
          ...attempt,
          result: { ...attempt.result!, output: connected },
        },
      },
      reviewsById: {
        ...approved.state.reviewsById,
        [approved.reviewId]: {
          ...approved.state.reviewsById[approved.reviewId]!,
          resultDigest: connected.resultDigest,
        },
      },
    }
    const reference = { ...approved.reference, resultDigest: connected.resultDigest }
    const reviewableState: WorkspaceState = {
      ...state,
      reviewsById: {},
      reviewOrder: [],
    }

    expect(selectExactResultTarget(reviewableState, reference).ok).toBe(true)
    expect(listReviewQueueItems(reviewableState)[0]).toMatchObject({
      canCreateReview: true,
      patientPreviewEligible: false,
    })
    const patient = evaluatePatientReportEligibility(state, approved.reviewId)
    expect(patient).toMatchObject({
      eligible: false,
      blockers: expect.arrayContaining([
        expect.objectContaining({ code: 'CONNECTED_OUTPUT_BLOCKED' }),
      ]),
    })
    expect(patient.blockers.map((entry) => entry.code)).not.toContain(
      'OUTPUT_ENVELOPE_INVALID',
    )
    expect(patient.blockers.map((entry) => entry.code)).not.toContain(
      'DETERMINISTIC_OUTPUT_MISMATCH',
    )
  })

  it('allows an approved current mock simulation with every research gate and structured note', () => {
    const { state, reviewId } = createApprovedReviewState()

    const result = evaluatePatientReportEligibility(state, reviewId)

    expect(result.eligible).toBe(true)
    if (result.eligible) {
      expect(result.output.origin).toBe('mock_simulation')
      expect(result.output.capabilityStatus).toBe('simulated_ui_only')
      expect(result.review.status).toBe('approved_for_research')
      expect(result.blockers).toEqual([])
    }
  })

  it('always blocks connected model output even if a review is marked approved', () => {
    const approved = createApprovedReviewState()
    const attempt = approved.state.attemptsById[approved.attemptId]!
    const mock = attempt.result!.output
    const connected = {
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
      watermark:
        'MODEL PREDICTION — RESEARCH UNVALIDATED — NOT HUMAN GAZE — CLINICAL USE BLOCKED',
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
        engineVersion: 'policy-test',
        canonicalSyntheticAsset: true,
        deterministic: true,
        networkAccessed: true,
        storageAccessed: false,
        observedGazePayloadIncluded: false,
        trainingDataProvenance: 'not_disclosed',
      },
    } as ConnectedInferenceOutput
    const state: WorkspaceState = {
      ...approved.state,
      attemptsById: {
        ...approved.state.attemptsById,
        [attempt.id]: { ...attempt, result: { ...attempt.result!, output: connected } },
      },
    }

    const result = evaluatePatientReportEligibility(state, approved.reviewId)

    expect(result.eligible).toBe(false)
    expect(result.blockers.map((blocker) => blocker.code)).toContain(
      'CONNECTED_OUTPUT_BLOCKED',
    )
  })

  it('fails closed after result revocation or review revocation', () => {
    const approved = createApprovedReviewState()
    const resultRevoked = workspaceReducer(approved.state, {
      type: 'result/revoke',
      runId: approved.binding.clientRunId,
      attemptId: approved.attemptId,
    })
    const reviewRevoked = workspaceReducer(approved.state, {
      type: 'review/revoke',
      reviewId: approved.reviewId,
      actorId: 'demo_reviewer',
      note: {
        rationale: 'Research display withdrawn.',
        limitations: 'Patient preview remains blocked.',
      },
    })

    expect(evaluatePatientReportEligibility(resultRevoked, approved.reviewId)).toMatchObject({
      eligible: false,
      blockers: expect.arrayContaining([
        expect.objectContaining({ code: 'RESULT_NOT_CURRENT' }),
      ]),
    })
    expect(evaluatePatientReportEligibility(reviewRevoked, approved.reviewId)).toMatchObject({
      eligible: false,
      blockers: expect.arrayContaining([
        expect.objectContaining({ code: 'REVIEW_REVOKED' }),
      ]),
    })
  })

  it('fails closed for missing, prototype-key, malformed-note, and failed quality-gate reviews', () => {
    const approved = createApprovedReviewState()
    const review = approved.state.reviewsById[approved.reviewId]!
    const attempt = approved.state.attemptsById[approved.attemptId]!
    const malformed: WorkspaceState = {
      ...approved.state,
      reviewsById: {
        ...approved.state.reviewsById,
        [review.id]: {
          ...review,
          events: review.events.map((event, index) =>
            index === review.events.length - 1
              ? { ...event, note: { ...event.note, limitations: '   ' } }
              : event,
          ),
        },
      },
      attemptsById: {
        ...approved.state.attemptsById,
        [attempt.id]: {
          ...attempt,
          result: {
            ...attempt.result!,
            output: {
              ...attempt.result!.output,
              qualityGates: {
                ...attempt.result!.output.qualityGates,
                researchDisplayEligible: false,
              },
            },
          },
        },
      },
    }

    expect(evaluatePatientReportEligibility(approved.state, 'missing').eligible).toBe(false)
    expect(evaluatePatientReportEligibility(approved.state, 'constructor').eligible).toBe(false)
    expect(evaluatePatientReportEligibility(malformed, approved.reviewId)).toMatchObject({
      eligible: false,
      blockers: expect.arrayContaining([
        expect.objectContaining({ code: 'REVIEW_NOTE_INVALID' }),
        expect.objectContaining({ code: 'OUTPUT_ENVELOPE_INVALID' }),
      ]),
    })
  })

  it('lists exact succeeded attempts as review candidates and exposes patient blockers', () => {
    const { state, binding, attemptId } = createSucceededReviewTarget()

    const queue = listReviewQueueItems(state)

    expect(queue).toHaveLength(1)
    expect(queue[0]).toMatchObject({
      runId: binding.clientRunId,
      attemptId,
      reviewId: undefined,
      canCreateReview: true,
      patientPreviewEligible: false,
    })
    expect(queue[0].blockers.map((blocker) => blocker.code)).toContain(
      'REVIEW_REQUIRED',
    )
  })

  it.each(['failed', 'blocked', 'cancelled'] as const)(
    'lists an active %s attempt as a blocker without review target placeholders',
    (status) => {
      const seeded = createSucceededReviewTarget()
      const attempt = seeded.state.attemptsById[seeded.attemptId]!
      const run = seeded.state.runsById[seeded.binding.clientRunId]!
      const state: WorkspaceState = {
        ...seeded.state,
        runsById: {
          ...seeded.state.runsById,
          [run.clientRunId]: { ...run, status },
        },
        attemptsById: {
          ...seeded.state.attemptsById,
          [attempt.id]: { ...attempt, status, result: undefined },
        },
      }

      const queue = listReviewQueueItems(state)

      expect(queue).toHaveLength(1)
      expect(queue[0]).toMatchObject({
        runId: run.clientRunId,
        attemptId: attempt.id,
        status,
        reviewId: undefined,
        resultDigest: undefined,
        inputFingerprint: undefined,
        canCreateReview: false,
        patientPreviewEligible: false,
      })
      expect(queue[0].blockers.map((entry) => entry.code)).toContain(
        'RESULT_NOT_SUCCEEDED',
      )
    },
  )

  it.each([
    ['watermark', (output: Record<string, unknown>) => ({ ...output, watermark: 'unsafe' })],
    ['heatmap', (output: Record<string, unknown>) => ({ ...output, heatmap: [] })],
    [
      'legacy metrics',
      (output: Record<string, unknown>) => ({
        ...output,
        metrics: {
          roiCoverage: 0.5,
          peakIntensity: 0.5,
          meanIntensity: 0.5,
          focusScore: 0.5,
        },
      }),
    ],
    [
      'binding',
      (output: Record<string, unknown>) => ({
        ...output,
        binding: { ...(output.binding as object), assetSha256: 'sha256_corrupt' },
      }),
    ],
    [
      'provenance',
      (output: Record<string, unknown>) => ({
        ...output,
        provenance: { ...(output.provenance as object), networkAccessed: true },
      }),
    ],
  ] as const)(
    'revalidates the complete mock output envelope when %s is corrupted',
    (_field, corrupt) => {
      const approved = createApprovedReviewState()
      const attempt = approved.state.attemptsById[approved.attemptId]!
      const state: WorkspaceState = {
        ...approved.state,
        attemptsById: {
          ...approved.state.attemptsById,
          [attempt.id]: {
            ...attempt,
            result: {
              ...attempt.result!,
              output: corrupt(
                attempt.result!.output as unknown as Record<string, unknown>,
              ) as unknown as InferenceOutput,
            },
          },
        },
      }

      expect(evaluatePatientReportEligibility(state, approved.reviewId)).toMatchObject({
        eligible: false,
        blockers: expect.arrayContaining([
          expect.objectContaining({ code: 'OUTPUT_ENVELOPE_INVALID' }),
        ]),
      })
    },
  )

  it('rejects corrupt review identity and a lifecycle that does not begin with author submission', () => {
    const approved = createApprovedReviewState()
    const review = approved.state.reviewsById[approved.reviewId]!
    const corruptReview = {
      ...review,
      id: 'different-review-id',
      authorId: 'demo_reviewer',
      reviewerId: 'demo_author',
      events: review.events.map((event, index) =>
        index === 0
          ? { ...event, decision: 'approved_for_research', actorId: 'demo_reviewer' }
          : event,
      ),
    } as unknown as typeof review
    const state: WorkspaceState = {
      ...approved.state,
      reviewsById: {
        ...approved.state.reviewsById,
        [approved.reviewId]: corruptReview,
      },
    }

    expect(evaluatePatientReportEligibility(state, approved.reviewId)).toMatchObject({
      eligible: false,
      blockers: expect.arrayContaining([
        expect.objectContaining({ code: 'REVIEW_IDENTITY_INVALID' }),
        expect.objectContaining({ code: 'REVIEW_EVENT_INVALID' }),
      ]),
    })
  })

  it('rejects an otherwise well-formed review event sequence with an illegal transition', () => {
    const approved = createApprovedReviewState()
    const review = approved.state.reviewsById[approved.reviewId]!
    const state: WorkspaceState = {
      ...approved.state,
      reviewsById: {
        ...approved.state.reviewsById,
        [review.id]: {
          ...review,
          status: 'changes_requested',
          decision: 'changes_requested',
          events: [
            ...review.events,
            {
              sequence: 3,
              decision: 'changes_requested',
              actorId: 'demo_reviewer',
              note: {
                rationale: 'Illegal transition after approval.',
                limitations: 'Must remain blocked.',
              },
            },
          ],
        },
      },
    }

    expect(evaluatePatientReportEligibility(state, approved.reviewId)).toMatchObject({
      eligible: false,
      blockers: expect.arrayContaining([
        expect.objectContaining({ code: 'REVIEW_EVENT_INVALID' }),
      ]),
    })
  })

  it.each([
    [
      'quality gates',
      (output: Record<string, unknown>) => {
        const { qualityGates: _qualityGates, ...rest } = output
        return rest
      },
    ],
    [
      'provenance',
      (output: Record<string, unknown>) => {
        const { provenance: _provenance, ...rest } = output
        return rest
      },
    ],
  ] as const)(
    'returns an envelope blocker without throwing when %s is missing',
    (_field, corrupt) => {
      const approved = createApprovedReviewState()
      const attempt = approved.state.attemptsById[approved.attemptId]!
      const state: WorkspaceState = {
        ...approved.state,
        attemptsById: {
          ...approved.state.attemptsById,
          [attempt.id]: {
            ...attempt,
            result: {
              ...attempt.result!,
              output: corrupt(
                attempt.result!.output as unknown as Record<string, unknown>,
              ) as unknown as InferenceOutput,
            },
          },
        },
      }

      expect(() =>
        evaluatePatientReportEligibility(state, approved.reviewId),
      ).not.toThrow()
      expect(evaluatePatientReportEligibility(state, approved.reviewId)).toMatchObject({
        eligible: false,
        blockers: expect.arrayContaining([
          expect.objectContaining({ code: 'OUTPUT_ENVELOPE_INVALID' }),
        ]),
      })
    },
  )

  it('fails closed without throwing when output binding or review events are structurally absent', () => {
    const approved = createApprovedReviewState()
    const attempt = approved.state.attemptsById[approved.attemptId]!
    const review = approved.state.reviewsById[approved.reviewId]!
    const state: WorkspaceState = {
      ...approved.state,
      attemptsById: {
        ...approved.state.attemptsById,
        [attempt.id]: {
          ...attempt,
          result: {
            ...attempt.result!,
            output: {
              ...attempt.result!.output,
              binding: null,
            } as unknown as InferenceOutput,
          },
        },
      },
      reviewsById: {
        ...approved.state.reviewsById,
        [review.id]: {
          ...review,
          events: undefined,
        } as unknown as typeof review,
      },
    }

    expect(() =>
      evaluatePatientReportEligibility(state, approved.reviewId),
    ).not.toThrow()
    expect(evaluatePatientReportEligibility(state, approved.reviewId).eligible).toBe(false)
  })

  it('keeps equal-digest results on distinct runs independently reviewable and eligible', () => {
    const first = createApprovedReviewState()
    const secondBinding = createInferenceBinding({
      clientRunId: 'run-policy-equivalent-002',
      attemptToken: 'token-policy-equivalent-002',
      caseId: first.asset.id,
      assetId: first.asset.id,
      assetSha256: first.asset.sha256,
      roi: first.roi,
      modelVersion: first.binding.modelVersion,
      modelMode: first.binding.modelMode,
      config: first.binding.config,
    })
    const secondAttemptId = 'attempt-policy-equivalent-002'
    const asyncBinding = {
      runId: secondBinding.clientRunId,
      attemptId: secondAttemptId,
      attemptToken: secondBinding.attemptToken,
      inputFingerprint: secondBinding.inputFingerprint,
    }
    let state = [
      {
        type: 'run/create',
        runId: secondBinding.clientRunId,
        attemptId: secondAttemptId,
        binding: secondBinding,
      },
      { type: 'run/validate', ...asyncBinding },
      { type: 'run/queue', ...asyncBinding },
      { type: 'run/start', ...asyncBinding },
      {
        type: 'run/succeed',
        ...asyncBinding,
        output: runMockEngine(secondBinding),
      },
    ].reduce(
      (current, action) => workspaceReducer(current, action as Parameters<typeof workspaceReducer>[1]),
      first.state,
    )

    expect(runMockEngine(secondBinding).resultDigest).toBe(first.reference.resultDigest)
    expect(secondBinding.inputFingerprint).toBe(first.binding.inputFingerprint)
    const queue = listReviewQueueItems(state)
    expect(queue.find((item) => item.runId === first.binding.clientRunId)).toMatchObject({
      reviewId: first.reviewId,
      patientPreviewEligible: true,
    })
    expect(queue.find((item) => item.runId === secondBinding.clientRunId)).toMatchObject({
      reviewId: undefined,
      canCreateReview: true,
      patientPreviewEligible: false,
    })

    state = workspaceReducer(state, {
      type: 'review/create',
      reviewId: 'review-2',
      runId: secondBinding.clientRunId,
      attemptId: secondAttemptId,
      resultDigest: runMockEngine(secondBinding).resultDigest,
      inputFingerprint: secondBinding.inputFingerprint,
      actorId: 'demo_author',
      note: {
        rationale: 'Second exact run is independently reviewable.',
        limitations: 'An equal digest does not share authorization.',
      },
    })
    state = workspaceReducer(state, {
      type: 'review/approve',
      reviewId: 'review-2',
      actorId: 'demo_reviewer',
      note: {
        rationale: 'Second exact run reviewed independently.',
        limitations: 'Research demonstration only.',
      },
    })

    expect(state.reviewOrder).toEqual([first.reviewId, 'review-2'])
    expect(evaluatePatientReportEligibility(state, first.reviewId).eligible).toBe(true)
    expect(evaluatePatientReportEligibility(state, 'review-2').eligible).toBe(true)
  })

  it('blocks review, queue, and patient preview when the current source binding is partial', () => {
    const approved = createApprovedReviewState()
    const roi = approved.state.roisByCase[approved.binding.caseId]!
    const state: WorkspaceState = {
      ...approved.state,
      roisByCase: {
        ...approved.state.roisByCase,
        [approved.binding.caseId]: {
          ...roi,
          geometry: { x: 0.05, y: 0.05, width: 0.9, height: 0.9 },
        },
      },
    }

    expect(selectExactResultTarget(state, approved.reference)).toMatchObject({
      ok: false,
      blockers: expect.arrayContaining([
        expect.objectContaining({
          code: 'FULL_IMAGE_SOURCE_BINDING_REQUIRED',
        }),
      ]),
    })
    expect(listReviewQueueItems(state)).toEqual([
      expect.objectContaining({
        canCreateReview: false,
        patientPreviewEligible: false,
        blockers: expect.arrayContaining([
          expect.objectContaining({
            code: 'FULL_IMAGE_SOURCE_BINDING_REQUIRED',
          }),
        ]),
      }),
    ])
    expect(evaluatePatientReportEligibility(state, approved.reviewId)).toMatchObject({
      eligible: false,
      blockers: expect.arrayContaining([
        expect.objectContaining({
          code: 'FULL_IMAGE_SOURCE_BINDING_REQUIRED',
        }),
      ]),
    })
  })

  it('blocks valid-shaped mock output when its digest or values differ from the deterministic engine', () => {
    const approved = createApprovedReviewState()
    const attempt = approved.state.attemptsById[approved.attemptId]!
    const firstPoint = attempt.result!.output.heatmap[0]!
    const state: WorkspaceState = {
      ...approved.state,
      attemptsById: {
        ...approved.state.attemptsById,
        [attempt.id]: {
          ...attempt,
          result: {
            ...attempt.result!,
            output: {
              ...attempt.result!.output,
              resultDigest: 'Patient name hidden in digest',
              heatmap: [
                { ...firstPoint, intensity: firstPoint.intensity === 1 ? 0.99 : 1 },
                ...attempt.result!.output.heatmap.slice(1),
              ],
            },
          },
        },
      },
      reviewsById: {
        ...approved.state.reviewsById,
        [approved.reviewId]: {
          ...approved.state.reviewsById[approved.reviewId]!,
          resultDigest: 'Patient name hidden in digest',
        },
      },
    }

    expect(evaluatePatientReportEligibility(state, approved.reviewId)).toMatchObject({
      eligible: false,
      blockers: expect.arrayContaining([
        expect.objectContaining({ code: 'DETERMINISTIC_OUTPUT_MISMATCH' }),
      ]),
    })
  })

  it('fails closed without throwing when a review event array contains null', () => {
    const approved = createApprovedReviewState()
    const review = approved.state.reviewsById[approved.reviewId]!
    const state: WorkspaceState = {
      ...approved.state,
      reviewsById: {
        ...approved.state.reviewsById,
        [review.id]: {
          ...review,
          events: [...review.events, null],
        } as unknown as typeof review,
      },
    }

    expect(() =>
      evaluatePatientReportEligibility(state, approved.reviewId),
    ).not.toThrow()
    expect(evaluatePatientReportEligibility(state, approved.reviewId)).toMatchObject({
      eligible: false,
      blockers: expect.arrayContaining([
        expect.objectContaining({ code: 'REVIEW_EVENT_INVALID' }),
      ]),
    })
  })
})
