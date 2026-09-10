import { describe, expect, it } from 'vitest'
import { DEMO_PATIENT_RECORDS } from '../data/demoPatientRecords'
import { createInitialPatientWorkflowState } from './reducer'
import { selectPatientComparisonState } from './selectors'
import type {
  AuthorizationSnapshot,
  CaptureAsset,
  PatientFaceRegistration,
  PatientResult,
  PatientRun,
  PatientRunBinding,
  PatientTimepoint,
  PatientVisit,
  PatientWorkflowState,
} from './types'
import {
  createAuthorizationSnapshotId,
  createCaptureAssetId,
  createPatientResultId,
  createPatientRunId,
  createPatientVisitId,
  createSessionMediaHandle,
} from './validation'

const PATIENT = DEMO_PATIENT_RECORDS[0]!

function visit(
  suffix: string,
  timepoint: PatientTimepoint,
  visitDate = '2026-07-20',
  createdAt = `${visitDate}T10:00:00.000Z`,
): PatientVisit {
  return {
    id: createPatientVisitId(`visit-comparison-${suffix}`),
    patientId: PATIENT.id,
    timepoint,
    visitDate,
    createdAt,
  }
}

function withVisits(
  visits: readonly PatientVisit[],
): PatientWorkflowState {
  const state = createInitialPatientWorkflowState(
    [PATIENT],
    '2026-07-27',
  )
  return {
    ...state,
    visitsById: Object.fromEntries(
      visits.map((item) => [item.id, item]),
    ),
    visitOrder: visits.map((item) => item.id),
  }
}

function completeVisitEntities(item: PatientVisit, suffix: string) {
  const capture: CaptureAsset = {
    id: createCaptureAssetId(`capture-comparison-${suffix}`),
    patientId: item.patientId,
    visitId: item.id,
    version: 1,
    status: 'current',
    source: 'upload',
    mediaHandle: createSessionMediaHandle(`media_comparison_${suffix}`),
    sha256: suffix.padEnd(64, 'a').slice(0, 64),
    mimeType: 'image/png',
    sizeBytes: 1_024,
    width: 1_024,
    height: 1_024,
    captureProtocol: 'frontal_relaxed_non_mirrored_v1',
    qualityChecks: {
      faceVisibleAndCentered: true,
      focusLightingAndOcclusionAcceptable: true,
      orientationConfirmed: true,
      authorizationDocumented: true,
    },
    capturedAt: '2026-07-27T10:01:00.000Z',
    qualityConfirmedAt: '2026-07-27T10:02:00.000Z',
  }
  const authorization: AuthorizationSnapshot = {
    id: createAuthorizationSnapshotId(
      `authorization-comparison-${suffix}`,
    ),
    patientId: item.patientId,
    visitId: item.id,
    revision: 1,
    status: 'documented',
    recordedAt: '2026-07-27T10:02:00.000Z',
  }
  const binding: PatientRunBinding = {
    patientId: capture.patientId,
    visitId: capture.visitId,
    captureId: capture.id,
    captureVersion: capture.version,
    captureSha256: capture.sha256,
    mediaHandle: capture.mediaHandle,
    authorizationRevision: authorization.revision,
    captureProtocol: capture.captureProtocol,
  }
  const run: PatientRun = {
    id: createPatientRunId(`run-comparison-${suffix}`),
    status: 'succeeded',
    binding,
    createdAt: '2026-07-27T10:03:00.000Z',
  }
  const points = [
    { x: 0.3, y: 0.3 },
    { x: 0.5, y: 0.7 },
    { x: 0.7, y: 0.3 },
  ] as const
  const faceRegistration: PatientFaceRegistration = {
    schemaVersion: 'patient-face-registration/1',
    source: 'on_device_face_landmarks',
    coordinateSpace: 'decoded_image_normalized_v1',
    captureSha256: binding.captureSha256,
    sourceWidth: capture.width,
    sourceHeight: capture.height,
    captureProtocol: binding.captureProtocol,
    detectorId: 'mediapipe_face_landmarker',
    detectorVersion: 'tasks-vision-1.0.0-model-float16-1',
    faceCount: 1,
    paths: [
      { feature: 'face_oval', closed: true, points },
      { feature: 'left_eye', closed: true, points },
      { feature: 'right_eye', closed: true, points },
      { feature: 'left_eyebrow', closed: false, points },
      { feature: 'right_eyebrow', closed: false, points },
      { feature: 'lips', closed: true, points },
    ],
  }
  const result: PatientResult = {
    id: createPatientResultId(`result-comparison-${suffix}`),
    runId: run.id,
    binding,
    freshness: 'current',
    createdAt: '2026-07-27T10:04:00.000Z',
    faceRegistration,
    output: {
      origin: 'workflow_simulation',
      points: [
        { x: 0.45, y: 0.5, intensity: 0.8, radius: 0.1 },
      ],
    },
  }
  return { capture, authorization, run, result }
}

function withEntities(
  state: PatientWorkflowState,
  entities: readonly ReturnType<typeof completeVisitEntities>[],
): PatientWorkflowState {
  return {
    ...state,
    capturesById: Object.fromEntries(
      entities.map(({ capture }) => [capture.id, capture]),
    ),
    captureOrder: entities.map(({ capture }) => capture.id),
    authorizationsById: Object.fromEntries(
      entities.map(({ authorization }) => [
        authorization.id,
        authorization,
      ]),
    ),
    authorizationOrder: entities.map(
      ({ authorization }) => authorization.id,
    ),
    runsById: Object.fromEntries(
      entities.map(({ run }) => [run.id, run]),
    ),
    runOrder: entities.map(({ run }) => run.id),
    resultsById: Object.fromEntries(
      entities.map(({ result }) => [result.id, result]),
    ),
    resultOrder: entities.map(({ result }) => result.id),
  }
}

describe('selectPatientComparisonState', () => {
  it('returns no_visits only when the patient has no visits', () => {
    const state = withVisits([])
    expect(selectPatientComparisonState(state, PATIENT.id)).toEqual({
      phase: 'no_visits',
    })

    const followUp = visit('follow-up', 'follow_up')
    expect(
      selectPatientComparisonState(withVisits([followUp]), PATIENT.id),
    ).toEqual({
      phase: 'missing_timepoint',
      missing: ['preoperative', 'postoperative'],
    })
  })

  it('requires both longitudinal timepoints before inspecting photos', () => {
    const preoperative = visit('pre', 'preoperative')
    expect(
      selectPatientComparisonState(
        withVisits([preoperative]),
        PATIENT.id,
      ),
    ).toEqual({
      phase: 'missing_timepoint',
      missing: ['postoperative'],
    })
  })

  it('reports exactly which current photos are missing', () => {
    const preoperative = visit('pre', 'preoperative')
    const postoperative = visit('post', 'postoperative')
    const visits = withVisits([preoperative, postoperative])
    expect(selectPatientComparisonState(visits, PATIENT.id)).toMatchObject({
      phase: 'needs_photos',
      missingPhotos: ['preoperative', 'postoperative'],
    })

    const pre = completeVisitEntities(preoperative, 'pre')
    expect(
      selectPatientComparisonState(withEntities(visits, [pre]), PATIENT.id),
    ).toMatchObject({
      phase: 'needs_photos',
      missingPhotos: ['postoperative'],
    })
  })

  it('requires both results bound to the two current captures', () => {
    const preoperative = visit('pre', 'preoperative')
    const postoperative = visit('post', 'postoperative')
    const pre = completeVisitEntities(preoperative, 'pre')
    const post = completeVisitEntities(postoperative, 'post')
    const state = withEntities(
      withVisits([preoperative, postoperative]),
      [pre, post],
    )
    const withoutPostResult: PatientWorkflowState = {
      ...state,
      resultsById: { [pre.result.id]: pre.result },
      resultOrder: [pre.result.id],
    }

    expect(
      selectPatientComparisonState(withoutPostResult, PATIENT.id),
    ).toMatchObject({
      phase: 'needs_results',
      missingResults: ['postoperative'],
    })
    expect(selectPatientComparisonState(state, PATIENT.id)).toMatchObject({
      phase: 'ready',
      pair: {
        preoperative: { result: { id: pre.result.id } },
        postoperative: { result: { id: post.result.id } },
      },
    })
  })

  it.each([
    ['stale result', (result: PatientResult) => ({ ...result, freshness: 'stale' as const })],
    [
      'superseded-capture binding',
      (result: PatientResult) => ({
        ...result,
        binding: {
          ...result.binding,
          captureVersion: result.binding.captureVersion - 1,
        },
      }),
    ],
    [
      'foreign-patient binding',
      (result: PatientResult) => ({
        ...result,
        binding: {
          ...result.binding,
          patientId: DEMO_PATIENT_RECORDS[1]!.id,
        },
      }),
    ],
  ])('rejects a %s as the current result', (_label, mutateResult) => {
    const preoperative = visit('pre', 'preoperative')
    const postoperative = visit('post', 'postoperative')
    const pre = completeVisitEntities(preoperative, 'pre')
    const post = completeVisitEntities(postoperative, 'post')
    const base = withEntities(
      withVisits([preoperative, postoperative]),
      [pre, post],
    )
    const invalidPostResult = mutateResult(post.result)
    const state: PatientWorkflowState = {
      ...base,
      resultsById: {
        [pre.result.id]: pre.result,
        [invalidPostResult.id]: invalidPostResult,
      },
    }

    expect(selectPatientComparisonState(state, PATIENT.id)).toMatchObject({
      phase: 'needs_results',
      missingResults: ['postoperative'],
    })
  })

  it('rejects a consistently foreign capture, authorization, run, and result chain', () => {
    const preoperative = visit('pre', 'preoperative')
    const postoperative = visit('post', 'postoperative')
    const pre = completeVisitEntities(preoperative, 'pre')
    const post = completeVisitEntities(postoperative, 'post')
    const foreignPatientId = DEMO_PATIENT_RECORDS[1]!.id
    const foreignCapture = {
      ...post.capture,
      patientId: foreignPatientId,
    }
    const foreignAuthorization = {
      ...post.authorization,
      patientId: foreignPatientId,
    }
    const foreignRun = {
      ...post.run,
      binding: {
        ...post.run.binding,
        patientId: foreignPatientId,
      },
    }
    const foreignResult = {
      ...post.result,
      binding: {
        ...post.result.binding,
        patientId: foreignPatientId,
      },
    }
    const base = withEntities(
      withVisits([preoperative, postoperative]),
      [pre, post],
    )
    const state: PatientWorkflowState = {
      ...base,
      capturesById: {
        [pre.capture.id]: pre.capture,
        [foreignCapture.id]: foreignCapture,
      },
      authorizationsById: {
        [pre.authorization.id]: pre.authorization,
        [foreignAuthorization.id]: foreignAuthorization,
      },
      runsById: {
        [pre.run.id]: pre.run,
        [foreignRun.id]: foreignRun,
      },
      resultsById: {
        [pre.result.id]: pre.result,
        [foreignResult.id]: foreignResult,
      },
    }

    expect(selectPatientComparisonState(state, PATIENT.id)).toMatchObject({
      phase: 'needs_photos',
      missingPhotos: ['postoperative'],
    })
  })

  it('rejects a foreign authorization even when capture and result bindings look current', () => {
    const preoperative = visit('pre', 'preoperative')
    const postoperative = visit('post', 'postoperative')
    const pre = completeVisitEntities(preoperative, 'pre')
    const post = completeVisitEntities(postoperative, 'post')
    const base = withEntities(
      withVisits([preoperative, postoperative]),
      [pre, post],
    )
    const state: PatientWorkflowState = {
      ...base,
      authorizationsById: {
        [pre.authorization.id]: pre.authorization,
        [post.authorization.id]: {
          ...post.authorization,
          patientId: DEMO_PATIENT_RECORDS[1]!.id,
        },
      },
    }

    expect(selectPatientComparisonState(state, PATIENT.id)).toMatchObject({
      phase: 'needs_results',
      missingResults: ['postoperative'],
    })
  })

  it('lets a newer incomplete visit block an older complete visit', () => {
    const olderPre = visit(
      'pre-old',
      'preoperative',
      '2026-07-10',
    )
    const newerPre = visit(
      'pre-new',
      'preoperative',
      '2026-07-25',
    )
    const postoperative = visit('post', 'postoperative', '2026-07-26')
    const old = completeVisitEntities(olderPre, 'pre-old')
    const post = completeVisitEntities(postoperative, 'post')
    const state = withEntities(
      withVisits([olderPre, postoperative, newerPre]),
      [old, post],
    )

    expect(selectPatientComparisonState(state, PATIENT.id)).toMatchObject({
      phase: 'needs_photos',
      missingPhotos: ['preoperative'],
      pair: { preoperative: { id: newerPre.id } },
    })
  })

  it('uses createdAt and then id to break equal-date ties deterministically', () => {
    const first = visit(
      'pre-a',
      'preoperative',
      '2026-07-20',
      '2026-07-20T09:00:00.000Z',
    )
    const laterCreated = visit(
      'pre-b',
      'preoperative',
      '2026-07-20',
      '2026-07-20T10:00:00.000Z',
    )
    const laterId = visit(
      'pre-z',
      'preoperative',
      '2026-07-20',
      '2026-07-20T10:00:00.000Z',
    )
    const postoperative = visit('post', 'postoperative')

    expect(
      selectPatientComparisonState(
        withVisits([laterId, first, postoperative, laterCreated]),
        PATIENT.id,
      ),
    ).toMatchObject({
      phase: 'needs_photos',
      pair: { preoperative: { id: laterId.id } },
    })
  })

  it('orders equal-date visits by the real timestamp across UTC offsets', () => {
    const olderInstant = visit(
      'pre-z-offset',
      'preoperative',
      '2026-11-01',
      '2026-11-01T01:30:00-04:00',
    )
    const newerInstant = visit(
      'pre-a-offset',
      'preoperative',
      '2026-11-01',
      '2026-11-01T01:15:00-05:00',
    )
    const postoperative = visit(
      'post-offset',
      'postoperative',
      '2026-11-01',
    )

    expect(
      selectPatientComparisonState(
        withVisits([newerInstant, postoperative, olderInstant]),
        PATIENT.id,
      ),
    ).toMatchObject({
      phase: 'needs_photos',
      pair: { preoperative: { id: newerInstant.id } },
    })
  })
})
