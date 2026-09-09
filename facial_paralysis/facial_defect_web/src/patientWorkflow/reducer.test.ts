import { describe, expect, it } from 'vitest'
import { DEMO_PATIENT_RECORDS } from '../data/demoPatientRecords'
import {
  createInitialPatientWorkflowState,
  patientWorkflowReducer,
} from './reducer'
import {
  selectCurrentCapture,
  selectCurrentResult,
  selectCurrentRun,
  selectPatientNextAction,
  selectPatientVisits,
  selectVisitNextAction,
} from './selectors'
import type {
  AuthorizationSnapshot,
  CaptureAsset,
  CaptureQualityChecks,
  PatientRecord,
  PatientFaceRegistration,
  PatientId,
  PatientResult,
  PatientRun,
  PatientRunBinding,
  PatientVisit,
  PatientVisitId,
  PatientWorkflowAction,
  PatientWorkflowState,
} from './types'
import {
  createAuthorizationSnapshotId,
  createCaptureAssetId,
  createPatientId,
  createPatientResultId,
  createPatientReviewId,
  createPatientRunId,
  createPatientVisitId,
  createSessionMediaHandle,
  isSessionMediaHandle,
  normalizeRecordNumber,
  validatePatientDraft,
  validateVisitDraft,
} from './validation'
import * as patientValidation from './validation'

function assertDistinctEntityIdTypes(
  patientId: PatientId,
  visitId: PatientVisitId,
): void {
  // @ts-expect-error Patient and visit IDs must not be assignable across entities.
  const crossEntityPatientId: PatientId = visitId
  void patientId
  void crossEntityPatientId
}

void assertDistinctEntityIdTypes

const PATIENT_ID = createPatientId('patient-001')
const VISIT_ID = createPatientVisitId('visit-001')
const AUTHORIZATION_ID = createAuthorizationSnapshotId('authorization-001')
const CAPTURE_ID = createCaptureAssetId('capture-001')
const RUN_ID = createPatientRunId('run-001')
const RESULT_ID = createPatientResultId('result-001')

const COMPLETE_QUALITY: CaptureQualityChecks = {
  faceVisibleAndCentered: true,
  focusLightingAndOcclusionAcceptable: true,
  orientationConfirmed: true,
  authorizationDocumented: true,
}

const INCOMPLETE_QUALITY: CaptureQualityChecks = {
  faceVisibleAndCentered: false,
  focusLightingAndOcclusionAcceptable: false,
  orientationConfirmed: false,
  authorizationDocumented: false,
}

function makePatient(
  overrides: Omit<Partial<PatientRecord>, 'id'> & {
    readonly id?: string
  } = {},
): PatientRecord {
  return {
    id: PATIENT_ID,
    displayName: 'Synthetic Test Patient',
    recordNumber: 'TEST-001',
    dateOfBirth: '1980-04-03',
    carePathway: 'Facial paralysis',
    recordKind: 'session_test',
    createdAt: '2026-07-27T13:00:00.000Z',
    ...overrides,
  } as PatientRecord
}

function makeVisit(
  overrides: Omit<Partial<PatientVisit>, 'id' | 'patientId'> & {
    readonly id?: string
    readonly patientId?: string
  } = {},
): PatientVisit {
  return {
    id: VISIT_ID,
    patientId: PATIENT_ID,
    timepoint: 'preoperative',
    visitDate: '2026-07-27',
    createdAt: '2026-07-27T13:01:00.000Z',
    ...overrides,
  } as PatientVisit
}

function createPatientAction(
  patient = makePatient(),
  syntheticTestAttestation = true,
  trustedToday = '2026-07-27',
): Extract<PatientWorkflowAction, { readonly type: 'patient/create' }> {
  return {
    type: 'patient/create',
    patient,
    trustedToday,
    syntheticTestAttestation,
  }
}

function createVisitAction(
  visit = makeVisit(),
  trustedToday = '2026-07-27',
): Extract<PatientWorkflowAction, { readonly type: 'visit/create' }> {
  return {
    type: 'visit/create',
    visit,
    trustedToday,
  }
}

function makeAuthorization(
  overrides: Omit<
    Partial<AuthorizationSnapshot>,
    'id' | 'patientId' | 'visitId'
  > & {
    readonly id?: string
    readonly patientId?: string
    readonly visitId?: string
  } = {},
): AuthorizationSnapshot {
  return {
    id: AUTHORIZATION_ID,
    patientId: PATIENT_ID,
    visitId: VISIT_ID,
    revision: 1,
    status: 'documented',
    recordedAt: '2026-07-27T13:02:00.000Z',
    ...overrides,
  } as AuthorizationSnapshot
}

function makeCapture(
  overrides: Omit<
    Partial<CaptureAsset>,
    'id' | 'patientId' | 'visitId'
  > & {
    readonly id?: string
    readonly patientId?: string
    readonly visitId?: string
  } = {},
): CaptureAsset {
  return {
    id: CAPTURE_ID,
    patientId: PATIENT_ID,
    visitId: VISIT_ID,
    version: 1,
    status: 'current',
    source: 'upload',
    mediaHandle: createSessionMediaHandle('opaque_token_001'),
    sha256: 'a'.repeat(64),
    mimeType: 'image/png',
    sizeBytes: 1_024_000,
    width: 1024,
    height: 1024,
    captureProtocol: 'frontal_relaxed_non_mirrored_v1',
    qualityChecks: INCOMPLETE_QUALITY,
    capturedAt: '2026-07-27T13:03:00.000Z',
    ...overrides,
  } as CaptureAsset
}

function bindingFor(
  capture: CaptureAsset,
  authorization = makeAuthorization(),
): PatientRunBinding {
  return {
    patientId: capture.patientId,
    visitId: capture.visitId,
    captureId: capture.id,
    captureVersion: capture.version,
    captureSha256: capture.sha256,
    mediaHandle: capture.mediaHandle,
    authorizationRevision: authorization.revision,
    captureProtocol: capture.captureProtocol,
  }
}

function makeRun(
  capture = makeCapture(),
  overrides: Omit<Partial<PatientRun>, 'id' | 'retryOfRunId'> & {
    readonly id?: string
    readonly retryOfRunId?: string
  } = {},
): PatientRun {
  return {
    id: RUN_ID,
    status: 'queued',
    binding: bindingFor(capture),
    createdAt: '2026-07-27T13:04:00.000Z',
    ...overrides,
  } as PatientRun
}

function makeResult(
  run = makeRun(),
  overrides: Omit<Partial<PatientResult>, 'id' | 'runId'> & {
    readonly id?: string
    readonly runId?: string
  } = {},
): PatientResult {
  const facePoints = [
    { x: 0.3, y: 0.2 },
    { x: 0.5, y: 0.8 },
    { x: 0.7, y: 0.2 },
  ] as const
  const faceRegistration: PatientFaceRegistration = {
    schemaVersion: 'patient-face-registration/1',
    source: 'on_device_face_landmarks',
    coordinateSpace: 'decoded_image_normalized_v1',
    captureSha256: run.binding.captureSha256,
    sourceWidth: 1_024,
    sourceHeight: 1_024,
    captureProtocol: run.binding.captureProtocol,
    detectorId: 'mediapipe_face_landmarker',
    detectorVersion: 'tasks-vision-1.0.0-model-float16-1',
    faceCount: 1,
    paths: [
      { feature: 'face_oval', closed: true, points: facePoints },
      { feature: 'left_eye', closed: true, points: facePoints },
      { feature: 'right_eye', closed: true, points: facePoints },
      {
        feature: 'left_eyebrow',
        closed: false,
        points: facePoints,
      },
      {
        feature: 'right_eyebrow',
        closed: false,
        points: facePoints,
      },
      { feature: 'lips', closed: true, points: facePoints },
    ],
  }

  return {
    id: RESULT_ID,
    runId: run.id,
    binding: run.binding,
    freshness: 'current',
    createdAt: '2026-07-27T13:05:00.000Z',
    faceRegistration,
    output: {
      origin: 'workflow_simulation',
      points: [],
    },
    ...overrides,
  } as PatientResult
}

function reduce(
  state: PatientWorkflowState,
  ...actions: readonly PatientWorkflowAction[]
): PatientWorkflowState {
  return actions.reduce(patientWorkflowReducer, state)
}

function stateThroughCapture(): PatientWorkflowState {
  return reduce(
    createInitialPatientWorkflowState(),
    createPatientAction(),
    createVisitAction(),
    { type: 'authorization/record', authorization: makeAuthorization() },
    { type: 'capture/add', capture: makeCapture() },
  )
}

function stateReadyToRun(): PatientWorkflowState {
  return patientWorkflowReducer(stateThroughCapture(), {
    type: 'capture/quality/set',
    captureId: CAPTURE_ID,
    checks: COMPLETE_QUALITY,
    confirmedAt: '2026-07-27T13:03:30.000Z',
  })
}

function stateWithQueuedRun(): PatientWorkflowState {
  const ready = stateReadyToRun()
  const capture = selectCurrentCapture(ready, 'visit-001')!
  return patientWorkflowReducer(ready, {
    type: 'run/create',
    run: makeRun(capture),
  })
}

describe('patient and visit validation', () => {
  it('normalizes record numbers and rejects a normalized duplicate', () => {
    const existing = reduce(
      createInitialPatientWorkflowState(),
      createPatientAction(makePatient({ recordNumber: 'CASE-001' })),
    )

    expect(normalizeRecordNumber('  case   001  ')).toBe('CASE-001')
    expect(
      validatePatientDraft(
        {
          displayName: 'Another Synthetic Patient',
          recordNumber: ' case_001 ',
          dateOfBirth: '1990-03-02',
          carePathway: 'Facial paralysis',
          syntheticTestAttestation: true,
        },
        existing,
        '2026-07-27',
      ),
    ).toEqual({
      ok: false,
      errors: {
        recordNumber: 'Record number is already in use in this session.',
      },
    })
  })

  it('requires display name, record number, date of birth, and care pathway', () => {
    expect(
      validatePatientDraft(
        {
          displayName: ' ',
          recordNumber: '',
          dateOfBirth: '',
          carePathway: ' ',
          syntheticTestAttestation: true,
        },
        createInitialPatientWorkflowState(),
        '2026-07-27',
      ),
    ).toEqual({
      ok: false,
      errors: {
        displayName: 'Display name is required.',
        recordNumber: 'Record number is required.',
        dateOfBirth: 'Date of birth is required.',
        carePathway: 'Care pathway is required.',
      },
    })
  })

  it('rejects a future date of birth and returns a trimmed valid draft', () => {
    const state = createInitialPatientWorkflowState()

    expect(
      validatePatientDraft(
        {
          displayName: 'Future Patient',
          recordNumber: 'FUTURE-001',
          dateOfBirth: '2026-07-28',
          carePathway: 'Facial paralysis',
          syntheticTestAttestation: true,
        },
        state,
        '2026-07-27',
      ),
    ).toEqual({
      ok: false,
      errors: {
        dateOfBirth: 'Date of birth cannot be in the future.',
      },
    })

    expect(
      validatePatientDraft(
        {
          displayName: '  Synthetic Test Patient  ',
          recordNumber: ' demo 002 ',
          dateOfBirth: '1980-04-03',
          carePathway: '  Facial paralysis  ',
          syntheticTestAttestation: true,
        },
        state,
        '2026-07-27',
      ),
    ).toEqual({
      ok: true,
      value: {
        displayName: 'Synthetic Test Patient',
        recordNumber: 'DEMO-002',
        dateOfBirth: '1980-04-03',
        carePathway: 'Facial paralysis',
      },
    })
  })

  it('requires a supported visit timepoint and rejects a future visit date', () => {
    expect(
      validateVisitDraft(
        { timepoint: '', visitDate: '2026-07-28' },
        '2026-07-27',
      ),
    ).toEqual({
      ok: false,
      errors: {
        timepoint: 'Timepoint is required.',
        visitDate: 'Visit date cannot be in the future.',
      },
    })

    expect(
      validateVisitDraft(
        { timepoint: 'follow_up', visitDate: '2026-07-27' },
        '2026-07-27',
      ),
    ).toEqual({
      ok: true,
      value: { timepoint: 'follow_up', visitDate: '2026-07-27' },
    })
  })
})

describe('patient workflow ownership and capture integrity', () => {
  it('creates a normalized patient whose first next action is to start a visit', () => {
    const initial = createInitialPatientWorkflowState()
    const next = patientWorkflowReducer(
      initial,
      createPatientAction(
        makePatient({ recordNumber: '  test 001 ' }),
      ),
    )

    expect(next.patientsById['patient-001']).toMatchObject({
      recordNumber: 'TEST-001',
      displayName: 'Synthetic Test Patient',
    })
    expect(selectPatientNextAction(next, 'patient-001')).toBe('start_visit')
    expect(initial.patientsById).toEqual({})
  })

  it('rejects an ID with surrounding whitespace', () => {
    const initial = reduce(
      createInitialPatientWorkflowState(),
      createPatientAction(),
    )
    const duplicate = patientWorkflowReducer(
      initial,
      createPatientAction(
        makePatient({
          id: ' patient-001 ',
          recordNumber: 'TEST-002',
        }),
      ),
    )

    expect(duplicate.lastFailure?.code).toBe('INVALID_PATIENT')
    expect(duplicate.patientOrder).toEqual(['patient-001'])
    expect(duplicate.patientsById['patient-001']?.recordNumber).toBe('TEST-001')
  })

  it('rejects an unknown visit owner and capture ownership mismatch', () => {
    const initial = reduce(
      createInitialPatientWorkflowState(),
      createPatientAction(),
      createPatientAction(
        makePatient({
          id: 'patient-002',
          recordNumber: 'TEST-002',
        }),
      ),
    )
    const unknownVisit = patientWorkflowReducer(
      initial,
      createVisitAction(
        makeVisit({ patientId: createPatientId('patient-missing') }),
      ),
    )
    expect(unknownVisit.lastFailure?.code).toBe('UNKNOWN_PATIENT')
    expect(unknownVisit.visitsById).toEqual({})

    const withVisit = patientWorkflowReducer(initial, createVisitAction())
    const mismatchedCapture = patientWorkflowReducer(withVisit, {
      type: 'capture/add',
      capture: makeCapture({ patientId: 'patient-002' }),
    })
    expect(mismatchedCapture.lastFailure?.code).toBe(
      'VISIT_OWNERSHIP_MISMATCH',
    )
    expect(mismatchedCapture.capturesById).toEqual({})
  })

  it('supports initial and subsequent visits only with explicit timepoints', () => {
    const state = reduce(
      createInitialPatientWorkflowState(),
      createPatientAction(),
      createVisitAction(),
      createVisitAction(
        makeVisit({
          id: 'visit-002',
          timepoint: 'postoperative',
          visitDate: '2026-07-28',
          createdAt: '2026-07-28T13:01:00.000Z',
        }),
        '2026-07-28',
      ),
    )

    expect(selectPatientVisits(state, 'patient-001').map((visit) => visit.timepoint))
      .toEqual(['preoperative', 'postoperative'])

    const invalid = patientWorkflowReducer(
      state,
      createVisitAction(
        makeVisit({
        id: 'visit-003',
        timepoint: '' as PatientVisit['timepoint'],
        }),
      ),
    )
    expect(invalid.lastFailure?.code).toBe('INVALID_VISIT')
    expect(invalid.visitsById['visit-003']).toBeUndefined()
  })

  it('stores an opaque media handle and metadata but strips media bytes and URLs', () => {
    const unsafeCapture = {
      ...makeCapture(),
      blob: new Blob(['patient-media']),
      bytes: new Uint8Array([1, 2, 3]),
      dataUrl: 'data:image/png;base64,cGF0aWVudA==',
      previewUrl: 'blob:https://example.test/patient-media',
    } as unknown as CaptureAsset
    const state = reduce(
      createInitialPatientWorkflowState(),
      createPatientAction(),
      createVisitAction(),
      { type: 'capture/add', capture: unsafeCapture },
    )
    const stored = state.capturesById['capture-001']

    expect(stored?.mediaHandle).toBe('session-media:opaque_token_001')
    expect(stored).not.toHaveProperty('blob')
    expect(stored).not.toHaveProperty('bytes')
    expect(stored).not.toHaveProperty('dataUrl')
    expect(stored).not.toHaveProperty('previewUrl')
    expect(stored).not.toBe(unsafeCapture)
  })

  it.each([
    'https://media.example.test/capture.png',
    'data:image/png;base64,cGF0aWVudA==',
    'blob:https://example.test/patient-media',
    'media:opaque-001',
    'arbitrary-handle',
  ])('rejects a non-vault media handle at capture registration: %s', (handle) => {
    const state = reduce(
      createInitialPatientWorkflowState(),
      createPatientAction(),
      createVisitAction(),
    )
    const rejected = patientWorkflowReducer(state, {
      type: 'capture/add',
      capture: makeCapture({
        mediaHandle: handle as CaptureAsset['mediaHandle'],
      }),
    })

    expect(rejected.lastFailure?.code).toBe('INVALID_CAPTURE')
    expect(rejected.capturesById).toEqual({})
  })

  it('exports a constructor and runtime guard for vault-issued handles', () => {
    const handle = createSessionMediaHandle('opaque_token_001')
    expect(handle).toBe('session-media:opaque_token_001')
    expect(isSessionMediaHandle(handle)).toBe(true)
    expect(isSessionMediaHandle('session-media:short')).toBe(false)
    expect(isSessionMediaHandle('https://example.test/media')).toBe(false)
    expect(() =>
      createSessionMediaHandle('blob:https://example.test/media'),
    ).toThrow('Invalid session media token.')
  })
})

describe('capture, run, result, and review lifecycle', () => {
  it('requires monotonic capture versions and supersedes a retake', () => {
    const initial = stateThroughCapture()
    const skippedVersion = patientWorkflowReducer(initial, {
      type: 'capture/add',
      capture: makeCapture({
        id: 'capture-003',
        version: 3,
        mediaHandle: createSessionMediaHandle('opaque_token_003'),
        sha256: 'c'.repeat(64),
      }),
    })
    expect(skippedVersion.lastFailure?.code).toBe('INVALID_CAPTURE_VERSION')
    expect(selectCurrentCapture(skippedVersion, 'visit-001')?.id).toBe(
      'capture-001',
    )

    const retaken = patientWorkflowReducer(initial, {
      type: 'capture/add',
      capture: makeCapture({
        id: 'capture-002',
        version: 2,
        mediaHandle: createSessionMediaHandle('opaque_token_002'),
        sha256: 'b'.repeat(64),
      }),
    })
    expect(retaken.capturesById['capture-001']).toMatchObject({
      status: 'superseded',
      supersededByCaptureId: 'capture-002',
    })
    expect(selectCurrentCapture(retaken, 'visit-001')).toMatchObject({
      id: 'capture-002',
      version: 2,
      status: 'current',
    })
  })

  it('accepts only an exact current capture and authorization run binding', () => {
    const ready = stateReadyToRun()
    const capture = selectCurrentCapture(ready, 'visit-001')!
    const invalidRun = makeRun(capture, {
      binding: {
        ...bindingFor(capture),
        captureSha256: 'f'.repeat(64),
      },
    })
    const rejected = patientWorkflowReducer(ready, {
      type: 'run/create',
      run: invalidRun,
    })
    expect(rejected.lastFailure?.code).toBe('INVALID_RUN_BINDING')
    expect(rejected.runsById).toEqual({})

    const accepted = patientWorkflowReducer(ready, {
      type: 'run/create',
      run: makeRun(capture),
    })
    expect(accepted.runsById['run-001']?.binding).toEqual({
      patientId: 'patient-001',
      visitId: 'visit-001',
      captureId: 'capture-001',
      captureVersion: 1,
      captureSha256: 'a'.repeat(64),
      mediaHandle: createSessionMediaHandle('opaque_token_001'),
      authorizationRevision: 1,
      captureProtocol: 'frontal_relaxed_non_mirrored_v1',
    })
  })

  it.each([
    {
      dependency: 'authorization revision',
      mutate: (state: PatientWorkflowState) =>
        patientWorkflowReducer(state, {
          type: 'authorization/record',
          authorization: makeAuthorization({
            id: 'authorization-002',
            revision: 2,
            recordedAt: '2026-07-27T13:05:00.000Z',
          }),
        }),
    },
    {
      dependency: 'authorization status',
      mutate: (state: PatientWorkflowState) =>
        patientWorkflowReducer(state, {
          type: 'authorization/record',
          authorization: makeAuthorization({
            id: 'authorization-002',
            revision: 2,
            status: 'withdrawn',
            recordedAt: '2026-07-27T13:05:00.000Z',
          }),
        }),
    },
    {
      dependency: 'active capture identity',
      mutate: (state: PatientWorkflowState) =>
        patientWorkflowReducer(state, {
          type: 'capture/add',
          capture: makeCapture({
            id: 'capture-002',
            version: 2,
            mediaHandle: createSessionMediaHandle('replacement_token_002'),
            sha256: 'b'.repeat(64),
            qualityChecks: COMPLETE_QUALITY,
            qualityConfirmedAt: '2026-07-27T13:05:00.000Z',
          }),
        }),
    },
    {
      dependency: 'capture version',
      mutate: (state: PatientWorkflowState) => {
        const capture = selectCurrentCapture(state, 'visit-001')!
        return {
          ...state,
          capturesById: {
            ...state.capturesById,
            [capture.id]: { ...capture, version: capture.version + 1 },
          },
        }
      },
    },
    {
      dependency: 'capture SHA-256',
      mutate: (state: PatientWorkflowState) => {
        const capture = selectCurrentCapture(state, 'visit-001')!
        return {
          ...state,
          capturesById: {
            ...state.capturesById,
            [capture.id]: { ...capture, sha256: 'b'.repeat(64) },
          },
        }
      },
    },
    {
      dependency: 'capture protocol',
      mutate: (state: PatientWorkflowState) => {
        const capture = selectCurrentCapture(state, 'visit-001')!
        return {
          ...state,
          capturesById: {
            ...state.capturesById,
            [capture.id]: {
              ...capture,
              captureProtocol:
                'frontal_smile_v2' as CaptureAsset['captureProtocol'],
            },
          },
        }
      },
    },
    {
      dependency: 'media handle',
      mutate: (state: PatientWorkflowState) => {
        const capture = selectCurrentCapture(state, 'visit-001')!
        return {
          ...state,
          capturesById: {
            ...state.capturesById,
            [capture.id]: {
              ...capture,
              mediaHandle: createSessionMediaHandle(
                'replacement_token_002',
              ),
            },
          },
        }
      },
    },
    {
      dependency: 'quality checks',
      mutate: (state: PatientWorkflowState) => {
        const capture = selectCurrentCapture(state, 'visit-001')!
        return {
          ...state,
          capturesById: {
            ...state.capturesById,
            [capture.id]: {
              ...capture,
              qualityChecks: {
                ...capture.qualityChecks,
                orientationConfirmed: false,
              },
            },
          },
        }
      },
    },
    {
      dependency: 'quality acceptance time',
      mutate: (state: PatientWorkflowState) => {
        const capture = selectCurrentCapture(state, 'visit-001')!
        const {
          qualityConfirmedAt: _qualityConfirmedAt,
          ...withoutConfirmation
        } = capture
        return {
          ...state,
          capturesById: {
            ...state.capturesById,
            [capture.id]: withoutConfirmation,
          },
        }
      },
    },
  ])(
    'revalidates $dependency immediately before queued analysis starts',
    ({ mutate }) => {
      const mutated = mutate(stateWithQueuedRun())
      const rejected = patientWorkflowReducer(mutated, {
        type: 'run/status/set',
        runId: RUN_ID,
        status: 'running',
      })

      expect(rejected.runsById['run-001']?.status).toBe('queued')
      expect(rejected.lastFailure?.code).toBe('INVALID_RUN_BINDING')
    },
  )

  it('keeps quality confirmation active until a completion time is recorded', () => {
    const state = patientWorkflowReducer(stateThroughCapture(), {
      type: 'capture/quality/set',
      captureId: CAPTURE_ID,
      checks: COMPLETE_QUALITY,
    })

    expect(selectVisitNextAction(state, 'visit-001')).toBe('confirm_quality')
  })

  it('derives the complete next-action sequence including failed retry', () => {
    let state = createInitialPatientWorkflowState()
    state = patientWorkflowReducer(state, createPatientAction())
    expect(selectPatientNextAction(state, 'patient-001')).toBe('start_visit')

    state = patientWorkflowReducer(state, createVisitAction())
    expect(selectVisitNextAction(state, 'visit-001')).toBe('capture_photo')

    state = patientWorkflowReducer(state, {
      type: 'capture/add',
      capture: makeCapture(),
    })
    expect(selectVisitNextAction(state, 'visit-001')).toBe('confirm_quality')

    state = patientWorkflowReducer(state, {
      type: 'authorization/record',
      authorization: makeAuthorization(),
    })
    state = patientWorkflowReducer(state, {
      type: 'capture/quality/set',
      captureId: CAPTURE_ID,
      checks: COMPLETE_QUALITY,
      confirmedAt: '2026-07-27T13:03:30.000Z',
    })
    expect(selectVisitNextAction(state, 'visit-001')).toBe('run_analysis')

    const capture = selectCurrentCapture(state, 'visit-001')!
    const firstRun = makeRun(capture)
    state = patientWorkflowReducer(state, {
      type: 'run/create',
      run: firstRun,
    })
    expect(selectVisitNextAction(state, 'visit-001')).toBe('processing')

    state = patientWorkflowReducer(state, {
      type: 'run/status/set',
      runId: firstRun.id,
      status: 'running',
    })
    expect(selectVisitNextAction(state, 'visit-001')).toBe('processing')

    state = patientWorkflowReducer(state, {
      type: 'run/status/set',
      runId: firstRun.id,
      status: 'failed',
      failure: {
        code: 'ANALYSIS_FAILED',
        message: 'The simulated analysis failed.',
      },
    })
    expect(selectVisitNextAction(state, 'visit-001')).toBe('retry_analysis')

    const retry = makeRun(capture, {
      id: 'run-002',
      retryOfRunId: firstRun.id,
      createdAt: '2026-07-27T13:06:00.000Z',
    })
    state = reduce(
      state,
      { type: 'run/create', run: retry },
      { type: 'run/status/set', runId: retry.id, status: 'running' },
      { type: 'run/status/set', runId: retry.id, status: 'succeeded' },
    )
    const result = makeResult(retry, {
      id: 'result-002',
      createdAt: '2026-07-27T13:07:00.000Z',
    })
    state = patientWorkflowReducer(state, {
      type: 'result/record',
      result,
    })
    expect(selectVisitNextAction(state, 'visit-001')).toBe('review_result')

    state = patientWorkflowReducer(state, {
      type: 'review/record',
      review: {
        id: createPatientReviewId('review-001'),
        patientId: PATIENT_ID,
        visitId: VISIT_ID,
        resultId: result.id,
        captureId: capture.id,
        decision: 'reviewed',
        note: 'Reviewed in the synthetic workflow.',
        completedAt: '2026-07-27T13:08:00.000Z',
      },
    })
    expect(selectVisitNextAction(state, 'visit-001')).toBe('visit_complete')
    expect(selectPatientNextAction(state, 'patient-001')).toBe('visit_complete')
  })

  it('rejects a retry whose immutable binding differs from its failed run', () => {
    const ready = stateReadyToRun()
    const capture = selectCurrentCapture(ready, 'visit-001')!
    const failed = reduce(
      ready,
      { type: 'run/create', run: makeRun(capture) },
      {
        type: 'run/status/set',
        runId: RUN_ID,
        status: 'failed',
        failure: { code: 'ANALYSIS_FAILED', message: 'Failed.' },
      },
    )
    const changedRetry = makeRun(capture, {
      id: 'run-002',
      retryOfRunId: 'run-001',
      binding: {
        ...bindingFor(capture),
        authorizationRevision: 2,
      },
    })
    const rejected = patientWorkflowReducer(failed, {
      type: 'run/create',
      run: changedRetry,
    })

    expect(rejected.lastFailure?.code).toBe('INVALID_RETRY_BINDING')
    expect(rejected.runsById['run-002']).toBeUndefined()
  })

  it('allows only one retry of the latest exact-bound failed run', () => {
    const ready = stateReadyToRun()
    const capture = selectCurrentCapture(ready, 'visit-001')!
    const failed = reduce(
      ready,
      { type: 'run/create', run: makeRun(capture) },
      {
        type: 'run/status/set',
        runId: RUN_ID,
        status: 'failed',
        failure: { code: 'ANALYSIS_FAILED', message: 'Failed.' },
      },
    )
    const withRetry = patientWorkflowReducer(failed, {
      type: 'run/create',
      run: makeRun(capture, {
        id: 'run-002',
        retryOfRunId: 'run-001',
      }),
    })
    const duplicateRetry = patientWorkflowReducer(withRetry, {
      type: 'run/create',
      run: makeRun(capture, {
        id: 'run-003',
        retryOfRunId: 'run-001',
      }),
    })

    expect(duplicateRetry.lastFailure?.code).toBe('INVALID_RETRY_BINDING')
    expect(duplicateRetry.runsById['run-003']).toBeUndefined()
    expect(selectCurrentRun(duplicateRetry, 'visit-001')?.id).toBe('run-002')
  })

  it('stops selecting a run and result after authorization revision changes', () => {
    const ready = stateReadyToRun()
    const capture = selectCurrentCapture(ready, 'visit-001')!
    const run = makeRun(capture)
    const succeeded = reduce(
      ready,
      { type: 'run/create', run },
      { type: 'run/status/set', runId: run.id, status: 'running' },
      { type: 'run/status/set', runId: run.id, status: 'succeeded' },
      { type: 'result/record', result: makeResult(run) },
    )
    const revised = patientWorkflowReducer(succeeded, {
      type: 'authorization/record',
      authorization: makeAuthorization({
        id: 'authorization-002',
        revision: 2,
        recordedAt: '2026-07-27T13:09:00.000Z',
      }),
    })

    expect(selectCurrentRun(revised, 'visit-001')).toBeUndefined()
    expect(selectCurrentResult(revised, 'visit-001')).toBeUndefined()
    expect(selectVisitNextAction(revised, 'visit-001')).toBe('run_analysis')
  })

  it('marks the old current result stale when a replacement capture is added', () => {
    const ready = stateReadyToRun()
    const capture = selectCurrentCapture(ready, 'visit-001')!
    const run = makeRun(capture)
    const succeeded = reduce(
      ready,
      { type: 'run/create', run },
      { type: 'run/status/set', runId: run.id, status: 'running' },
      { type: 'run/status/set', runId: run.id, status: 'succeeded' },
      { type: 'result/record', result: makeResult(run) },
    )
    expect(succeeded.resultsById['result-001']?.freshness).toBe('current')

    const retaken = patientWorkflowReducer(succeeded, {
      type: 'capture/add',
      capture: makeCapture({
        id: 'capture-002',
        version: 2,
        mediaHandle: createSessionMediaHandle('opaque_token_002'),
        sha256: 'b'.repeat(64),
      }),
    })

    expect(retaken.resultsById['result-001']?.freshness).toBe('stale')
    expect(selectVisitNextAction(retaken, 'visit-001')).toBe('confirm_quality')
  })

  it('derives retake after a repeat-photo review and requires its reason', () => {
    const ready = stateReadyToRun()
    const capture = selectCurrentCapture(ready, 'visit-001')!
    const run = makeRun(capture)
    const result = makeResult(run)
    const succeeded = reduce(
      ready,
      { type: 'run/create', run },
      { type: 'run/status/set', runId: run.id, status: 'running' },
      { type: 'run/status/set', runId: run.id, status: 'succeeded' },
      { type: 'result/record', result },
    )
    const missingReason = patientWorkflowReducer(succeeded, {
      type: 'review/record',
      review: {
        id: createPatientReviewId('review-001'),
        patientId: PATIENT_ID,
        visitId: VISIT_ID,
        resultId: result.id,
        captureId: capture.id,
        decision: 'repeat_photo',
        note: ' ',
        completedAt: '2026-07-27T13:08:00.000Z',
      },
    })
    expect(missingReason.lastFailure?.code).toBe('REPEAT_REASON_REQUIRED')

    const repeat = patientWorkflowReducer(succeeded, {
      type: 'review/record',
      review: {
        id: createPatientReviewId('review-001'),
        patientId: PATIENT_ID,
        visitId: VISIT_ID,
        resultId: result.id,
        captureId: capture.id,
        decision: 'repeat_photo',
        note: 'Image orientation needs to be reconfirmed.',
        completedAt: '2026-07-27T13:08:00.000Z',
      },
    })
    expect(selectVisitNextAction(repeat, 'visit-001')).toBe('retake')
  })
})

describe('synthetic demo patient seeds', () => {
  it('provides three explicit standalone synthetic identities without pairing', () => {
    expect(DEMO_PATIENT_RECORDS).toHaveLength(3)
    expect(new Set(DEMO_PATIENT_RECORDS.map((record) => record.id)).size).toBe(3)
    expect(
      new Set(DEMO_PATIENT_RECORDS.map((record) => record.recordNumber)).size,
    ).toBe(3)

    for (const record of DEMO_PATIENT_RECORDS) {
      expect(record.recordKind).toBe('synthetic_demo')
      expect(record.displayName).toMatch(/Synthetic Demo/)
      expect(record).not.toHaveProperty('assetId')
      expect(record).not.toHaveProperty('pairedPatientId')
      expect(record).not.toHaveProperty('visits')
      expect(Object.isFrozen(record)).toBe(true)
    }
  })
})

describe('prototype-safe entity stores', () => {
  it.each(['constructor', 'toString', '__proto__'])(
    'rejects a patient whose ID is an Object prototype key: %s',
    (patientId) => {
      const state = patientWorkflowReducer(
        createInitialPatientWorkflowState(),
        createPatientAction(
          makePatient({
            id: patientId,
            recordNumber: `SAFE-${patientId.length}`,
          }),
        ),
      )

      expect(state.lastFailure?.code).toBe('INVALID_PATIENT')
      expect(Object.hasOwn(state.patientsById, patientId)).toBe(false)
    },
  )

  it('does not resolve missing patient, visit, capture, or run IDs through Object.prototype', () => {
    const state = createInitialPatientWorkflowState()

    expect(selectPatientNextAction(state, 'constructor')).toBeUndefined()
    expect(selectVisitNextAction(state, 'toString')).toBeUndefined()

    const captureLookup = patientWorkflowReducer(state, {
      type: 'capture/quality/set',
      captureId: createCaptureAssetId('capture-missing'),
      checks: COMPLETE_QUALITY,
    })
    expect(captureLookup.lastFailure?.code).toBe('UNKNOWN_CAPTURE')

    const runLookup = patientWorkflowReducer(state, {
      type: 'run/status/set',
      runId: createPatientRunId('run-missing'),
      status: 'running',
    })
    expect(runLookup.lastFailure?.code).toBe('UNKNOWN_RUN')
  })

  it('rejects a visit whose owner ID is an Object prototype key', () => {
    const state = patientWorkflowReducer(
      createInitialPatientWorkflowState(),
      createVisitAction(
        makeVisit({
          id: 'visit-prototype-owner',
          patientId: 'toString',
        }),
      ),
    )

    expect(state.lastFailure?.code).toBe('INVALID_VISIT')
    expect(Object.hasOwn(state.visitsById, 'visit-prototype-owner')).toBe(
      false,
    )
  })
})

describe('post-run quality integrity', () => {
  it('locks the accepted capture quality once a run exists', () => {
    const ready = stateReadyToRun()
    const capture = selectCurrentCapture(ready, 'visit-001')!
    const queued = patientWorkflowReducer(ready, {
      type: 'run/create',
      run: makeRun(capture),
    })
    const rejected = patientWorkflowReducer(queued, {
      type: 'capture/quality/set',
      captureId: capture.id,
      checks: {
        ...COMPLETE_QUALITY,
        orientationConfirmed: false,
      },
    })

    expect(rejected.lastFailure?.code).toBe('CAPTURE_QUALITY_LOCKED')
    expect(rejected.capturesById[capture.id]?.qualityChecks).toEqual(
      COMPLETE_QUALITY,
    )
    expect(rejected.runsById['run-001']?.status).toBe('queued')
  })

  it('rejects quality changes after a result without hiding a still-valid review target', () => {
    const ready = stateReadyToRun()
    const capture = selectCurrentCapture(ready, 'visit-001')!
    const run = makeRun(capture)
    const result = makeResult(run)
    const succeeded = reduce(
      ready,
      { type: 'run/create', run },
      { type: 'run/status/set', runId: run.id, status: 'running' },
      { type: 'run/status/set', runId: run.id, status: 'succeeded' },
      { type: 'result/record', result },
    )
    const rejected = patientWorkflowReducer(succeeded, {
      type: 'capture/quality/set',
      captureId: capture.id,
      checks: INCOMPLETE_QUALITY,
    })

    expect(rejected.lastFailure?.code).toBe('CAPTURE_QUALITY_LOCKED')
    expect(selectCurrentResult(rejected, 'visit-001')?.id).toBe(result.id)
    expect(selectVisitNextAction(rejected, 'visit-001')).toBe('review_result')
  })
})

describe('trusted date and synthetic/test attestation', () => {
  it('fails patient and visit draft validation when trusted today is invalid', () => {
    const state = createInitialPatientWorkflowState()

    expect(
      validatePatientDraft(
        {
          displayName: 'Synthetic Test Patient',
          recordNumber: 'TEST-100',
          dateOfBirth: '1980-04-03',
          carePathway: 'Facial paralysis',
          syntheticTestAttestation: true,
        } as Parameters<typeof validatePatientDraft>[0],
        state,
        'invalid-today',
      ),
    ).toEqual({
      ok: false,
      errors: {
        dateOfBirth:
          'Date of birth cannot be validated because the trusted current date is invalid.',
      },
    })

    expect(
      validateVisitDraft(
        { timepoint: 'follow_up', visitDate: '2026-07-27' },
        'invalid-today',
      ),
    ).toEqual({
      ok: false,
      errors: {
        visitDate:
          'Visit date cannot be validated because the trusted current date is invalid.',
      },
    })
  })

  it('requires the synthetic/test attestation but does not persist it', () => {
    const state = createInitialPatientWorkflowState()
    const validation = validatePatientDraft(
      {
        displayName: 'Synthetic Test Patient',
        recordNumber: 'TEST-100',
        dateOfBirth: '1980-04-03',
        carePathway: 'Facial paralysis',
        syntheticTestAttestation: false,
      } as Parameters<typeof validatePatientDraft>[0],
      state,
      '2026-07-27',
    )
    expect(validation).toEqual({
      ok: false,
      errors: {
        syntheticTestAttestation:
          'Confirm that only synthetic/test information is being entered.',
      },
    })

    const rejected = patientWorkflowReducer(
      state,
      createPatientAction(makePatient(), false),
    )
    expect(rejected.lastFailure?.code).toBe('INVALID_PATIENT')
    expect(rejected.patientsById['patient-001']).toBeUndefined()

    const accepted = patientWorkflowReducer(
      state,
      createPatientAction(),
    )
    expect(accepted.patientsById['patient-001']).not.toHaveProperty(
      'syntheticTestAttestation',
    )
  })

  it('uses trusted today rather than caller-supplied creation timestamps', () => {
    const futurePatient = patientWorkflowReducer(
      createInitialPatientWorkflowState(),
      createPatientAction(
        makePatient({
          dateOfBirth: '2026-07-28',
          createdAt: '2026-07-28T10:00:00.000Z',
        }),
      ),
    )
    expect(futurePatient.lastFailure?.code).toBe('INVALID_PATIENT')
    expect(futurePatient.patientsById['patient-001']).toBeUndefined()

    const withPatient = patientWorkflowReducer(
      createInitialPatientWorkflowState(),
      createPatientAction(),
    )
    const futureVisit = patientWorkflowReducer(
      withPatient,
      createVisitAction(
        makeVisit({
          visitDate: '2026-07-28',
          createdAt: '2026-07-28T10:00:00.000Z',
        }),
      ),
    )
    expect(futureVisit.lastFailure?.code).toBe('INVALID_VISIT')
    expect(futureVisit.visitsById['visit-001']).toBeUndefined()
  })
})

describe('entity ID contracts', () => {
  it('exports distinct prefix-validating factories and guards', () => {
    const idApi = patientValidation as typeof patientValidation &
      Record<string, unknown>
    const factoryNames = [
      'createPatientId',
      'createPatientVisitId',
      'createAuthorizationSnapshotId',
      'createCaptureAssetId',
      'createPatientRunId',
      'createPatientResultId',
      'createPatientReviewId',
    ] as const
    const guardNames = [
      'isPatientId',
      'isPatientVisitId',
      'isAuthorizationSnapshotId',
      'isCaptureAssetId',
      'isPatientRunId',
      'isPatientResultId',
      'isPatientReviewId',
    ] as const

    for (const name of [...factoryNames, ...guardNames]) {
      expect(typeof idApi[name]).toBe('function')
    }
    if (
      factoryNames.some((name) => typeof idApi[name] !== 'function') ||
      guardNames.some((name) => typeof idApi[name] !== 'function')
    ) {
      return
    }

    const createPatientId = idApi.createPatientId as (value: string) => string
    const isPatientId = idApi.isPatientId as (value: unknown) => boolean
    const isPatientVisitId = idApi.isPatientVisitId as (
      value: unknown,
    ) => boolean
    const patientId = createPatientId('patient-safe_001')

    expect(patientId).toBe('patient-safe_001')
    expect(isPatientId(patientId)).toBe(true)
    expect(isPatientVisitId(patientId)).toBe(false)
    expect(() => createPatientId(' visit-safe_001 ')).toThrow(
      'Invalid patient ID.',
    )
    expect(() => createPatientId(`patient-${'a'.repeat(65)}`)).toThrow(
      'Invalid patient ID.',
    )
  })

  it.each([
    'constructor',
    'toString',
    '__proto__',
    'visit-cross-entity',
    ' patient-leading-space',
    'patient-unsafe/slash',
    `patient-${'a'.repeat(65)}`,
  ])('rejects an unsafe or cross-entity patient ID: %s', (id) => {
    const rejected = patientWorkflowReducer(
      createInitialPatientWorkflowState(),
      createPatientAction(
        makePatient({
          id: id as PatientRecord['id'],
          recordNumber: `INVALID-${id.length}`,
        }),
      ),
    )

    expect(rejected.lastFailure?.code).toBe('INVALID_PATIENT')
    expect(Object.hasOwn(rejected.patientsById, id)).toBe(false)
  })
})

describe('one immutable result per run', () => {
  it('rejects result substitution after the original result was reviewed', () => {
    const ready = stateReadyToRun()
    const capture = selectCurrentCapture(ready, 'visit-001')!
    const run = makeRun(capture)
    const result = makeResult(run)
    const reviewed = reduce(
      ready,
      { type: 'run/create', run },
      { type: 'run/status/set', runId: run.id, status: 'running' },
      { type: 'run/status/set', runId: run.id, status: 'succeeded' },
      { type: 'result/record', result },
      {
        type: 'review/record',
        review: {
          id: createPatientReviewId('review-reviewed-result'),
          patientId: PATIENT_ID,
          visitId: VISIT_ID,
          resultId: result.id,
          captureId: capture.id,
          decision: 'reviewed',
          completedAt: '2026-07-27T13:08:00.000Z',
        },
      },
    )
    const substituted = patientWorkflowReducer(reviewed, {
      type: 'result/record',
      result: makeResult(run, {
        id: 'result-substitute',
        createdAt: '2026-07-27T13:09:00.000Z',
      }),
    })

    expect(substituted.lastFailure?.code).toBe('INVALID_RESULT')
    expect(substituted.resultsById['result-substitute']).toBeUndefined()
    expect(substituted.resultsById[result.id]?.freshness).toBe('current')
    expect(selectVisitNextAction(substituted, 'visit-001')).toBe(
      'visit_complete',
    )
  })
})

describe('run failure ownership', () => {
  it('copies and freezes a valid failure without freezing the caller object', () => {
    const queued = stateWithQueuedRun()
    const failure = {
      code: 'ANALYSIS_FAILED' as const,
      message: 'The simulated analysis failed.',
      field: 'gateway',
    }
    const failed = patientWorkflowReducer(queued, {
      type: 'run/status/set',
      runId: RUN_ID,
      status: 'failed',
      failure,
    })

    expect(failed.runsById['run-001']?.failure).toEqual(failure)
    expect(failed.runsById['run-001']?.failure).not.toBe(failure)
    expect(Object.isFrozen(failure)).toBe(false)
    expect(Object.isFrozen(failed.runsById['run-001']?.failure)).toBe(true)
  })

  it('fails closed on an invalid supplied run failure', () => {
    const queued = stateWithQueuedRun()
    const rejected = patientWorkflowReducer(queued, {
      type: 'run/status/set',
      runId: RUN_ID,
      status: 'failed',
      failure: {
        code: 'INVALID_PATIENT',
        message: ' ',
      },
    })

    expect(rejected.lastFailure?.code).toBe('INVALID_RUN_FAILURE')
    expect(rejected.runsById['run-001']?.status).toBe('queued')
  })
})

describe('seeded patient initialization', () => {
  it('populates all valid demo seeds using an explicit trusted date', () => {
    const state = createInitialPatientWorkflowState(
      DEMO_PATIENT_RECORDS,
      '2026-07-27',
    )

    expect(state.lastFailure).toBeUndefined()
    expect(state.patientOrder).toEqual(
      DEMO_PATIENT_RECORDS.map((patient) => patient.id),
    )
    expect(Object.values(state.patientsById).filter(Boolean)).toHaveLength(3)
  })

  it('requires a trusted date whenever seeds are supplied', () => {
    if (false) {
      // @ts-expect-error Seeded initialization requires trustedToday.
      createInitialPatientWorkflowState(DEMO_PATIENT_RECORDS)
    }
    expect(createInitialPatientWorkflowState()).toMatchObject({
      patientOrder: [],
    })
  })
})
