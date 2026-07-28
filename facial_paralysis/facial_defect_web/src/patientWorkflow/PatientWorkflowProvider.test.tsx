import { act, render } from '@testing-library/react'
import type { ReactNode } from 'react'
import {
  afterEach,
  beforeEach,
  describe,
  expect,
  it,
  vi,
} from 'vitest'
import { approvedAssets } from '../data/approvedAssetManifest'
import { SessionMediaVault } from './SessionMediaVault'
import {
  CaptureFileValidationError,
  type CaptureFileValidationResult,
} from './captureFile'
import {
  PatientWorkflowProvider,
  PatientWorkflowProviderError,
  usePatientWorkflow,
  type PatientWorkflowContextValue,
  type PatientWorkflowRuntime,
  type PatientWorkflowRuntimeIdKind,
  type PatientWorkflowProviderProps,
} from './PatientWorkflowProvider'
import {
  selectCurrentAuthorization,
  selectCurrentCapture,
  selectCurrentResult,
  selectCurrentReview,
  selectCurrentRun,
} from './selectors'
import type {
  AuthorizationSnapshot,
  CaptureAsset,
  CaptureQualityChecks,
  PatientRunBinding,
  PatientSimulationOutput,
  PatientWorkflowFailureCode,
  PatientWorkflowState,
} from './types'
import { createSessionMediaHandle } from './validation'

const COMPLETE_QUALITY: CaptureQualityChecks = {
  faceVisibleAndCentered: true,
  focusLightingAndOcclusionAcceptable: true,
  orientationConfirmed: true,
  authorizationDocumented: true,
}

const DEFAULT_SHA = 'a'.repeat(64)

type TestMedia = Blob & {
  readonly testSha256?: string
  readonly testWidth?: number
  readonly testHeight?: number
}

function makeMedia(
  label: string,
  sha256 = DEFAULT_SHA,
  type: 'image/jpeg' | 'image/png' | 'image/webp' = 'image/png',
): TestMedia {
  const blob = new Blob([label], { type }) as TestMedia
  Object.defineProperties(blob, {
    testSha256: { value: sha256 },
    testWidth: { value: 1_024 },
    testHeight: { value: 900 },
  })
  return blob
}

function makeFile(
  label: string,
  sha256 = DEFAULT_SHA,
): File & TestMedia {
  const file = new File([label], 'phi-must-not-escape.png', {
    type: 'image/png',
  }) as File & TestMedia
  Object.defineProperties(file, {
    testSha256: { value: sha256 },
    testWidth: { value: 1_024 },
    testHeight: { value: 900 },
  })
  return file
}

function createCapturePreparer() {
  return vi.fn(
    async (media: Blob): Promise<CaptureFileValidationResult> => {
      const tagged = media as TestMedia
      const vaultMedia = makeMedia(
        await media.text(),
        tagged.testSha256 ?? DEFAULT_SHA,
        media.type as 'image/jpeg' | 'image/png' | 'image/webp',
      )
      return {
        ok: true,
        value: Object.freeze({
          metadata: Object.freeze({
            sha256: tagged.testSha256 ?? DEFAULT_SHA,
            mimeType:
              media.type as 'image/jpeg' | 'image/png' | 'image/webp',
            sizeBytes: media.size,
            width: tagged.testWidth ?? 1_024,
            height: tagged.testHeight ?? 900,
          }),
          vaultMedia,
        }),
      }
    },
  )
}

function createObjectUrlApi() {
  let sequence = 0
  return {
    createObjectURL: vi.fn(
      () => `blob:patient-provider-${++sequence}`,
    ),
    revokeObjectURL: vi.fn(),
  }
}

function createRuntime(
  tokenOverrides: Partial<
    Record<PatientWorkflowRuntimeIdKind, readonly string[]>
  > = {},
): PatientWorkflowRuntime {
  const counters = new Map<PatientWorkflowRuntimeIdKind, number>()
  return {
    nextIdToken(kind) {
      const next = (counters.get(kind) ?? 0) + 1
      counters.set(kind, next)
      return (
        tokenOverrides[kind]?.[next - 1] ??
        `${kind}_${String(next).padStart(4, '0')}`
      )
    },
    now: () => '2026-07-27T14:00:00.000Z',
    today: () => '2026-07-27',
    reset: vi.fn(),
  }
}

function validOutput(
  binding: PatientRunBinding,
): PatientSimulationOutput {
  void binding
  return {
    origin: 'workflow_simulation',
    points: [
      { x: 0.4, y: 0.5, intensity: 0.8, radius: 0.1 },
    ],
  }
}

let latest: PatientWorkflowContextValue

function Probe() {
  latest = usePatientWorkflow()
  return null
}

type RenderProviderOptions = Omit<
  PatientWorkflowProviderProps,
  'children'
> & {
  readonly children?: ReactNode
}

function renderProvider(options: RenderProviderOptions = {}) {
  return render(
    <PatientWorkflowProvider {...options}>
      <Probe />
      {options.children}
    </PatientWorkflowProvider>,
  )
}

function patientInput(
  overrides: Partial<
    Parameters<
      PatientWorkflowContextValue['actions']['createPatient']
    >[0]
  > = {},
) {
  return {
    displayName: 'Synthetic Test Patient',
    recordNumber: ' test 100 ',
    dateOfBirth: '1980-04-03',
    carePathway: 'Facial paralysis',
    syntheticTestAttestation: true,
    initialVisit: {
      timepoint: 'preoperative' as const,
      visitDate: '2026-07-27',
    },
    ...overrides,
  }
}

function expectProviderFailure(
  operation: () => unknown,
  code: PatientWorkflowFailureCode,
): PatientWorkflowProviderError {
  try {
    operation()
  } catch (error) {
    expect(error).toBeInstanceOf(PatientWorkflowProviderError)
    const providerError = error as PatientWorkflowProviderError
    expect(providerError.failure.code).toBe(code)
    return providerError
  }
  throw new Error(`Expected provider failure ${code}.`)
}

async function expectProviderFailureAsync(
  operation: () => Promise<unknown>,
  code: PatientWorkflowFailureCode,
): Promise<PatientWorkflowProviderError> {
  try {
    await operation()
  } catch (error) {
    expect(error).toBeInstanceOf(PatientWorkflowProviderError)
    const providerError = error as PatientWorkflowProviderError
    expect(providerError.failure.code).toBe(code)
    return providerError
  }
  throw new Error(`Expected provider failure ${code}.`)
}

function createPatientAndVisit() {
  let identifiers!: ReturnType<
    PatientWorkflowContextValue['actions']['createPatient']
  >
  act(() => {
    identifiers = latest.actions.createPatient(patientInput())
  })
  return identifiers
}

async function attachCapture(
  visitId: string,
  media: Blob = makeMedia('first capture'),
  source: 'camera' | 'upload' = 'upload',
) {
  let captureId = ''
  await act(async () => {
    captureId = await latest.actions.attachSessionCapture(
      visitId,
      media,
      source,
    )
  })
  return captureId
}

function confirmAllQuality(visitId: string) {
  act(() => {
    for (const check of Object.keys(
      COMPLETE_QUALITY,
    ) as (keyof CaptureQualityChecks)[]) {
      latest.actions.setQualityCheck(visitId, check, true)
    }
  })
}

async function advance(ms: number) {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms)
  })
}

function mutableState(state: PatientWorkflowState) {
  return state as {
    authorizationsById: Record<string, AuthorizationSnapshot | undefined>
    capturesById: Record<string, CaptureAsset | undefined>
  }
}

async function createReadyVisit(
  media: Blob = makeMedia('ready capture'),
) {
  const identifiers = createPatientAndVisit()
  await attachCapture(identifiers.visitId, media)
  confirmAllQuality(identifiers.visitId)
  return identifiers
}

beforeEach(() => {
  latest = undefined as unknown as PatientWorkflowContextValue
  vi.useFakeTimers()
})

afterEach(() => {
  vi.useRealTimers()
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
  vi.unstubAllEnvs()
})

describe('PatientWorkflowProvider patient and visit ownership', () => {
  it('exposes only the simulation-only, memory-only prototype boundary', () => {
    renderProvider()

    expect(latest.mode).toBe('simulation_only')
    expect(latest.persistence).toBe('memory_only')
    expect(Object.keys(latest).sort()).toEqual([
      'actions',
      'mode',
      'patientListQuery',
      'persistence',
      'setPatientListQuery',
      'state',
    ])
  })

  it('keeps patient search in React memory and clears it with the session', () => {
    renderProvider()

    act(() => {
      latest.setPatientListQuery('synthetic record')
    })
    expect(latest.patientListQuery).toBe('synthetic record')

    act(() => {
      latest.actions.resetSession()
    })
    expect(latest.patientListQuery).toBe('')
  })

  it('creates a normalized patient and validated initial visit atomically', () => {
    renderProvider({ runtime: createRuntime() })

    const invalid = patientInput({
      initialVisit: { timepoint: '', visitDate: '2026-07-27' },
    })
    expectProviderFailure(
      () => latest.actions.createPatient(invalid),
      'INVALID_VISIT',
    )
    expect(latest.state.patientOrder).toEqual([])
    expect(latest.state.visitOrder).toEqual([])

    const identifiers = createPatientAndVisit()
    expect(identifiers).toEqual({
      patientId: 'patient-patient_0001',
      visitId: 'visit-visit_0001',
    })
    expect(latest.state.patientsById[identifiers.patientId]).toMatchObject({
      displayName: 'Synthetic Test Patient',
      recordNumber: 'TEST-100',
    })
    expect(latest.state.visitsById[identifiers.visitId]).toMatchObject({
      patientId: identifiers.patientId,
      timepoint: 'preoperative',
    })
  })

  it('rejects normalized duplicate record numbers without creating a patient or visit', () => {
    renderProvider({ runtime: createRuntime() })
    createPatientAndVisit()

    expectProviderFailure(
      () =>
        latest.actions.createPatient(
          patientInput({
            displayName: 'Second Synthetic Patient',
            recordNumber: 'TEST_100',
          }),
        ),
      'DUPLICATE_RECORD_NUMBER',
    )
    expect(latest.state.patientOrder).toHaveLength(1)
    expect(latest.state.visitOrder).toHaveLength(1)
  })

  it('uses the New York local calendar date when UTC is already the next day', () => {
    vi.stubEnv('TZ', 'America/New_York')
    vi.setSystemTime(new Date('2026-07-28T01:30:00.000Z'))
    const localNow = new Date()
    expect([
      localNow.getFullYear(),
      localNow.getMonth() + 1,
      localNow.getDate(),
    ]).toEqual([2026, 7, 27])
    renderProvider()

    let localToday!: ReturnType<
      PatientWorkflowContextValue['actions']['createPatient']
    >
    act(() => {
      localToday = latest.actions.createPatient(
        patientInput({
          recordNumber: 'LOCAL-TODAY',
          initialVisit: {
            timepoint: 'follow_up',
            visitDate: '2026-07-27',
          },
        }),
      )
    })
    expect(
      latest.state.visitsById[localToday.visitId]?.visitDate,
    ).toBe('2026-07-27')

    expectProviderFailure(
      () =>
        latest.actions.createPatient(
          patientInput({
            recordNumber: 'LOCAL-FUTURE-VISIT',
            initialVisit: {
              timepoint: 'follow_up',
              visitDate: '2026-07-28',
            },
          }),
        ),
      'INVALID_VISIT',
    )
    expectProviderFailure(
      () =>
        latest.actions.createPatient(
          patientInput({
            recordNumber: 'LOCAL-FUTURE-DOB',
            dateOfBirth: '2026-07-28',
          }),
        ),
      'INVALID_PATIENT',
    )
  })

  it('creates later visits only for a known patient and a valid date', () => {
    renderProvider({ runtime: createRuntime() })
    const { patientId } = createPatientAndVisit()

    let visitId = ''
    act(() => {
      visitId = latest.actions.createVisit(patientId, {
        timepoint: 'follow_up',
        visitDate: '2026-07-27',
      })
    })
    expect(latest.state.visitsById[visitId]?.patientId).toBe(patientId)

    expectProviderFailure(
      () =>
        latest.actions.createVisit('patient-missing', {
          timepoint: 'follow_up',
          visitDate: '2026-07-27',
        }),
      'UNKNOWN_PATIENT',
    )
    expectProviderFailure(
      () =>
        latest.actions.createVisit(patientId, {
          timepoint: 'follow_up',
          visitDate: '2026-07-28',
        }),
      'INVALID_VISIT',
    )
  })
})

describe('capture preparation, vault ownership, and retakes', () => {
  it('attaches upload Blob and camera File captures with preview URLs', async () => {
    const prepareCapture = createCapturePreparer()
    const objectUrls = createObjectUrlApi()
    const vault = new SessionMediaVault(objectUrls)
    renderProvider({
      runtime: createRuntime(),
      mediaVault: vault,
      prepareCapture,
    })
    const { visitId } = createPatientAndVisit()

    const firstId = await attachCapture(
      visitId,
      makeMedia('upload capture'),
      'upload',
    )
    expect(latest.state.capturesById[firstId]?.source).toBe('upload')
    expect(latest.actions.getCapturePreviewUrl(visitId)).toBe(
      'blob:patient-provider-1',
    )

    const secondId = await attachCapture(
      visitId,
      makeFile('camera capture', 'b'.repeat(64)),
      'camera',
    )
    expect(latest.state.capturesById[firstId]).toMatchObject({
      status: 'superseded',
      supersededByCaptureId: secondId,
    })
    expect(latest.state.capturesById[secondId]).toMatchObject({
      source: 'camera',
      version: 2,
      sha256: 'b'.repeat(64),
    })
    expect(objectUrls.revokeObjectURL).toHaveBeenCalledWith(
      'blob:patient-provider-1',
    )
    expect(latest.actions.getCapturePreviewUrl(visitId)).toBe(
      'blob:patient-provider-2',
    )
  })

  it('retains old media until the reducer accepts a replacement and cleans rejected new media', async () => {
    const prepareCapture = createCapturePreparer()
    const objectUrls = createObjectUrlApi()
    const vault = new SessionMediaVault(objectUrls)
    const runtime = createRuntime({
      capture: ['duplicate_0001', 'duplicate_0001'],
    })
    renderProvider({
      runtime,
      mediaVault: vault,
      prepareCapture,
    })
    const { visitId } = createPatientAndVisit()
    await attachCapture(visitId, makeMedia('accepted'))

    await expectProviderFailureAsync(
      () =>
        latest.actions.attachSessionCapture(
          visitId,
          makeMedia('rejected', 'b'.repeat(64)),
          'upload',
        ),
      'DUPLICATE_CAPTURE_ID',
    )

    expect(selectCurrentCapture(latest.state, visitId)?.version).toBe(1)
    expect(latest.actions.getCapturePreviewUrl(visitId)).toBe(
      'blob:patient-provider-1',
    )
    expect(objectUrls.revokeObjectURL.mock.calls).toEqual([
      ['blob:patient-provider-2'],
    ])
  })

  it('surfaces typed preparation errors without allocating vault media', async () => {
    const objectUrls = createObjectUrlApi()
    const vault = new SessionMediaVault(objectUrls)
    const validationError = new CaptureFileValidationError(
      'RESOLUTION_TOO_LOW',
      'Choose an image at least 640 × 640 pixels.',
    )
    const prepareCapture = vi.fn(async () => ({
      ok: false as const,
      error: validationError,
    }))
    renderProvider({
      runtime: createRuntime(),
      mediaVault: vault,
      prepareCapture,
    })
    const { visitId } = createPatientAndVisit()

    await expect(
      latest.actions.attachSessionCapture(
        visitId,
        makeMedia('undersized'),
      ),
    ).rejects.toBe(validationError)
    expect(latest.state.captureOrder).toEqual([])
    expect(objectUrls.createObjectURL).not.toHaveBeenCalled()
  })

  it('loads only approved synthetic media and verifies its actual blob hash', async () => {
    const approved = approvedAssets[0]
    const prepareCapture = createCapturePreparer()
    const loadSyntheticMedia = vi.fn(
      async () => makeMedia('approved synthetic', approved.sha256),
    )
    renderProvider({
      runtime: createRuntime(),
      prepareCapture,
      loadSyntheticMedia,
    })
    const { visitId } = createPatientAndVisit()

    let captureId = ''
    await act(async () => {
      captureId = await latest.actions.attachSyntheticCapture(
        visitId,
        approved.id,
      )
    })
    expect(loadSyntheticMedia).toHaveBeenCalledWith(approved)
    expect(latest.state.capturesById[captureId]).toMatchObject({
      source: 'synthetic_demo',
      syntheticSourceAssetId: approved.id,
      sha256: approved.sha256,
    })

    await expectProviderFailureAsync(
      () =>
        latest.actions.attachSyntheticCapture(
          visitId,
          'SYN-NOT-APPROVED',
        ),
      'INVALID_CAPTURE',
    )
    expect(loadSyntheticMedia).toHaveBeenCalledOnce()
  })

  it('does not create a new capture version from the same current catalog demo', async () => {
    const approved = approvedAssets[0]
    const loadSyntheticMedia = vi.fn(
      async () => makeMedia('approved synthetic', approved.sha256),
    )
    renderProvider({
      runtime: createRuntime(),
      prepareCapture: createCapturePreparer(),
      loadSyntheticMedia,
    })
    const { visitId } = createPatientAndVisit()

    await act(async () => {
      await latest.actions.attachSyntheticCapture(
        visitId,
        approved.id,
      )
    })
    const failure = await expectProviderFailureAsync(
      () =>
        latest.actions.attachSyntheticCapture(
          visitId,
          approved.id,
        ),
      'INVALID_CAPTURE',
    )

    expect(failure.message).toMatch(/already uses.+catalog demo/i)
    expect(loadSyntheticMedia).toHaveBeenCalledOnce()
    expect(latest.state.captureOrder).toHaveLength(1)
  })

  it('prevents standalone catalog media from creating a longitudinal patient series', async () => {
    const approved = approvedAssets[0]
    const loadSyntheticMedia = vi.fn(
      async () => makeMedia('approved synthetic', approved.sha256),
    )
    renderProvider({
      runtime: createRuntime(),
      prepareCapture: createCapturePreparer(),
      loadSyntheticMedia,
    })
    const { patientId, visitId } = createPatientAndVisit()

    await act(async () => {
      await latest.actions.attachSyntheticCapture(
        visitId,
        approved.id,
      )
    })
    let laterVisitId = ''
    act(() => {
      laterVisitId = latest.actions.createVisit(patientId, {
        timepoint: 'postoperative',
        visitDate: '2026-07-27',
      })
    })

    const failure = await expectProviderFailureAsync(
      () =>
        latest.actions.attachSyntheticCapture(
          laterVisitId,
          approved.id,
        ),
      'INVALID_CAPTURE',
    )
    expect(failure.message).toMatch(/standalone.+another visit/i)
    expect(loadSyntheticMedia).toHaveBeenCalledOnce()
    expect(latest.state.captureOrder).toHaveLength(1)
  })

  it('rejects a synthetic loader blob whose actual hash differs from the canonical asset', async () => {
    const objectUrls = createObjectUrlApi()
    const vault = new SessionMediaVault(objectUrls)
    renderProvider({
      runtime: createRuntime(),
      mediaVault: vault,
      prepareCapture: createCapturePreparer(),
      loadSyntheticMedia: async () =>
        makeMedia('tampered synthetic', 'f'.repeat(64)),
    })
    const { visitId } = createPatientAndVisit()

    await expectProviderFailureAsync(
      () =>
        latest.actions.attachSyntheticCapture(
          visitId,
          approvedAssets[0].id,
        ),
      'INVALID_CAPTURE',
    )
    expect(latest.state.captureOrder).toEqual([])
    expect(objectUrls.createObjectURL).not.toHaveBeenCalled()
  })
})

describe('quality authorization and background simulation', () => {
  it('blocks submission until all checks and an authorization snapshot are current', async () => {
    renderProvider({
      runtime: createRuntime(),
      prepareCapture: createCapturePreparer(),
    })
    const { visitId } = createPatientAndVisit()
    await attachCapture(visitId)

    expectProviderFailure(
      () => latest.actions.submitAnalysis(visitId),
      'CAPTURE_QUALITY_INCOMPLETE',
    )
    expect(latest.state.runOrder).toEqual([])

    confirmAllQuality(visitId)
    expect(selectCurrentCapture(latest.state, visitId)).toMatchObject({
      qualityChecks: COMPLETE_QUALITY,
      qualityConfirmedAt: '2026-07-27T14:00:00.000Z',
    })
    expect(selectCurrentAuthorization(latest.state, visitId)).toMatchObject({
      revision: 1,
      status: 'documented',
    })
  })

  it('moves queued to running to succeeded and passes only the exact binding to simulation', async () => {
    const simulationRunner = vi.fn(validOutput)
    renderProvider({
      runtime: createRuntime(),
      prepareCapture: createCapturePreparer(),
      simulationRunner,
      queueDelayMs: 20,
      analysisDelayMs: 30,
    })
    const { visitId } = await createReadyVisit()

    let runId = ''
    act(() => {
      runId = latest.actions.submitAnalysis(visitId)
    })
    expect(latest.state.runsById[runId]?.status).toBe('queued')
    expect(simulationRunner).not.toHaveBeenCalled()

    await advance(20)
    expect(latest.state.runsById[runId]?.status).toBe('running')
    expect(simulationRunner).toHaveBeenCalledOnce()
    const request = simulationRunner.mock.calls[0]?.[0]
    expect(Object.keys(request).sort()).toEqual([
      'authorizationRevision',
      'captureId',
      'captureProtocol',
      'captureSha256',
      'captureVersion',
      'mediaHandle',
      'patientId',
      'visitId',
    ])
    const serialized = JSON.stringify(request)
    expect(serialized).not.toMatch(
      /Synthetic Test Patient|TEST-100|1980-04-03|data:|blob:|phi-must-not-escape/i,
    )
    expect(request).not.toBeInstanceOf(Blob)

    await advance(29)
    expect(latest.state.runsById[runId]?.status).toBe('running')
    await advance(1)
    expect(latest.state.runsById[runId]?.status).toBe('succeeded')
    expect(selectCurrentResult(latest.state, visitId)).toMatchObject({
      runId,
      binding: request,
      output: validOutput(request),
    })
  })

  it.each([
    {
      label: 'withdrawn authorization',
      mutate(state: PatientWorkflowState, visitId: string) {
        const authorization = selectCurrentAuthorization(state, visitId)!
        mutableState(state).authorizationsById[authorization.id] = {
          ...authorization,
          status: 'withdrawn',
        }
      },
    },
    {
      label: 'changed authorization revision',
      mutate(state: PatientWorkflowState, visitId: string) {
        const authorization = selectCurrentAuthorization(state, visitId)!
        mutableState(state).authorizationsById[authorization.id] = {
          ...authorization,
          revision: authorization.revision + 1,
        }
      },
    },
    {
      label: 'incomplete quality',
      mutate(state: PatientWorkflowState, visitId: string) {
        const capture = selectCurrentCapture(state, visitId)!
        mutableState(state).capturesById[capture.id] = {
          ...capture,
          qualityChecks: {
            ...capture.qualityChecks,
            orientationConfirmed: false,
          },
        }
      },
    },
    {
      label: 'non-current capture',
      mutate(state: PatientWorkflowState, visitId: string) {
        const capture = selectCurrentCapture(state, visitId)!
        mutableState(state).capturesById[capture.id] = {
          ...capture,
          status: 'superseded',
        }
      },
    },
    {
      label: 'changed capture version',
      mutate(state: PatientWorkflowState, visitId: string) {
        const capture = selectCurrentCapture(state, visitId)!
        mutableState(state).capturesById[capture.id] = {
          ...capture,
          version: capture.version + 1,
        }
      },
    },
    {
      label: 'changed capture hash',
      mutate(state: PatientWorkflowState, visitId: string) {
        const capture = selectCurrentCapture(state, visitId)!
        mutableState(state).capturesById[capture.id] = {
          ...capture,
          sha256: 'f'.repeat(64),
        }
      },
    },
    {
      label: 'changed media handle',
      mutate(state: PatientWorkflowState, visitId: string) {
        const capture = selectCurrentCapture(state, visitId)!
        mutableState(state).capturesById[capture.id] = {
          ...capture,
          mediaHandle: createSessionMediaHandle('drifted_media_0001'),
        }
      },
    },
  ])(
    'fails $label before making any inference call',
    async ({ mutate }) => {
      const simulationRunner = vi.fn(validOutput)
      renderProvider({
        runtime: createRuntime(),
        prepareCapture: createCapturePreparer(),
        simulationRunner,
        queueDelayMs: 20,
      })
      const { visitId } = await createReadyVisit()
      let runId = ''
      act(() => {
        runId = latest.actions.submitAnalysis(visitId)
      })

      mutate(latest.state, visitId)
      await advance(20)

      expect(simulationRunner).not.toHaveBeenCalled()
      expect(latest.state.runsById[runId]).toMatchObject({
        status: 'failed',
        failure: {
          code: 'ANALYSIS_FAILED',
          field: 'binding',
        },
      })
    },
  )

  it('fails a missing or replaced vault entry before inference', async () => {
    const simulationRunner = vi.fn(validOutput)
    const vault = new SessionMediaVault(createObjectUrlApi())
    renderProvider({
      runtime: createRuntime(),
      mediaVault: vault,
      prepareCapture: createCapturePreparer(),
      simulationRunner,
      queueDelayMs: 20,
    })
    const { visitId } = await createReadyVisit()
    const capture = selectCurrentCapture(latest.state, visitId)!
    let runId = ''
    act(() => {
      runId = latest.actions.submitAnalysis(visitId)
    })
    vault.delete(capture.mediaHandle)

    await advance(20)
    expect(simulationRunner).not.toHaveBeenCalled()
    expect(latest.state.runsById[runId]).toMatchObject({
      status: 'failed',
      failure: { code: 'ANALYSIS_FAILED', field: 'mediaHandle' },
    })
  })

  it('revalidates actual vault media hash immediately before inference', async () => {
    const simulationRunner = vi.fn(validOutput)
    const vault = new SessionMediaVault(createObjectUrlApi())
    renderProvider({
      runtime: createRuntime(),
      mediaVault: vault,
      prepareCapture: createCapturePreparer(),
      simulationRunner,
      queueDelayMs: 20,
    })
    const { visitId } = await createReadyVisit()
    const capture = selectCurrentCapture(latest.state, visitId)!
    let runId = ''
    act(() => {
      runId = latest.actions.submitAnalysis(visitId)
    })
    vault.set(
      capture.mediaHandle,
      makeMedia('different bytes', 'f'.repeat(64)),
    )

    await advance(20)
    expect(simulationRunner).not.toHaveBeenCalled()
    expect(latest.state.runsById[runId]).toMatchObject({
      status: 'failed',
      failure: { code: 'ANALYSIS_FAILED', field: 'captureSha256' },
    })
  })

  it('rejects invalid simulation output before success or result recording', async () => {
    const invalidOutput = {
      origin: 'workflow_simulation',
      points: [
        { x: Number.NaN, y: 2, intensity: 0.5, radius: 0 },
      ],
    } as PatientSimulationOutput
    renderProvider({
      runtime: createRuntime(),
      prepareCapture: createCapturePreparer(),
      simulationRunner: () => invalidOutput,
    })
    const { visitId } = await createReadyVisit()
    let runId = ''
    act(() => {
      runId = latest.actions.submitAnalysis(visitId)
    })

    await advance(0)
    expect(latest.state.runsById[runId]).toMatchObject({
      status: 'failed',
      failure: { code: 'ANALYSIS_FAILED', field: 'output' },
    })
    expect(latest.state.resultOrder).toEqual([])
  })

  it('retries only the exact failed binding and revalidates the retry launch', async () => {
    const simulationRunner = vi
      .fn()
      .mockRejectedValueOnce(new Error('first simulation failed'))
      .mockImplementationOnce(validOutput)
    const vault = new SessionMediaVault(createObjectUrlApi())
    renderProvider({
      runtime: createRuntime(),
      mediaVault: vault,
      prepareCapture: createCapturePreparer(),
      simulationRunner,
      queueDelayMs: 10,
    })
    const { visitId } = await createReadyVisit()
    let firstRunId = ''
    act(() => {
      firstRunId = latest.actions.submitAnalysis(visitId)
    })
    await advance(10)
    await advance(0)
    const firstRun = latest.state.runsById[firstRunId]!
    expect(firstRun.status).toBe('failed')

    let retryId = ''
    act(() => {
      retryId = latest.actions.retryAnalysis(visitId)
    })
    const retry = latest.state.runsById[retryId]!
    expect(retry).toMatchObject({
      status: 'queued',
      retryOfRunId: firstRunId,
      binding: firstRun.binding,
    })
    expect(retry.binding).toEqual(firstRun.binding)

    const capture = selectCurrentCapture(latest.state, visitId)!
    vault.delete(capture.mediaHandle)
    await advance(10)
    expect(simulationRunner).toHaveBeenCalledTimes(1)
    expect(latest.state.runsById[retryId]).toMatchObject({
      status: 'failed',
      failure: { code: 'ANALYSIS_FAILED', field: 'mediaHandle' },
    })
  })

  it('rejects retry after replacement and submits a fresh run bound only to the new capture', async () => {
    const simulationRunner = vi
      .fn()
      .mockRejectedValueOnce(new Error('first simulation failed'))
      .mockImplementation(validOutput)
    renderProvider({
      runtime: createRuntime(),
      prepareCapture: createCapturePreparer(),
      simulationRunner,
    })
    const { visitId } = await createReadyVisit()
    let failedRunId = ''
    act(() => {
      failedRunId = latest.actions.submitAnalysis(visitId)
    })
    await advance(0)

    const failedRun = latest.state.runsById[failedRunId]!
    const oldBinding = failedRun.binding
    expect(failedRun.status).toBe('failed')

    const replacementId = await attachCapture(
      visitId,
      makeMedia('replacement capture', 'b'.repeat(64)),
    )
    const replacement = selectCurrentCapture(latest.state, visitId)!
    expect(replacement).toMatchObject({
      id: replacementId,
      version: oldBinding.captureVersion + 1,
      sha256: 'b'.repeat(64),
      status: 'current',
    })
    expect(replacement.mediaHandle).not.toBe(oldBinding.mediaHandle)

    expectProviderFailure(
      () => latest.actions.retryAnalysis(visitId),
      'INVALID_RETRY_BINDING',
    )
    expect(latest.state.runOrder).toEqual([failedRunId])
    expect(simulationRunner).toHaveBeenCalledOnce()

    confirmAllQuality(visitId)
    let freshRunId = ''
    act(() => {
      freshRunId = latest.actions.submitAnalysis(visitId)
    })
    const freshRun = latest.state.runsById[freshRunId]!
    expect(freshRun.retryOfRunId).toBeUndefined()
    expect(freshRun.binding).toEqual({
      patientId: replacement.patientId,
      visitId: replacement.visitId,
      captureId: replacement.id,
      captureVersion: replacement.version,
      captureSha256: replacement.sha256,
      mediaHandle: replacement.mediaHandle,
      authorizationRevision: 2,
      captureProtocol: replacement.captureProtocol,
    })
    expect(freshRun.binding).not.toEqual(oldBinding)

    await advance(0)
    expect(latest.state.runsById[freshRunId]?.status).toBe('succeeded')
    expect(simulationRunner).toHaveBeenCalledTimes(2)
  })
})

describe('review, reset, and lifecycle cleanup', () => {
  async function completeResult() {
    const { visitId } = await createReadyVisit()
    act(() => {
      latest.actions.submitAnalysis(visitId)
    })
    await advance(0)
    return visitId
  }

  it('records Reviewed or Repeat photo and requires a repeat reason', async () => {
    renderProvider({
      runtime: createRuntime(),
      prepareCapture: createCapturePreparer(),
      simulationRunner: validOutput,
    })
    const visitId = await completeResult()

    expectProviderFailure(
      () =>
        latest.actions.completeReview(
          visitId,
          'repeat_photo',
          '   ',
        ),
      'REPEAT_REASON_REQUIRED',
    )
    act(() => {
      latest.actions.completeReview(
        visitId,
        'repeat_photo',
        '  Orientation needs another photograph.  ',
      )
    })
    expect(selectCurrentReview(latest.state, visitId)).toMatchObject({
      decision: 'repeat_photo',
      note: 'Orientation needs another photograph.',
    })
  })

  it('cleans requested-retake media and makes the old result stale only after replacement', async () => {
    const objectUrls = createObjectUrlApi()
    renderProvider({
      runtime: createRuntime(),
      mediaVault: new SessionMediaVault(objectUrls),
      prepareCapture: createCapturePreparer(),
      simulationRunner: validOutput,
    })
    const visitId = await completeResult()
    const oldResult = selectCurrentResult(latest.state, visitId)!
    act(() => {
      latest.actions.completeReview(
        visitId,
        'repeat_photo',
        'Repeat for orientation.',
      )
      latest.actions.requestRetake(visitId)
    })
    expect(latest.actions.getCapturePreviewUrl(visitId)).toBeUndefined()
    expect(latest.state.resultsById[oldResult.id]?.freshness).toBe(
      'current',
    )

    await attachCapture(
      visitId,
      makeMedia('replacement', 'b'.repeat(64)),
    )
    expect(latest.state.resultsById[oldResult.id]?.freshness).toBe(
      'stale',
    )
    expect(selectCurrentCapture(latest.state, visitId)?.version).toBe(2)
    expect(objectUrls.revokeObjectURL).toHaveBeenCalledWith(
      'blob:patient-provider-1',
    )
  })

  it('reset cancels queued work, clears all state/media, and prevents late inference', async () => {
    const objectUrls = createObjectUrlApi()
    const runtime = createRuntime()
    const simulationRunner = vi.fn(validOutput)
    renderProvider({
      runtime,
      mediaVault: new SessionMediaVault(objectUrls),
      prepareCapture: createCapturePreparer(),
      simulationRunner,
      queueDelayMs: 50,
    })
    const { visitId } = await createReadyVisit()
    act(() => {
      latest.actions.submitAnalysis(visitId)
      latest.actions.resetSession()
    })

    expect(latest.state.patientOrder).toEqual([])
    expect(latest.state.runOrder).toEqual([])
    expect(objectUrls.revokeObjectURL).toHaveBeenCalledOnce()
    expect(runtime.reset).toHaveBeenCalledOnce()
    await advance(100)
    expect(simulationRunner).not.toHaveBeenCalled()
  })

  it('preserves state, media, and queued work when runtime reset throws', async () => {
    const resetFailure = new Error('runtime reset failed')
    const runtime = {
      ...createRuntime(),
      reset: vi.fn(() => {
        throw resetFailure
      }),
    }
    const objectUrls = createObjectUrlApi()
    const simulationRunner = vi.fn(validOutput)
    renderProvider({
      runtime,
      mediaVault: new SessionMediaVault(objectUrls),
      prepareCapture: createCapturePreparer(),
      simulationRunner,
      queueDelayMs: 50,
    })
    const { visitId } = await createReadyVisit()
    let runId = ''
    act(() => {
      runId = latest.actions.submitAnalysis(visitId)
    })
    const stateBeforeReset = latest.state
    const previewBeforeReset =
      latest.actions.getCapturePreviewUrl(visitId)

    expect(() => {
      act(() => latest.actions.resetSession())
    }).toThrow(resetFailure)
    expect(runtime.reset).toHaveBeenCalledOnce()
    expect(latest.state).toBe(stateBeforeReset)
    expect(latest.state.runsById[runId]?.status).toBe('queued')
    expect(latest.actions.getCapturePreviewUrl(visitId)).toBe(
      previewBeforeReset,
    )
    expect(objectUrls.revokeObjectURL).not.toHaveBeenCalled()

    await advance(50)
    expect(simulationRunner).toHaveBeenCalledOnce()
    expect(latest.state.runsById[runId]?.status).toBe('succeeded')
    expect(selectCurrentResult(latest.state, visitId)?.runId).toBe(
      runId,
    )
  })

  it('reset invalidates an in-flight synthetic media load even when IDs are reused', async () => {
    let resolveMedia!: (media: Blob) => void
    const pendingMedia = new Promise<Blob>((resolve) => {
      resolveMedia = resolve
    })
    const objectUrls = createObjectUrlApi()
    renderProvider({
      runtime: createRuntime(),
      mediaVault: new SessionMediaVault(objectUrls),
      prepareCapture: createCapturePreparer(),
      loadSyntheticMedia: () => pendingMedia,
    })
    const first = createPatientAndVisit()
    let pendingCapture!: Promise<string>
    act(() => {
      pendingCapture = latest.actions.attachSyntheticCapture(
        first.visitId,
        approvedAssets[0].id,
      )
      latest.actions.resetSession()
      latest.actions.createPatient(patientInput())
    })

    resolveMedia(
      makeMedia('late synthetic media', approvedAssets[0].sha256),
    )
    await expectProviderFailureAsync(
      () => pendingCapture,
      'INVALID_CAPTURE',
    )
    expect(latest.state.captureOrder).toEqual([])
    expect(objectUrls.createObjectURL).not.toHaveBeenCalled()
  })

  it('unmount clears media and ignores a late async simulation completion', async () => {
    let resolveSimulation!: (output: PatientSimulationOutput) => void
    const pending = new Promise<PatientSimulationOutput>((resolve) => {
      resolveSimulation = resolve
    })
    const objectUrls = createObjectUrlApi()
    const simulationRunner = vi.fn(() => pending)
    const rendered = renderProvider({
      runtime: createRuntime(),
      mediaVault: new SessionMediaVault(objectUrls),
      prepareCapture: createCapturePreparer(),
      simulationRunner,
    })
    const { visitId } = await createReadyVisit()
    act(() => {
      latest.actions.submitAnalysis(visitId)
    })
    await advance(0)
    expect(selectCurrentRun(latest.state, visitId)?.status).toBe('running')
    const stateAtUnmount = latest.state

    rendered.unmount()
    expect(objectUrls.revokeObjectURL).toHaveBeenCalledOnce()
    resolveSimulation(
      validOutput(
        stateAtUnmount.runsById[stateAtUnmount.runOrder[0]]!.binding,
      ),
    )
    await advance(100)
    expect(stateAtUnmount.resultOrder).toEqual([])
    expect(
      stateAtUnmount.runsById[stateAtUnmount.runOrder[0]]?.status,
    ).toBe('running')
  })

  it('never writes patient workflow data to localStorage, sessionStorage, or IndexedDB', async () => {
    const storageSet = vi.spyOn(Storage.prototype, 'setItem')
    const storageRemove = vi.spyOn(Storage.prototype, 'removeItem')
    const storageClear = vi.spyOn(Storage.prototype, 'clear')
    const indexedDbOpen = vi.fn()
    vi.stubGlobal('indexedDB', { open: indexedDbOpen })
    renderProvider({
      runtime: createRuntime(),
      prepareCapture: createCapturePreparer(),
      simulationRunner: validOutput,
    })

    const visitId = await completeResult()
    act(() => {
      latest.actions.completeReview(visitId, 'reviewed', 'Reviewed.')
    })

    expect(storageSet).not.toHaveBeenCalled()
    expect(storageRemove).not.toHaveBeenCalled()
    expect(storageClear).not.toHaveBeenCalled()
    expect(indexedDbOpen).not.toHaveBeenCalled()
  })
})
