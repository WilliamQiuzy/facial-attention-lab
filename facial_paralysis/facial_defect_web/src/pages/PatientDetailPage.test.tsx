import { render, screen, within } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { describe, expect, it, vi } from 'vitest'
import { DEMO_PATIENT_RECORDS } from '../data/demoPatientRecords'
import { PatientWorkflowProvider } from '../patientWorkflow/PatientWorkflowProvider'
import { SessionMediaVault } from '../patientWorkflow/SessionMediaVault'
import {
  createInitialPatientWorkflowState,
  patientWorkflowReducer,
} from '../patientWorkflow/reducer'
import type {
  AuthorizationSnapshot,
  CaptureAsset,
  PatientFaceRegistration,
  PatientResult,
  PatientRun,
  PatientRunBinding,
  PatientVisit,
  PatientWorkflowState,
} from '../patientWorkflow/types'
import {
  createAuthorizationSnapshotId,
  createCaptureAssetId,
  createPatientResultId,
  createPatientRunId,
  createPatientVisitId,
  createSessionMediaHandle,
} from '../patientWorkflow/validation'
import { PatientDetailPage } from './PatientDetailPage'

function detailState(): PatientWorkflowState {
  let state = createInitialPatientWorkflowState(
    [DEMO_PATIENT_RECORDS[0]!],
    '2026-07-27',
  )
  state = patientWorkflowReducer(state, {
    type: 'visit/create',
    visit: {
      id: createPatientVisitId('visit-detail-later'),
      patientId: DEMO_PATIENT_RECORDS[0]!.id,
      timepoint: 'follow_up',
      visitDate: '2026-06-20',
      createdAt: '2026-06-20T09:00:00.000Z',
    },
    trustedToday: '2026-07-27',
  })
  return patientWorkflowReducer(state, {
    type: 'visit/create',
    visit: {
      id: createPatientVisitId('visit-detail-earlier'),
      patientId: DEMO_PATIENT_RECORDS[0]!.id,
      timepoint: 'preoperative',
      visitDate: '2026-01-05',
      createdAt: '2026-01-05T09:00:00.000Z',
    },
    trustedToday: '2026-07-27',
  })
}

function renderPage(
  path: string,
  initialState: PatientWorkflowState = detailState(),
  mediaVault?: SessionMediaVault,
) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <PatientWorkflowProvider
        initialState={initialState}
        mediaVault={mediaVault}
      >
        <Routes>
          <Route
            path="/patients/:patientId"
            element={<PatientDetailPage />}
          />
        </Routes>
      </PatientWorkflowProvider>
    </MemoryRouter>,
  )
}

function completeVisitEntities(
  visit: PatientVisit,
  suffix: string,
) {
  const capture: CaptureAsset = {
    id: createCaptureAssetId(`capture-detail-${suffix}`),
    patientId: visit.patientId,
    visitId: visit.id,
    version: 1,
    status: 'current',
    source: 'upload',
    mediaHandle: createSessionMediaHandle(`media_detail_${suffix}`),
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
    capturedAt: `${visit.visitDate}T10:01:00.000Z`,
    qualityConfirmedAt: `${visit.visitDate}T10:02:00.000Z`,
  }
  const authorization: AuthorizationSnapshot = {
    id: createAuthorizationSnapshotId(
      `authorization-detail-${suffix}`,
    ),
    patientId: visit.patientId,
    visitId: visit.id,
    revision: 1,
    status: 'documented',
    recordedAt: `${visit.visitDate}T10:02:00.000Z`,
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
    id: createPatientRunId(`run-detail-${suffix}`),
    status: 'succeeded',
    binding,
    createdAt: `${visit.visitDate}T10:03:00.000Z`,
  }
  const points = [
    { x: 0.3, y: 0.3 },
    { x: 0.5, y: 0.75 },
    { x: 0.7, y: 0.3 },
  ] as const
  const faceRegistration: PatientFaceRegistration = {
    schemaVersion: 'patient-face-registration/1',
    source: 'on_device_face_landmarks',
    coordinateSpace: 'decoded_image_normalized_v1',
    captureSha256: capture.sha256,
    sourceWidth: capture.width,
    sourceHeight: capture.height,
    captureProtocol: capture.captureProtocol,
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
    id: createPatientResultId(`result-detail-${suffix}`),
    runId: run.id,
    binding,
    freshness: 'current',
    createdAt: `${visit.visitDate}T10:04:00.000Z`,
    faceRegistration,
    output: {
      origin: 'workflow_simulation',
      points: [
        { x: 0.42, y: 0.5, intensity: 0.85, radius: 0.12 },
      ],
    },
  }
  return { capture, authorization, run, result }
}

function readyComparisonState(): {
  readonly state: PatientWorkflowState
  readonly preoperative: ReturnType<typeof completeVisitEntities>
  readonly postoperative: ReturnType<typeof completeVisitEntities>
} {
  const patient = DEMO_PATIENT_RECORDS[0]!
  const preVisit: PatientVisit = {
    id: createPatientVisitId('visit-detail-pre'),
    patientId: patient.id,
    timepoint: 'preoperative',
    visitDate: '2026-07-20',
    createdAt: '2026-07-20T10:00:00.000Z',
  }
  const postVisit: PatientVisit = {
    id: createPatientVisitId('visit-detail-post'),
    patientId: patient.id,
    timepoint: 'postoperative',
    visitDate: '2026-07-25',
    createdAt: '2026-07-25T10:00:00.000Z',
  }
  const preoperative = completeVisitEntities(preVisit, 'pre')
  const postoperative = completeVisitEntities(postVisit, 'post')
  const base = createInitialPatientWorkflowState(
    [patient],
    '2026-07-27',
  )
  const entities = [preoperative, postoperative]
  return {
    state: {
      ...base,
      visitsById: {
        [preVisit.id]: preVisit,
        [postVisit.id]: postVisit,
      },
      visitOrder: [preVisit.id, postVisit.id],
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
    },
    preoperative,
    postoperative,
  }
}

function mediaVaultWith(
  ...captures: readonly CaptureAsset[]
): SessionMediaVault {
  let index = 0
  const vault = new SessionMediaVault({
    createObjectURL: vi.fn(() => `blob:detail-${index++}`),
    revokeObjectURL: vi.fn(),
  })
  for (const capture of captures) {
    vault.set(capture.mediaHandle, new Blob(['patient-photo']))
  }
  return vault
}

function reverseOrderCapturedState(): {
  readonly state: PatientWorkflowState
  readonly postoperative: ReturnType<typeof completeVisitEntities>
  readonly preoperative: ReturnType<typeof completeVisitEntities>
} {
  const patient = DEMO_PATIENT_RECORDS[0]!
  const postoperativeVisit: PatientVisit = {
    id: createPatientVisitId('visit-detail-post-first'),
    patientId: patient.id,
    timepoint: 'postoperative',
    visitDate: '2026-07-20',
    createdAt: '2026-07-20T10:00:00.000Z',
  }
  const preoperativeVisit: PatientVisit = {
    id: createPatientVisitId('visit-detail-pre-second'),
    patientId: patient.id,
    timepoint: 'preoperative',
    visitDate: '2026-07-21',
    createdAt: '2026-07-21T10:00:00.000Z',
  }
  const postoperative = completeVisitEntities(
    postoperativeVisit,
    'post-first',
  )
  const preoperative = completeVisitEntities(
    preoperativeVisit,
    'pre-second',
  )
  const base = createInitialPatientWorkflowState(
    [patient],
    '2026-07-27',
  )

  return {
    state: {
      ...base,
      visitsById: {
        [postoperativeVisit.id]: postoperativeVisit,
        [preoperativeVisit.id]: preoperativeVisit,
      },
      visitOrder: [postoperativeVisit.id, preoperativeVisit.id],
      capturesById: {
        [postoperative.capture.id]: postoperative.capture,
        [preoperative.capture.id]: preoperative.capture,
      },
      captureOrder: [
        postoperative.capture.id,
        preoperative.capture.id,
      ],
    },
    postoperative,
    preoperative,
  }
}

describe('PatientDetailPage', () => {
  it('keeps identity visible and lists visits in chronological order with one next action each', () => {
    const patient = DEMO_PATIENT_RECORDS[0]!
    const view = renderPage(`/patients/${patient.id}`)
    expect(view.container.querySelector('main')).toBeNull()

    expect(
      screen.getByRole('heading', { name: patient.displayName, level: 1 }),
    ).toBeVisible()
    const identity = screen.getByRole('region', {
      name: 'Patient identity',
    })
    expect(within(identity).getByText(patient.recordNumber)).toBeVisible()
    expect(within(identity).getByText('Mar 14, 1962')).toBeVisible()
    expect(within(identity).getByText(patient.carePathway)).toBeVisible()
    expect(within(identity).getByText('Sample record')).toBeVisible()
    expect(
      screen.getByRole('link', { name: 'Add postoperative visit' }),
    ).toHaveAttribute('href', `/patients/${patient.id}/visits/new`)

    const timeline = screen.getByRole('region', {
      name: 'Visit timeline',
    })
    const visits = within(timeline).getAllByRole('listitem')
    expect(visits).toHaveLength(2)
    expect(within(visits[0]!).getByText('Preoperative')).toBeVisible()
    expect(within(visits[0]!).getByText('Jan 5, 2026')).toBeVisible()
    expect(within(visits[1]!).getByText('Follow-up')).toBeVisible()
    expect(within(visits[1]!).getByText('Jun 20, 2026')).toBeVisible()
    for (const visit of visits) {
      expect(within(visit).getByText('Photo needed')).toBeVisible()
      expect(
        within(visit).getByRole('link', { name: 'Add photo' }),
      ).toBeVisible()
    }
  })

  it('does not invent a comparison for a patient with no visits', () => {
    const patient = DEMO_PATIENT_RECORDS[0]!
    const emptyState = createInitialPatientWorkflowState(
      [patient],
      '2026-07-27',
    )
    renderPage(`/patients/${patient.id}`, emptyState)

    expect(
      screen.queryByRole('region', {
        name: 'Patient before and after comparison',
      }),
    ).not.toBeInTheDocument()
    expect(
      screen.queryByRole('region', {
        name: 'Before and after readiness',
      }),
    ).not.toBeInTheDocument()
    expect(
      screen.getByRole('link', { name: 'Add photo visit' }),
    ).toBeVisible()
  })

  it('shows the missing postoperative step instead of a simulated comparison', () => {
    const patient = DEMO_PATIENT_RECORDS[0]!
    renderPage(`/patients/${patient.id}`)

    const readiness = screen.getByRole('region', {
      name: 'Before and after readiness',
    })
    expect(
      within(readiness).getByText('Postoperative visit needed'),
    ).toBeVisible()
    expect(
      screen.queryByRole('link', { name: 'Add photo visit' }),
    ).not.toBeInTheDocument()
    expect(
      screen.queryByRole('region', {
        name: 'Patient before and after comparison',
      }),
    ).not.toBeInTheDocument()
  })

  it('shows the captured photograph when only a postoperative visit exists', () => {
    const patient = DEMO_PATIENT_RECORDS[0]!
    const reverse = reverseOrderCapturedState()
    const state: PatientWorkflowState = {
      ...reverse.state,
      visitsById: {
        [reverse.postoperative.capture.visitId]:
          reverse.state.visitsById[
            reverse.postoperative.capture.visitId
          ],
      },
      visitOrder: [reverse.postoperative.capture.visitId],
      capturesById: {
        [reverse.postoperative.capture.id]:
          reverse.postoperative.capture,
      },
      captureOrder: [reverse.postoperative.capture.id],
    }
    const vault = mediaVaultWith(reverse.postoperative.capture)
    renderPage(`/patients/${patient.id}`, state, vault)

    const timeline = screen.getByRole('region', {
      name: 'Visit timeline',
    })
    expect(
      within(timeline).getByRole('img', {
        name: 'Postoperative visit photograph',
      }),
    ).toHaveAttribute('src', 'blob:detail-0')
    expect(
      screen.queryByRole('region', {
        name: 'Patient before and after comparison',
      }),
    ).not.toBeInTheDocument()
  })

  it('keeps the postoperative photograph after preoperative is added second', () => {
    const patient = DEMO_PATIENT_RECORDS[0]!
    const reverse = reverseOrderCapturedState()
    const vault = mediaVaultWith(
      reverse.postoperative.capture,
      reverse.preoperative.capture,
    )
    renderPage(`/patients/${patient.id}`, reverse.state, vault)

    const timeline = screen.getByRole('region', {
      name: 'Visit timeline',
    })
    expect(within(timeline).getAllByRole('img')).toHaveLength(2)
    expect(
      within(timeline).getByRole('img', {
        name: 'Postoperative visit photograph',
      }),
    ).toHaveAttribute('src', 'blob:detail-0')
    expect(
      within(timeline).getByRole('img', {
        name: 'Preoperative visit photograph',
      }),
    ).toHaveAttribute('src', 'blob:detail-1')
  })

  it('renders only the current patient media when both results are ready', () => {
    const patient = DEMO_PATIENT_RECORDS[0]!
    const ready = readyComparisonState()
    const vault = mediaVaultWith(
      ready.preoperative.capture,
      ready.postoperative.capture,
    )
    renderPage(`/patients/${patient.id}`, ready.state, vault)

    const comparison = screen.getByRole('region', {
      name: 'Patient before and after comparison',
    })
    expect(
      within(comparison).getByRole('img', {
        name: 'Preoperative patient photograph with illustrative attention',
      }),
    ).toHaveAttribute('src', 'blob:detail-0')
    expect(
      within(comparison).getByRole('img', {
        name: 'Postoperative patient photograph with illustrative attention',
      }),
    ).toHaveAttribute('src', 'blob:detail-1')
    expect(
      screen.getByRole('link', { name: 'Add another visit' }),
    ).toHaveAttribute('href', `/patients/${patient.id}/visits/new`)
    expect(
      screen.queryByRole('region', {
        name: 'Before and after readiness',
      }),
    ).not.toBeInTheDocument()
  })

  it('fails closed when one ready capture is unavailable in session media', () => {
    const patient = DEMO_PATIENT_RECORDS[0]!
    const ready = readyComparisonState()
    const vault = mediaVaultWith(ready.preoperative.capture)
    renderPage(`/patients/${patient.id}`, ready.state, vault)

    const recovery = screen.getByRole('region', {
      name: 'Comparison media unavailable',
    })
    expect(
      within(recovery).getByText(
        'Both current photos are required before this comparison can be shown.',
      ),
    ).toBeVisible()
    expect(within(recovery).getAllByRole('link')).toHaveLength(2)
    expect(
      screen.queryByRole('region', {
        name: 'Patient before and after comparison',
      }),
    ).not.toBeInTheDocument()
  })

  it('lets a newer incomplete preoperative visit replace the older ready one', () => {
    const patient = DEMO_PATIENT_RECORDS[0]!
    const ready = readyComparisonState()
    const newerVisit: PatientVisit = {
      id: createPatientVisitId('visit-detail-pre-new'),
      patientId: patient.id,
      timepoint: 'preoperative',
      visitDate: '2026-07-27',
      createdAt: '2026-07-27T10:00:00.000Z',
    }
    const state: PatientWorkflowState = {
      ...ready.state,
      visitsById: {
        ...ready.state.visitsById,
        [newerVisit.id]: newerVisit,
      },
      visitOrder: [...ready.state.visitOrder, newerVisit.id],
    }
    renderPage(`/patients/${patient.id}`, state)

    const readiness = screen.getByRole('region', {
      name: 'Before and after readiness',
    })
    expect(
      within(readiness).getByRole('link', {
        name: 'Add preoperative photo',
      }),
    ).toHaveAttribute(
      'href',
      `/patients/${patient.id}/visits/${newerVisit.id}`,
    )
    expect(
      screen.queryByRole('region', {
        name: 'Patient before and after comparison',
      }),
    ).not.toBeInTheDocument()
  })

  it('fails closed for an unknown patient', () => {
    renderPage('/patients/patient-does-not-exist')

    expect(
      screen.getByRole('heading', {
        name: 'Patient record unavailable',
        level: 1,
      }),
    ).toBeVisible()
    expect(
      screen.getByText(
        'This patient record is not available in the current session.',
      ),
    ).toBeVisible()
    expect(
      screen.getByRole('link', { name: 'Back to patients' }),
    ).toHaveAttribute('href', '/patients')
    expect(
      screen.queryByRole('link', { name: 'Add photo visit' }),
    ).not.toBeInTheDocument()
  })
})
