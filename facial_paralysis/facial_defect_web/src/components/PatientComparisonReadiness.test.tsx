import { render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it } from 'vitest'
import { DEMO_PATIENT_RECORDS } from '../data/demoPatientRecords'
import { createInitialPatientWorkflowState } from '../patientWorkflow/reducer'
import type {
  AuthorizationSnapshot,
  CaptureAsset,
  PatientFaceRegistration,
  PatientComparisonState,
  PatientResult,
  PatientRun,
  PatientRunBinding,
  PatientVisit,
  PatientWorkflowState,
} from '../patientWorkflow/types'
import {
  createAuthorizationSnapshotId,
  createCaptureAssetId,
  createPatientRunId,
  createPatientResultId,
  createPatientVisitId,
  createSessionMediaHandle,
} from '../patientWorkflow/validation'
import { PatientComparisonReadiness } from './PatientComparisonReadiness'

const PATIENT = DEMO_PATIENT_RECORDS[0]!

function visits() {
  const preoperative: PatientVisit = {
    id: createPatientVisitId('visit-readiness-pre'),
    patientId: PATIENT.id,
    timepoint: 'preoperative',
    visitDate: '2026-07-20',
    createdAt: '2026-07-20T10:00:00.000Z',
  }
  const postoperative: PatientVisit = {
    id: createPatientVisitId('visit-readiness-post'),
    patientId: PATIENT.id,
    timepoint: 'postoperative',
    visitDate: '2026-07-25',
    createdAt: '2026-07-25T10:00:00.000Z',
  }
  return { preoperative, postoperative }
}

function workflowState(
  patientVisits: readonly PatientVisit[],
  captures: readonly CaptureAsset[] = [],
): PatientWorkflowState {
  const state = createInitialPatientWorkflowState(
    [PATIENT],
    '2026-07-27',
  )
  return {
    ...state,
    visitsById: Object.fromEntries(
      patientVisits.map((visit) => [visit.id, visit]),
    ),
    visitOrder: patientVisits.map((visit) => visit.id),
    capturesById: Object.fromEntries(
      captures.map((capture) => [capture.id, capture]),
    ),
    captureOrder: captures.map((capture) => capture.id),
  }
}

function capture(
  item: PatientVisit,
  suffix: string,
  qualityComplete = false,
): CaptureAsset {
  return {
    id: createCaptureAssetId(`capture-readiness-${suffix}`),
    patientId: item.patientId,
    visitId: item.id,
    version: 1,
    status: 'current',
    source: 'upload',
    mediaHandle: createSessionMediaHandle(`media_readiness_${suffix}`),
    sha256: suffix.padEnd(64, 'a').slice(0, 64),
    mimeType: 'image/png',
    sizeBytes: 1_024,
    width: 1_024,
    height: 1_024,
    captureProtocol: 'frontal_relaxed_non_mirrored_v1',
    qualityChecks: {
      faceVisibleAndCentered: qualityComplete,
      focusLightingAndOcclusionAcceptable: qualityComplete,
      orientationConfirmed: qualityComplete,
      authorizationDocumented: qualityComplete,
    },
    capturedAt: '2026-07-27T10:00:00.000Z',
    qualityConfirmedAt: qualityComplete
      ? '2026-07-27T10:01:00.000Z'
      : undefined,
  }
}

function resultPendingState(
  runStatus?: PatientRun['status'],
): {
  readonly comparison: Extract<
    PatientComparisonState,
    { readonly phase: 'needs_results' }
  >
  readonly state: PatientWorkflowState
} {
  const pair = visits()
  const preCapture = capture(pair.preoperative, 'pre-ready', true)
  const postCapture = capture(pair.postoperative, 'post-ready', true)
  const authorizations: readonly AuthorizationSnapshot[] = [
    {
      id: createAuthorizationSnapshotId('authorization-readiness-pre'),
      patientId: PATIENT.id,
      visitId: pair.preoperative.id,
      revision: 1,
      status: 'documented',
      recordedAt: '2026-07-27T10:01:00.000Z',
    },
    {
      id: createAuthorizationSnapshotId('authorization-readiness-post'),
      patientId: PATIENT.id,
      visitId: pair.postoperative.id,
      revision: 1,
      status: 'documented',
      recordedAt: '2026-07-27T10:01:00.000Z',
    },
  ]
  const binding: PatientRunBinding = {
    patientId: PATIENT.id,
    visitId: pair.postoperative.id,
    captureId: postCapture.id,
    captureVersion: postCapture.version,
    captureSha256: postCapture.sha256,
    mediaHandle: postCapture.mediaHandle,
    authorizationRevision: 1,
    captureProtocol: postCapture.captureProtocol,
  }
  const run: PatientRun | undefined = runStatus
    ? {
        id: createPatientRunId('run-readiness-post'),
        status: runStatus,
        binding,
        createdAt: '2026-07-27T10:02:00.000Z',
        failure:
          runStatus === 'failed'
            ? {
                code: 'ANALYSIS_FAILED',
                message: 'Illustrative failure for action coverage.',
              }
            : undefined,
      }
    : undefined
  const base = workflowState(
    [pair.preoperative, pair.postoperative],
    [preCapture, postCapture],
  )
  return {
    comparison: {
      phase: 'needs_results',
      pair: {
        preoperative: {
          visit: pair.preoperative,
          capture: preCapture,
        },
        postoperative: {
          visit: pair.postoperative,
          capture: postCapture,
        },
      },
      missingResults: ['postoperative'],
    },
    state: {
      ...base,
      authorizationsById: Object.fromEntries(
        authorizations.map((authorization) => [
          authorization.id,
          authorization,
        ]),
      ),
      authorizationOrder: authorizations.map(
        (authorization) => authorization.id,
      ),
      runsById: run ? { [run.id]: run } : {},
      runOrder: run ? [run.id] : [],
    },
  }
}

function preoperativeReviewPendingState(): PatientWorkflowState {
  const { preoperative } = visits()
  const preoperativeCapture = capture(
    preoperative,
    'pre-review-pending',
    true,
  )
  const authorization: AuthorizationSnapshot = {
    id: createAuthorizationSnapshotId(
      'authorization-readiness-pre-review-pending',
    ),
    patientId: PATIENT.id,
    visitId: preoperative.id,
    revision: 1,
    status: 'documented',
    recordedAt: '2026-07-27T10:01:00.000Z',
  }
  const binding: PatientRunBinding = {
    patientId: PATIENT.id,
    visitId: preoperative.id,
    captureId: preoperativeCapture.id,
    captureVersion: preoperativeCapture.version,
    captureSha256: preoperativeCapture.sha256,
    mediaHandle: preoperativeCapture.mediaHandle,
    authorizationRevision: authorization.revision,
    captureProtocol: preoperativeCapture.captureProtocol,
  }
  const run: PatientRun = {
    id: createPatientRunId('run-readiness-pre-review-pending'),
    status: 'succeeded',
    binding,
    createdAt: '2026-07-27T10:02:00.000Z',
  }
  const points = [
    { x: 0.25, y: 0.25 },
    { x: 0.5, y: 0.75 },
    { x: 0.75, y: 0.25 },
  ] as const
  const registration: PatientFaceRegistration = {
    schemaVersion: 'patient-face-registration/1',
    source: 'on_device_face_landmarks',
    coordinateSpace: 'decoded_image_normalized_v1',
    captureSha256: preoperativeCapture.sha256,
    sourceWidth: preoperativeCapture.width,
    sourceHeight: preoperativeCapture.height,
    captureProtocol: preoperativeCapture.captureProtocol,
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
    id: createPatientResultId('result-readiness-pre-review-pending'),
    runId: run.id,
    binding,
    freshness: 'current',
    createdAt: '2026-07-27T10:03:00.000Z',
    faceRegistration: registration,
    output: {
      origin: 'workflow_simulation',
      points: [
        { x: 0.42, y: 0.5, intensity: 0.85, radius: 0.12 },
      ],
    },
  }
  const base = workflowState(
    [preoperative],
    [preoperativeCapture],
  )
  return {
    ...base,
    authorizationsById: { [authorization.id]: authorization },
    authorizationOrder: [authorization.id],
    runsById: { [run.id]: run },
    runOrder: [run.id],
    resultsById: { [result.id]: result },
    resultOrder: [result.id],
  }
}

function renderReadiness(
  comparison: Exclude<
    PatientComparisonState,
    { readonly phase: 'no_visits' | 'ready' }
  >,
  state: PatientWorkflowState,
) {
  return render(
    <MemoryRouter>
      <PatientComparisonReadiness
        comparison={comparison}
        patientId={PATIENT.id}
        workflowState={state}
      />
    </MemoryRouter>,
  )
}

describe('PatientComparisonReadiness', () => {
  it('identifies a missing timepoint and presents one direct next step', () => {
    const { preoperative } = visits()
    renderReadiness(
      {
        phase: 'missing_timepoint',
        missing: ['postoperative'],
      },
      workflowState([preoperative]),
    )

    const panel = screen.getByRole('region', {
      name: 'Before and after readiness',
    })
    expect(within(panel).getByText('Postoperative visit needed')).toBeVisible()
    expect(
      within(panel).getByRole('link', {
        name: 'Add postoperative visit',
      }),
    ).toHaveAttribute('href', `/patients/${PATIENT.id}/visits/new`)
    expect(within(panel).getAllByRole('link')).toHaveLength(1)
  })

  it('requires review of an existing result before advancing to the missing postoperative visit', () => {
    renderReadiness(
      {
        phase: 'missing_timepoint',
        missing: ['postoperative'],
      },
      preoperativeReviewPendingState(),
    )

    const panel = screen.getByRole('region', {
      name: 'Before and after readiness',
    })
    expect(within(panel).getByText('Review needed')).toBeVisible()
    expect(
      within(panel).getByRole('link', {
        name: 'Review preoperative result',
      }),
    ).toHaveAttribute(
      'href',
      `/patients/${PATIENT.id}/visits/${visits().preoperative.id}`,
    )
    expect(
      within(panel).queryByRole('link', {
        name: 'Add postoperative visit',
      }),
    ).not.toBeInTheDocument()
    const preoperativeStatus = within(panel)
      .getByText('Preoperative')
      .closest('li')
    expect(
      preoperativeStatus?.querySelector(
        '.patient-comparison-readiness__marker--complete',
      ),
    ).not.toBeInTheDocument()
  })

  it('prioritizes the preoperative side when both photos are missing', () => {
    const pair = visits()
    renderReadiness(
      {
        phase: 'needs_photos',
        pair,
        missingPhotos: ['preoperative', 'postoperative'],
      },
      workflowState([pair.preoperative, pair.postoperative]),
    )

    const panel = screen.getByRole('region', {
      name: 'Before and after readiness',
    })
    expect(within(panel).getByText('Preoperative')).toBeVisible()
    expect(within(panel).getByText('Postoperative')).toBeVisible()
    expect(within(panel).getAllByText('Photo needed')).toHaveLength(2)
    expect(
      within(panel).getByRole('link', { name: 'Add preoperative photo' }),
    ).toHaveAttribute(
      'href',
      `/patients/${PATIENT.id}/visits/${pair.preoperative.id}`,
    )
    expect(within(panel).getAllByRole('link')).toHaveLength(1)

    for (const marker of panel.querySelectorAll(
      '.patient-comparison-readiness__marker',
    )) {
      expect(
        marker.querySelector(
          '.patient-comparison-readiness__marker-dot',
        ),
      ).toBeInTheDocument()
      expect(marker.textContent).toBe('')
    }
  })

  it('uses the visit workflow action for results that are not ready', () => {
    const pair = visits()
    const preCapture = capture(pair.preoperative, 'pre')
    const postCapture = capture(pair.postoperative, 'post')
    const state = workflowState(
      [pair.preoperative, pair.postoperative],
      [preCapture, postCapture],
    )
    renderReadiness(
      {
        phase: 'needs_results',
        pair: {
          preoperative: {
            visit: pair.preoperative,
            capture: preCapture,
          },
          postoperative: {
            visit: pair.postoperative,
            capture: postCapture,
          },
        },
        missingResults: ['preoperative', 'postoperative'],
      },
      state,
    )

    const panel = screen.getByRole('region', {
      name: 'Before and after readiness',
    })
    expect(
      within(panel).getByRole('link', { name: 'Confirm preoperative quality' }),
    ).toHaveAttribute(
      'href',
      `/patients/${PATIENT.id}/visits/${pair.preoperative.id}`,
    )
    expect(within(panel).getAllByRole('link')).toHaveLength(1)
  })

  it.each([
    [undefined, 'Run postoperative analysis'],
    ['running', 'View postoperative progress'],
    ['failed', 'Retry postoperative analysis'],
  ] as const)(
    'maps the postoperative %s state to one precise action',
    (runStatus, expectedLabel) => {
      const pending = resultPendingState(runStatus)
      renderReadiness(pending.comparison, pending.state)

      const panel = screen.getByRole('region', {
        name: 'Before and after readiness',
      })
      expect(
        within(panel).getByRole('link', { name: expectedLabel }),
      ).toHaveAttribute(
        'href',
        `/patients/${PATIENT.id}/visits/${pending.comparison.pair.postoperative.visit.id}`,
      )
      expect(within(panel).getAllByRole('link')).toHaveLength(1)
    },
  )
})
