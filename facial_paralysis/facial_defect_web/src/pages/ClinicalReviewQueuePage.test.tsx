import { render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it } from 'vitest'
import { DEMO_PATIENT_RECORDS } from '../data/demoPatientRecords'
import { PatientWorkflowProvider } from '../patientWorkflow/PatientWorkflowProvider'
import {
  createInitialPatientWorkflowState,
  patientWorkflowReducer,
} from '../patientWorkflow/reducer'
import type {
  CaptureQualityChecks,
  PatientFaceRegistration,
  PatientRecord,
  PatientRunBinding,
  PatientWorkflowState,
} from '../patientWorkflow/types'
import {
  createAuthorizationSnapshotId,
  createCaptureAssetId,
  createPatientResultId,
  createPatientReviewId,
  createPatientRunId,
  createPatientVisitId,
  createSessionMediaHandle,
} from '../patientWorkflow/validation'
import { ClinicalReviewQueuePage } from './ClinicalReviewQueuePage'

const COMPLETE_QUALITY: CaptureQualityChecks = {
  faceVisibleAndCentered: true,
  focusLightingAndOcclusionAcceptable: true,
  orientationConfirmed: true,
  authorizationDocumented: true,
}

function faceRegistrationFor(
  binding: PatientRunBinding,
): PatientFaceRegistration {
  const points = [
    { x: 0.3, y: 0.3 },
    { x: 0.5, y: 0.7 },
    { x: 0.7, y: 0.3 },
  ] as const
  return {
    schemaVersion: 'patient-face-registration/1',
    source: 'on_device_face_landmarks',
    coordinateSpace: 'decoded_image_normalized_v1',
    captureSha256: binding.captureSha256,
    sourceWidth: 1_024,
    sourceHeight: 1_024,
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
}

function addResult(
  state: PatientWorkflowState,
  patient: PatientRecord,
  suffix: string,
  reviewed: boolean,
): PatientWorkflowState {
  const visitId = createPatientVisitId(`visit-queue-${suffix}`)
  const captureId = createCaptureAssetId(`capture-queue-${suffix}`)
  const authorizationId = createAuthorizationSnapshotId(
    `authorization-queue-${suffix}`,
  )
  const runId = createPatientRunId(`run-queue-${suffix}`)
  const resultId = createPatientResultId(`result-queue-${suffix}`)
  const mediaHandle = createSessionMediaHandle(`media_queue_${suffix}`)
  const binding: PatientRunBinding = {
    patientId: patient.id,
    visitId,
    captureId,
    captureVersion: 1,
    captureSha256: suffix.repeat(64).slice(0, 64),
    mediaHandle,
    authorizationRevision: 1,
    captureProtocol: 'frontal_relaxed_non_mirrored_v1',
  }

  let next = patientWorkflowReducer(state, {
    type: 'visit/create',
    visit: {
      id: visitId,
      patientId: patient.id,
      timepoint: suffix === 'a' ? 'preoperative' : 'follow_up',
      visitDate: suffix === 'a' ? '2026-07-20' : '2026-07-21',
      createdAt: `2026-07-2${suffix === 'a' ? '0' : '1'}T10:00:00.000Z`,
    },
    trustedToday: '2026-07-27',
  })
  next = patientWorkflowReducer(next, {
    type: 'capture/add',
    capture: {
      id: captureId,
      patientId: patient.id,
      visitId,
      version: 1,
      status: 'current',
      source: 'upload',
      mediaHandle,
      sha256: binding.captureSha256,
      mimeType: 'image/png',
      sizeBytes: 1_024,
      width: 1_024,
      height: 1_024,
      captureProtocol: 'frontal_relaxed_non_mirrored_v1',
      qualityChecks: COMPLETE_QUALITY,
      capturedAt: '2026-07-27T10:00:00.000Z',
      qualityConfirmedAt: '2026-07-27T10:01:00.000Z',
    },
  })
  next = patientWorkflowReducer(next, {
    type: 'authorization/record',
    authorization: {
      id: authorizationId,
      patientId: patient.id,
      visitId,
      revision: 1,
      status: 'documented',
      recordedAt: '2026-07-27T10:01:00.000Z',
    },
  })
  next = patientWorkflowReducer(next, {
    type: 'run/create',
    run: {
      id: runId,
      status: 'queued',
      binding,
      createdAt: '2026-07-27T10:02:00.000Z',
    },
  })
  next = patientWorkflowReducer(next, {
    type: 'run/status/set',
    runId,
    status: 'running',
  })
  next = patientWorkflowReducer(next, {
    type: 'run/status/set',
    runId,
    status: 'succeeded',
  })
  next = patientWorkflowReducer(next, {
    type: 'result/record',
    result: {
      id: resultId,
      runId,
      binding,
      freshness: 'current',
      createdAt: '2026-07-27T10:03:00.000Z',
      faceRegistration: faceRegistrationFor(binding),
      output: {
        origin: 'workflow_simulation',
        points: [
          { x: 0.4, y: 0.5, intensity: 0.8, radius: 0.1 },
        ],
      },
    },
  })

  return reviewed
    ? patientWorkflowReducer(next, {
        type: 'review/record',
        review: {
          id: createPatientReviewId(`review-queue-${suffix}`),
          patientId: patient.id,
          visitId,
          resultId,
          captureId,
          decision: 'reviewed',
          completedAt: '2026-07-27T10:04:00.000Z',
        },
      })
    : next
}

function queueState(): PatientWorkflowState {
  let state = createInitialPatientWorkflowState(
    DEMO_PATIENT_RECORDS.slice(0, 2),
    '2026-07-27',
  )
  state = addResult(state, DEMO_PATIENT_RECORDS[0]!, 'a', false)
  return addResult(state, DEMO_PATIENT_RECORDS[1]!, 'b', true)
}

function renderQueue(initialState = queueState()) {
  return render(
    <MemoryRouter>
      <PatientWorkflowProvider initialState={initialState}>
        <ClinicalReviewQueuePage />
      </PatientWorkflowProvider>
    </MemoryRouter>,
  )
}

describe('ClinicalReviewQueuePage', () => {
  it('lists only unreviewed current results and links each row back to the same visit', () => {
    renderQueue()

    expect(
      screen.getByRole('heading', { name: 'Reviews', level: 1 }),
    ).toBeVisible()
    const rows = screen.getAllByRole('listitem')
    expect(rows).toHaveLength(1)
    expect(
      within(rows[0]!).getByText(DEMO_PATIENT_RECORDS[0]!.displayName),
    ).toBeVisible()
    expect(
      within(rows[0]!).getByText(DEMO_PATIENT_RECORDS[0]!.recordNumber),
    ).toBeVisible()
    expect(within(rows[0]!).getByText('Preoperative')).toBeVisible()
    expect(
      within(rows[0]!).getByRole('link', { name: 'Review result' }),
    ).toHaveAttribute(
      'href',
      `/patients/${DEMO_PATIENT_RECORDS[0]!.id}/visits/visit-queue-a`,
    )
    expect(document.body).not.toHaveTextContent(
      /run-queue|result-queue|capture-queue|[ab]{64}/i,
    )
    expect(
      screen.queryByText(DEMO_PATIENT_RECORDS[1]!.displayName),
    ).not.toBeInTheDocument()
  })

  it('shows a compact empty state with a direct return to patients', () => {
    const reviewedState = addResult(
      createInitialPatientWorkflowState(
        [DEMO_PATIENT_RECORDS[0]!],
        '2026-07-27',
      ),
      DEMO_PATIENT_RECORDS[0]!,
      'a',
      true,
    )
    renderQueue(reviewedState)

    const emptyState = screen.getByRole('status', {
      name: 'Review queue status',
    })
    expect(emptyState).toHaveTextContent('No results are waiting for review.')
    expect(
      within(emptyState).getByRole('link', { name: 'View patients' }),
    ).toHaveAttribute('href', '/patients')
    expect(screen.queryByRole('listitem')).not.toBeInTheDocument()
  })
})
